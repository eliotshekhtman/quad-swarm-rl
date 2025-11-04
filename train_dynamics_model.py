#!/usr/bin/env python3
"""
Train a feed-forward neural network that predicts the next quadrotor state from the current state
and planner outputs using datasets produced by collect_dynamics_dataset.py.

The dataset must provide the fields: position, velocity, angular_velocity, rotation (flattened 3x3),
thrusts, delta_position, delta_velocity, delta_angular_velocity, and delta_rotation (flattened 3x3).

Place this file at: quad-swarm-rl/scripts/train_dynamics_model.py
"""

import argparse
import glob
import os
import sys
from datetime import datetime
from typing import Dict, Optional, Tuple

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Subset, TensorDataset
from tqdm import tqdm


REQUIRED_KEYS = (
    "position",
    "velocity",
    "angular_velocity",
    "rotation",
    "thrusts",
    "delta_position",
    "delta_velocity",
    "delta_angular_velocity",
    "delta_rotation",
)


def set_seed(seed: int) -> None:
    """Seed numpy and torch for reproducible splits."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_dataset(npz_path: str) -> Dict[str, np.ndarray]:
    """Load and validate the NPZ dataset."""
    data = np.load(npz_path)
    missing = [key for key in REQUIRED_KEYS if key not in data]
    if missing:
        raise KeyError(f"Dataset is missing required keys: {missing}")
    return {key: data[key] for key in REQUIRED_KEYS}


def assemble_features_targets(
    data: Dict[str, np.ndarray],
    delta_position_only: bool,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, slice]]:
    """Flatten and concatenate arrays into model inputs and targets."""
    def _ensure_2d(array: np.ndarray) -> np.ndarray:
        array = np.asarray(array)
        return array.reshape(array.shape[0], -1).astype(np.float32)

    position = _ensure_2d(data["position"])
    velocity = _ensure_2d(data["velocity"])
    angular_velocity = _ensure_2d(data["angular_velocity"])
    rotation = _ensure_2d(data["rotation"])
    thrusts = _ensure_2d(data["thrusts"])

    delta_position = _ensure_2d(data["delta_position"])
    delta_velocity = _ensure_2d(data["delta_velocity"])
    delta_angular_velocity = _ensure_2d(data["delta_angular_velocity"])
    delta_rotation = _ensure_2d(data["delta_rotation"])

    inputs = np.concatenate(
        [position, velocity, angular_velocity, rotation, thrusts],
        axis=1,
    )
    if delta_position_only:
        targets = delta_position
    else:
        targets = np.concatenate(
            [delta_position, delta_velocity, delta_angular_velocity, delta_rotation],
            axis=1,
        )

    if inputs.shape[0] != targets.shape[0]:
        raise ValueError(
            f"Input count ({inputs.shape[0]}) does not match target count ({targets.shape[0]})."
        )

    delta_slice = slice(0, delta_position.shape[1])
    return inputs, targets, {"delta_position_slice": delta_slice}


def build_model(input_dim: int, output_dim: int, hidden_size: int, num_layers: int) -> nn.Module:
    """Construct an MLP with the requested depth."""
    if num_layers < 1:
        raise ValueError("num_layers must be at least 1 (hidden layers).")

    layers = []
    in_dim = input_dim
    for _ in range(num_layers):
        layers.append(nn.Linear(in_dim, hidden_size))
        layers.append(nn.ReLU())
        in_dim = hidden_size
    layers.append(nn.Linear(in_dim, output_dim))
    return nn.Sequential(*layers)


def train(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: Optional[DataLoader],
    device: torch.device,
    epochs: int,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    *,
    start_epoch: int = 0,
    target_mean: Optional[np.ndarray] = None,
    target_std: Optional[np.ndarray] = None,
    delta_slice: Optional[slice] = None,
    log_fn=None,
    epoch_callback=None,
) -> Tuple[list, list]:
    """Execute the training loop and return loss histories."""
    model.to(device)
    train_losses = []
    val_losses = []
    if log_fn is None:
        log_fn = tqdm.write

    log_delta = (
        delta_slice is not None
        and target_mean is not None
        and target_std is not None
    )
    if log_delta:
        model_dtype = next(model.parameters()).dtype
        delta_mean_tensor = torch.from_numpy(target_mean[delta_slice]).to(device=device, dtype=model_dtype)
        delta_std_tensor = torch.from_numpy(target_std[delta_slice]).to(device=device, dtype=model_dtype)
        if delta_mean_tensor.dim() == 1:
            delta_mean_tensor = delta_mean_tensor.unsqueeze(0)
        if delta_std_tensor.dim() == 1:
            delta_std_tensor = delta_std_tensor.unsqueeze(0)
    else:
        delta_mean_tensor = delta_std_tensor = None

    for epoch in range(1, epochs + 1):
        actual_epoch = start_epoch + epoch
        model.train()
        epoch_loss = 0.0
        epoch_delta_sum = 0.0
        for batch_inputs, batch_targets in train_loader:
            batch_inputs = batch_inputs.to(device)
            batch_targets = batch_targets.to(device)

            optimizer.zero_grad(set_to_none=True)
            preds = model(batch_inputs)
            loss = criterion(preds, batch_targets)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item() * batch_inputs.size(0)

            if log_delta:
                preds_delta = preds[:, delta_slice]
                targets_delta = batch_targets[:, delta_slice]
                denorm_preds = preds_delta * delta_std_tensor + delta_mean_tensor
                denorm_targets = targets_delta * delta_std_tensor + delta_mean_tensor
                delta_loss = torch.mean((denorm_preds - denorm_targets) ** 2)
                epoch_delta_sum += delta_loss.item() * batch_inputs.size(0)

        epoch_loss /= len(train_loader.dataset)
        train_losses.append(epoch_loss)
        epoch_delta_avg = None
        if log_delta and len(train_loader.dataset) > 0:
            epoch_delta_avg = epoch_delta_sum / len(train_loader.dataset)

        val_loss_value = None
        val_delta_avg = None
        if val_loader is not None and len(val_loader.dataset) > 0:
            model.eval()
            val_loss = 0.0
            val_delta_sum = 0.0
            with torch.no_grad():
                for val_inputs, val_targets in val_loader:
                    val_inputs = val_inputs.to(device)
                    val_targets = val_targets.to(device)
                    preds = model(val_inputs)
                    loss = criterion(preds, val_targets)
                    val_loss += loss.item() * val_inputs.size(0)

                    if log_delta:
                        preds_delta = preds[:, delta_slice]
                        targets_delta = val_targets[:, delta_slice]
                        denorm_preds = preds_delta * delta_std_tensor + delta_mean_tensor
                        denorm_targets = targets_delta * delta_std_tensor + delta_mean_tensor
                        delta_loss = torch.mean((denorm_preds - denorm_targets) ** 2)
                        val_delta_sum += delta_loss.item() * val_inputs.size(0)
            val_loss /= len(val_loader.dataset)
            val_losses.append(val_loss)
            val_loss_value = val_loss
            if log_delta and len(val_loader.dataset) > 0:
                val_delta_avg = val_delta_sum / len(val_loader.dataset)
        else:
            val_loss_value = None

        log_parts = [f"Epoch {actual_epoch:03d}", f"train_loss={epoch_loss:.6f}"]
        if val_loss_value is not None:
            log_parts.append(f"val_loss={val_loss_value:.6f}")
        if log_delta and epoch_delta_avg is not None:
            log_parts.append(f"train_delta_pos_loss={epoch_delta_avg:.6f}")
            if val_delta_avg is not None:
                log_parts.append(f"val_delta_pos_loss={val_delta_avg:.6f}")
        log_fn(" | ".join(log_parts))

        if epoch_callback is not None:
            epoch_callback(actual_epoch, epoch_loss, val_loss_value)

    return train_losses, val_losses


def main():
    parser = argparse.ArgumentParser(description="Train a dynamics FFN on collected quadrotor data")
    parser.add_argument("--dataset_path", required=True,
                        help="Path to the NPZ dataset produced by collect_dynamics_dataset.py")
    parser.add_argument("--train_dir", default='train_dir',
                        help="Directory that will hold trained dynamics models")
    parser.add_argument("--experiment_name", required=True,
                        help="Subdirectory within train_dir to store checkpoints")
    parser.add_argument("--load_best", action="store_true",
                        help="When resuming an existing experiment, load the best checkpoint instead of the latest")
    parser.add_argument("--hidden_size", type=int, default=256,
                        help="Width of each hidden layer")
    parser.add_argument("--num_layers", type=int, default=3,
                        help="Number of hidden layers")
    parser.add_argument("--batch_size", type=int, default=1024,
                        help="Training batch size")
    parser.add_argument("--epochs", type=int, default=50,
                        help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=1e-3,
                        help="Learning rate for Adam optimizer")
    parser.add_argument("--val_fraction", type=float, default=0.1,
                        help="Fraction of samples reserved for validation (must be > 0)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for the train/val split")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu",
                        help="Computation device")
    parser.add_argument("--delta_position_only", action="store_true",
                        help="Train the dynamics model to predict only delta_position")
    args = parser.parse_args()

    set_seed(args.seed)

    data = load_dataset(args.dataset_path)
    inputs, targets, target_meta = assemble_features_targets(data, args.delta_position_only)
    delta_slice = target_meta.get("delta_position_slice")

    num_samples = inputs.shape[0]
    if num_samples == 0:
        raise ValueError("Dataset is empty; cannot train a model.")
    if num_samples < 2:
        raise ValueError("Dataset must contain at least two samples to create validation splits.")

    val_fraction = max(0.0, min(1.0, args.val_fraction))
    if val_fraction <= 0.0:
        raise ValueError("val_fraction must be greater than 0 to compute validation losses for checkpointing.")

    val_size = max(1, int(round(num_samples * val_fraction)))
    val_size = min(val_size, num_samples - 1)
    if val_size <= 0:
        raise ValueError("Validation split is empty; increase dataset size or val_fraction.")

    indices = np.random.permutation(num_samples)
    val_indices = indices[:val_size]
    train_indices = indices[val_size:]

    output_dir = os.path.join(args.train_dir, args.experiment_name)
    experiment_dir_exists = os.path.isdir(output_dir)
    os.makedirs(output_dir, exist_ok=True)

    run_log_path = os.path.join(output_dir, "training_runs.log")
    log_file = open(run_log_path, "a", encoding="utf-8")

    def log(message: str) -> None:
        tqdm.write(message)
        log_file.write(message + "\n")
        log_file.flush()

    log("")
    log("=" * 80)
    log(f"Run started: {datetime.now().isoformat(timespec='seconds')}")
    log(f"Command: {' '.join(sys.argv)}")
    log(f"Arguments: {vars(args)}")
    log(f"delta_position_only={args.delta_position_only}")

    try:
        existing_best_path: Optional[str] = None
        existing_best_loss = float("inf")
        existing_best_data: Optional[Dict[str, object]] = None
        existing_latest_path: Optional[str] = None
        latest_checkpoint_data: Optional[Dict[str, object]] = None

        if experiment_dir_exists:
            best_ckpts = sorted(glob.glob(os.path.join(output_dir, "checkpoint_best_val*.pt")))
            if best_ckpts:
                existing_best_path = best_ckpts[-1]
                existing_best_data = torch.load(existing_best_path, map_location="cpu")
                existing_best_loss = float(existing_best_data.get("val_loss", float("inf")))

            latest_ckpts = sorted(glob.glob(os.path.join(output_dir, "checkpoint_latest_val*.pt")))
            if latest_ckpts:
                existing_latest_path = latest_ckpts[-1]
                if existing_latest_path == existing_best_path:
                    latest_checkpoint_data = existing_best_data
                else:
                    latest_checkpoint_data = torch.load(existing_latest_path, map_location="cpu")

        resume_checkpoint_path: Optional[str] = None
        resume_checkpoint_data: Optional[Dict[str, object]] = None
        if experiment_dir_exists:
            if args.load_best and existing_best_data is not None:
                resume_checkpoint_path = existing_best_path
                resume_checkpoint_data = existing_best_data
            elif (not args.load_best) and latest_checkpoint_data is not None:
                resume_checkpoint_path = existing_latest_path
                resume_checkpoint_data = latest_checkpoint_data
            elif existing_best_data is not None:
                resume_checkpoint_path = existing_best_path
                resume_checkpoint_data = existing_best_data
            elif latest_checkpoint_data is not None:
                resume_checkpoint_path = existing_latest_path
                resume_checkpoint_data = latest_checkpoint_data

        if resume_checkpoint_data is not None:
            saved_input_dim = int(resume_checkpoint_data.get("input_dim", inputs.shape[1]))
            saved_output_dim = int(resume_checkpoint_data.get("output_dim", targets.shape[1]))
            if saved_input_dim != inputs.shape[1]:
                raise ValueError(
                    f"Checkpoint input_dim ({saved_input_dim}) does not match dataset input dim ({inputs.shape[1]})."
                )
            if saved_output_dim != targets.shape[1]:
                raise ValueError(
                    f"Checkpoint output_dim ({saved_output_dim}) does not match dataset target dim ({targets.shape[1]})."
                )

        input_mean = np.zeros(inputs.shape[1], dtype=np.float32)
        input_std = np.ones(inputs.shape[1], dtype=np.float32)
        target_mean = np.zeros(targets.shape[1], dtype=np.float32)
        target_std = np.ones(targets.shape[1], dtype=np.float32)

        if delta_slice is not None:
            target_mean = target_mean.copy()
            target_std = target_std.copy()
            target_mean[delta_slice] = 0.0
            target_std[delta_slice] = 1.0

        inputs_norm = inputs.astype(np.float32)
        targets_norm = targets.astype(np.float32)

        inputs_tensor = torch.from_numpy(inputs_norm)
        targets_tensor = torch.from_numpy(targets_norm)

        dataset = TensorDataset(inputs_tensor, targets_tensor)
        train_subset = Subset(dataset, train_indices.tolist())

        val_subset = Subset(dataset, val_indices.tolist())
        val_loader: Optional[DataLoader] = DataLoader(
            val_subset, batch_size=args.batch_size, shuffle=False, drop_last=False
        )

        train_loader = DataLoader(train_subset, batch_size=args.batch_size, shuffle=True, drop_last=False)

        device = torch.device(args.device)
        model = build_model(inputs.shape[1], targets.shape[1], args.hidden_size, args.num_layers)
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
        criterion = nn.MSELoss()

        start_epoch = 0
        if resume_checkpoint_data is not None:
            model.load_state_dict(resume_checkpoint_data["model_state_dict"])
            start_epoch = int(resume_checkpoint_data.get("epochs_trained", 0))

        if resume_checkpoint_path:
            resume_val_loss = float(resume_checkpoint_data.get("val_loss", float("nan")))
            val_loss_str = f"{resume_val_loss:.6f}" if np.isfinite(resume_val_loss) else "n/a"
            log(
                f"[train_dynamics_model] Resuming from {resume_checkpoint_path} "
                f"(epochs_trained={start_epoch}, val_loss={val_loss_str})"
            )
        else:
            log(f"[train_dynamics_model] Starting new experiment at {output_dir}")

        def save_checkpoint(path: str, val_loss_value: float, epoch_idx: int) -> None:
            state_dict = {k: v.detach().cpu() for k, v in model.state_dict().items()}
            torch.save(
                {
                    "model_state_dict": state_dict,
                    "input_mean": input_mean,
                    "input_std": input_std,
                    "target_mean": target_mean,
                    "target_std": target_std,
                    "input_dim": inputs.shape[1],
                    "output_dim": targets.shape[1],
                    "hidden_size": args.hidden_size,
                    "num_layers": args.num_layers,
                    "epochs_trained": epoch_idx,
                    "learning_rate": args.lr,
                    "batch_size": args.batch_size,
                    "val_fraction": val_fraction,
                    "seed": args.seed,
                    "val_loss": float(val_loss_value),
                },
                path,
            )

        best_val_loss = existing_best_loss if np.isfinite(existing_best_loss) else float("inf")
        best_checkpoint_path: Optional[str] = existing_best_path
        latest_checkpoint_path: Optional[str] = existing_latest_path

        def epoch_callback(epoch_idx: int, train_loss_value: float, val_loss_value: Optional[float]) -> None:
            nonlocal best_val_loss, best_checkpoint_path, latest_checkpoint_path
            if val_loss_value is None:
                raise RuntimeError(
                    "Validation loss is required for checkpoint naming; ensure val_fraction > 0."
                )

            latest_filename = f"checkpoint_latest_val{val_loss_value:.6f}.pt"
            latest_path = os.path.join(output_dir, latest_filename)
            if latest_checkpoint_path and latest_checkpoint_path != latest_path and os.path.exists(latest_checkpoint_path):
                os.remove(latest_checkpoint_path)
            save_checkpoint(latest_path, val_loss_value, epoch_idx)
            latest_checkpoint_path = latest_path

            if val_loss_value < best_val_loss:
                best_filename = f"checkpoint_best_val{val_loss_value:.6f}.pt"
                best_path = os.path.join(output_dir, best_filename)
                if best_checkpoint_path and best_checkpoint_path != best_path and os.path.exists(best_checkpoint_path):
                    os.remove(best_checkpoint_path)
                save_checkpoint(best_path, val_loss_value, epoch_idx)
                best_checkpoint_path = best_path
                best_val_loss = val_loss_value
                log(f"[train_dynamics_model] New best checkpoint at epoch {epoch_idx} (val_loss={val_loss_value:.6f})")

        log(
            f"[train_dynamics_model] Dataset size: train={len(train_subset)}, val={len(val_subset)}, total={num_samples}"
        )

        train(
            model,
            train_loader,
            val_loader,
            device,
            args.epochs,
            optimizer,
            criterion,
            start_epoch=start_epoch,
            target_mean=target_mean,
            target_std=target_std,
            delta_slice=delta_slice,
            log_fn=log,
            epoch_callback=epoch_callback,
        )

        log(
            f"[train_dynamics_model] Latest checkpoint: {os.path.abspath(latest_checkpoint_path) if latest_checkpoint_path else 'N/A'}"
        )
        if best_checkpoint_path:
            log(
                f"[train_dynamics_model] Best checkpoint: {os.path.abspath(best_checkpoint_path)} "
                f"(val_loss={best_val_loss:.6f})"
            )
    finally:
        log_file.close()


if __name__ == "__main__":
    main()
