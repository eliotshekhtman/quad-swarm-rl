#!/bin/bash

set -euo pipefail

usage() {
  printf 'Usage: %s --seed SEED --device {cpu|gpu} [--kappa KAPPA]\n' "$0"
}

die() {
  printf 'Error: %s\n' "$1" >&2
  usage >&2
  exit 2
}

SEED=""
DEVICE=""
KAPPA="0.6"
while [[ $# -gt 0 ]]; do
  case "$1" in
    --seed)
      [[ $# -ge 2 ]] || die "--seed requires a value"
      SEED="$2"
      shift 2
      ;;
    --device)
      [[ $# -ge 2 ]] || die "--device requires a value"
      DEVICE="$2"
      shift 2
      ;;
    --kappa)
      [[ $# -ge 2 ]] || die "--kappa requires a value"
      KAPPA="$2"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      die "unknown argument: $1"
      ;;
  esac
done

[[ "$SEED" =~ ^[0-9]+$ ]] || die "--seed must be a non-negative integer"
[[ "$DEVICE" == "cpu" || "$DEVICE" == "gpu" ]] \
  || die "--device must be either cpu or gpu"
[[ "$KAPPA" =~ ^(0([.][0-9]+)?|[.][0-9]+)$ ]] \
  || die "--kappa must lie in [0, 1)"
KAPPA_TAG="${KAPPA/./p}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKSPACE_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$WORKSPACE_DIR"

run_python() {
  if [[ "$DEVICE" == "gpu" ]]; then
    CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" python -u "$@"
  else
    python -u "$@"
  fi
}

PREDICTION_NAME="homog_gtg_pred_traj_data_seed${SEED}"
PREDICTION_PATH="train_dir/${PREDICTION_NAME}/pred_trajectories.npz"
if [[ -f "$PREDICTION_PATH" ]]; then
  printf 'Reusing predictions from %s\n' "$PREDICTION_PATH"
else
  printf 'Collecting predictions for seed %s\n' "$SEED"
  run_python l4dc/quad-swarm-rl/collect_homogeneous_predictions.py \
    --experiment_name "$PREDICTION_NAME" \
    --seed "$SEED" \
    --device "$DEVICE"
fi

COMMON_ARGS=(
  --multi_experiment multi_ddg_6
  --solo_experiment solo_ddg
  --episode_length 300
  --num_multi_agents 5
  --init_predictions "$PREDICTION_NAME"
  --delta 0.05
  --kappa "$KAPPA"
  --deterministic
  --seed "$SEED"
  --device "$DEVICE"
)

PIDS=()
NAMES=()

run_python l4dc/quad-swarm-rl/conformal.py \
  "${COMMON_ARGS[@]}" \
  --experiment_name "gtg_s${SEED}_paperstyle_robust_r8_n1000_e20_kappa${KAPPA_TAG}" \
  --update-procedure robust \
  --r_init 8 \
  --num_trajectories 1000 \
  --num_episodes 20 \
  > "train_dir/gtg_s${SEED}_robust_kappa${KAPPA_TAG}.log" 2>&1 &
PIDS+=("$!")
NAMES+=("robust")

run_python l4dc/quad-swarm-rl/conformal.py \
  "${COMMON_ARGS[@]}" \
  --experiment_name "gtg_s${SEED}_paperstyle_naive_r8_n1000_e20_kappa${KAPPA_TAG}" \
  --update-procedure naive \
  --r_init 8 \
  --num_trajectories 1000 \
  --num_episodes 20 \
  > "train_dir/gtg_s${SEED}_naive_kappa${KAPPA_TAG}.log" 2>&1 &
PIDS+=("$!")
NAMES+=("naive")

run_python l4dc/quad-swarm-rl/conformal.py \
  "${COMMON_ARGS[@]}" \
  --experiment_name "gtg_s${SEED}_paperstyle_noninteractive_r8_n1000_kappa${KAPPA_TAG}" \
  --update-procedure non-interactive \
  --r_init 8 \
  --num_trajectories 1000 \
  > "train_dir/gtg_s${SEED}_noninteractive_kappa${KAPPA_TAG}.log" 2>&1 &
PIDS+=("$!")
NAMES+=("non-interactive")

run_python l4dc/quad-swarm-rl/conformal.py \
  "${COMMON_ARGS[@]}" \
  --experiment_name "gtg_s${SEED}_paperstyle_calonce_r8_n1000_u20_kappa${KAPPA_TAG}" \
  --update-procedure calibrate-once \
  --r_init 8 \
  --num_trajectories 1000 \
  --num_episodes 20 \
  > "train_dir/gtg_s${SEED}_calonce_kappa${KAPPA_TAG}.log" 2>&1 &
PIDS+=("$!")
NAMES+=("calibrate-once")

run_python l4dc/quad-swarm-rl/conformal.py \
  "${COMMON_ARGS[@]}" \
  --experiment_name "gtg_s${SEED}_revised_anytime_r2_e50_kappa${KAPPA_TAG}" \
  --update-procedure anytime \
  --r_init 2 \
  --num_trajectories 8000 \
  --num_episodes 50 \
  --anytime-check-every 10 \
  --anytime-init-trajectories 1000 \
  --anytime-delta-init 0.05 \
  > "train_dir/gtg_s${SEED}_anytime_kappa${KAPPA_TAG}.log" 2>&1 &
PIDS+=("$!")
NAMES+=("anytime")

run_python l4dc/quad-swarm-rl/conformal.py \
  "${COMMON_ARGS[@]}" \
  --experiment_name "gtg_s${SEED}_revised_anytime_nonparametric_r2_e50_kappa${KAPPA_TAG}" \
  --update-procedure anytime-nonparametric \
  --r_init 2 \
  --num_trajectories 8000 \
  --num_episodes 50 \
  --anytime-check-every 10 \
  --anytime-init-trajectories 1000 \
  --anytime-delta-init 0.05 \
  > "train_dir/gtg_s${SEED}_anytime_nonparametric_kappa${KAPPA_TAG}.log" 2>&1 &
PIDS+=("$!")
NAMES+=("anytime-nonparametric")

status=0
for index in "${!PIDS[@]}"; do
  if wait "${PIDS[$index]}"; then
    printf '%s completed successfully\n' "${NAMES[$index]}"
  else
    printf '%s failed; see its log in train_dir\n' "${NAMES[$index]}" >&2
    status=1
  fi
done

exit "$status"
