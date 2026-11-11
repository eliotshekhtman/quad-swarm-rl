#!/bin/bash

set -uo pipefail

SEED=439
GPU=0
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKSPACE_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$WORKSPACE_DIR"

PREDICTION_NAME="homog_gtg_pred_traj_data_seed${SEED}"
PREDICTION_PATH="train_dir/${PREDICTION_NAME}/pred_trajectories.npz"
if [[ ! -f "$PREDICTION_PATH" ]]; then
  printf 'Missing prediction artifact: %s\n' "$PREDICTION_PATH" >&2
  exit 1
fi

COMMON_ARGS=(
  --multi_experiment multi_ddg_6
  --solo_experiment solo_ddg
  --episode_length 300
  --num_multi_agents 5
  --init_predictions "$PREDICTION_NAME"
  --alpha 0.1
  --delta 0.05
  --num_trajectories 8000
  --num_episodes 50
  --anytime-check-every 10
  --anytime-init-trajectories 1000
  --anytime-delta-init 0.05
  --deterministic
  --seed "$SEED"
  --device gpu
)

overall_status=0

run_stage() {
  local stage="$1"
  shift
  local -a configs=("$@")
  local -a pids=()
  local -a names=()

  printf 'Starting nonparametric anytime stage %s (%s runs on GPU %s)\n' \
    "$stage" "${#configs[@]}" "$GPU"

  local config kappa tolerance r_init
  local kappa_tag tolerance_tag r_init_tag run_tag experiment_name log_path
  for config in "${configs[@]}"; do
    read -r kappa tolerance r_init <<< "$config"
    kappa_tag="${kappa/./p}"
    tolerance_tag="${tolerance/./p}"
    r_init_tag="${r_init/./p}"
    run_tag="s${stage}_k${kappa_tag}_eps${tolerance_tag}_r${r_init_tag}"
    experiment_name="gtg_s${SEED}_anytime_np_ablation_${run_tag}"
    log_path="train_dir/${experiment_name}.log"

    CUDA_VISIBLE_DEVICES="$GPU" python -u l4dc/quad-swarm-rl/conformal.py \
      "${COMMON_ARGS[@]}" \
      --experiment_name "$experiment_name" \
      --update-procedure anytime-nonparametric \
      --kappa "$kappa" \
      --r_init "$r_init" \
      --anytime-radius-tolerance "$tolerance" \
      > "$log_path" 2>&1 &
    pids+=("$!")
    names+=("$experiment_name")
    printf '  launched %s (PID %s)\n' "$experiment_name" "$!"
  done

  local index
  for index in "${!pids[@]}"; do
    if wait "${pids[$index]}"; then
      printf '  completed %s\n' "${names[$index]}"
    else
      printf '  failed %s; see train_dir/%s.log\n' \
        "${names[$index]}" "${names[$index]}" >&2
      overall_status=1
    fi
  done
  printf 'Stage %s finished.\n' "$stage"
}

# Most promising settings: preserve the lower fixed point, tighten the final
# bracket, and test whether a closer certified initial radius helps.
run_stage 1 \
  "0.67 0.05 2.0" \
  "0.70 0.05 2.0" \
  "0.70 0.05 1.8"

# Follow-up ablations: lower/higher sensitivity and a looser stopping tolerance.
run_stage 2 \
  "0.60 0.05 2.0" \
  "0.70 0.10 2.0" \
  "0.75 0.05 2.0"

exit "$overall_status"
