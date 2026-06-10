#!/bin/bash

set -euo pipefail

LOG_DIR="./logs"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
COMPUTE_LOG="$LOG_DIR/compute_norm_stats_$TIMESTAMP.log"
TRAIN_LOG="$LOG_DIR/train_$TIMESTAMP.log"
CONVERT_LOG="$LOG_DIR/convert_$TIMESTAMP.log"
SUMMARY_LOG="$LOG_DIR/summary_$TIMESTAMP.log"

mkdir -p "$LOG_DIR"

log_message() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a "$SUMMARY_LOG"
}

run_command() {
    local log_file="$1"
    local step_name="$2"
    shift 2

    log_message "Starting: $step_name"
    log_message "Command: $*"
    log_message "Log file: $log_file"

    if "$@" > "$log_file" 2>&1; then
        log_message "$step_name completed successfully"
        return 0
    fi

    local exit_code=$?
    log_message "$step_name failed (exit code: $exit_code)"
    log_message "Error log saved to: $log_file"
    {
        echo "=== Last 50 log lines ==="
        tail -n 50 "$log_file"
        echo "========================="
    } >> "$SUMMARY_LOG"
    return "$exit_code"
}

setup_gpu_environment() {
    log_message "Setting GPU environment variables"
    export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
    export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-max_split_size_mb:512}"
    export XLA_FLAGS="${XLA_FLAGS:---xla_gpu_force_compilation_parallelism=8}"
    export XLA_PYTHON_CLIENT_MEM_FRACTION="${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.9}"
    export CUDNN_ENABLED="${CUDNN_ENABLED:-1}"
    export CUDNN_BENCHMARK="${CUDNN_BENCHMARK:-true}"

    log_message "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
    log_message "PYTORCH_CUDA_ALLOC_CONF: $PYTORCH_CUDA_ALLOC_CONF"
    log_message "XLA_FLAGS: $XLA_FLAGS"
    log_message "XLA_PYTHON_CLIENT_MEM_FRACTION: $XLA_PYTHON_CLIENT_MEM_FRACTION"
    log_message "CUDNN_ENABLED: $CUDNN_ENABLED"
    log_message "CUDNN_BENCHMARK: $CUDNN_BENCHMARK"
}

main() {
    log_message "Starting data conversion, normalization, and training"
    log_message "Working directory: $(pwd)"

    source .venv/bin/activate
    setup_gpu_environment

    log_message "Environment variables:"
    echo "  HF_HOME: ${HF_HOME:-}" | tee -a "$SUMMARY_LOG"
    echo "  HF_DATASETS_CACHE: ${HF_DATASETS_CACHE:-}" | tee -a "$SUMMARY_LOG"
    echo "  TMPDIR: ${TMPDIR:-}" | tee -a "$SUMMARY_LOG"
    echo "  LEROBOT_HOME: ${LEROBOT_HOME:-}" | tee -a "$SUMMARY_LOG"

    RAW_DIR="${RAW_DIR:-/path/to/hdf5_files}"
    REPO_ID="${REPO_ID:-zerith/test}"
    CONFIG_NAME="${CONFIG_NAME:-test}"
    EXP_NAME="${EXP_NAME:-zerithtest}"

    # REPO_ID controls the LeRobot dataset path under LEROBOT_HOME and should match the config repo_id.
    run_command "$CONVERT_LOG" "Convert HDF5 data to LeRobot" \
        uv run scripts/convert_new.py --raw_dir "$RAW_DIR" --repo_id "$REPO_ID"

    # CONFIG_NAME selects the training config in src/openpi/training/config.py.
    run_command "$COMPUTE_LOG" "Compute normalization statistics" \
        uv run scripts/compute_norm_stats.py --config_name "$CONFIG_NAME"

    # EXP_NAME controls the final checkpoint directory: <checkpoint_base_dir>/<CONFIG_NAME>/<EXP_NAME>/.
    run_command "$TRAIN_LOG" "Train policy" \
        uv run scripts/train.py "$CONFIG_NAME" --exp_name "$EXP_NAME" --overwrite

    log_message "All tasks completed successfully"
    log_message "Conversion log: $CONVERT_LOG"
    log_message "Normalization log: $COMPUTE_LOG"
    log_message "Training log: $TRAIN_LOG"
}

trap 'log_message "Interrupted by user"; exit 130' INT TERM

main "$@"
