#!/bin/bash
# One-click launch script for distributed training on rank0 node
# Usage: ./run.sh <train.sh> [options]
# 
# This script is used in debug mode:
# - Cluster is already running and waiting (via wait.sh)
# - Cluster info (hostnames) should be saved during initialization
# - This script reads saved cluster info and launches train.sh on all nodes

set -e

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
PYTHON_SCRIPT="${PROJECT_ROOT}/run.py"
# Fallback: try auto_launch/run.py if run.py not in root
if [ ! -f "$PYTHON_SCRIPT" ]; then
    PYTHON_SCRIPT="${PROJECT_ROOT}/auto_launch/run.py"
fi

# Check if train script is provided
if [ $# -eq 0 ]; then
    echo "Error: train script is required"
    echo "Usage: $0 <train.sh> [options]"
    exit 1
fi

TRAIN_SCRIPT="$1"
shift  # Remove train_script from arguments

# Default configuration - can be overridden by environment variables
MASTER_ADDR="${MASTER_ADDR:-}"
WORLD_SIZE="${WORLD_SIZE:-}"
MASTER_PORT="${MASTER_PORT:-23456}"
NODES="${NODES:-}"  # Auto-discover if not set
SSH_KEY="${SSH_KEY:-/mnt/3fs/dots-pretrain/weishi/release/public/ssh-key/id_rsa}"
SSH_PORT="${SSH_PORT:-2025}"
SSH_USER="${SSH_USER:-root}"

# Check if Python script exists
if [ ! -f "$PYTHON_SCRIPT" ]; then
    echo "Error: $PYTHON_SCRIPT not found"
    exit 1
fi

# Build command arguments
CMD_ARGS=()

if [ -n "$MASTER_ADDR" ]; then
    CMD_ARGS+=(--master-addr "$MASTER_ADDR")
fi

if [ -n "$WORLD_SIZE" ]; then
    CMD_ARGS+=(--world-size "$WORLD_SIZE")
fi

if [ -n "$MASTER_PORT" ]; then
    CMD_ARGS+=(--master-port "$MASTER_PORT")
fi

if [ -n "$NODES" ]; then
    CMD_ARGS+=(--nodes "$NODES")
fi

CMD_ARGS+=(--ssh-key "$SSH_KEY")
CMD_ARGS+=(--ssh-port "$SSH_PORT")
CMD_ARGS+=(--ssh-user "$SSH_USER")

# Execute Python script with train_script as first argument
python3 "$PYTHON_SCRIPT" "$TRAIN_SCRIPT" "${CMD_ARGS[@]}" "$@"

