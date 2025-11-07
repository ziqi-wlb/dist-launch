#!/bin/bash
# NCCL tests wrapper script - executed on each node
# This script is launched by dist-launch run

set -e

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
NCCL_TESTS_PY="${PROJECT_ROOT}/nccl_tests.py"

# Fallback: try dist_launch/nccl_tests.py if not in root
if [ ! -f "$NCCL_TESTS_PY" ]; then
    NCCL_TESTS_PY="${PROJECT_ROOT}/dist_launch/nccl_tests.py"
fi

# Check if Python script exists
if [ ! -f "$NCCL_TESTS_PY" ]; then
    echo "Error: $NCCL_TESTS_PY not found"
    exit 1
fi

# Execute nccl_tests.py
# Environment variables (RANK, WORLD_SIZE, MASTER_ADDR, MASTER_PORT) are set by run.py
python3 "$NCCL_TESTS_PY" "$@"

