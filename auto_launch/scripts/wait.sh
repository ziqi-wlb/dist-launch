#!/bin/bash
# Wait script for debug mode - Run on all nodes to keep them alive
# Usage: ./wait.sh [timeout_seconds]

set -e

TIMEOUT=${1:-0}
CLUSTER_INFO_FILE="${CLUSTER_INFO_FILE:-/tmp/cluster_info.json}"

# Set NCCL environment variables for GB200 (if not already set)
if [ -z "$NCCL_SOCKET_IFNAME" ]; then
    if command -v ifconfig >/dev/null 2>&1; then
        DETECTED_IFNAME=$(ifconfig -a | grep -E '^[a-zA-Z0-9]+:' | grep -v '^lo:' | awk -F: '{print $1}' | grep -E 'enP|enp|eth' | head -5 | tr '\n' ',' | sed 's/,$//')
    elif command -v ip >/dev/null 2>&1; then
        DETECTED_IFNAME=$(ip addr show | grep -E '^[0-9]+:' | grep -v 'lo:' | awk -F: '{print $2}' | tr -d ' ' | grep -E 'enP|enp|eth' | head -5 | tr '\n' ',' | sed 's/,$//')
    fi
    
    if [ -n "$DETECTED_IFNAME" ]; then
        export NCCL_SOCKET_IFNAME="$DETECTED_IFNAME"
    else
        export NCCL_SOCKET_IFNAME="enP22p3s0f0np0,enP6p3s0f0np0"
    fi
fi
export NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-1}"

cleanup() {
    echo ""
    echo "Received signal, cleaning up..."
    jobs -p | xargs -r kill 2>/dev/null || true
    wait
    echo "Cleanup completed"
    exit 0
}

trap cleanup SIGTERM SIGINT SIGQUIT

echo "=== Cluster Initialization ==="
export CLUSTER_INFO_FILE

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
INIT_SCRIPT="${SCRIPT_DIR}/init_cluster.py"

if [ -f "$INIT_SCRIPT" ]; then
    echo "Initializing cluster..."
    python3 "$INIT_SCRIPT"
    if [ $? -eq 0 ] && [ -f "$CLUSTER_INFO_FILE" ]; then
        echo "Cluster initialization completed"
    else
        echo "Warning: Cluster initialization failed, but continuing..."
    fi
fi

echo ""
echo "=== Entering wait mode ==="
echo "Waiting for debug... (send SIGTERM/SIGINT to exit)"

if [ "$TIMEOUT" -gt 0 ]; then
    timeout $TIMEOUT sleep infinity 2>/dev/null || timeout $TIMEOUT tail -f /dev/null || true
else
    if command -v sleep >/dev/null 2>&1 && sleep infinity 2>/dev/null; then
        exec sleep infinity
    else
        exec tail -f /dev/null
    fi
fi

