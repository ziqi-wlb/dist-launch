#!/bin/bash
# Fix SSH keys on all nodes - Distribute public key to all nodes manually
# This script should be run on rank0 after cluster is initialized
# Usage: ./fix_ssh_keys.sh [hostname1] [hostname2] ...

set -e

CLUSTER_INFO_FILE="${CLUSTER_INFO_FILE:-/tmp/cluster_info.json}"
SSH_PUBLIC_KEY="${SSH_PUBLIC_KEY:-/mnt/3fs/dots-pretrain/weishi/release/public/ssh-key/id_rsa.pub}"
SSH_KEY="${SSH_KEY:-/mnt/3fs/dots-pretrain/weishi/release/public/ssh-key/id_rsa}"
SSH_PORT="${SSH_PORT:-2025}"
SSH_USER="${SSH_USER:-root}"

if [ ! -f "$SSH_PUBLIC_KEY" ]; then
    echo "Error: SSH public key not found: $SSH_PUBLIC_KEY"
    exit 1
fi

# Read public key
PUBLIC_KEY=$(cat "$SSH_PUBLIC_KEY" | tr -d '\n')

# Get hostnames
if [ $# -gt 0 ]; then
    # Use provided hostnames
    HOSTNAMES=("$@")
else
    # Read from cluster info file
    if [ ! -f "$CLUSTER_INFO_FILE" ]; then
        echo "Error: Cluster info file not found: $CLUSTER_INFO_FILE"
        echo "Please provide hostnames as arguments or run cluster initialization first"
        exit 1
    fi
    
    HOSTNAMES_STR=$(python3 -c "import json; f=open('$CLUSTER_INFO_FILE'); d=json.load(f); print(','.join(d['hostnames']))" 2>/dev/null || echo "")
    if [ -z "$HOSTNAMES_STR" ]; then
        echo "Error: Could not read hostnames from $CLUSTER_INFO_FILE"
        exit 1
    fi
    
    IFS=',' read -ra HOSTNAMES <<< "$HOSTNAMES_STR"
fi

echo "=== Fixing SSH Keys on All Nodes ==="
echo "Public key file: $SSH_PUBLIC_KEY"
echo "Number of nodes: ${#HOSTNAMES[@]}"
echo ""

# First, add key to current node (rank0)
CURRENT_HOSTNAME=$(hostname)
echo "Adding key to current node: $CURRENT_HOSTNAME"
mkdir -p ~/.ssh
chmod 700 ~/.ssh
if ! grep -q "$(echo "$PUBLIC_KEY" | awk '{print $2}')" ~/.ssh/authorized_keys 2>/dev/null; then
    echo "$PUBLIC_KEY" >> ~/.ssh/authorized_keys
    chmod 600 ~/.ssh/authorized_keys
    echo "✓ Added key to $CURRENT_HOSTNAME"
else
    echo "✓ Key already exists on $CURRENT_HOSTNAME"
fi

echo ""

# For other nodes, we need to use a method that works
# Since we may not have passwordless login yet, we'll use a different approach
# Option 1: If password is available, use sshpass or expect
# Option 2: Use scp with password (if available)
# Option 3: Manual instructions

echo "For other nodes, you have two options:"
echo ""
echo "Option 1: Manual setup (recommended if password is available)"
echo "Run the following on each node (or via password SSH):"
echo ""
for hostname in "${HOSTNAMES[@]}"; do
    hostname=$(echo "$hostname" | xargs)
    if [ "$hostname" != "$CURRENT_HOSTNAME" ]; then
        echo "  # On $hostname:"
        echo "  mkdir -p ~/.ssh"
        echo "  chmod 700 ~/.ssh"
        echo "  echo '$PUBLIC_KEY' >> ~/.ssh/authorized_keys"
        echo "  chmod 600 ~/.ssh/authorized_keys"
        echo ""
    fi
done

echo "Option 2: Use password SSH once to add key"
echo "After adding key manually on one node, you can use that node to distribute to others"
echo ""
echo "To test if key works, run:"
echo "  bash dist_launch/scripts/test_ssh.sh <hostname>"

