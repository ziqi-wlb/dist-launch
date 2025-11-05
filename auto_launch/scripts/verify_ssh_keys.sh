#!/bin/bash
# Verify SSH keys are properly distributed on all nodes
# This script should be run on rank0 after cluster initialization

set -e

CLUSTER_INFO_FILE="${CLUSTER_INFO_FILE:-/tmp/cluster_info.json}"
SSH_PUBLIC_KEY="${SSH_PUBLIC_KEY:-/mnt/3fs/dots-pretrain/weishi/release/public/ssh-key/id_rsa.pub}"
SSH_KEY="${SSH_KEY:-/mnt/3fs/dots-pretrain/weishi/release/public/ssh-key/id_rsa}"
SSH_PORT="${SSH_PORT:-2025}"
SSH_USER="${SSH_USER:-root}"

if [ ! -f "$CLUSTER_INFO_FILE" ]; then
    echo "Error: Cluster info file not found: $CLUSTER_INFO_FILE"
    echo "Please run cluster initialization first (auto-launch wait)"
    exit 1
fi

if [ ! -f "$SSH_PUBLIC_KEY" ]; then
    echo "Error: SSH public key not found: $SSH_PUBLIC_KEY"
    exit 1
fi

# Read public key content
PUBLIC_KEY_CONTENT=$(cat "$SSH_PUBLIC_KEY" | tr -d '\n')
KEY_TYPE=$(echo "$PUBLIC_KEY_CONTENT" | awk '{print $1}')
KEY_DATA=$(echo "$PUBLIC_KEY_CONTENT" | awk '{print $2}')

echo "=== Verifying SSH Keys on All Nodes ==="
echo "Public key: $SSH_PUBLIC_KEY"
echo "Key type: $KEY_TYPE"
echo "Key data: ${KEY_DATA:0:20}..."
echo ""

# Read cluster info
HOSTNAMES=$(python3 -c "import json; f=open('$CLUSTER_INFO_FILE'); d=json.load(f); print(','.join(d['hostnames']))" 2>/dev/null || echo "")

if [ -z "$HOSTNAMES" ]; then
    echo "Error: Could not read hostnames from $CLUSTER_INFO_FILE"
    exit 1
fi

# Convert to array
IFS=',' read -ra HOSTS <<< "$HOSTNAMES"

echo "Checking SSH keys on ${#HOSTS[@]} nodes..."
echo ""

all_success=true
for hostname in "${HOSTS[@]}"; do
    hostname=$(echo "$hostname" | xargs)  # trim whitespace
    echo -n "Checking $hostname... "
    
    # Check if key exists in authorized_keys on remote node
    key_found=$(ssh -i "$SSH_KEY" \
        -p "$SSH_PORT" \
        -o "StrictHostKeyChecking=no" \
        -o "UserKnownHostsFile=/dev/null" \
        -o "ConnectTimeout=5" \
        "$SSH_USER@$hostname" \
        "grep -q '$KEY_DATA' ~/.ssh/authorized_keys 2>/dev/null && echo 'found' || echo 'not found'" 2>/dev/null || echo "connection_failed")
    
    if [ "$key_found" == "found" ]; then
        echo "✓ Key found"
    elif [ "$key_found" == "connection_failed" ]; then
        echo "✗ Connection failed (may need password for first time)"
        all_success=false
    else
        echo "✗ Key not found"
        all_success=false
    fi
done

echo ""
if [ "$all_success" == "true" ]; then
    echo "✓ All nodes have SSH keys properly configured"
else
    echo "✗ Some nodes are missing SSH keys"
    echo ""
    echo "To manually add the key, run on each node:"
    echo "  echo '$PUBLIC_KEY_CONTENT' >> ~/.ssh/authorized_keys"
    echo "  chmod 600 ~/.ssh/authorized_keys"
    echo "  chmod 700 ~/.ssh"
fi

