#!/bin/bash
# Test SSH connection to a node using the configured SSH key
# Usage: ./test_ssh.sh <hostname>

set -e

if [ $# -eq 0 ]; then
    echo "Usage: $0 <hostname>"
    echo "Example: $0 BYAW071-JPI2-A-4-420B-L-L01"
    exit 1
fi

HOSTNAME="$1"
SSH_KEY="${SSH_KEY:-/mnt/3fs/dots-pretrain/weishi/release/public/ssh-key/id_rsa}"
SSH_PORT="${SSH_PORT:-2025}"
SSH_USER="${SSH_USER:-root}"

echo "Testing SSH connection to $SSH_USER@$HOSTNAME:$SSH_PORT"
echo "Using SSH key: $SSH_KEY"

if [ ! -f "$SSH_KEY" ]; then
    echo "Error: SSH key not found at $SSH_KEY"
    exit 1
fi

# Check key permissions (should be 600)
key_perms=$(stat -c "%a" "$SSH_KEY" 2>/dev/null || stat -f "%OLp" "$SSH_KEY" 2>/dev/null)
if [ "$key_perms" != "600" ]; then
    echo "Warning: SSH key permissions are $key_perms, should be 600"
    echo "Fixing permissions..."
    chmod 600 "$SSH_KEY"
fi

echo ""
echo "Connecting..."
ssh -i "$SSH_KEY" \
    -p "$SSH_PORT" \
    -o "StrictHostKeyChecking=no" \
    -o "UserKnownHostsFile=/dev/null" \
    "$SSH_USER@$HOSTNAME" \
    "echo 'SSH connection successful!' && hostname"

echo ""
echo "✓ SSH connection test passed"

