#!/usr/bin/env python3
"""
Cluster initialization script - Run when DLC task starts
Discovers all node hostnames using PyTorch allgather and saves to file
Distributes SSH public key to all nodes for passwordless login
This should be executed on all nodes when the cluster starts
"""
import os
import sys
import json
import torch
import torch.distributed as dist
import subprocess
import re
import socket
from pathlib import Path


def detect_network_interfaces():
    """
    Detect network interfaces using ifconfig or ip command
    Returns a comma-separated string of interface names, excluding loopback
    
    Returns:
        str: Comma-separated interface names, or None if detection fails
    """
    interfaces = []
    
    # Try ifconfig first
    try:
        result = subprocess.run(['ifconfig', '-a'], 
                              capture_output=True, 
                              text=True, 
                              timeout=5)
        if result.returncode == 0:
            # Parse ifconfig output
            for line in result.stdout.split('\n'):
                # Match interface name (e.g., "eth0:", "enP22p3s0f0np0:")
                match = re.match(r'^(\S+):', line.strip())
                if match:
                    ifname = match.group(1)
                    # Skip loopback
                    if ifname != 'lo' and not ifname.startswith('lo:'):
                        # Check if interface has an IP address (not just link-local)
                        if 'inet ' in line or 'inet6 ' in line:
                            interfaces.append(ifname)
    except (subprocess.TimeoutExpired, FileNotFoundError, Exception) as e:
        pass
    
    # If ifconfig failed or found nothing, try ip command
    if not interfaces:
        try:
            result = subprocess.run(['ip', 'addr', 'show'], 
                                  capture_output=True, 
                                  text=True, 
                                  timeout=5)
            if result.returncode == 0:
                # Parse ip addr output
                current_if = None
                for line in result.stdout.split('\n'):
                    # Match interface name (e.g., "1: eth0:")
                    match = re.match(r'^\d+:\s+(\S+):', line.strip())
                    if match:
                        current_if = match.group(1)
                        # Skip loopback
                        if current_if == 'lo':
                            current_if = None
                    # Check if interface has inet address
                    elif current_if and ('inet ' in line or 'inet6 ' in line):
                        if current_if not in interfaces:
                            interfaces.append(current_if)
        except (subprocess.TimeoutExpired, FileNotFoundError, Exception) as e:
            pass
    
    # Filter interfaces: prefer GB200-style interfaces (enP* pattern) if available
    # Otherwise use all non-loopback interfaces
    gb200_interfaces = [ifname for ifname in interfaces if 'enP' in ifname or 'enp' in ifname.lower()]
    if gb200_interfaces:
        interfaces = gb200_interfaces
    
    # Remove duplicates and sort
    interfaces = sorted(list(set(interfaces)))
    
    if interfaces:
        return ','.join(interfaces)
    return None


def update_hosts_file(hostnames):
    """
    Update /etc/hosts file on rank0 to add rank-0, rank-1, etc. aliases
    This allows easy login via ssh root@rank-1
    
    Args:
        hostnames: List of hostnames, ordered by rank
    
    Returns:
        True if successful
    """
    try:
        hosts_file = '/etc/hosts'
        hosts_backup = '/etc/hosts.dist-launch.backup'
        
        # Read current hosts file
        if os.path.exists(hosts_file):
            with open(hosts_file, 'r') as f:
                current_hosts = f.read()
        else:
            current_hosts = ''
        
        # Create backup
        try:
            with open(hosts_backup, 'w') as f:
                f.write(current_hosts)
            print(f'Created backup: {hosts_backup}')
        except Exception as e:
            print(f'Warning: Could not create backup: {e}', file=sys.stderr)
        
        # Remove old dist-launch entries
        lines = current_hosts.split('\n')
        filtered_lines = [line for line in lines if 'dist-launch' not in line.lower() and 'auto-launch' not in line.lower()]
        
        # Get IP addresses for each hostname
        rank_entries = []
        for rank, hostname in enumerate(hostnames):
            try:
                # Try to get IP address
                ip = socket.gethostbyname(hostname)
                rank_alias = f'rank-{rank}'
                entry = f'{ip}\t{rank_alias}\t# dist-launch: rank{rank} -> {hostname}'
                rank_entries.append(entry)
            except socket.gaierror:
                # If hostname resolution fails, try to get IP from current node's hostname
                print(f'Warning: Could not resolve {hostname}, skipping hosts entry', file=sys.stderr)
        
        # Write updated hosts file
        new_hosts = '\n'.join(filtered_lines)
        if rank_entries:
            new_hosts += '\n\n# Dist-launch cluster node aliases (added automatically)\n'
            new_hosts += '\n'.join(rank_entries)
            new_hosts += '\n'
        
        try:
            with open(hosts_file, 'w') as f:
                f.write(new_hosts)
            print(f'Updated /etc/hosts with rank aliases')
            return True
        except PermissionError:
            print(f'Warning: Permission denied writing to {hosts_file}', file=sys.stderr)
            print(f'You may need to run with sudo or manually add entries:', file=sys.stderr)
            for entry in rank_entries:
                print(f'  {entry}', file=sys.stderr)
            return False
        except Exception as e:
            print(f'Error updating hosts file: {e}', file=sys.stderr)
            return False
            
    except Exception as e:
        print(f'Error updating hosts file: {e}', file=sys.stderr)
        import traceback
        traceback.print_exc()
        return False


def distribute_ssh_key(hostnames, public_key_path):
    """
    Distribute SSH public key to all nodes using allgather
    
    Args:
        hostnames: List of all node hostnames
        public_key_path: Path to SSH public key file
    
    Returns:
        True if successful
    """
    try:
        # Read public key
        if not os.path.exists(public_key_path):
            print(f'Warning: SSH public key not found at {public_key_path}', file=sys.stderr)
            return False
        
        with open(public_key_path, 'r') as f:
            public_key = f.read().strip()
        
        if not public_key:
            print(f'Warning: SSH public key is empty', file=sys.stderr)
            return False
        
        # Get current hostname
        current_hostname = os.environ.get('HOSTNAME', '')
        if not current_hostname:
            import socket
            current_hostname = socket.gethostname()
        
        # Add public key to authorized_keys on current node
        ssh_dir = os.path.expanduser('~/.ssh')
        authorized_keys_file = os.path.join(ssh_dir, 'authorized_keys')
        
        # Create .ssh directory if not exists
        os.makedirs(ssh_dir, mode=0o700, exist_ok=True)
        
        # Read existing authorized_keys
        existing_keys = set()
        if os.path.exists(authorized_keys_file):
            with open(authorized_keys_file, 'r') as f:
                existing_keys = set(line.strip() for line in f if line.strip())
        
        # Add new key if not already present
        # Check if key exists (by comparing key content, not exact match)
        # SSH public key format: "ssh-rsa AAAAB3NzaC1yc2E... comment" or "ssh-ed25519 AAAAC3... comment"
        public_key_parts = public_key.strip().split()
        if len(public_key_parts) < 2:
            print(f'Warning: Invalid public key format on {current_hostname}', file=sys.stderr)
            return False
        
        key_type = public_key_parts[0]  # e.g., "ssh-rsa", "ssh-ed25519"
        key_data = public_key_parts[1]   # The actual key data
        
        key_exists = False
        for existing_key in existing_keys:
            existing_parts = existing_key.strip().split()
            if len(existing_parts) >= 2:
                existing_data = existing_parts[1]
                # Compare key data (the actual cryptographic key)
                if key_data == existing_data:
                    key_exists = True
                    break
        
        if not key_exists:
            with open(authorized_keys_file, 'a') as f:
                try:
                    if os.path.getsize(authorized_keys_file) > 0:
                        with open(authorized_keys_file, 'rb') as rf:
                            rf.seek(-1, os.SEEK_END)
                            if rf.read(1) != b'\n':
                                f.write('\n')
                except Exception:
                    pass
                f.write(f'{public_key.strip()}\n')
            os.chmod(authorized_keys_file, 0o600)
        
        os.chmod(ssh_dir, 0o700)
        if os.path.exists(authorized_keys_file):
            os.chmod(authorized_keys_file, 0o600)
        
        return True
        
    except Exception as e:
        print(f'Error distributing SSH key: {e}', file=sys.stderr)
        return False


def discover_and_save_hostnames():
    """
    Discover all hostnames using PyTorch allgather and save to file
    This should be called on all nodes simultaneously
    """
    # Save original NCCL environment variables
    nccl_vars = ['NCCL_SOCKET_IFNAME', 'NCCL_IB_DISABLE', 'NCCL_DEBUG']
    original_nccl_env = {}
    for var in nccl_vars:
        if var in os.environ:
            original_nccl_env[var] = os.environ[var]
    
    try:
        # Set NCCL environment variables for GB200
        # These should be set before initializing process group
        if 'NCCL_SOCKET_IFNAME' not in os.environ:
            # Try to detect network interfaces dynamically
            detected_ifname = detect_network_interfaces()
            if detected_ifname:
                os.environ['NCCL_SOCKET_IFNAME'] = detected_ifname
            else:
                os.environ['NCCL_SOCKET_IFNAME'] = 'enP22p3s0f0np0,enP6p3s0f0np0'
        
        if 'NCCL_IB_DISABLE' not in os.environ:
            os.environ['NCCL_IB_DISABLE'] = '1'
        os.environ['NCCL_DEBUG'] = os.environ.get('NCCL_DEBUG', 'WARN')
        
        # Get environment variables from pytorch-job
        master_addr = os.environ.get('MASTER_ADDR', 'localhost')
        # Use a different port for initialization to avoid conflict with training
        # Use INIT_MASTER_PORT if set, otherwise use MASTER_PORT + 1
        training_master_port = int(os.environ.get('MASTER_PORT', '23456'))
        init_master_port = int(os.environ.get('INIT_MASTER_PORT', str(training_master_port + 1)))
        world_size = int(os.environ.get('WORLD_SIZE', 1))
        rank = int(os.environ.get('RANK', 0))
        
        # Initialize process group
        backend = 'nccl' if torch.cuda.is_available() else 'gloo'
        
        # Increase timeout for GB200 network initialization
        init_timeout = torch.distributed.default_pg_timeout
        if init_timeout.total_seconds() < 1800:  # 30 minutes
            from datetime import timedelta
            init_timeout = timedelta(seconds=1800)
        
        dist.init_process_group(
            backend=backend,
            init_method=f'tcp://{master_addr}:{init_master_port}',
            world_size=world_size,
            rank=rank,
            timeout=init_timeout
        )
        
        world_size = dist.get_world_size()
        rank = dist.get_rank()
        
        # Get current hostname
        current_hostname = os.environ.get('HOSTNAME', '')
        if not current_hostname:
            import socket
            current_hostname = socket.gethostname()
        
        # Convert hostname to tensor for allgather
        hostname_bytes = current_hostname.encode('utf-8')
        max_len = 256
        hostname_bytes = hostname_bytes[:max_len].ljust(max_len, b'\0')
        
        # Create tensor for allgather
        if torch.cuda.is_available():
            local_tensor = torch.ByteTensor(list(hostname_bytes)).cuda()
            gathered_tensors = [torch.zeros_like(local_tensor) for _ in range(world_size)]
        else:
            local_tensor = torch.ByteTensor(list(hostname_bytes))
            gathered_tensors = [torch.zeros_like(local_tensor) for _ in range(world_size)]
        
        dist.all_gather(gathered_tensors, local_tensor)
        
        # Extract hostnames from gathered tensors
        hostnames = []
        for tensor in gathered_tensors:
            hostname_bytes = bytes(tensor.cpu().tolist() if torch.cuda.is_available() else tensor.tolist()).rstrip(b'\0')
            hostname = hostname_bytes.decode('utf-8', errors='ignore')
            if hostname:
                hostnames.append(hostname)
        
        # Distribute SSH public key to all nodes
        ssh_public_key = os.environ.get('SSH_PUBLIC_KEY', 
                                       '/mnt/3fs/dots-pretrain/weishi/release/public/ssh-key/id_rsa.pub')
        print(f'Distributing SSH public key from {ssh_public_key}...')
        print(f'Public key file exists: {os.path.exists(ssh_public_key)}')
        if os.path.exists(ssh_public_key):
            with open(ssh_public_key, 'r') as f:
                key_preview = f.read().strip()[:50]
                print(f'Public key preview: {key_preview}...')
        
        success = distribute_ssh_key(hostnames, ssh_public_key)
        if success:
            print(f'✓ SSH public key distribution completed on rank {rank}')
        else:
            print(f'✗ Warning: SSH public key distribution may have failed on rank {rank}', file=sys.stderr)
        
        dist.barrier()

        # Only rank 0 saves the result
        if rank == 0:
            # Save to file (use training master_port, not init port)
            cluster_info = {
                'master_addr': master_addr,
                'master_port': training_master_port,  # Save training port, not init port
                'world_size': world_size,
                'hostnames': hostnames
            }
            
            # Save to shared location (assume all nodes can access)
            info_file = os.environ.get('CLUSTER_INFO_FILE', '/tmp/cluster_info.json')
            with open(info_file, 'w') as f:
                json.dump(cluster_info, f, indent=2)
            
            print(f'Cluster info saved to {info_file}')
            print(f'Discovered {len(hostnames)} nodes: {", ".join(hostnames)}')
            
            update_hosts_file(hostnames)
            
            dist.barrier()
            return hostnames
        
        dist.barrier()
        return None
        
    except Exception as e:
        print(f'Error discovering hosts: {e}', file=sys.stderr)
        import traceback
        traceback.print_exc()
        return None
    finally:
        # Restore original NCCL environment variables to avoid conflicts with train.sh
        nccl_vars = ['NCCL_SOCKET_IFNAME', 'NCCL_IB_DISABLE', 'NCCL_DEBUG']
        for var in nccl_vars:
            if var in original_nccl_env:
                os.environ[var] = original_nccl_env[var]
            elif var in os.environ:
                # Remove if it was added by us
                if var not in original_nccl_env:
                    del os.environ[var]
        
        # Clean up process group if initialized
        # This is critical to free the port for training phase
        try:
            if dist.is_initialized():
                print(f'[rank{rank}] Destroying process group to free port for training...')
                dist.destroy_process_group()
                print(f'[rank{rank}] Process group destroyed, port freed')
                # Give a small delay to ensure port is fully released
                import time
                time.sleep(0.5)
        except Exception as e:
            print(f'Warning: Failed to destroy process group: {e}', file=sys.stderr)


def main():
    """Main entry point"""
    try:
        hostnames = discover_and_save_hostnames()
        if hostnames:
            sys.exit(0)
        else:
            sys.exit(0)  # Other ranks also exit successfully
    except Exception as e:
        print(f'Error: {e}', file=sys.stderr)
        sys.exit(1)


if __name__ == '__main__':
    main()

