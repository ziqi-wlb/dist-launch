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
from datetime import datetime


# Default SSH key paths - can be overridden via SSH_KEY and SSH_PUBLIC_KEY environment variables
DEFAULT_SSH_KEY_PATH = '/mnt/3fs/dots-pretrain/weishi/release/public/ssh-key/id_rsa'
DEFAULT_SSH_PUBLIC_KEY_PATH = '/mnt/3fs/dots-pretrain/weishi/release/public/ssh-key/id_rsa.pub'

# Log file path - can be configured via DIST_LAUNCH_LOG_FILE environment variable
# Default: /tmp/dist-launch-init.log
# If using shared storage, you can set it to a shared path like:
#   export DIST_LAUNCH_LOG_FILE="/shared/path/dist-launch-init-${HOSTNAME}.log"
# Or use a single file (each node will append):
#   export DIST_LAUNCH_LOG_FILE="/shared/path/dist-launch-init.log"
def get_log_file_path():
    """Get log file path from environment or use default"""
    log_file = os.environ.get('DIST_LAUNCH_LOG_FILE', '/tmp/dist-launch-init.log')
    
    # If the path contains ${HOSTNAME} or ${RANK}, replace them
    hostname = os.environ.get('HOSTNAME', '')
    if not hostname:
        import socket
        hostname = socket.gethostname()
    
    rank = os.environ.get('RANK', '0')
    
    log_file = log_file.replace('${HOSTNAME}', hostname)
    log_file = log_file.replace('${RANK}', str(rank))
    log_file = log_file.replace('$HOSTNAME', hostname)
    log_file = log_file.replace('$RANK', str(rank))
    
    return log_file

LOG_FILE = None  # Will be set in setup_logging


class LogWriter:
    """Write logs to both file and stdout/stderr"""
    _log_file_handle = None
    _lock = None
    
    def __init__(self, log_file, original_stream, is_stderr=False):
        self.log_file = log_file
        self.original_stream = original_stream
        self.is_stderr = is_stderr
        
        # Initialize file handle if not exists
        if LogWriter._log_file_handle is None:
            try:
                LogWriter._log_file_handle = open(log_file, 'a', encoding='utf-8', buffering=1)  # Line buffered
            except Exception as e:
                # If can't open file, log to stderr
                original_stream.write(f'Warning: Could not open log file {log_file}: {e}\n')
        
    def write(self, message):
        """Write to both file and original stream"""
        if not message:  # Skip empty messages
            return
            
        # Always write to original stream
        self.original_stream.write(message)
        
        # Write to log file if available
        if LogWriter._log_file_handle is not None:
            try:
                timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                prefix = '[ERROR]' if self.is_stderr else '[INFO]'
                LogWriter._log_file_handle.write(f'[{timestamp}] {prefix} {message}')
                LogWriter._log_file_handle.flush()  # Force flush to ensure immediate write
            except Exception as e:
                # If write fails, try to write error to original stream
                try:
                    self.original_stream.write(f'Warning: Log write failed: {e}\n')
                except Exception:
                    pass
    
    def flush(self):
        """Flush both file and original stream"""
        self.original_stream.flush()
        if LogWriter._log_file_handle is not None:
            try:
                LogWriter._log_file_handle.flush()
            except Exception:
                pass
    
    @classmethod
    def close_log_file(cls):
        """Close log file handle"""
        if cls._log_file_handle is not None:
            try:
                cls._log_file_handle.close()
            except Exception:
                pass
            cls._log_file_handle = None


def setup_logging():
    """Setup logging to file and stdout/stderr"""
    global LOG_FILE
    LOG_FILE = get_log_file_path()
    
    try:
        # Create directory if it doesn't exist
        log_dir = os.path.dirname(LOG_FILE)
        if log_dir and not os.path.exists(log_dir):
            try:
                os.makedirs(log_dir, mode=0o755, exist_ok=True)
            except Exception as e:
                sys.stderr.write(f'Warning: Could not create log directory {log_dir}: {e}\n')
                # Fallback to /tmp
                LOG_FILE = '/tmp/dist-launch-init.log'
        
        # Get hostname and rank for log header
        hostname = os.environ.get('HOSTNAME', '')
        if not hostname:
            import socket
            hostname = socket.gethostname()
        rank = os.environ.get('RANK', 'N/A')
        
        # Clear existing log file (or append if it's a shared file)
        mode = 'a' if os.path.exists(LOG_FILE) else 'w'
        with open(LOG_FILE, mode, encoding='utf-8') as f:
            if mode == 'w':
                f.write(f'=== Dist-Launch Init Log ===\n')
            f.write(f'=== Started at {datetime.now().strftime("%Y-%m-%d %H:%M:%S")} ===\n')
            f.write(f'=== Hostname: {hostname}, Rank: {rank} ===\n')
        
        # Replace stdout and stderr with LogWriter
        sys.stdout = LogWriter(LOG_FILE, sys.stdout, is_stderr=False)
        sys.stderr = LogWriter(LOG_FILE, sys.stderr, is_stderr=True)
        print(f'Logging initialized, log file: {LOG_FILE}')
        print(f'Log file directory: {os.path.dirname(LOG_FILE)}')
    except Exception as e:
        # If logging setup fails, continue without file logging
        original_stderr = sys.stderr
        try:
            original_stderr.write(f'Warning: Could not setup file logging: {e}\n')
            original_stderr.write(f'Attempted log file path: {LOG_FILE}\n')
        except Exception:
            pass


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


def update_hosts_file(hostnames, hostname_to_ip=None):
    """
    Update /etc/hosts file on rank0 to add rank-0, rank-1, etc. aliases
    This allows easy login via ssh root@rank-1
    
    Args:
        hostnames: List of hostnames, ordered by rank
        hostname_to_ip: Dictionary mapping hostname to IP address (optional, from allgather)
    
    Returns:
        True if successful
    """
    if hostname_to_ip is None:
        hostname_to_ip = {}
    
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
        
        # Get IP addresses for each hostname (only from allgather, no DNS)
        rank_entries = []
        for rank, hostname in enumerate(hostnames):
            # Only use IP from allgather, no DNS resolution
            if hostname in hostname_to_ip:
                ip = hostname_to_ip[hostname]
                rank_alias = f'rank-{rank}'
                entry = f'{ip}\t{rank_alias}\t# dist-launch: rank{rank} -> {hostname}'
                rank_entries.append(entry)
                print(f'Added hosts entry: rank{rank} ({hostname}) -> {ip}')
            else:
                # If IP not available from allgather, skip this entry
                print(f'Warning: No IP address available for {hostname} (rank {rank}), skipping hosts entry', file=sys.stderr)
        
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


def update_ssh_config(hostnames, ssh_key_path, ssh_port=2025, ssh_user='root'):
    """
    Update ~/.ssh/config file on rank0 to add SSH configuration for rank-0, rank-1, etc.
    This allows passwordless login via ssh rank-0 without specifying -i key
    
    Args:
        hostnames: List of hostnames, ordered by rank
        ssh_key_path: Path to SSH private key
        ssh_port: SSH port (default 2025)
        ssh_user: SSH username (default root)
    
    Returns:
        True if successful
    """
    try:
        ssh_dir = os.path.expanduser('~/.ssh')
        ssh_config_file = os.path.join(ssh_dir, 'config')
        ssh_config_backup = os.path.join(ssh_dir, 'config.dist-launch.backup')
        
        # Create .ssh directory if not exists
        os.makedirs(ssh_dir, mode=0o700, exist_ok=True)
        
        # Read current SSH config
        current_config = ''
        if os.path.exists(ssh_config_file):
            with open(ssh_config_file, 'r') as f:
                current_config = f.read()
        
        # Create backup
        try:
            with open(ssh_config_backup, 'w') as f:
                f.write(current_config)
            print(f'Created SSH config backup: {ssh_config_backup}')
        except Exception as e:
            print(f'Warning: Could not create SSH config backup: {e}', file=sys.stderr)
        
        # Remove old dist-launch entries
        # SSH config format: Host name\n    option value\n    option value\n
        lines = current_config.split('\n')
        filtered_lines = []
        skip_block = False
        
        for i, line in enumerate(lines):
            stripped = line.strip()
            
            # Check if this is a dist-launch Host block or comment
            if stripped.startswith('Host ') and 'rank-' in stripped.lower():
                skip_block = True
                continue
            if 'dist-launch' in stripped.lower() and 'cluster node ssh' in stripped.lower():
                skip_block = True
                continue
            
            # If we're in a skip block, check if we should stop skipping
            if skip_block:
                # Stop skipping when we hit the next Host entry (non-dist-launch)
                if stripped.startswith('Host ') and 'rank-' not in stripped.lower():
                    skip_block = False
                    filtered_lines.append(line)
                # Stop skipping when we hit a non-indented, non-empty, non-comment line that's not a Host
                elif stripped and not line.startswith(' ') and not line.startswith('\t') and not stripped.startswith('#'):
                    skip_block = False
                    filtered_lines.append(line)
                # Otherwise continue skipping (this line is part of the dist-launch block)
            else:
                filtered_lines.append(line)
        
        # Build new SSH config entries
        config_entries = []
        config_entries.append('\n# Dist-launch cluster node SSH configuration (added automatically)')
        config_entries.append('# This allows passwordless login via: ssh rank-0, ssh rank-1, etc.')
        
        for rank, hostname in enumerate(hostnames):
            rank_alias = f'rank-{rank}'
            config_entries.append(f'\nHost {rank_alias}')
            config_entries.append(f'    HostName {hostname}')
            config_entries.append(f'    User {ssh_user}')
            config_entries.append(f'    Port {ssh_port}')
            config_entries.append(f'    IdentityFile {ssh_key_path}')
            config_entries.append('    StrictHostKeyChecking no')
            config_entries.append('    UserKnownHostsFile /dev/null')
            config_entries.append('    LogLevel ERROR')
        
        # Write updated SSH config
        new_config = '\n'.join(filtered_lines)
        if config_entries:
            new_config += '\n' + '\n'.join(config_entries) + '\n'
        
        try:
            with open(ssh_config_file, 'w') as f:
                f.write(new_config)
            os.chmod(ssh_config_file, 0o600)
            print(f'Updated ~/.ssh/config with rank aliases')
            return True
        except PermissionError:
            print(f'Warning: Permission denied writing to {ssh_config_file}', file=sys.stderr)
            return False
        except Exception as e:
            print(f'Error updating SSH config file: {e}', file=sys.stderr)
            return False
            
    except Exception as e:
        print(f'Error updating SSH config file: {e}', file=sys.stderr)
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
        try:
            os.makedirs(ssh_dir, mode=0o700, exist_ok=True)
            print(f'Created/verified .ssh directory: {ssh_dir}')
        except Exception as e:
            print(f'Error creating .ssh directory {ssh_dir}: {e}', file=sys.stderr)
            return False
        
        # Read existing authorized_keys
        existing_keys = set()
        if os.path.exists(authorized_keys_file):
            try:
                with open(authorized_keys_file, 'r') as f:
                    existing_keys = set(line.strip() for line in f if line.strip())
                print(f'Read existing authorized_keys file with {len(existing_keys)} keys')
            except Exception as e:
                print(f'Warning: Could not read existing authorized_keys: {e}', file=sys.stderr)
                # Continue anyway, will create new file
        else:
            print(f'authorized_keys file does not exist, will create new one: {authorized_keys_file}')
        
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
            try:
                # Use 'a' mode to append, but if file doesn't exist, 'a' will create it
                mode = 'a' if os.path.exists(authorized_keys_file) else 'w'
                with open(authorized_keys_file, mode) as f:
                    # If appending to existing file, ensure newline before adding
                    if mode == 'a' and os.path.getsize(authorized_keys_file) > 0:
                        try:
                            with open(authorized_keys_file, 'rb') as rf:
                                rf.seek(-1, os.SEEK_END)
                                if rf.read(1) != b'\n':
                                    f.write('\n')
                        except Exception:
                            pass
                    f.write(f'{public_key.strip()}\n')
                print(f'Added SSH public key to {authorized_keys_file}')
            except Exception as e:
                print(f'Error writing to authorized_keys file {authorized_keys_file}: {e}', file=sys.stderr)
                import traceback
                traceback.print_exc()
                return False
        
        # Ensure correct permissions
        try:
            os.chmod(ssh_dir, 0o700)
            if os.path.exists(authorized_keys_file):
                os.chmod(authorized_keys_file, 0o600)
                print(f'Set permissions on {authorized_keys_file}: 0o600')
            else:
                print(f'Warning: authorized_keys file was not created: {authorized_keys_file}', file=sys.stderr)
                return False
        except Exception as e:
            print(f'Error setting permissions: {e}', file=sys.stderr)
            return False
        
        # Verify the file exists and has content
        if os.path.exists(authorized_keys_file):
            file_size = os.path.getsize(authorized_keys_file)
            print(f'Verified authorized_keys file exists: {authorized_keys_file} (size: {file_size} bytes)')
        else:
            print(f'Error: authorized_keys file does not exist after creation: {authorized_keys_file}', file=sys.stderr)
            return False
        
        return True
        
    except Exception as e:
        print(f'Error distributing SSH key: {e}', file=sys.stderr)
        return False


def discover_and_save_hostnames():
    """
    Discover all hostnames using PyTorch allgather and save to file
    This should be called on all nodes simultaneously
    """
    print(f'Starting cluster discovery...')
    print(f'Environment: RANK={os.environ.get("RANK", "N/A")}, WORLD_SIZE={os.environ.get("WORLD_SIZE", "N/A")}')
    
    # Save original NCCL environment variables
    nccl_vars = ['NCCL_SOCKET_IFNAME', 'NCCL_IB_DISABLE', 'NCCL_DEBUG']
    original_nccl_env = {}
    for var in nccl_vars:
        if var in os.environ:
            original_nccl_env[var] = os.environ[var]
    
    try:
        print(f'Initializing PyTorch distributed...')
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
        
        print(f'Configuration: master_addr={master_addr}, init_port={init_master_port}, world_size={world_size}, rank={rank}')
        
        # Initialize process group
        backend = 'nccl' if torch.cuda.is_available() else 'gloo'
        print(f'Using backend: {backend} (CUDA available: {torch.cuda.is_available()})')
        
        # Increase timeout for GB200 network initialization
        init_timeout = torch.distributed.default_pg_timeout
        if init_timeout.total_seconds() < 1800:  # 30 minutes
            from datetime import timedelta
            init_timeout = timedelta(seconds=1800)
        
        print(f'Calling dist.init_process_group: backend={backend}, master={master_addr}:{init_master_port}, world_size={world_size}, rank={rank}, timeout={init_timeout.total_seconds()}s')
        dist.init_process_group(
            backend=backend,
            init_method=f'tcp://{master_addr}:{init_master_port}',
            world_size=world_size,
            rank=rank,
            timeout=init_timeout
        )
        print(f'✓ Process group initialized successfully')
        
        world_size = dist.get_world_size()
        rank = dist.get_rank()
        print(f'Confirmed: world_size={world_size}, rank={rank}')
        
        # Get current hostname and IP address
        current_hostname = os.environ.get('HOSTNAME', '')
        if not current_hostname:
            import socket
            current_hostname = socket.gethostname()
        print(f'Current hostname: {current_hostname}')
        
        # Get current node's IP address (no DNS resolution)
        # Method: Get IP from network interfaces by connecting to external address
        current_ip = None
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            try:
                # Connect to a remote address (doesn't actually send data)
                # This gets the local IP address used for the connection
                s.connect(('8.8.8.8', 80))
                current_ip = s.getsockname()[0]
            except Exception as e:
                print(f'Warning: Could not get IP via socket connection: {e}', file=sys.stderr)
            finally:
                s.close()
        except Exception as e:
            print(f'Warning: Could not create socket to get IP: {e}', file=sys.stderr)
        
        if not current_ip:
            # Fallback: try to get IP from hostname (but this might use DNS)
            # Only as last resort
            try:
                current_ip = socket.gethostbyname(current_hostname)
                print(f'Got IP via hostname resolution: {current_ip}')
            except (socket.gaierror, socket.herror):
                # If hostname is already an IP address, use it directly
                import re
                ip_pattern = r'^(\d{1,3}\.){3}\d{1,3}$'
                if re.match(ip_pattern, current_hostname):
                    current_ip = current_hostname
                    print(f'Hostname appears to be an IP address: {current_ip}')
                else:
                    print(f'Error: Could not determine IP address for {current_hostname}', file=sys.stderr)
                    current_ip = current_hostname  # Use hostname as fallback
        else:
            print(f'Current IP address: {current_ip}')
        
        # Convert hostname and IP to bytes for allgather
        hostname_bytes = current_hostname.encode('utf-8')
        ip_bytes = current_ip.encode('utf-8')
        max_len = 256
        hostname_bytes = hostname_bytes[:max_len].ljust(max_len, b'\0')
        ip_bytes = ip_bytes[:max_len].ljust(max_len, b'\0')
        
        # Create tensors for allgather (hostname and IP)
        if torch.cuda.is_available():
            local_hostname_tensor = torch.ByteTensor(list(hostname_bytes)).cuda()
            local_ip_tensor = torch.ByteTensor(list(ip_bytes)).cuda()
            gathered_hostname_tensors = [torch.zeros_like(local_hostname_tensor) for _ in range(world_size)]
            gathered_ip_tensors = [torch.zeros_like(local_ip_tensor) for _ in range(world_size)]
        else:
            local_hostname_tensor = torch.ByteTensor(list(hostname_bytes))
            local_ip_tensor = torch.ByteTensor(list(ip_bytes))
            gathered_hostname_tensors = [torch.zeros_like(local_hostname_tensor) for _ in range(world_size)]
            gathered_ip_tensors = [torch.zeros_like(local_ip_tensor) for _ in range(world_size)]
        
        print(f'Gathering hostnames and IP addresses from all nodes...')
        dist.all_gather(gathered_hostname_tensors, local_hostname_tensor)
        dist.all_gather(gathered_ip_tensors, local_ip_tensor)
        print(f'✓ Hostname and IP gathering completed')
        
        # Extract hostnames and IPs from gathered tensors
        hostnames = []
        hostname_to_ip = {}
        for i, (hostname_tensor, ip_tensor) in enumerate(zip(gathered_hostname_tensors, gathered_ip_tensors)):
            hostname_bytes = bytes(hostname_tensor.cpu().tolist() if torch.cuda.is_available() else hostname_tensor.tolist()).rstrip(b'\0')
            ip_bytes = bytes(ip_tensor.cpu().tolist() if torch.cuda.is_available() else ip_tensor.tolist()).rstrip(b'\0')
            hostname = hostname_bytes.decode('utf-8', errors='ignore')
            ip = ip_bytes.decode('utf-8', errors='ignore')
            if hostname:
                hostnames.append(hostname)
                if ip:
                    hostname_to_ip[hostname] = ip
                    print(f'  Rank {i}: {hostname} -> {ip}')
                else:
                    print(f'  Rank {i}: {hostname} -> (no IP)')
        
        print(f'Discovered {len(hostnames)} hostnames: {", ".join(hostnames)}')
        
        # Distribute SSH public key to all nodes
        # Support both SSH_KEY and SSH_PUBLIC_KEY environment variables
        ssh_key = os.environ.get('SSH_KEY', '')
        ssh_public_key = os.environ.get('SSH_PUBLIC_KEY', '')
        
        # If SSH_KEY is set, derive public key path
        if ssh_key and not ssh_public_key:
            ssh_public_key = ssh_key + '.pub'
            print(f'Derived SSH public key path from SSH_KEY: {ssh_public_key}')
        elif not ssh_public_key:
            # Default fallback
            ssh_public_key = DEFAULT_SSH_PUBLIC_KEY_PATH
        
        print(f'Distributing SSH public key from {ssh_public_key}...')
        print(f'Public key file exists: {os.path.exists(ssh_public_key)}')
        if os.path.exists(ssh_public_key):
            with open(ssh_public_key, 'r') as f:
                key_preview = f.read().strip()[:50]
                print(f'Public key preview: {key_preview}...')
        else:
            print(f'Warning: SSH public key file not found at {ssh_public_key}')
            if ssh_key:
                print(f'SSH_KEY was set to: {ssh_key}')
            if ssh_public_key != ssh_key + '.pub':
                print(f'Trying alternative: {ssh_key}.pub' if ssh_key else 'No SSH_KEY set')
        
        print(f'Calling distribute_ssh_key...')
        success = distribute_ssh_key(hostnames, ssh_public_key)
        if success:
            print(f'✓ SSH public key distribution completed on rank {rank}')
        else:
            print(f'✗ Warning: SSH public key distribution may have failed on rank {rank}', file=sys.stderr)
        
        print(f'Waiting for barrier...')
        dist.barrier()
        print(f'✓ Barrier completed')

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
            
            update_hosts_file(hostnames, hostname_to_ip)
            
            # Update SSH config for easy login without specifying key
            # Use the same SSH_KEY that was used for public key distribution
            ssh_key_path = os.environ.get('SSH_KEY', '')
            if not ssh_key_path:
                # If SSH_KEY not set, try to derive from SSH_PUBLIC_KEY
                ssh_public_key_env = os.environ.get('SSH_PUBLIC_KEY', '')
                if ssh_public_key_env and ssh_public_key_env.endswith('.pub'):
                    ssh_key_path = ssh_public_key_env[:-4]  # Remove .pub extension
                else:
                    # Default fallback
                    ssh_key_path = DEFAULT_SSH_KEY_PATH
            
            ssh_port = int(os.environ.get('SSH_PORT', '2025'))
            ssh_user = os.environ.get('SSH_USER', 'root')
            print(f'Updating SSH config with key: {ssh_key_path}, port: {ssh_port}, user: {ssh_user}')
            update_ssh_config(hostnames, ssh_key_path, ssh_port, ssh_user)
            
            dist.barrier()
            return hostnames
        
        dist.barrier()
        return None
        
    except Exception as e:
        print(f'Error discovering hosts: {e}', file=sys.stderr)
        import traceback
        print(f'Traceback:', file=sys.stderr)
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
    # Setup logging to file
    setup_logging()
    
    try:
        hostnames = discover_and_save_hostnames()
        if hostnames:
            print(f'Cluster initialization completed successfully on rank 0')
            print(f'Log file: {LOG_FILE}')
            # Close log file handle
            LogWriter.close_log_file()
            sys.exit(0)
        else:
            print(f'Cluster initialization completed (rank > 0)')
            # Close log file handle
            LogWriter.close_log_file()
            sys.exit(0)  # Other ranks also exit successfully
    except Exception as e:
        print(f'Error: {e}', file=sys.stderr)
        import traceback
        traceback.print_exc()
        # Close log file handle even on error
        LogWriter.close_log_file()
        sys.exit(1)


if __name__ == '__main__':
    main()

