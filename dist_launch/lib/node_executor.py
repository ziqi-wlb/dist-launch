"""
Node Executor - Executes commands on cluster nodes via SSH
"""
import subprocess
import os
from typing import List, Dict, Optional
from cluster_manager import NodeConfig

# Import function to get project SSH key path
try:
    from dist_launch import get_project_ssh_key_path
except ImportError:
    # Fallback for direct import
    import sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    from dist_launch import get_project_ssh_key_path


class NodeExecutor:
    """Executes commands on remote nodes via SSH"""
    
    def __init__(self, ssh_key_path: str = None,
                 ssh_port: int = 2025, ssh_user: str = 'root'):
        """
        Initialize node executor
        
        Args:
            ssh_key_path: Path to SSH private key (default: from project ssh-key)
            ssh_port: SSH port (default 2025)
            ssh_user: SSH username
        """
        if ssh_key_path is None:
            ssh_key_path = get_project_ssh_key_path()
        
        # Ensure SSH key permissions are correct (required by SSH)
        try:
            from dist_launch import _fix_ssh_key_permissions
            if not _fix_ssh_key_permissions(ssh_key_path):
                import sys
                print(f'Warning: SSH key permissions may be incorrect for {ssh_key_path}. '
                      f'SSH may fail. Run: chmod 600 {ssh_key_path}', file=sys.stderr)
        except Exception:
            pass  # If we can't fix permissions, let SSH handle the error
        
        self.ssh_key_path = ssh_key_path
        self.ssh_port = ssh_port
        self.ssh_user = ssh_user
        
    def _resolve_hostname(self, hostname: str) -> str:
        """
        Resolve hostname to IP address via DNS or /etc/hosts
        
        Args:
            hostname: Hostname or IP address
            
        Returns:
            IP address if hostname, or original string if already IP or resolution fails
        """
        import socket
        import re
        import os
        
        # Check if already an IP address
        ip_pattern = r'^(\d{1,3}\.){3}\d{1,3}$'
        if re.match(ip_pattern, hostname):
            return hostname  # Already an IP, return as-is
        
        # First, try to resolve from /etc/hosts file
        hosts_file = '/etc/hosts'
        if os.path.exists(hosts_file):
            try:
                with open(hosts_file, 'r') as f:
                    for line in f:
                        line = line.strip()
                        if not line or line.startswith('#'):
                            continue
                        
                        # Parse line: IP hostname1 hostname2 ...
                        parts = line.split()
                        if len(parts) < 2:
                            continue
                        
                        ip = parts[0]
                        # Check if hostname matches any of the hostnames in this line
                        for part in parts[1:]:
                            # Skip comments
                            if part.startswith('#'):
                                break
                            if part == hostname:
                                # Validate IP format
                                if re.match(ip_pattern, ip):
                                    return ip
            except Exception:
                pass  # If reading /etc/hosts fails, fall back to DNS
        
        # Try DNS resolution
        try:
            ip = socket.gethostbyname(hostname)
            return ip
        except (socket.gaierror, socket.herror, OSError):
            # DNS resolution failed, return original hostname
            # SSH will try to resolve it
            return hostname
    
    def _build_ssh_command(self, hostname: str, command: str, env_vars: Optional[Dict[str, str]] = None,
                          work_dir: Optional[str] = None) -> List[str]:
        """
        Build SSH command
        
        Args:
            hostname: Target hostname (will be resolved to IP via DNS)
            command: Command to execute
            env_vars: Environment variables to set
            work_dir: Working directory to cd into before executing command
            
        Returns:
            SSH command as list of arguments
        """
        # Resolve hostname to IP address via DNS
        resolved_hostname = self._resolve_hostname(hostname)
        
        ssh_cmd = [
            'ssh',
            '-i', self.ssh_key_path,
            '-p', str(self.ssh_port),  # Ensure port is explicitly set
            '-o', 'StrictHostKeyChecking=no',
            '-o', 'UserKnownHostsFile=/dev/null',
            '-o', 'LogLevel=ERROR',
            f'{self.ssh_user}@{resolved_hostname}',
        ]
        
        # Debug: Log the SSH command (without password-sensitive info)
        # This helps verify the port is being used correctly
        
        # Build command with optional cd and env vars
        parts = []
        if work_dir:
            parts.append(f'cd {work_dir}')
        if env_vars:
            # Export environment variables so they are available in the script
            export_cmds = ' && '.join([f'export {k}="{v}"' for k, v in env_vars.items()])
            parts.append(export_cmds)
        parts.append(command)
        
        full_command = ' && '.join(parts) if len(parts) > 1 else parts[0]
        ssh_cmd.append(full_command)
        return ssh_cmd
    
    def execute(self, node: NodeConfig, command: str, env_vars: Optional[Dict[str, str]] = None,
                background: bool = False, work_dir: Optional[str] = None) -> subprocess.Popen:
        """
        Execute command on a node
        
        Args:
            node: Target node
            command: Command to execute
            env_vars: Environment variables to set
            background: Whether to run in background
            work_dir: Working directory to cd into before executing
            
        Returns:
            Process object
        """
        ssh_cmd = self._build_ssh_command(node.hostname, command, env_vars, work_dir)
        
        if background:
            # For background processes, redirect output to terminal so logs are visible
            # Use None to inherit parent's stdout/stderr, allowing real-time log viewing
            process = subprocess.Popen(
                ssh_cmd,
                stdout=None,  # Output directly to terminal for real-time viewing
                stderr=subprocess.STDOUT,  # Merge stderr to stdout
                preexec_fn=os.setsid if hasattr(os, 'setsid') else None
            )
        else:
            process = subprocess.Popen(
                ssh_cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE
            )
            
        return process
    
    def execute_sync(self, node: NodeConfig, command: str, env_vars: Optional[Dict[str, str]] = None,
                    work_dir: Optional[str] = None) -> tuple:
        """
        Execute command synchronously and return result
        
        Args:
            node: Target node
            command: Command to execute
            env_vars: Environment variables to set
            work_dir: Working directory to cd into before executing
            
        Returns:
            Tuple of (returncode, stdout, stderr)
        """
        ssh_cmd = self._build_ssh_command(node.hostname, command, env_vars, work_dir)
        
        result = subprocess.run(
            ssh_cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        
        return result.returncode, result.stdout, result.stderr
    
    def test_connection(self, node: NodeConfig) -> bool:
        """
        Test SSH connection to a node
        
        Args:
            node: Target node
            
        Returns:
            True if connection successful
        """
        try:
            returncode, _, _ = self.execute_sync(node, 'echo "test"')
            return returncode == 0
        except Exception:
            return False
