"""
Host Discovery - Automatically discover cluster node hostnames
"""
import os
import subprocess
import socket
import sys
from typing import List, Optional


class HostDiscovery:
    """Discover cluster node hostnames automatically"""
    
    @staticmethod
    def discover_from_torchrun() -> Optional[List[str]]:
        """
        Discover nodes using PyTorch distributed communication (allgather)
        Directly uses PyTorch distributed without torchrun, assuming all nodes
        can connect to MASTER_ADDR:MASTER_PORT
        
        Returns:
            List of hostnames or None if discovery fails
        """
        try:
            # Check if we have required environment variables
            master_addr = os.environ.get('MASTER_ADDR', '')
            master_port = os.environ.get('MASTER_PORT', '')
            world_size = os.environ.get('WORLD_SIZE', '')
            rank = os.environ.get('RANK', '')
            
            if not all([master_addr, master_port, world_size, rank]):
                return None
            
            # Import PyTorch distributed modules
            try:
                import torch
                import torch.distributed as dist
            except ImportError:
                return None
            
            # Runtime discovery via torchrun is not supported
            # Hostname discovery should be done during cluster initialization (init_cluster.py)
            # This method is kept for fallback but will likely fail
            # The preferred method is to use discover_from_cluster_info()
            return None
            
        except Exception as e:
            print(f'Discovery exception: {e}', file=sys.stderr)
            return None
    
    @staticmethod
    def discover_from_hostname_pattern() -> Optional[List[str]]:
        """
        Discover nodes by inferring hostname pattern
        
        Returns:
            List of hostnames or None if pattern cannot be inferred
        """
        try:
            current_hostname = socket.gethostname()
            world_size = int(os.environ.get('WORLD_SIZE', '0'))
            
            if world_size == 0 or not current_hostname:
                return None
            
            # Try to extract rank from hostname
            # Common patterns: hostname-0, hostname-1, or hostname-0-xxx
            parts = current_hostname.rsplit('-', 1)
            if len(parts) != 2:
                return None
            
            base_name = parts[0]
            try:
                current_rank = int(parts[1])
            except ValueError:
                return None
            
            # Construct hostnames for all ranks
            hostnames = []
            for i in range(world_size):
                hostname = f'{base_name}-{i}'
                hostnames.append(hostname)
            
            return hostnames
            
        except Exception:
            return None
    
    @staticmethod
    def discover_from_dns() -> Optional[List[str]]:
        """
        Discover nodes by DNS resolution (if using service discovery)
        
        Returns:
            List of hostnames or None
        """
        try:
            master_addr = os.environ.get('MASTER_ADDR', '')
            world_size = int(os.environ.get('WORLD_SIZE', '0'))
            
            if not master_addr or world_size == 0:
                return None
            
            # Try to extract base name from MASTER_ADDR
            # Common pattern: job-name-master-0 or job-name-0
            parts = master_addr.rsplit('-', 1)
            if len(parts) != 2:
                return None
            
            base_part = parts[0]
            # Remove common suffixes
            for suffix in ['-master', '-worker']:
                if base_part.endswith(suffix):
                    base_part = base_part[:-len(suffix)]
                    break
            
            hostnames = []
            for i in range(world_size):
                # Try different patterns
                patterns = [
                    f'{base_part}-{i}',
                    f'{base_part}-master-{i}' if i == 0 else f'{base_part}-worker-{i-1}',
                ]
                for pattern in patterns:
                    try:
                        socket.gethostbyname(pattern)
                        hostnames.append(pattern)
                        break
                    except socket.gaierror:
                        continue
            
            if len(hostnames) == world_size:
                return hostnames
                
        except Exception:
            pass
        
        return None
    
    @staticmethod
    def discover_from_env() -> Optional[List[str]]:
        """
        Discover nodes from environment variables (if explicitly set)
        
        Returns:
            List of hostnames or None
        """
        # Check for explicit node list in environment
        node_list = os.environ.get('NODE_LIST', '')
        if node_list:
            return [n.strip() for n in node_list.split(',')]
        
        # Check for HOSTFILE
        hostfile = os.environ.get('HOSTFILE', '')
        if hostfile and os.path.exists(hostfile):
            try:
                with open(hostfile, 'r') as f:
                    hosts = [line.strip() for line in f if line.strip() and not line.startswith('#')]
                    if hosts:
                        return hosts
            except Exception:
                pass
        
        return None
    
    @staticmethod
    def discover_from_cluster_info() -> Optional[List[str]]:
        """
        Discover nodes from cluster info file (saved during initialization)
        
        Returns:
            List of hostnames or None
        """
        try:
            # Check for cluster info file
            info_file = os.environ.get('CLUSTER_INFO_FILE', '/tmp/cluster_info.json')
            if os.path.exists(info_file):
                import json
                with open(info_file, 'r') as f:
                    cluster_info = json.load(f)
                    hostnames = cluster_info.get('hostnames', [])
                    if hostnames:
                        return hostnames
        except Exception as e:
            print(f'Error reading cluster info file: {e}', file=sys.stderr)
        
        return None
    
    @staticmethod
    def discover_all() -> Optional[List[str]]:
        """
        Try all discovery methods in order
        
        Returns:
            List of hostnames or None if all methods fail
        """
        methods = [
            HostDiscovery.discover_from_env,  # Explicit NODE_LIST or HOSTFILE
            HostDiscovery.discover_from_cluster_info,  # Saved cluster info from initialization
            HostDiscovery.discover_from_torchrun,  # Runtime discovery via torchrun
            HostDiscovery.discover_from_hostname_pattern,
            HostDiscovery.discover_from_dns,
        ]
        
        for method in methods:
            try:
                hosts = method()
                if hosts:
                    return hosts
            except Exception:
                continue
        
        return None

