"""
Cluster Manager - Manages cluster nodes and their configurations
"""
import os
from typing import List, Dict, Optional
from dataclasses import dataclass


@dataclass
class NodeConfig:
    """Node configuration"""
    name: str
    rank: int
    node_rank: int
    hostname: str
    ip: Optional[str] = None


class ClusterManager:
    """Manages cluster nodes and their environment variables"""
    
    def __init__(self, master_addr: str, world_size: int, master_port: int = 23456):
        """
        Initialize cluster manager
        
        Args:
            master_addr: Master node address (hostname or IP)
            world_size: Total number of nodes
            master_port: Master port for communication
        """
        self.master_addr = master_addr
        self.world_size = world_size
        self.master_port = master_port
        self.nodes: List[NodeConfig] = []
        
    def add_node(self, name: str, rank: int, node_rank: int, hostname: str, ip: Optional[str] = None):
        """Add a node to the cluster"""
        node = NodeConfig(name=name, rank=rank, node_rank=node_rank, hostname=hostname, ip=ip)
        self.nodes.append(node)
        return node
    
    def get_node_env(self, node: NodeConfig, use_existing: bool = False) -> Dict[str, str]:
        """
        Get environment variables for a specific node
        
        Args:
            node: Node configuration
            use_existing: If True, use existing env vars from image as base, but always set node-specific vars
            
        Returns:
            Dictionary of environment variables
        """
        # Always set node-specific environment variables explicitly
        # For remote nodes via SSH, we must pass env vars explicitly because SSH non-interactive
        # shell doesn't inherit environment variables from pytorch-job
        env_vars = {
            'PET_NODE_RANK': str(node.node_rank),
            'RANK': str(node.rank),
            'WORLD_SIZE': str(self.world_size),
            'MASTER_PORT': str(self.master_port),
            'PET_MASTER_PORT': str(self.master_port),
            'PET_MASTER_ADDR': self.master_addr,
            'MASTER_ADDR': self.master_addr,
        }
        
        if use_existing:
            # If use_existing is True, we still set the env vars above, but for local rank0
            # we can optionally merge with existing env vars if they exist
            # For remote nodes, we always need to pass env vars via SSH
            import os
            # Only for local rank0, we might want to preserve some existing env vars
            # But for now, we always set explicitly to ensure correctness
            pass
        
        return env_vars
    
    def get_rank0_node(self) -> Optional[NodeConfig]:
        """Get rank 0 node"""
        for node in self.nodes:
            if node.rank == 0:
                return node
        return None
    
    def get_all_nodes(self) -> List[NodeConfig]:
        """Get all nodes"""
        return sorted(self.nodes, key=lambda x: x.rank)

