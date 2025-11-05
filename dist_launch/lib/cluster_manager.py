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
            use_existing: If True, use existing env vars from image and only set missing ones
            
        Returns:
            Dictionary of environment variables
        """
        env_vars = {}
        
        if use_existing:
            # Use existing env from pytorch-job image if available
            import os
            existing_rank = os.environ.get('RANK', None)
            existing_node_rank = os.environ.get('PET_NODE_RANK', None)
            existing_world_size = os.environ.get('WORLD_SIZE', None)
            existing_master_addr = os.environ.get('MASTER_ADDR', None)
            existing_master_port = os.environ.get('MASTER_PORT', None)
            
            # Only set if not already set by image
            if existing_node_rank is None:
                env_vars['PET_NODE_RANK'] = str(node.node_rank)
            if existing_rank is None:
                env_vars['RANK'] = str(node.rank)
            if existing_world_size is None:
                env_vars['WORLD_SIZE'] = str(self.world_size)
            if existing_master_port is None:
                env_vars['MASTER_PORT'] = str(self.master_port)
                env_vars['PET_MASTER_PORT'] = str(self.master_port)
            if existing_master_addr is None:
                env_vars['PET_MASTER_ADDR'] = self.master_addr
                env_vars['MASTER_ADDR'] = self.master_addr
        else:
            # Set all env vars explicitly
            env_vars = {
                'PET_NODE_RANK': str(node.node_rank),
                'RANK': str(node.rank),
                'WORLD_SIZE': str(self.world_size),
                'MASTER_PORT': str(self.master_port),
                'PET_MASTER_PORT': str(self.master_port),
                'PET_MASTER_ADDR': self.master_addr,
                'MASTER_ADDR': self.master_addr,
            }
        
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

