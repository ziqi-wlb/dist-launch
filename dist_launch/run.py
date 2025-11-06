#!/usr/bin/env python3
"""
Main entry script for launching distributed training across cluster nodes
"""
import sys
import os
import argparse
import time
import subprocess
import json
from typing import List, Optional

# Add lib directory to path
# Support both direct execution and package installation
lib_path = os.path.join(os.path.dirname(__file__), 'lib')
if not os.path.exists(lib_path):
    # Try installed package path
    import importlib.util
    spec = importlib.util.find_spec('dist_launch')
    if spec and spec.origin:
        lib_path = os.path.join(os.path.dirname(spec.origin), 'lib')
sys.path.insert(0, lib_path)

from cluster_manager import ClusterManager, NodeConfig
from node_executor import NodeExecutor
from host_discovery import HostDiscovery


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Launch distributed training across cluster nodes')
    
    # Auto-detect from environment if available
    env_master_addr = os.environ.get('MASTER_ADDR', '')
    env_world_size = os.environ.get('WORLD_SIZE', '')
    env_master_port = os.environ.get('MASTER_PORT', '23456')
    
    # train-script as first positional argument
    parser.add_argument('train_script', type=str,
                       help='Path to training script on each node')
    parser.add_argument('--master-addr', type=str, default=env_master_addr,
                       help='Master node address (default: from MASTER_ADDR env)')
    parser.add_argument('--world-size', type=int, default=int(env_world_size) if env_world_size else None,
                       help='Total number of nodes (default: from WORLD_SIZE env)')
    parser.add_argument('--master-port', type=int, default=int(env_master_port) if env_master_port else 23456,
                       help='Master port (default: from MASTER_PORT env or 23456)')
    parser.add_argument('--nodes', type=str, default=None,
                       help='Comma-separated list of node hostnames (auto-discover if not provided). '
                            'Useful for excluding bad nodes. Example: --nodes "node1,node2,node3" --world-size 3')
    # SSH configuration - allow override from environment variables
    env_ssh_key = os.environ.get('SSH_KEY', '')
    env_ssh_port = os.environ.get('SSH_PORT', '')
    env_ssh_user = os.environ.get('SSH_USER', '')
    
    parser.add_argument('--ssh-key', type=str,
                       default=env_ssh_key if env_ssh_key else '/mnt/3fs/dots-pretrain/weishi/release/public/ssh-key/id_rsa',
                       help='Path to SSH private key (default: from SSH_KEY env or /mnt/3fs/...)')
    parser.add_argument('--ssh-port', type=int, 
                       default=int(env_ssh_port) if env_ssh_port and env_ssh_port.isdigit() else 2025,
                       help='SSH port (default: from SSH_PORT env or 2025)')
    parser.add_argument('--ssh-user', type=str, 
                       default=env_ssh_user if env_ssh_user else 'root',
                       help='SSH username (default: from SSH_USER env or root)')
    parser.add_argument('--dry-run', action='store_true',
                       help='Dry run mode - show commands without executing')
    parser.add_argument('--wait', action='store_true', default=True,
                       help='Wait for all processes to complete (default: True)')
    parser.add_argument('--no-wait', dest='wait', action='store_false',
                       help='Do not wait for processes to complete')
    
    return parser.parse_args()


def create_cluster(nodes: List[str], master_addr: str, world_size: int) -> ClusterManager:
    """
    Create cluster configuration from node list
    
    Args:
        nodes: List of node hostnames
        master_addr: Master node address
        world_size: Total number of nodes
        
    Returns:
        ClusterManager instance
    """
    cluster = ClusterManager(master_addr, world_size)
    
    for idx, hostname in enumerate(nodes):
        cluster.add_node(
            name=f'node{idx}',
            rank=idx,
            node_rank=idx,
            hostname=hostname
        )
    
    return cluster


def launch_training(cluster: ClusterManager, executor: NodeExecutor, train_script: str,
                   dry_run: bool = False, wait: bool = False, use_existing_env: bool = True) -> tuple:
    """
    Launch training on all nodes
    
    Args:
        cluster: ClusterManager instance
        executor: NodeExecutor instance
        train_script: Path to training script
        dry_run: Whether to only show commands
        wait: Whether to wait for completion
        use_existing_env: If True, use existing env vars from image
        
    Returns:
        List of process objects
    """
    processes = []
    all_nodes = cluster.get_all_nodes()
    
    # Convert train_script to absolute path if it's relative
    if not os.path.isabs(train_script):
        # Use current working directory
        cwd = os.getcwd()
        train_script_abs = os.path.join(cwd, train_script)
    else:
        train_script_abs = train_script
    
    print(f'Launching training on {len(all_nodes)} nodes...')
    
    # Prepare all nodes first (collect commands and env vars)
    # This ensures all nodes start as simultaneously as possible
    node_commands = []
    for node in all_nodes:
        env_vars = cluster.get_node_env(node, use_existing=use_existing_env)
        env_str = ' '.join([f'{k}="{v}"' for k, v in env_vars.items()]) if env_vars else ''
        command = f'{env_str} bash {train_script_abs}'.strip()
        
        if dry_run:
            print(f'[DRY RUN] Node {node.name} (rank {node.rank}): {command}')
            continue
        
        if node.rank == 0:
            # Rank0 will be executed locally, store for later launch
            node_commands.append({
                'node': node,
                'type': 'local',
                'env_vars': env_vars,
                'command': command,
                'train_script_abs': train_script_abs
            })
            continue
        
        # Remote nodes - prepare for async launch
        # Always pass env vars to remote nodes via SSH, because SSH non-interactive shell
        # doesn't automatically inherit environment variables from pytorch-job
        # We need to explicitly pass the correct env vars for each node
        exec_env = env_vars if env_vars else None
        # Use absolute path for remote nodes to ensure script is found
        # Assume all nodes have the same directory structure
        node_commands.append({
            'node': node,
            'type': 'remote',
            'env_vars': exec_env,
            'command': f'bash {train_script_abs}',
            'work_dir': None  # Use absolute path, no need to cd
        })
    
    # Launch all remote nodes asynchronously (non-blocking)
    # This ensures all nodes start as simultaneously as possible for distributed training
    print('Launching all remote nodes asynchronously (non-blocking)...')
    remote_nodes_to_launch = [cmd_info for cmd_info in node_commands if cmd_info.get('type') == 'remote']
    
    # Launch all remote nodes in quick succession (all are non-blocking)
    for cmd_info in remote_nodes_to_launch:
        node = cmd_info['node']
        print(f'Launching on node {node.name} (rank {node.rank})...')
        try:
            process = executor.execute(
                node, 
                cmd_info['command'], 
                cmd_info['env_vars'], 
                background=True, 
                work_dir=cmd_info['work_dir']
            )
            processes.append((node, process))
            print(f'  ✓ Launched (PID: {process.pid})')
        except Exception as e:
            print(f'  ✗ Failed to launch: {e}')
    
    return processes, node_commands


def main():
    """Main entry point"""
    args = parse_args()
    
    # Validate required parameters
    if not args.master_addr:
        print('Error: --master-addr is required or MASTER_ADDR env must be set')
        sys.exit(1)
    
    if args.world_size is None or args.world_size <= 0:
        print('Error: --world-size is required or WORLD_SIZE env must be set')
        sys.exit(1)
    
    # Get nodes list
    if args.nodes:
        # Manual node list specified
        nodes = [n.strip() for n in args.nodes.split(',')]
        nodes = [n for n in nodes if n]  # Remove empty strings
        print(f'Using manually specified {len(nodes)} nodes: {", ".join(nodes)}')
        
        # Validate node count matches world_size
        if len(nodes) != args.world_size:
            print(f'Warning: Number of specified nodes ({len(nodes)}) does not match world_size ({args.world_size})')
            print(f'This is allowed for scenarios like excluding bad nodes, but ensure your training script handles this correctly')
            
            # If nodes count is less than world_size, warn but continue
            if len(nodes) < args.world_size:
                print(f'Note: Only {len(nodes)} nodes will be used, but WORLD_SIZE={args.world_size} will be set')
                print(f'Make sure your training script can handle fewer nodes than WORLD_SIZE')
            # If nodes count is more than world_size, use only first world_size nodes
            elif len(nodes) > args.world_size:
                print(f'Warning: More nodes specified than world_size, using first {args.world_size} nodes')
                nodes = nodes[:args.world_size]
    else:
        # Auto-discover nodes
        nodes = HostDiscovery.discover_all()
        if not nodes:
            print('Error: Cannot auto-discover nodes.')
            print('Make sure cluster was initialized (dist-launch wait) or provide --nodes argument')
            print('Example: dist-launch run train.sh --nodes "node1,node2,node3" --world-size 3')
            sys.exit(1)
        print(f'Discovered {len(nodes)} nodes: {", ".join(nodes)}')
        
        # Validate discovered node count matches world_size
        if len(nodes) != args.world_size:
            print(f'Warning: Number of discovered nodes ({len(nodes)}) does not match world_size ({args.world_size})')
            print(f'If you have bad nodes, use --nodes to specify only working nodes')
    
    # Validate that we have at least one node
    if not nodes:
        print('Error: No nodes specified or discovered')
        sys.exit(1)
    
    # Ensure rank0 node is correctly set
    # Rank0 should be the node where this script is running (current hostname)
    current_hostname = os.environ.get('HOSTNAME', '')
    if not current_hostname:
        import socket
        current_hostname = socket.gethostname()
    
    # If manually specifying nodes, ensure current hostname is rank0
    if args.nodes:
        if current_hostname not in nodes:
            print(f'Warning: Current hostname ({current_hostname}) is not in the specified node list')
            print(f'Adding current hostname as rank0 (first node)')
            nodes.insert(0, current_hostname)
            # Update world_size if needed
            if len(nodes) > args.world_size:
                print(f'Note: Node count ({len(nodes)}) now exceeds world_size ({args.world_size})')
        elif nodes[0] != current_hostname:
            # Current hostname is in list but not first - move it to first position
            nodes.remove(current_hostname)
            nodes.insert(0, current_hostname)
            print(f'Reordered nodes to put current hostname ({current_hostname}) as rank0')
    else:
        # Auto-discovered: current hostname should already be rank0 from discovery
        if current_hostname in nodes and nodes[0] != current_hostname:
            nodes.remove(current_hostname)
            nodes.insert(0, current_hostname)
            print(f'Reordered discovered nodes to put current hostname ({current_hostname}) as rank0')
    
    # Create cluster
    cluster = create_cluster(nodes, args.master_addr, args.world_size)
    
    executor = NodeExecutor(
        ssh_key_path=args.ssh_key,
        ssh_port=args.ssh_port,
        ssh_user=args.ssh_user
    )
    
    if not args.dry_run:
        rank0_node = cluster.get_rank0_node()
        if not rank0_node:
            print('Error: Rank 0 node not found')
            sys.exit(1)
        
        all_connected = True
        for node in cluster.get_all_nodes():
            if node.rank == 0:
                continue
            if not executor.test_connection(node):
                print(f'Error: Cannot connect to {node.hostname}')
                all_connected = False
        
        if not all_connected:
            print('Error: Some nodes are not reachable')
            sys.exit(1)
    
    # Default to True: use existing env vars from pytorch-job
    # Only set to False if explicitly requested (e.g., for testing without pytorch-job)
    use_existing_env = True
    if os.environ.get('DIST_LAUNCH_OVERRIDE_ENV', '').lower() in ('1', 'true', 'yes'):
        use_existing_env = False
    
    # Launch training - returns (processes, node_commands)
    processes, node_commands = launch_training(cluster, executor, args.train_script,
                                               dry_run=args.dry_run, wait=args.wait,
                                               use_existing_env=use_existing_env)
    
    # Launch rank0 locally immediately after remote nodes (for synchronous startup)
    if not args.dry_run:
        # Find rank0 command info
        rank0_cmd_info = None
        for cmd_info in node_commands:
            if cmd_info.get('type') == 'local':
                rank0_cmd_info = cmd_info
                break
        
        if rank0_cmd_info:
            rank0_node = rank0_cmd_info['node']
            env_vars = rank0_cmd_info['env_vars']
            train_script_abs = rank0_cmd_info['train_script_abs']
            
            # Prepare environment for rank0
            # If use_existing_env is True, use existing env vars from pytorch-job
            # Only add missing env vars, don't override existing ones
            if use_existing_env:
                full_env = os.environ.copy()
                # Only add env vars that are not already set
                for key, value in env_vars.items():
                    if key not in full_env:
                        full_env[key] = value
            else:
                full_env = os.environ.copy()
                full_env.update(env_vars)
            
            print(f'\nLaunching rank0 locally...')
            
            # Save process info for kill command
            pid_info_file = '/tmp/dist-launch-pids.json'
            pid_info = {
                'train_script': train_script_abs,
                'rank0_pid': None,
                'remote_processes': []
            }
            
            if args.wait:
                # Execute rank0 immediately (all nodes should start around the same time)
                # This ensures distributed training can establish connections properly
                rank0_process = subprocess.Popen(
                    ['bash', train_script_abs],
                    env=full_env
                )
                pid_info['rank0_pid'] = rank0_process.pid
                
                # Save remote process info
                for node, process in processes:
                    pid_info['remote_processes'].append({
                        'hostname': node.hostname,
                        'rank': node.rank,
                        'pid': process.pid
                    })
                
                # Write PID info file
                try:
                    with open(pid_info_file, 'w') as f:
                        json.dump(pid_info, f, indent=2)
                    print(f'\nProcess info saved to {pid_info_file}')
                    print(f'Use "dist-launch kill" to stop all training processes')
                except Exception as e:
                    print(f'Warning: Could not save process info: {e}', file=sys.stderr)
                
                # Wait for all processes to complete
                print('\nWaiting for all processes to complete...')
                
                # Wait for rank0
                rank0_process.wait()
                if rank0_process.returncode == 0:
                    print(f'  ✓ rank0 completed successfully')
                else:
                    print(f'  ✗ rank0 failed with return code {rank0_process.returncode}')
                
                # Wait for other nodes
                # Note: stdout/stderr are None (redirected to terminal), so logs are already visible
                for node, process in processes:
                    try:
                        print(f'\nWaiting for {node.name} (rank {node.rank}) to complete...')
                        process.wait()
                        if process.returncode == 0:
                            print(f'  ✓ {node.name} (rank {node.rank}) completed successfully')
                        else:
                            print(f'  ✗ {node.name} (rank {node.rank}) failed with return code {process.returncode}')
                    except Exception as e:
                        print(f'  ✗ {node.name} (rank {node.rank}) error: {e}')
            else:
                # Execute rank0 in background
                rank0_process = subprocess.Popen(
                    ['bash', train_script_abs],
                    env=full_env,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE
                )
                pid_info['rank0_pid'] = rank0_process.pid
                
                # Save remote process info
                for node, process in processes:
                    pid_info['remote_processes'].append({
                        'hostname': node.hostname,
                        'rank': node.rank,
                        'pid': process.pid
                    })
                
                # Write PID info file
                try:
                    with open(pid_info_file, 'w') as f:
                        json.dump(pid_info, f, indent=2)
                    print(f'Use "dist-launch kill" to stop all training processes')
                except Exception as e:
                    print(f'Warning: Could not save process info: {e}', file=sys.stderr)
                
                print(f'✓ Rank0 launched in background (PID: {rank0_process.pid})')
                print(f'All nodes launched. Use --wait to wait for completion.')
    
    if args.dry_run:
        rank0_node = cluster.get_rank0_node()
        if rank0_node:
            # Convert train_script to absolute path if it's relative
            if not os.path.isabs(args.train_script):
                cwd = os.getcwd()
                train_script_abs = os.path.join(cwd, args.train_script)
            else:
                train_script_abs = args.train_script
            env_vars = cluster.get_node_env(rank0_node, use_existing=use_existing_env)
            env_str = ' '.join([f'{k}="{v}"' for k, v in env_vars.items()]) if env_vars else ''
            command = f'{env_str} bash {train_script_abs}'.strip()
            print(f'\n[DRY RUN] Rank0 local execution: {command}')


if __name__ == '__main__':
    main()

