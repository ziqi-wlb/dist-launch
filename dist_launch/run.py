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
    from dist_launch import get_project_ssh_key_path
    env_ssh_key = os.environ.get('SSH_KEY', '')
    env_ssh_port = os.environ.get('SSH_PORT', '')
    env_ssh_user = os.environ.get('SSH_USER', '')
    
    parser.add_argument('--ssh-key', type=str,
                       default=env_ssh_key if env_ssh_key else get_project_ssh_key_path(),
                       help='Path to SSH private key (default: from SSH_KEY env or project ssh-key)')
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
    parser.add_argument('--nper-node', type=int, default=None,
                       help='Number of GPUs per node (default: 1). If specified, WORLD_SIZE will be num_nodes * nper_node')
    
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
                   dry_run: bool = False, wait: bool = False, use_existing_env: bool = True,
                   nper_node: int = 1) -> tuple:
    """
    Launch training on all nodes
    
    Args:
        cluster: ClusterManager instance
        executor: NodeExecutor instance
        train_script: Path to training script
        dry_run: Whether to only show commands
        wait: Whether to wait for completion
        use_existing_env: If True, use existing env vars from image
        nper_node: Number of GPUs per node (default: 1)
        
    Returns:
        List of process objects
    """
    processes = []
    all_nodes = cluster.get_all_nodes()
    
    # Get current working directory (master's working directory)
    # All nodes should execute commands/scripts in the same directory
    master_work_dir = os.getcwd()
    
    # Check if train_script is a file or a command
    # If it's a file, use absolute path; if it's a command (like 'pwd'), use it directly
    if os.path.isfile(train_script):
        # It's a file - convert to absolute path if relative
        if not os.path.isabs(train_script):
            train_script_abs = os.path.join(master_work_dir, train_script)
        else:
            train_script_abs = train_script
        # Use bash to execute the script
        command_template = 'bash {script}'
        is_command = False
    elif os.path.isabs(train_script) and os.path.isfile(train_script):
        # Absolute path to existing file
        train_script_abs = train_script
        command_template = 'bash {script}'
        is_command = False
    else:
        # It's a command (like 'pwd', 'ls -la', etc.) - use it directly
        train_script_abs = train_script
        command_template = '{script}'  # Execute directly, not via bash
        is_command = True
    
    print(f'Launching training on {len(all_nodes)} nodes with {nper_node} GPU(s) per node...')
    print(f'Working directory: {master_work_dir}')
    
    # Prepare all processes (one per GPU per node)
    # This ensures all processes start as simultaneously as possible
    node_commands = []
    for node in all_nodes:
        for local_rank in range(nper_node):
            # Calculate global rank: node_rank * nper_node + local_rank
            global_rank = node.node_rank * nper_node + local_rank
            
            # Get base env vars for this node
            env_vars = cluster.get_node_env(node, use_existing=use_existing_env)
            # Override RANK with global rank
            env_vars['RANK'] = str(global_rank)
            # Set LOCAL_RANK
            env_vars['LOCAL_RANK'] = str(local_rank)
            # Update WORLD_SIZE to total number of processes
            env_vars['WORLD_SIZE'] = str(len(all_nodes) * nper_node)
            
            env_str = ' '.join([f'{k}="{v}"' for k, v in env_vars.items()]) if env_vars else ''
            # Use command_template to handle both scripts and commands
            script_cmd = command_template.format(script=train_script_abs)
            command = f'{env_str} {script_cmd}'.strip()
            
            if dry_run:
                print(f'[DRY RUN] Node {node.name} (node_rank {node.node_rank}, local_rank {local_rank}, global_rank {global_rank}): {command}')
                continue
            
            if node.node_rank == 0:
                # All processes on rank0 node are local
                node_commands.append({
                    'node': node,
                    'local_rank': local_rank,
                    'global_rank': global_rank,
                    'type': 'local',
                    'env_vars': env_vars,
                    'command': command,
                    'train_script_abs': train_script_abs,
                    'is_command': is_command,
                    'command_template': command_template,
                    'work_dir': master_work_dir
                })
                continue
            
            # Remote nodes - prepare for async launch
            # Always pass env vars to remote nodes via SSH, because SSH non-interactive shell
            # doesn't automatically inherit environment variables from pytorch-job
            exec_env = env_vars if env_vars else None
            # Use command_template for remote nodes too
            remote_script_cmd = command_template.format(script=train_script_abs)
            node_commands.append({
                'node': node,
                'local_rank': local_rank,
                'global_rank': global_rank,
                'type': 'remote',
                'env_vars': exec_env,
                'command': remote_script_cmd,
                'work_dir': master_work_dir,  # All nodes execute in same directory as master
                'is_command': is_command,
                'command_template': command_template
            })
    
    # Launch all remote nodes asynchronously (non-blocking)
    # This ensures all nodes start as simultaneously as possible for distributed training
    print('Launching all remote nodes asynchronously (non-blocking)...')
    remote_nodes_to_launch = [cmd_info for cmd_info in node_commands if cmd_info.get('type') == 'remote']
    
    # Launch all remote nodes in quick succession (all are non-blocking)
    for cmd_info in remote_nodes_to_launch:
        node = cmd_info['node']
        local_rank = cmd_info.get('local_rank', 0)
        global_rank = cmd_info.get('global_rank', node.rank)
        print(f'Launching on node {node.name} (node_rank {node.node_rank}, local_rank {local_rank}, global_rank {global_rank})...')
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
    
    return processes, node_commands, master_work_dir


def main():
    """Main entry point"""
    args = parse_args()
    
    # Validate required parameters
    if not args.master_addr:
        print('Error: --master-addr is required or MASTER_ADDR env must be set')
        sys.exit(1)
    
    # Handle nper_node parameter
    nper_node = args.nper_node if args.nper_node is not None else 1
    
    # If nper_node is specified, world_size should be num_nodes * nper_node
    # But if world_size is explicitly set, we use it as total number of processes
    # Otherwise, world_size is number of nodes
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
    
    # Calculate actual world_size: if nper_node > 1, world_size should be num_nodes * nper_node
    # But if world_size is explicitly set, we treat it as total number of processes
    num_nodes = len(nodes)
    if nper_node > 1:
        # If nper_node is specified, world_size should be num_nodes * nper_node
        actual_world_size = num_nodes * nper_node
        if args.world_size != actual_world_size:
            print(f'Note: With {num_nodes} nodes and {nper_node} GPUs per node, total processes = {actual_world_size}')
            print(f'Updating world_size from {args.world_size} to {actual_world_size}')
            args.world_size = actual_world_size
    else:
        actual_world_size = args.world_size
    
    # Create cluster (world_size here is number of nodes, not total processes)
    cluster = create_cluster(nodes, args.master_addr, num_nodes)
    
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
    
    # Launch training - returns (processes, node_commands, master_work_dir)
    processes, node_commands, master_work_dir = launch_training(cluster, executor, args.train_script,
                                                                dry_run=args.dry_run, wait=args.wait,
                                                                use_existing_env=use_existing_env,
                                                                nper_node=nper_node)
    
    # Launch all local processes (rank0 node with potentially multiple GPUs)
    if not args.dry_run:
        # Find all local command info (all processes on rank0 node)
        local_cmd_infos = [cmd_info for cmd_info in node_commands if cmd_info.get('type') == 'local']
        
        if local_cmd_infos:
            # Save process info for kill command
            pid_info_file = '/tmp/dist-launch-pids.json'
            pid_info = {
                'train_script': local_cmd_infos[0]['train_script_abs'],
                'local_pids': [],
                'remote_processes': []
            }
            
            print(f'\nLaunching {len(local_cmd_infos)} local process(es) on rank0 node...')
            
            local_processes = []  # Track all local processes
            
            if args.wait:
                # Launch all local processes (for multi-GPU, launch all GPUs on rank0 node)
                for cmd_info in local_cmd_infos:
                    env_vars = cmd_info['env_vars']
                    train_script_abs = cmd_info['train_script_abs']
                    local_rank = cmd_info.get('local_rank', 0)
                    global_rank = cmd_info.get('global_rank', 0)
                    
                    # Prepare environment for this process
                    # If use_existing_env is True, use existing env vars from pytorch-job
                    # But always override critical distributed training vars (RANK, WORLD_SIZE, LOCAL_RANK, etc.)
                    if use_existing_env:
                        full_env = os.environ.copy()
                        # Critical distributed training env vars that must be overridden
                        critical_vars = ['RANK', 'WORLD_SIZE', 'LOCAL_RANK', 'MASTER_ADDR', 'MASTER_PORT', 
                                        'PET_NODE_RANK', 'PET_MASTER_ADDR', 'PET_MASTER_PORT']
                        for key, value in env_vars.items():
                            # Always override critical vars, even if they exist
                            if key in critical_vars or key not in full_env:
                                full_env[key] = value
                    else:
                        full_env = os.environ.copy()
                        full_env.update(env_vars)
                    
                    print(f'  Launching local process (local_rank={local_rank}, global_rank={global_rank})...')
                    print(f'    Env: RANK={full_env.get("RANK")}, WORLD_SIZE={full_env.get("WORLD_SIZE")}, LOCAL_RANK={full_env.get("LOCAL_RANK")}, MASTER_ADDR={full_env.get("MASTER_ADDR")}, MASTER_PORT={full_env.get("MASTER_PORT")}')
                    # Use command_template to handle both scripts and commands
                    is_cmd = cmd_info.get('is_command', False)
                    work_dir = cmd_info.get('work_dir', master_work_dir)
                    if is_cmd:
                        # For commands, use shell=True and execute directly
                        local_process = subprocess.Popen(
                            train_script_abs,
                            env=full_env,
                            shell=True,
                            cwd=work_dir
                        )
                    else:
                        # For scripts, use bash
                        local_process = subprocess.Popen(
                            ['bash', train_script_abs],
                            env=full_env,
                            cwd=work_dir
                        )
                    local_processes.append((cmd_info, local_process))
                    pid_info['local_pids'].append({
                        'pid': local_process.pid,
                        'local_rank': local_rank,
                        'global_rank': global_rank
                    })
                    print(f'    ✓ Launched (PID: {local_process.pid})')
                
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
                
                # Wait for all local processes
                all_local_success = True
                for cmd_info, local_process in local_processes:
                    local_rank = cmd_info.get('local_rank', 0)
                    global_rank = cmd_info.get('global_rank', 0)
                    local_process.wait()
                    if local_process.returncode == 0:
                        print(f'  ✓ Local process (local_rank={local_rank}, global_rank={global_rank}) completed successfully')
                    else:
                        print(f'  ✗ Local process (local_rank={local_rank}, global_rank={global_rank}) failed with return code {local_process.returncode}')
                        all_local_success = False
                
                # Wait for all remote processes
                all_remote_success = True
                for node, process in processes:
                    try:
                        # Calculate local_rank and global_rank for remote processes
                        node_rank = node.node_rank
                        local_rank = (node.rank % nper_node) if nper_node > 1 else 0
                        global_rank = node.rank
                        print(f'\nWaiting for {node.name} (node_rank {node_rank}, local_rank {local_rank}, global_rank {global_rank}) to complete...')
                        process.wait()
                        if process.returncode == 0:
                            print(f'  ✓ {node.name} (node_rank {node_rank}, local_rank {local_rank}, global_rank {global_rank}) completed successfully')
                        else:
                            print(f'  ✗ {node.name} (node_rank {node_rank}, local_rank {local_rank}, global_rank {global_rank}) failed with return code {process.returncode}')
                            all_remote_success = False
                    except Exception as e:
                        print(f'  ✗ {node.name} (rank {node.rank}) error: {e}')
                        all_remote_success = False
                
                if all_local_success and all_remote_success:
                    print('\n✓ All processes completed successfully')
                    sys.exit(0)
                else:
                    print('\n✗ Some processes failed')
                    sys.exit(1)
            else:
                # Execute all local processes in background
                for cmd_info in local_cmd_infos:
                    env_vars = cmd_info['env_vars']
                    train_script_abs = cmd_info['train_script_abs']
                    local_rank = cmd_info.get('local_rank', 0)
                    global_rank = cmd_info.get('global_rank', 0)
                    
                    # Prepare environment
                    # Critical distributed training env vars that must be overridden
                    critical_vars = ['RANK', 'WORLD_SIZE', 'LOCAL_RANK', 'MASTER_ADDR', 'MASTER_PORT', 
                                    'PET_NODE_RANK', 'PET_MASTER_ADDR', 'PET_MASTER_PORT']
                    if use_existing_env:
                        full_env = os.environ.copy()
                        for key, value in env_vars.items():
                            # Always override critical vars, even if they exist
                            if key in critical_vars or key not in full_env:
                                full_env[key] = value
                    else:
                        full_env = os.environ.copy()
                        full_env.update(env_vars)
                    
                    print(f'  Launching local process (local_rank={local_rank}, global_rank={global_rank}) in background...')
                    print(f'    Env: RANK={full_env.get("RANK")}, WORLD_SIZE={full_env.get("WORLD_SIZE")}, LOCAL_RANK={full_env.get("LOCAL_RANK")}, MASTER_ADDR={full_env.get("MASTER_ADDR")}, MASTER_PORT={full_env.get("MASTER_PORT")}')
                    # Use command_template to handle both scripts and commands
                    is_cmd = cmd_info.get('is_command', False)
                    work_dir = cmd_info.get('work_dir', master_work_dir)
                    if is_cmd:
                        # For commands, use shell=True and execute directly
                        local_process = subprocess.Popen(
                            train_script_abs,
                            env=full_env,
                            shell=True,
                            cwd=work_dir,
                            stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE
                        )
                    else:
                        # For scripts, use bash
                        local_process = subprocess.Popen(
                            ['bash', train_script_abs],
                            env=full_env,
                            cwd=work_dir,
                            stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE
                        )
                    local_processes.append((cmd_info, local_process))
                    pid_info['local_pids'].append({
                        'pid': local_process.pid,
                        'local_rank': local_rank,
                        'global_rank': global_rank
                    })
                
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

