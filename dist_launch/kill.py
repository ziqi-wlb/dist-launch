#!/usr/bin/env python3
"""
Kill all training processes started by dist-launch run
This script kills all processes recorded during the last dist-launch run execution
"""
import os
import sys
import json
import subprocess
import argparse
from typing import List, Dict, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed

# Add lib directory to path
lib_path = os.path.join(os.path.dirname(__file__), 'lib')
if not os.path.exists(lib_path):
    # Try installed package path
    import importlib.util
    spec = importlib.util.find_spec('dist_launch')
    if spec and spec.origin:
        lib_path = os.path.join(os.path.dirname(spec.origin), 'lib')
sys.path.insert(0, lib_path)

from node_executor import NodeExecutor
from cluster_manager import NodeConfig


PID_FILE = '/tmp/dist-launch-pids.json'


def load_process_info() -> Optional[Dict]:
    """Load process information from PID file"""
    if not os.path.exists(PID_FILE):
        print(f'Error: Process info file not found: {PID_FILE}')
        print('No training processes were started by dist-launch run')
        return None
    
    try:
        with open(PID_FILE, 'r') as f:
            return json.load(f)
    except Exception as e:
        print(f'Error reading process info file: {e}', file=sys.stderr)
        return None


def kill_local_process(pid: int, name: str = '', force: bool = False, kill_tree: bool = True) -> bool:
    """Kill a local process by PID, optionally killing the entire process tree"""
    try:
        # Check if process exists and is not the wait process
        try:
            # Read /proc/pid/cmdline to check if it's a wait process
            cmdline_file = f'/proc/{pid}/cmdline'
            if os.path.exists(cmdline_file):
                with open(cmdline_file, 'r') as f:
                    cmdline = f.read()
                    # Check if it's wait.sh or dist-launch wait
                    if 'wait.sh' in cmdline or 'dist-launch wait' in cmdline:
                        print(f'  Skipping {name} (PID {pid}): This is a wait process, not a training process')
                        return False
        except Exception:
            pass
        
        # Kill the entire process tree if requested
        if kill_tree:
            try:
                # Use pkill to kill process group (all children)
                # First get the process group ID
                pgid = os.getpgid(pid)
                signal = 9 if force else 15
                signal_name = 'SIGKILL' if force else 'SIGTERM'
                
                # Kill the process group (all processes in the tree)
                os.killpg(pgid, signal)
                print(f'  ✓ Sent {signal_name} to process group {pgid} (PID {pid} and all children)')
                return True
            except (ProcessLookupError, OSError):
                # If process group kill fails, fall back to single process
                pass
        
        # Try SIGTERM first (graceful shutdown), or SIGKILL if force
        signal = 9 if force else 15
        os.kill(pid, signal)
        signal_name = 'SIGKILL' if force else 'SIGTERM'
        print(f'  ✓ Sent {signal_name} to {name} (PID {pid})')
        return True
    except ProcessLookupError:
        print(f'  - Process {name} (PID {pid}) already terminated')
        return False
    except PermissionError:
        print(f'  ✗ Permission denied killing {name} (PID {pid})')
        return False
    except Exception as e:
        print(f'  ✗ Error killing {name} (PID {pid}): {e}')
        return False


def kill_remote_process(executor: NodeExecutor, node: NodeConfig, pid: int, name: str = '', force: bool = False, kill_tree: bool = True, process_info: Optional[Dict] = None) -> bool:
    """Kill a remote process by PID via SSH, optionally killing the entire process tree"""
    try:
        # Note: The saved PID might be the SSH connection PID, not the actual training process
        # We need to find the actual training process on the remote node
        
        # First, try to find the actual training process
        # The saved PID might be the SSH connection PID, so we need to find the real training process
        train_script = process_info.get('train_script', '') if process_info else ''
        if train_script:
            script_name = os.path.basename(train_script)
            # Find processes matching the training script, excluding SSH and wait processes
            find_training_cmd = f'ps aux | grep -E "{script_name}|train[.]sh|torchrun|nccl_tests[.]sh|nccl_tests[.]py" | grep -v grep | grep -v "wait.sh" | grep -v "ssh.*root@" | head -1'
        else:
            find_training_cmd = f'ps aux | grep -E "train[.]sh|torchrun|debug_redmoe|nccl_tests[.]sh|nccl_tests[.]py" | grep -v grep | grep -v "wait.sh" | grep -v "ssh.*root@" | head -1'
        
        result = executor.execute_sync(node, find_training_cmd)
        
        actual_pid = pid  # Default to saved PID
        if result[0] == 0 and result[1].strip():
            # Extract PID from ps output (second column)
            try:
                parts = result[1].strip().split()
                if len(parts) >= 2:
                    actual_pid = int(parts[1])
                    print(f'  Found training process on {node.hostname}: PID {actual_pid} (saved PID was {pid})')
            except (ValueError, IndexError):
                pass
        
        # If we couldn't find the training process, try to find any process in the process tree of the saved PID
        if actual_pid == pid:
            # Try to find child processes of the saved PID
            find_children_cmd = f'ps --ppid {pid} -o pid= 2>/dev/null | head -1'
            result = executor.execute_sync(node, find_children_cmd)
            if result[0] == 0 and result[1].strip():
                try:
                    actual_pid = int(result[1].strip())
                except ValueError:
                    pass
        
        # Check if process exists and is not a wait process
        check_cmd = f'if [ -f /proc/{actual_pid}/cmdline ]; then cat /proc/{actual_pid}/cmdline | grep -q "wait.sh\\|dist-launch wait" && echo "wait" || echo "train"; else echo "notfound"; fi'
        
        result = executor.execute_sync(node, check_cmd)
        # result[1] is stdout (already a string because text=True in execute_sync)
        if result[0] == 0 and 'wait' in result[1]:
            print(f'  Skipping {name} on {node.hostname} (PID {actual_pid}): This is a wait process')
            return False
        
        # Kill the process tree (all children)
        if kill_tree:
            # Get process group ID and kill the entire process group
            if force:
                kill_cmd = f'pkill -g $(ps -o pgid= -p {actual_pid} 2>/dev/null | tr -d " ") -9 2>/dev/null || kill -9 {actual_pid} 2>/dev/null || true'
            else:
                kill_cmd = f'pkill -g $(ps -o pgid= -p {actual_pid} 2>/dev/null | tr -d " ") -15 2>/dev/null || kill -15 {actual_pid} 2>/dev/null || true'
        else:
            # Kill only the specific process
            if force:
                kill_cmd = f'kill -9 {actual_pid} 2>/dev/null || true'
            else:
                kill_cmd = f'kill -15 {actual_pid} 2>/dev/null || true'
        
        result = executor.execute_sync(node, kill_cmd)
        
        if result[0] == 0:
            signal_name = 'SIGKILL' if force else 'SIGTERM'
            tree_info = ' (process tree)' if kill_tree else ''
            print(f'  ✓ Killed {name} on {node.hostname} (PID {actual_pid}){tree_info} with {signal_name}')
            return True
        else:
            # Check if process still exists
            check_cmd = f'kill -0 {actual_pid} 2>/dev/null && echo "exists" || echo "notfound"'
            result = executor.execute_sync(node, check_cmd)
            # result[1] is stdout (already a string because text=True in execute_sync)
            if 'notfound' in result[1]:
                return False
            return False
    except Exception as e:
        print(f'  ✗ Error killing {name} on {node.hostname} (PID {pid}): {e}')
        return False


def kill_all_processes(force: bool = False, executor: Optional[NodeExecutor] = None) -> bool:
    """Kill all processes recorded in PID file"""
    process_info = load_process_info()
    if not process_info:
        return False
    
    print('Killing all training processes started by dist-launch run...')
    print(f'Process info file: {PID_FILE}\n')
    
    killed_count = 0
    total_count = 0
    processed_pids = set()  # Track PIDs we've already processed to avoid duplicates
    
    # Kill local processes (support both old rank0_pid and new local_pids format)
    train_script = process_info.get('train_script', '')
    
    # Handle new format: local_pids (for multi-GPU scenarios)
    if 'local_pids' in process_info:
        local_pids = process_info['local_pids']
        for proc_info in local_pids:
            total_count += 1
            pid = proc_info.get('pid', -1)
            local_rank = proc_info.get('local_rank', 0)
            global_rank = proc_info.get('global_rank', 0)
            
            if pid == -1:
                continue
            
            # Try to find the actual training process (not just the bash wrapper)
            # The saved PID might be the bash process, but we need to kill the training script
            if train_script:
                script_name = os.path.basename(train_script)
                # Find processes matching the training script, including nccl_tests
                try:
                    result = subprocess.run(
                        ['ps', 'aux'],
                        capture_output=True,
                        text=True,
                        timeout=5
                    )
                    if result.returncode == 0:
                        for line in result.stdout.split('\n'):
                            if (script_name in line or 'nccl_tests' in line) and 'grep' not in line and 'wait.sh' not in line:
                                parts = line.split()
                                if len(parts) >= 2:
                                    try:
                                        found_pid = int(parts[1])
                                        # Use this PID if it's different from saved PID or if it's a child process
                                        if found_pid != pid:
                                            # Check if it's a child of the saved PID
                                            try:
                                                parent_pid = int(parts[1]) if len(parts) > 1 else -1
                                                # If found PID is a child or parent, use it
                                                if found_pid != pid:
                                                    print(f'  Found training process on local node: PID {found_pid} (saved PID was {pid})')
                                                    pid = found_pid
                                                    break
                                            except (ValueError, IndexError):
                                                pass
                                    except (ValueError, IndexError):
                                        pass
                except Exception:
                    pass
            
            name = f'rank{global_rank} (local_rank={local_rank})'
            if pid not in processed_pids:
                processed_pids.add(pid)
                if kill_local_process(pid, name, force=force, kill_tree=True):
                    killed_count += 1
    
    # Handle old format: rank0_pid (for backward compatibility)
    elif 'rank0_pid' in process_info:
        total_count += 1
        rank0_pid = process_info['rank0_pid']
        
        # Try to find the actual training process (not just the bash wrapper)
        if train_script:
            script_name = os.path.basename(train_script)
            # Find processes matching the training script, including nccl_tests
            try:
                result = subprocess.run(
                    ['ps', 'aux'],
                    capture_output=True,
                    text=True,
                    timeout=5
                )
                if result.returncode == 0:
                    for line in result.stdout.split('\n'):
                        if (script_name in line or 'nccl_tests' in line) and 'grep' not in line and 'wait.sh' not in line:
                            parts = line.split()
                            if len(parts) >= 2:
                                try:
                                    found_pid = int(parts[1])
                                    if found_pid != rank0_pid:
                                        print(f'  Found training process on local node: PID {found_pid} (saved PID was {rank0_pid})')
                                        rank0_pid = found_pid
                                        break
                                except (ValueError, IndexError):
                                    pass
            except Exception:
                pass
        
        if rank0_pid not in processed_pids:
            processed_pids.add(rank0_pid)
            if kill_local_process(rank0_pid, 'rank0', force=force, kill_tree=True):
                killed_count += 1
    
    # Also kill any remaining training processes that might not be in the PID file
    # This handles cases where processes were started but not properly recorded
    if train_script:
        script_name = os.path.basename(train_script)
        # Find all processes matching the script name or nccl_tests
        try:
            result = subprocess.run(
                ['ps', 'aux'],
                capture_output=True,
                text=True,
                timeout=5
            )
            if result.returncode == 0:
                for line in result.stdout.split('\n'):
                    if (script_name in line or 'nccl_tests' in line) and 'grep' not in line and 'wait.sh' not in line:
                        parts = line.split()
                        if len(parts) >= 2:
                            try:
                                found_pid = int(parts[1])
                                # Skip if we've already processed this PID
                                if found_pid in processed_pids:
                                    continue
                                # Check if this PID is a wait process
                                try:
                                    cmdline_file = f'/proc/{found_pid}/cmdline'
                                    if os.path.exists(cmdline_file):
                                        with open(cmdline_file, 'r') as f:
                                            cmdline = f.read()
                                            if 'wait.sh' not in cmdline and 'dist-launch wait' not in cmdline:
                                                # Kill this process
                                                processed_pids.add(found_pid)
                                                if kill_local_process(found_pid, f'training process (PID {found_pid})', force=force, kill_tree=True):
                                                    killed_count += 1
                                                    total_count += 1
                                except Exception:
                                    pass
                            except (ValueError, IndexError):
                                pass
        except Exception:
            pass
    
    # Kill remote processes in parallel
    if 'remote_processes' in process_info and executor:
        remote_procs = process_info['remote_processes']
        
        # Prepare tasks for parallel execution
        kill_tasks = []
        for proc_info in remote_procs:
            total_count += 1
            hostname = proc_info.get('hostname', '')
            rank = proc_info.get('rank', -1)
            pid = proc_info.get('pid', -1)
            
            if pid == -1:
                print(f'  ✗ Invalid PID for rank {rank} on {hostname}')
                continue
            
            # Create a temporary NodeConfig for this node
            node = NodeConfig(
                name=f'node{rank}',
                rank=rank,
                node_rank=rank,  # Assuming node_rank == rank for simplicity
                hostname=hostname
            )
            
            kill_tasks.append((executor, node, pid, f'rank{rank}', force, process_info))
        
        # Execute kill operations in parallel
        if kill_tasks:
            with ThreadPoolExecutor(max_workers=len(kill_tasks)) as executor_pool:
                futures = {
                    executor_pool.submit(
                        kill_remote_process, 
                        task[0], task[1], task[2], task[3], 
                        force=task[4], kill_tree=True, process_info=task[5]
                    ): task for task in kill_tasks
                }
                
                for future in as_completed(futures):
                    try:
                        if future.result():
                            killed_count += 1
                    except Exception as e:
                        task = futures[future]
                        print(f'  ✗ Error killing {task[3]} on {task[1].hostname}: {e}')
    
    print(f'\nSummary: {killed_count}/{total_count} processes killed')
    
    if killed_count > 0:
        try:
            os.remove(PID_FILE)
        except Exception:
            pass
    
    return killed_count > 0


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description='Kill all training processes started by dist-launch run')
    parser.add_argument('--force', '-f', action='store_true',
                       help='Force kill (SIGKILL) instead of graceful (SIGTERM)')
    from dist_launch import get_project_ssh_key_path
    parser.add_argument('--ssh-key', type=str,
                       default=os.environ.get('SSH_KEY', get_project_ssh_key_path()),
                       help='Path to SSH private key (default: from SSH_KEY env or project ssh-key)')
    parser.add_argument('--ssh-port', type=int,
                       default=int(os.environ.get('SSH_PORT', '2025')),
                       help='SSH port')
    parser.add_argument('--ssh-user', type=str,
                       default=os.environ.get('SSH_USER', 'root'),
                       help='SSH username')
    args = parser.parse_args()
    
    # Create executor for remote processes
    executor = NodeExecutor(
        ssh_key_path=args.ssh_key,
        ssh_port=args.ssh_port,
        ssh_user=args.ssh_user
    )
    
    success = kill_all_processes(force=args.force, executor=executor)
    sys.exit(0 if success else 1)


if __name__ == '__main__':
    main()

