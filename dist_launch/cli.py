#!/usr/bin/env python3
"""
Command line interface for dist-launch
"""
import sys
import os
import subprocess
import argparse
import importlib.util

def get_package_dir():
    """Get the directory where this package is installed"""
    # Try to find the package location
    spec = importlib.util.find_spec('dist_launch')
    if spec and spec.origin:
        return os.path.dirname(spec.origin)
    
    # Fallback to current file's directory
    return os.path.dirname(os.path.abspath(__file__))

def get_scripts_dir():
    """Get the scripts directory"""
    package_dir = get_package_dir()
    scripts_dir = os.path.join(package_dir, 'scripts')
    
    # Fallback for development mode
    if not os.path.exists(scripts_dir):
        scripts_dir = os.path.join(os.path.dirname(package_dir), 'scripts')
    
    return scripts_dir

def get_run_py():
    """Get the run.py path"""
    # Try to import run module and get its path
    try:
        from dist_launch import run
        if hasattr(run, '__file__'):
            return run.__file__
    except ImportError:
        try:
            # Fallback: try importing as run module
            import run
            if hasattr(run, '__file__'):
                return run.__file__
        except ImportError:
            pass
    
    # Fallback: try various paths
    package_dir = get_package_dir()
    run_py = os.path.join(package_dir, 'run.py')
    
    if not os.path.exists(run_py):
        run_py = os.path.join(os.path.dirname(package_dir), 'dist_launch', 'run.py')
        if not os.path.exists(run_py):
            run_py = os.path.join(package_dir, '..', 'run.py')
            run_py = os.path.abspath(run_py)
    
    return run_py

def setup_ssh_server(ssh_port=2025):
    """
    Configure and start SSH server
    
    Args:
        ssh_port: SSH server port (default: 2025)
    """
    sshd_config = '/etc/ssh/sshd_config'
    ssh_config_content = f'''# ====== custom override ======
Port {ssh_port}
PermitRootLogin prohibit-password
PasswordAuthentication yes
PubkeyAuthentication yes
AllowUsers bushou root
PermitEmptyPasswords no
# =============================
'''
    
    try:
        # Check if running as root
        if os.geteuid() != 0:
            print(f'Warning: SSH server setup requires root privileges. Skipping...')
            return False
        
        # Append SSH config to /etc/ssh/sshd_config
        print(f'Configuring SSH server (port {ssh_port})...')
        with open(sshd_config, 'a') as f:
            f.write(ssh_config_content)
        print(f'✓ SSH configuration added to {sshd_config}')
        
        # Restart SSH service
        print('Restarting SSH service...')
        print(f'Executing: service ssh restart')
        result = subprocess.run(['service', 'ssh', 'restart'], 
                              capture_output=True, text=True, timeout=30)
        if result.stdout:
            print(f'Command stdout: {result.stdout}')
        if result.stderr:
            print(f'Command stderr: {result.stderr}')
        print(f'Command return code: {result.returncode}')
        
        if result.returncode == 0:
            print('✓ SSH service restarted successfully')
            return True
        else:
            print(f'Warning: Failed to restart SSH service (return code: {result.returncode})')
            if result.stderr:
                print(f'Error output: {result.stderr}')
            if result.stdout:
                print(f'Output: {result.stdout}')
            
            # Try alternative command
            print(f'Trying alternative: systemctl restart ssh')
            result = subprocess.run(['systemctl', 'restart', 'ssh'], 
                                  capture_output=True, text=True, timeout=30)
            if result.stdout:
                print(f'Command stdout: {result.stdout}')
            if result.stderr:
                print(f'Command stderr: {result.stderr}')
            print(f'Command return code: {result.returncode}')
            
            if result.returncode == 0:
                print('✓ SSH service restarted successfully (via systemctl)')
                return True
            else:
                print(f'Warning: Failed to restart SSH service via systemctl (return code: {result.returncode})')
                if result.stderr:
                    print(f'Error output: {result.stderr}')
                if result.stdout:
                    print(f'Output: {result.stdout}')
                return False
    except Exception as e:
        print(f'Warning: Failed to setup SSH server: {e}')
        return False

def cmd_wait(args):
    """Execute wait.sh with optional SSH server setup"""
    # Get default SSH port from environment variable or use 2025
    default_ssh_port = 2025
    if 'SSH_PORT' in os.environ:
        try:
            default_ssh_port = int(os.environ['SSH_PORT'])
        except ValueError:
            print(f'Warning: Invalid SSH_PORT environment variable value: {os.environ["SSH_PORT"]}, using default 2025')
            default_ssh_port = 2025
    
    parser = argparse.ArgumentParser(description='Wait for debug mode')
    parser.add_argument('--ssh-port', type=int, default=None,
                       help=f'SSH server port to configure (default: {default_ssh_port} from SSH_PORT env or 2025, set to 0 to skip SSH setup)')
    parser.add_argument('--no-ssh-setup', action='store_true',
                       help='Skip SSH server setup')
    
    # Parse known args to extract --ssh-port and --no-ssh-setup
    parsed_args, remaining_args = parser.parse_known_args(args)
    
    # Setup SSH server if requested
    if not parsed_args.no_ssh_setup:
        ssh_port = parsed_args.ssh_port if parsed_args.ssh_port is not None else default_ssh_port
        if ssh_port > 0:
            setup_ssh_server(ssh_port)
    
    # Execute wait.sh with remaining arguments
    wait_script = os.path.join(get_scripts_dir(), 'wait.sh')
    if not os.path.exists(wait_script):
        print(f'Error: {wait_script} not found')
        sys.exit(1)
    
    cmd = ['bash', wait_script] + remaining_args
    os.execvp('bash', cmd)

def cmd_run(args):
    """Execute run.py with train script"""
    if not args:
        print('Error: train script is required')
        print('Usage: dist-launch run <train.sh> [options]')
        sys.exit(1)
    
    train_script = args[0]
    remaining_args = args[1:]
    
    # Execute run.py
    run_py = get_run_py()
    if not os.path.exists(run_py):
        print(f'Error: {run_py} not found')
        sys.exit(1)
    
    cmd = ['python3', run_py, train_script] + remaining_args
    os.execvp('python3', cmd)

def cmd_kill(args):
    """Execute kill.py to stop all training processes"""
    # Get kill.py path
    package_dir = get_package_dir()
    kill_py = os.path.join(package_dir, 'kill.py')
    
    if not os.path.exists(kill_py):
        print(f'Error: {kill_py} not found')
        sys.exit(1)
    
    cmd = ['python3', kill_py] + args
    os.execvp('python3', cmd)

def cmd_nccl_tests(args):
    """Execute NCCL tests via run.py"""
    # Handle --help directly to show nccl_tests.py help instead of run.py help
    if '--help' in args or '-h' in args:
        # Import and show nccl_tests.py help directly
        try:
            # Get nccl_tests.py path
            package_dir = get_package_dir()
            nccl_tests_py = os.path.join(package_dir, 'nccl_tests.py')
            
            # Fallback for development mode
            if not os.path.exists(nccl_tests_py):
                nccl_tests_py = os.path.join(os.path.dirname(package_dir), 'nccl_tests.py')
            
            if os.path.exists(nccl_tests_py):
                # Run nccl_tests.py with --help to show its help
                cmd = ['python3', nccl_tests_py, '--help']
                os.execvp('python3', cmd)
            else:
                # Fallback: try to import and call main with --help
                import importlib.util
                spec = importlib.util.spec_from_file_location("nccl_tests", nccl_tests_py)
                if spec and spec.loader:
                    module = importlib.util.module_from_spec(spec)
                    # Temporarily set sys.argv to show help
                    old_argv = sys.argv
                    sys.argv = ['nccl_tests.py', '--help']
                    try:
                        spec.loader.exec_module(module)
                    finally:
                        sys.argv = old_argv
                    sys.exit(0)
        except Exception as e:
            print(f'Error showing help: {e}', file=sys.stderr)
            # Fall through to normal execution
    
    # Get nccl_tests.sh wrapper script path
    scripts_dir = get_scripts_dir()
    nccl_tests_sh = os.path.join(scripts_dir, 'nccl_tests.sh')
    
    if not os.path.exists(nccl_tests_sh):
        print(f'Error: {nccl_tests_sh} not found')
        sys.exit(1)
    
    # Get run.py path
    run_py = get_run_py()
    if not os.path.exists(run_py):
        print(f'Error: {run_py} not found')
        sys.exit(1)
    
    # Launch via run.py, passing nccl_tests.sh as the script and args as additional options
    # The args will be passed to nccl_tests.sh, which will pass them to nccl_tests.py
    cmd = ['python3', run_py, nccl_tests_sh] + args
    os.execvp('python3', cmd)

def main():
    """Main entry point"""
    if len(sys.argv) < 2 or sys.argv[1] in ['-h', '--help', 'help']:
        print('Dist Launch - 一键启动集群训练工具（Debug模式）')
        print()
        print('Usage: dist-launch <command> [arguments]')
        print()
        print('Commands:')
        print('  wait [--ssh-port PORT] [--no-ssh-setup]')
        print('      Wait for debug mode (auto-setup SSH server)')
        print('      This command initializes the cluster and keeps nodes alive for debugging.')
        print('      It should be run on all nodes when the cluster starts.')
        print()
        print('      Options:')
        print('        --ssh-port PORT     SSH server port to configure (default: 2025)')
        print('        --no-ssh-setup      Skip SSH server setup')
        print()
        print('  run <train.sh> [options]')
        print('      Launch training script on all nodes')
        print('      This command reads cluster info and executes the script on all nodes via SSH.')
        print('      It should be run on rank0 node after cluster initialization.')
        print()
        print('      Options:')
        print('        --master-addr ADDR   Master node address (default: from MASTER_ADDR env)')
        print('        --world-size N      Total number of nodes (default: from WORLD_SIZE env)')
        print('        --master-port PORT  Master port (default: 23456)')
        print('        --nodes NODES        Comma-separated node list (default: auto-discover)')
        print('        --ssh-key PATH      SSH private key path (default: from project)')
        print('        --ssh-port PORT     SSH port (default: 2025)')
        print('        --ssh-user USER     SSH username (default: root)')
        print('        --nper-node N       Number of GPUs per node (default: 1)')
        print('        --wait               Wait for all processes to complete (default: True)')
        print('        --no-wait            Launch processes in background')
        print('        --dry-run            Show commands without executing')
        print()
        print('      Examples:')
        print('        dist-launch run train.sh')
        print('        dist-launch run train.sh --nper-node 4 --world-size 2')
        print('        dist-launch run "pwd"  # Execute command directly')
        print()
        print('  kill [--force]')
        print('      Kill all training processes started by dist-launch run')
        print('      This command stops all processes on all nodes that were started via dist-launch run.')
        print('      It does not stop wait processes.')
        print()
        print('      Options:')
        print('        --force             Force kill processes (SIGKILL instead of SIGTERM)')
        print()
        print('  nccl-tests [options]')
        print('      Run NCCL performance tests using PyTorch distributed')
        print('      This command runs collective communication tests (Allreduce, Allgather, Broadcast)')
        print('      across multiple GPUs and nodes to measure network bandwidth.')
        print()
        print('      Options:')
        print('        --operations OPS    Comma-separated list of operations to test: allreduce, allgather, broadcast')
        print('                            (default: allreduce)')
        print('        --sizes SIZES       Comma-separated list of sizes in MB to test (default: 1,10,100,1000)')
        print('        --iterations N      Number of iterations per test (default: 20)')
        print('        --dtype TYPE       Data type to use: float32, float16, or int32 (default: float32)')
        print('        --nper-node N      Number of GPUs per node. If not specified, auto-detects from')
        print('                            CUDA_VISIBLE_DEVICES or uses 1')
        print()
        print('      Examples:')
        print('        # Run allreduce test with default settings')
        print('        dist-launch nccl-tests')
        print()
        print('        # Test multiple operations with custom sizes')
        print('        dist-launch nccl-tests --operations allreduce,allgather --sizes 10,100,1000')
        print()
        print('        # 2 nodes, 4 GPUs per node')
        print('        dist-launch nccl-tests --nper-node 4 --world-size 2')
        print()
        print('      Note: This command must be run via "dist-launch nccl-tests" or')
        print('            "dist-launch run nccl_tests.sh". Environment variables')
        print('            (RANK, WORLD_SIZE, MASTER_ADDR, MASTER_PORT) are set automatically.')
        print()
        print('For more information, see README.md')
        sys.exit(0 if sys.argv[1] in ['-h', '--help', 'help'] else 1)
    
    command = sys.argv[1]
    args = sys.argv[2:]
    
    if command == 'wait':
        cmd_wait(args)
    elif command == 'run':
        cmd_run(args)
    elif command == 'kill':
        cmd_kill(args)
    elif command == 'nccl-tests':
        cmd_nccl_tests(args)
    else:
        print(f'Error: Unknown command: {command}')
        print('Available commands: wait, run, kill, nccl-tests')
        sys.exit(1)

if __name__ == '__main__':
    main()
