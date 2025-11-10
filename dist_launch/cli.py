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
    if len(sys.argv) < 2:
        print('Usage: dist-launch <command> [arguments]')
        print('Commands:')
        print('  wait [--ssh-port PORT] [--no-ssh-setup]  Wait for debug mode (auto-setup SSH server)')
        print('  run <train.sh> [opts]                     Launch training on all nodes')
        print('  kill [--force]                            Kill all training processes started by dist-launch run')
        print('  nccl-tests [opts]                         Run NCCL performance tests using PyTorch distributed')
        sys.exit(1)
    
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
