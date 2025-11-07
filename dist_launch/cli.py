#!/usr/bin/env python3
"""
Command line interface for dist-launch
"""
import sys
import os
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

def cmd_wait(args):
    """Execute wait.sh"""
    wait_script = os.path.join(get_scripts_dir(), 'wait.sh')
    if not os.path.exists(wait_script):
        print(f'Error: {wait_script} not found')
        sys.exit(1)
    
    cmd = ['bash', wait_script] + args
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
        print('  wait                    Wait for debug mode')
        print('  run <train.sh> [opts]   Launch training on all nodes')
        print('  kill [--force]          Kill all training processes started by dist-launch run')
        print('  nccl-tests [opts]       Run NCCL performance tests using PyTorch distributed')
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
