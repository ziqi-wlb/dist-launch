#!/usr/bin/env python3
"""
Command line interface for auto-launch
"""
import sys
import os
import importlib.util

def get_package_dir():
    """Get the directory where this package is installed"""
    # Try to find the package location
    spec = importlib.util.find_spec('auto_launch')
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
        from auto_launch import run
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
        run_py = os.path.join(os.path.dirname(package_dir), 'auto_launch', 'run.py')
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
        print('Usage: auto-launch run <train.sh> [options]')
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

def main():
    """Main entry point"""
    if len(sys.argv) < 2:
        print('Usage: auto-launch <command> [arguments]')
        print('Commands:')
        print('  wait                    Wait for debug mode')
        print('  run <train.sh> [opts]   Launch training on all nodes')
        print('  kill [--force]          Kill all training processes started by auto-launch run')
        sys.exit(1)
    
    command = sys.argv[1]
    args = sys.argv[2:]
    
    if command == 'wait':
        cmd_wait(args)
    elif command == 'run':
        cmd_run(args)
    elif command == 'kill':
        cmd_kill(args)
    else:
        print(f'Error: Unknown command: {command}')
        print('Available commands: wait, run, kill')
        sys.exit(1)

if __name__ == '__main__':
    main()
