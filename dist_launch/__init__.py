"""
Auto Launch - One-click launch tool for distributed training
"""
import os
import sys
import importlib.util
from pathlib import Path

__version__ = '0.1.0'


def get_package_dir():
    """Get the directory where this package is installed"""
    # Try to find the package location
    spec = importlib.util.find_spec('dist_launch')
    if spec and spec.origin:
        return os.path.dirname(spec.origin)
    
    # Fallback to current file's directory
    return os.path.dirname(os.path.abspath(__file__))


def get_project_ssh_key_path():
    """
    Get the SSH private key path from project directory
    Priority: SSH_KEY env > package ssh-key/id_rsa > project root ssh-key/id_rsa > default fallback
    Automatically fixes file permissions if needed (SSH requires 600 for private keys)
    
    Returns:
        Path to SSH private key
    """
    # First check environment variable
    env_ssh_key = os.environ.get('SSH_KEY', '')
    if env_ssh_key and os.path.exists(env_ssh_key):
        _fix_ssh_key_permissions(env_ssh_key)
        return env_ssh_key
    
    # Try to find ssh-key in package directory (for installed package)
    package_dir = get_package_dir()
    package_ssh_key = os.path.join(package_dir, 'ssh-key', 'id_rsa')
    if os.path.exists(package_ssh_key):
        _fix_ssh_key_permissions(package_ssh_key)
        return package_ssh_key
    
    # Try to find ssh-key in project root (for development mode)
    project_root = os.path.dirname(package_dir)
    project_ssh_key = os.path.join(project_root, 'ssh-key', 'id_rsa')
    if os.path.exists(project_ssh_key):
        _fix_ssh_key_permissions(project_ssh_key)
        return project_ssh_key
    
    # Fallback to default location
    default_path = '/mnt/3fs/dots-pretrain/weishi/release/public/ssh-key/id_rsa'
    if os.path.exists(default_path):
        _fix_ssh_key_permissions(default_path)
    return default_path


def _fix_ssh_key_permissions(key_path):
    """
    Fix SSH private key file permissions to 600 (required by SSH)
    
    Args:
        key_path: Path to SSH private key file
    
    Returns:
        True if permissions are correct or were fixed, False if fix failed
    """
    try:
        if not os.path.exists(key_path):
            return False
        
        # Get current permissions
        current_mode = os.stat(key_path).st_mode
        # Extract permission bits (last 3 octal digits)
        # 0o600 = rw------- (owner read/write only)
        # We want: owner can read/write (0o600), group and others have no permissions
        desired_mode = 0o600
        
        # Check if permissions are correct
        # Mask to get only permission bits (0o777)
        current_perms = current_mode & 0o777
        if current_perms == desired_mode:
            return True  # Permissions are already correct
        
        # Try to fix permissions
        try:
            os.chmod(key_path, desired_mode)
            # Verify the change
            new_mode = os.stat(key_path).st_mode
            new_perms = new_mode & 0o777
            if new_perms == desired_mode:
                return True
            else:
                # Permission change failed - might need root or file is read-only
                import sys
                print(f'Warning: Could not set permissions to 600 for {key_path} (current: {oct(new_perms)}). '
                      f'You may need to run: chmod 600 {key_path}', file=sys.stderr)
                return False
        except (OSError, PermissionError) as e:
            # If we can't fix permissions, print a helpful message
            import sys
            print(f'Warning: Could not fix permissions for {key_path}: {e}. '
                  f'You may need to run: chmod 600 {key_path}', file=sys.stderr)
            return False
    except Exception as e:
        import sys
        print(f'Warning: Error checking/fixing permissions for {key_path}: {e}', file=sys.stderr)
        return False


def get_project_ssh_public_key_path():
    """
    Get the SSH public key path from project directory
    Priority: SSH_PUBLIC_KEY env > package ssh-key/id_rsa.pub > project root ssh-key/id_rsa.pub > derived from SSH_KEY > default fallback
    
    Returns:
        Path to SSH public key
    """
    # First check environment variable
    env_ssh_public_key = os.environ.get('SSH_PUBLIC_KEY', '')
    if env_ssh_public_key and os.path.exists(env_ssh_public_key):
        return env_ssh_public_key
    
    # Try to derive from SSH_KEY if set
    env_ssh_key = os.environ.get('SSH_KEY', '')
    if env_ssh_key and os.path.exists(env_ssh_key):
        derived_public_key = env_ssh_key + '.pub'
        if os.path.exists(derived_public_key):
            return derived_public_key
    
    # Try to find ssh-key in package directory (for installed package)
    package_dir = get_package_dir()
    package_ssh_public_key = os.path.join(package_dir, 'ssh-key', 'id_rsa.pub')
    if os.path.exists(package_ssh_public_key):
        return package_ssh_public_key
    
    # Try to find ssh-key in project root (for development mode)
    project_root = os.path.dirname(package_dir)
    project_ssh_public_key = os.path.join(project_root, 'ssh-key', 'id_rsa.pub')
    if os.path.exists(project_ssh_public_key):
        return project_ssh_public_key
    
    # Fallback to default location
    default_path = '/mnt/3fs/dots-pretrain/weishi/release/public/ssh-key/id_rsa.pub'
    return default_path

