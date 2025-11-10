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
    """
    try:
        if os.path.exists(key_path):
            # Get current permissions
            current_mode = os.stat(key_path).st_mode
            # Check if permissions are too open (not 600)
            # 0o600 = rw------- (owner read/write only)
            if current_mode & 0o077 != 0:  # If group or others have any permissions
                try:
                    os.chmod(key_path, 0o600)
                    # Only log if we actually changed permissions (avoid spam)
                    if os.stat(key_path).st_mode & 0o077 == 0:
                        pass  # Success, no need to log
                except (OSError, PermissionError) as e:
                    # If we can't fix permissions, SSH will fail with a clear error
                    # Don't raise exception, let SSH handle it
                    pass
    except Exception:
        # Silently fail - if permissions can't be fixed, SSH will give a clear error
        pass


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

