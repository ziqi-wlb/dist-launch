#!/usr/bin/env python3
"""
Setup script for dist-launch package
"""
from setuptools import setup, find_packages
from setuptools.command.install import install
from pathlib import Path
import os
import stat
import subprocess

# Read README
readme_file = Path(__file__).parent / 'README.md'
long_description = readme_file.read_text(encoding='utf-8') if readme_file.exists() else ''


class PostInstallCommand(install):
    """Post-installation hook to fix SSH key permissions"""
    
    def run(self):
        # Run the standard install
        install.run(self)
        
        # Fix SSH key permissions after installation
        try:
            # Find the installed package directory
            import dist_launch
            package_dir = os.path.dirname(dist_launch.__file__)
            ssh_key_path = os.path.join(package_dir, 'ssh-key', 'id_rsa')
            
            if os.path.exists(ssh_key_path):
                # Try to fix permissions using os.chmod
                try:
                    os.chmod(ssh_key_path, 0o600)
                    print(f'✓ Set permissions to 600 for {ssh_key_path}')
                except (OSError, PermissionError):
                    # If os.chmod fails, try using subprocess
                    try:
                        subprocess.run(['chmod', '600', ssh_key_path], check=True, timeout=5)
                        print(f'✓ Set permissions to 600 for {ssh_key_path} (via chmod)')
                    except (subprocess.CalledProcessError, subprocess.TimeoutExpired, FileNotFoundError):
                        print(f'⚠ Warning: Could not set permissions for {ssh_key_path}. '
                              f'Please run: chmod 600 {ssh_key_path}')
        except Exception as e:
            # Don't fail installation if permission fix fails
            print(f'⚠ Warning: Could not fix SSH key permissions: {e}')


setup(
    name='dist-launch',
    version='0.1.10',
    description='One-click launch tool for distributed training on cluster nodes',
    long_description=long_description,
    long_description_content_type='text/markdown',
    author='ziqi-wlb',
    author_email="550461053@qq.com",
    python_requires='>=3.6',
    packages=['dist_launch', 'dist_launch.lib'],
    package_dir={
        'dist_launch': 'dist_launch',
        'dist_launch.lib': 'dist_launch/lib',
    },
    install_requires=[
        'torch>=1.8.0',
    ],
    entry_points={
        'console_scripts': [
            'dist-launch=dist_launch.cli:main',
        ],
    },
    include_package_data=True,
    package_data={
        'dist_launch': [
            'scripts/*.sh',
            'scripts/*.py',
            'ssh-key/id_rsa',
            'ssh-key/id_rsa.pub',
        ],
    },
    cmdclass={
        'install': PostInstallCommand,
    },
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
        'Programming Language :: Python :: 3.12',
    ],
)

