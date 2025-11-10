#!/usr/bin/env python3
"""
Setup script for dist-launch package
"""
from setuptools import setup, find_packages
from pathlib import Path

# Read README
readme_file = Path(__file__).parent / 'README.md'
long_description = readme_file.read_text(encoding='utf-8') if readme_file.exists() else ''

setup(
    name='dist-launch',
    version='0.1.6',
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

