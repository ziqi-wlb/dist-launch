#!/usr/bin/env python3
"""
NCCL tests using PyTorch distributed - Similar to nccl-tests but using PyTorch process group
Usage: dist-launch nccl-tests [options]
"""
import os
import sys
import argparse
import time
import torch
import torch.distributed as dist
from typing import List, Optional

# Add lib directory to path
lib_path = os.path.join(os.path.dirname(__file__), 'lib')
if not os.path.exists(lib_path):
    import importlib.util
    spec = importlib.util.find_spec('dist_launch')
    if spec and spec.origin:
        lib_path = os.path.join(os.path.dirname(spec.origin), 'lib')
sys.path.insert(0, lib_path)

from cluster_manager import ClusterManager, NodeConfig
from node_executor import NodeExecutor
from host_discovery import HostDiscovery


def test_allreduce(size_mb: int, iterations: int, dtype: str = 'float32'):
    """
    Test Allreduce operation
    
    Args:
        size_mb: Size of tensor in MB
        iterations: Number of iterations
        dtype: Data type (float32, float16, int32)
    """
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    
    # Convert size to elements based on dtype
    bytes_per_element = 4 if dtype in ['float32', 'int32'] else 2  # float16 is 2 bytes
    size_elements = (size_mb * 1024 * 1024) // bytes_per_element
    
    # Create tensor
    if dtype == 'float32':
        tensor = torch.ones(size_elements, dtype=torch.float32).cuda()
    elif dtype == 'float16':
        tensor = torch.ones(size_elements, dtype=torch.float16).cuda()
    elif dtype == 'int32':
        tensor = torch.ones(size_elements, dtype=torch.int32).cuda()
    else:
        raise ValueError(f'Unsupported dtype: {dtype}')
    
    # Warmup
    for _ in range(3):
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    
    torch.cuda.synchronize()
    
    # Actual test
    start_time = time.time()
    for i in range(iterations):
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    torch.cuda.synchronize()
    end_time = time.time()
    
    elapsed = end_time - start_time
    avg_time = elapsed / iterations
    # Bandwidth calculation: data transferred per second
    # Allreduce: each node sends and receives data, total data = size_mb * 2 * (world_size - 1) / world_size
    # Convert MB to GB and ms to seconds
    bandwidth_gbps = (size_mb * 2 * (world_size - 1) / world_size) / (avg_time)  # GB/s
    
    if rank == 0:
        print(f'Allreduce test: {size_mb}MB, {iterations} iterations, dtype={dtype}')
        print(f'  Average time: {avg_time*1000:.2f} ms')
        print(f'  Bandwidth: {bandwidth_gbps:.2f} GB/s')
        print(f'  Total time: {elapsed:.2f} s')
    
    return avg_time, bandwidth_gbps


def test_allgather(size_mb: int, iterations: int, dtype: str = 'float32'):
    """
    Test Allgather operation
    
    Args:
        size_mb: Size of tensor in MB
        iterations: Number of iterations
        dtype: Data type (float32, float16, int32)
    """
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    
    # Convert size to elements based on dtype
    bytes_per_element = 4 if dtype in ['float32', 'int32'] else 2  # float16 is 2 bytes
    size_elements = (size_mb * 1024 * 1024) // bytes_per_element
    
    # Create tensor
    if dtype == 'float32':
        tensor = torch.ones(size_elements, dtype=torch.float32).cuda()
        output_list = [torch.zeros_like(tensor) for _ in range(world_size)]
    elif dtype == 'float16':
        tensor = torch.ones(size_elements, dtype=torch.float16).cuda()
        output_list = [torch.zeros_like(tensor) for _ in range(world_size)]
    elif dtype == 'int32':
        tensor = torch.ones(size_elements, dtype=torch.int32).cuda()
        output_list = [torch.zeros_like(tensor) for _ in range(world_size)]
    else:
        raise ValueError(f'Unsupported dtype: {dtype}')
    
    # Warmup
    for _ in range(3):
        dist.all_gather(output_list, tensor)
    
    torch.cuda.synchronize()
    
    # Actual test
    start_time = time.time()
    for i in range(iterations):
        dist.all_gather(output_list, tensor)
    torch.cuda.synchronize()
    end_time = time.time()
    
    elapsed = end_time - start_time
    avg_time = elapsed / iterations
    # Allgather: each node receives data from all other nodes
    bandwidth_gbps = (size_mb * (world_size - 1)) / (avg_time)  # GB/s
    
    if rank == 0:
        print(f'Allgather test: {size_mb}MB, {iterations} iterations, dtype={dtype}')
        print(f'  Average time: {avg_time*1000:.2f} ms')
        print(f'  Bandwidth: {bandwidth_gbps:.2f} GB/s')
        print(f'  Total time: {elapsed:.2f} s')
    
    return avg_time, bandwidth_gbps


def test_broadcast(size_mb: int, iterations: int, dtype: str = 'float32'):
    """
    Test Broadcast operation
    
    Args:
        size_mb: Size of tensor in MB
        iterations: Number of iterations
        dtype: Data type (float32, float16, int32)
    """
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    
    # Convert size to elements based on dtype
    bytes_per_element = 4 if dtype in ['float32', 'int32'] else 2  # float16 is 2 bytes
    size_elements = (size_mb * 1024 * 1024) // bytes_per_element
    
    # Create tensor
    if dtype == 'float32':
        tensor = torch.ones(size_elements, dtype=torch.float32).cuda()
    elif dtype == 'float16':
        tensor = torch.ones(size_elements, dtype=torch.float16).cuda()
    elif dtype == 'int32':
        tensor = torch.ones(size_elements, dtype=torch.int32).cuda()
    else:
        raise ValueError(f'Unsupported dtype: {dtype}')
    
    # Warmup
    for _ in range(3):
        dist.broadcast(tensor, src=0)
    
    torch.cuda.synchronize()
    
    # Actual test
    start_time = time.time()
    for i in range(iterations):
        dist.broadcast(tensor, src=0)
    torch.cuda.synchronize()
    end_time = time.time()
    
    elapsed = end_time - start_time
    avg_time = elapsed / iterations
    # Broadcast: rank0 sends to all other nodes
    bandwidth_gbps = (size_mb * (world_size - 1)) / (avg_time)  # GB/s
    
    if rank == 0:
        print(f'Broadcast test: {size_mb}MB, {iterations} iterations, dtype={dtype}')
        print(f'  Average time: {avg_time*1000:.2f} ms')
        print(f'  Bandwidth: {bandwidth_gbps:.2f} GB/s')
        print(f'  Total time: {elapsed:.2f} s')
    
    return avg_time, bandwidth_gbps


def run_nccl_tests(operations: List[str], sizes_mb: List[int], iterations: int, dtype: str):
    """
    Run NCCL tests
    
    Args:
        operations: List of operations to test (allreduce, allgather, broadcast)
        sizes_mb: List of sizes in MB to test
        iterations: Number of iterations per test
        dtype: Data type to use
    """
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    
    if rank == 0:
        print(f'=== NCCL Tests ===')
        print(f'World size: {world_size}')
        print(f'Operations: {", ".join(operations)}')
        print(f'Sizes: {sizes_mb} MB')
        print(f'Iterations: {iterations}')
        print(f'Data type: {dtype}')
        print(f'Device: {torch.cuda.get_device_name(0)}')
        print()
    
    results = {}
    
    for op in operations:
        if rank == 0:
            print(f'\n--- Testing {op.upper()} ---')
        results[op] = {}
        
        for size_mb in sizes_mb:
            try:
                if op == 'allreduce':
                    avg_time, bandwidth = test_allreduce(size_mb, iterations, dtype)
                elif op == 'allgather':
                    avg_time, bandwidth = test_allgather(size_mb, iterations, dtype)
                elif op == 'broadcast':
                    avg_time, bandwidth = test_broadcast(size_mb, iterations, dtype)
                else:
                    if rank == 0:
                        print(f'Unknown operation: {op}')
                    continue
                
                results[op][size_mb] = {
                    'avg_time_ms': avg_time * 1000,
                    'bandwidth_gbps': bandwidth
                }
            except Exception as e:
                if rank == 0:
                    print(f'Error testing {op} with size {size_mb}MB: {e}')
                import traceback
                traceback.print_exc()
    
    if rank == 0:
        print(f'\n=== Summary ===')
        for op in operations:
            if op in results:
                print(f'\n{op.upper()}:')
                for size_mb in sorted(results[op].keys()):
                    r = results[op][size_mb]
                    print(f'  {size_mb}MB: {r["avg_time_ms"]:.2f} ms, {r["bandwidth_gbps"]:.2f} GB/s')


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description='NCCL tests using PyTorch distributed')
    
    parser.add_argument('--operations', type=str, default='allreduce',
                       help='Comma-separated list of operations to test (allreduce, allgather, broadcast)')
    parser.add_argument('--sizes', type=str, default='1,10,100,1000',
                       help='Comma-separated list of sizes in MB to test')
    parser.add_argument('--iterations', type=int, default=20,
                       help='Number of iterations per test (default: 20)')
    parser.add_argument('--dtype', type=str, default='float32',
                       choices=['float32', 'float16', 'int32'],
                       help='Data type to use (default: float32)')
    args = parser.parse_args()
    
    # Parse operations
    operations = [op.strip() for op in args.operations.split(',')]
    valid_ops = ['allreduce', 'allgather', 'broadcast']
    for op in operations:
        if op not in valid_ops:
            print(f'Error: Invalid operation: {op}. Valid operations: {", ".join(valid_ops)}')
            sys.exit(1)
    
    # Parse sizes
    try:
        sizes_mb = [int(s.strip()) for s in args.sizes.split(',')]
    except ValueError:
        print(f'Error: Invalid sizes format: {args.sizes}')
        sys.exit(1)
    
    # Get cluster info from environment (set by run.py when launched via dist-launch run)
    master_addr = os.environ.get('MASTER_ADDR', '')
    if not master_addr:
        print('Error: MASTER_ADDR environment variable must be set')
        print('This script should be launched via: dist-launch run <nccl_tests.sh> [options]')
        sys.exit(1)
    
    world_size = int(os.environ.get('WORLD_SIZE', '1'))
    master_port = int(os.environ.get('MASTER_PORT', '23456'))
    rank = int(os.environ.get('RANK', '0'))
    
    # Initialize PyTorch distributed
    if not torch.cuda.is_available():
        print('Error: CUDA is not available')
        sys.exit(1)
    
    # Set NCCL environment if not set
    if 'NCCL_SOCKET_IFNAME' not in os.environ:
        # Try to detect network interfaces
        import subprocess
        try:
            result = subprocess.run(['ifconfig', '-a'], capture_output=True, text=True, timeout=5)
            if result.returncode == 0:
                interfaces = []
                for line in result.stdout.split('\n'):
                    if ':' in line and not line.strip().startswith('lo'):
                        ifname = line.split(':')[0].strip()
                        if 'enP' in ifname or 'enp' in ifname.lower():
                            interfaces.append(ifname)
                if interfaces:
                    os.environ['NCCL_SOCKET_IFNAME'] = ','.join(interfaces[:5])
        except Exception:
            pass
    
    if 'NCCL_IB_DISABLE' not in os.environ:
        os.environ['NCCL_IB_DISABLE'] = '1'
    
    print(f'Initializing PyTorch distributed...')
    print(f'Master: {master_addr}:{master_port}, World size: {world_size}, Rank: {rank}')
    
    dist.init_process_group(
        backend='nccl',
        init_method=f'tcp://{master_addr}:{master_port}',
        world_size=world_size,
        rank=rank
    )
    
    try:
        # Run tests
        run_nccl_tests(operations, sizes_mb, args.iterations, args.dtype)
    finally:
        dist.destroy_process_group()


if __name__ == '__main__':
    main()

