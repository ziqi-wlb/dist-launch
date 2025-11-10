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
    local_rank = int(os.environ.get('LOCAL_RANK', '0'))
    
    # Convert size to elements based on dtype
    bytes_per_element = 4 if dtype in ['float32', 'int32'] else 2  # float16 is 2 bytes
    size_elements = (size_mb * 1024 * 1024) // bytes_per_element
    
    # Create tensor on the correct GPU device
    device = torch.device(f'cuda:{local_rank}')
    if dtype == 'float32':
        tensor = torch.ones(size_elements, dtype=torch.float32, device=device)
    elif dtype == 'float16':
        tensor = torch.ones(size_elements, dtype=torch.float16, device=device)
    elif dtype == 'int32':
        tensor = torch.ones(size_elements, dtype=torch.int32, device=device)
    else:
        raise ValueError(f'Unsupported dtype: {dtype}')
    
    # Warmup
    for _ in range(3):
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    
    torch.cuda.synchronize(device)
    
    # Actual test
    start_time = time.time()
    for i in range(iterations):
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    torch.cuda.synchronize(device)
    end_time = time.time()
    
    elapsed = end_time - start_time
    avg_time = elapsed / iterations
    # Bandwidth calculation: distinguish between algorithm bandwidth and bus bandwidth
    # Convert MB to GB: divide by 1024
    size_gb = size_mb / 1024.0
    
    # Algorithm bandwidth: data processed per second (actual data size / time)
    # This represents the effective bandwidth from the algorithm's perspective
    algo_bw_gbps = size_gb / avg_time
    
    # Bus bandwidth: total data transferred on bus per second
    # For Allreduce: each node sends and receives data, total bus traffic = size_GB * 2 * (nRanks - 1) / nRanks
    # Formula matches NVIDIA official nccl-tests: bus_bw = (size_GB * 2 * (nRanks - 1) / nRanks) / time
    bus_bw_gbps = (size_gb * 2 * (world_size - 1) / world_size) / avg_time
    
    if rank == 0:
        print(f'Allreduce test: {size_mb}MB, {iterations} iterations, dtype={dtype}')
        print(f'  Average time: {avg_time*1000:.2f} ms')
        print(f'  Algorithm bandwidth: {algo_bw_gbps:.2f} GB/s')
        print(f'  Bus bandwidth: {bus_bw_gbps:.2f} GB/s')
        print(f'  Total time: {elapsed:.2f} s')
    
    return avg_time, algo_bw_gbps, bus_bw_gbps


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
    local_rank = int(os.environ.get('LOCAL_RANK', '0'))
    
    # Convert size to elements based on dtype
    bytes_per_element = 4 if dtype in ['float32', 'int32'] else 2  # float16 is 2 bytes
    size_elements = (size_mb * 1024 * 1024) // bytes_per_element
    
    # Create tensor on the correct GPU device
    device = torch.device(f'cuda:{local_rank}')
    if dtype == 'float32':
        tensor = torch.ones(size_elements, dtype=torch.float32, device=device)
        output_list = [torch.zeros_like(tensor) for _ in range(world_size)]
    elif dtype == 'float16':
        tensor = torch.ones(size_elements, dtype=torch.float16, device=device)
        output_list = [torch.zeros_like(tensor) for _ in range(world_size)]
    elif dtype == 'int32':
        tensor = torch.ones(size_elements, dtype=torch.int32, device=device)
        output_list = [torch.zeros_like(tensor) for _ in range(world_size)]
    else:
        raise ValueError(f'Unsupported dtype: {dtype}')
    
    # Warmup
    for _ in range(3):
        dist.all_gather(output_list, tensor)
    
    torch.cuda.synchronize(device)
    
    # Actual test
    start_time = time.time()
    for i in range(iterations):
        dist.all_gather(output_list, tensor)
    torch.cuda.synchronize(device)
    end_time = time.time()
    
    elapsed = end_time - start_time
    avg_time = elapsed / iterations
    # Bandwidth calculation: distinguish between algorithm bandwidth and bus bandwidth
    # Convert MB to GB: divide by 1024
    size_gb = size_mb / 1024.0
    
    # Algorithm bandwidth: data processed per second (actual data size / time)
    algo_bw_gbps = size_gb / avg_time
    
    # Bus bandwidth: total data transferred on bus per second
    # For Allgather: each node receives data from all other nodes
    # Formula matches NVIDIA official nccl-tests: bus_bw = (size_GB * (nRanks - 1)) / time
    bus_bw_gbps = (size_gb * (world_size - 1)) / avg_time
    
    if rank == 0:
        print(f'Allgather test: {size_mb}MB, {iterations} iterations, dtype={dtype}')
        print(f'  Average time: {avg_time*1000:.2f} ms')
        print(f'  Algorithm bandwidth: {algo_bw_gbps:.2f} GB/s')
        print(f'  Bus bandwidth: {bus_bw_gbps:.2f} GB/s')
        print(f'  Total time: {elapsed:.2f} s')
    
    return avg_time, algo_bw_gbps, bus_bw_gbps


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
    local_rank = int(os.environ.get('LOCAL_RANK', '0'))
    
    # Convert size to elements based on dtype
    bytes_per_element = 4 if dtype in ['float32', 'int32'] else 2  # float16 is 2 bytes
    size_elements = (size_mb * 1024 * 1024) // bytes_per_element
    
    # Create tensor on the correct GPU device
    device = torch.device(f'cuda:{local_rank}')
    if dtype == 'float32':
        tensor = torch.ones(size_elements, dtype=torch.float32, device=device)
    elif dtype == 'float16':
        tensor = torch.ones(size_elements, dtype=torch.float16, device=device)
    elif dtype == 'int32':
        tensor = torch.ones(size_elements, dtype=torch.int32, device=device)
    else:
        raise ValueError(f'Unsupported dtype: {dtype}')
    
    # Warmup
    for _ in range(3):
        dist.broadcast(tensor, src=0)
    
    torch.cuda.synchronize(device)
    
    # Actual test
    start_time = time.time()
    for i in range(iterations):
        dist.broadcast(tensor, src=0)
    torch.cuda.synchronize(device)
    end_time = time.time()
    
    elapsed = end_time - start_time
    avg_time = elapsed / iterations
    # Bandwidth calculation: distinguish between algorithm bandwidth and bus bandwidth
    # Convert MB to GB: divide by 1024
    size_gb = size_mb / 1024.0
    
    # Algorithm bandwidth: data processed per second (actual data size / time)
    algo_bw_gbps = size_gb / avg_time
    
    # Bus bandwidth: total data transferred on bus per second
    # For Broadcast: rank0 sends to all other nodes
    # Formula matches NVIDIA official nccl-tests: bus_bw = (size_GB * (nRanks - 1)) / time
    bus_bw_gbps = (size_gb * (world_size - 1)) / avg_time
    
    if rank == 0:
        print(f'Broadcast test: {size_mb}MB, {iterations} iterations, dtype={dtype}')
        print(f'  Average time: {avg_time*1000:.2f} ms')
        print(f'  Algorithm bandwidth: {algo_bw_gbps:.2f} GB/s')
        print(f'  Bus bandwidth: {bus_bw_gbps:.2f} GB/s')
        print(f'  Total time: {elapsed:.2f} s')
    
    return avg_time, algo_bw_gbps, bus_bw_gbps


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
                    avg_time, algo_bw, bus_bw = test_allreduce(size_mb, iterations, dtype)
                elif op == 'allgather':
                    avg_time, algo_bw, bus_bw = test_allgather(size_mb, iterations, dtype)
                elif op == 'broadcast':
                    avg_time, algo_bw, bus_bw = test_broadcast(size_mb, iterations, dtype)
                else:
                    if rank == 0:
                        print(f'Unknown operation: {op}')
                    continue
                
                results[op][size_mb] = {
                    'avg_time_ms': avg_time * 1000,
                    'algo_bw_gbps': algo_bw,
                    'bus_bw_gbps': bus_bw
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
                    print(f'  {size_mb}MB: {r["avg_time_ms"]:.2f} ms, algo_bw: {r["algo_bw_gbps"]:.2f} GB/s, bus_bw: {r["bus_bw_gbps"]:.2f} GB/s')


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        prog='dist-launch nccl-tests',
        description='NCCL performance tests using PyTorch distributed. '
                    'This command runs collective communication tests (Allreduce, Allgather, Broadcast) '
                    'across multiple GPUs and nodes to measure network bandwidth.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
Examples:
  # Run allreduce test with default settings
  dist-launch nccl-tests

  # Test multiple operations with custom sizes
  dist-launch nccl-tests --operations allreduce,allgather --sizes 10,100,1000

  # 2 nodes, 4 GPUs per node
  dist-launch nccl-tests --nper-node 4 --world-size 2

Note: This command must be run via "dist-launch nccl-tests" or "dist-launch run nccl_tests.sh".
Environment variables (RANK, WORLD_SIZE, MASTER_ADDR, MASTER_PORT) are set automatically.
        '''
    )
    
    parser.add_argument('--operations', type=str, default='allreduce',
                       help='Comma-separated list of operations to test: allreduce, allgather, broadcast (default: allreduce)')
    parser.add_argument('--sizes', type=str, default='1,10,100,1000',
                       help='Comma-separated list of sizes in MB to test (default: 1,10,100,1000)')
    parser.add_argument('--iterations', type=int, default=20,
                       help='Number of iterations per test (default: 20)')
    parser.add_argument('--dtype', type=str, default='float32',
                       choices=['float32', 'float16', 'int32'],
                       help='Data type to use: float32, float16, or int32 (default: float32)')
    parser.add_argument('--nper-node', type=int, default=None,
                       dest='nper_node',
                       help='Number of GPUs per node. If not specified, auto-detects from CUDA_VISIBLE_DEVICES or uses 1')
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
    
    # Handle nper_node parameter
    # If nper_node is specified, world_size should be num_nodes * nper_node
    # But in this script, world_size is already the total number of processes (including all GPUs)
    # So we just need to set LOCAL_RANK if not already set
    nper_node = args.nper_node
    if nper_node is None:
        # Try to auto-detect from CUDA_VISIBLE_DEVICES
        cuda_visible = os.environ.get('CUDA_VISIBLE_DEVICES', '')
        if cuda_visible:
            nper_node = len([x for x in cuda_visible.split(',') if x.strip()])
        else:
            nper_node = torch.cuda.device_count() if torch.cuda.is_available() else 1
    
    # Calculate local_rank from global rank and nper_node
    local_rank = rank % nper_node
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(local_rank)
    
    # Initialize PyTorch distributed
    if not torch.cuda.is_available():
        print('Error: CUDA is not available')
        sys.exit(1)
    
    # Set CUDA device for this process
    torch.cuda.set_device(local_rank)
    
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
    print(f'Master: {master_addr}:{master_port}, World size: {world_size}, Rank: {rank}, Local rank: {local_rank}, GPUs per node: {nper_node}')
    print(f'Environment check - RANK={os.environ.get("RANK")}, WORLD_SIZE={os.environ.get("WORLD_SIZE")}, LOCAL_RANK={os.environ.get("LOCAL_RANK")}')
    print(f'Process PID: {os.getpid()}, Parent PID: {os.getppid()}')
    
    # Verify rank matches environment variable
    env_rank = int(os.environ.get('RANK', '-1'))
    if env_rank != rank:
        print(f'ERROR: Rank mismatch! env RANK={env_rank}, but calculated rank={rank}')
        sys.exit(1)
    
    # Only rank 0 should bind the port, others should connect
    if rank == 0:
        print(f'Rank 0: Will bind to port {master_port} as master')
    else:
        print(f'Rank {rank}: Will connect to master at {master_addr}:{master_port}')
    
    dist.init_process_group(
        backend='nccl',
        init_method=f'tcp://{master_addr}:{master_port}',
        world_size=world_size,
        rank=rank,
        timeout=torch.distributed.default_pg_timeout
    )
    print(f'✓ Process group initialized successfully (rank {rank})')
    
    try:
        # Run tests
        run_nccl_tests(operations, sizes_mb, args.iterations, args.dtype)
    finally:
        dist.destroy_process_group()


if __name__ == '__main__':
    main()

