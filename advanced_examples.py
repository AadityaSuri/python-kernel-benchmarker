#!/usr/bin/env python3
"""
Advanced KernelBenchmarker Usage Examples

Shows more sophisticated use cases and features.
"""

import KernelBenchmarker as kb

# ============================================================================
# Example 1: Simple Vector Addition
# ============================================================================

def example_vector_add():
    """Benchmark vector addition kernels."""
    print("\n" + "="*70)
    print("Example 1: Vector Addition")
    print("="*70)
    
    # Load kernels
    vec_basic = kb.load_kernel("vector_kernels.cu", "vector_add_basic")
    vec_optimized = kb.load_kernel("vector_kernels.cu", "vector_add_optimized")
    
    # Single parameter (vector size)
    N = [2**i for i in range(20, 26)]  # 1M to 64M elements
    
    benchmarker = kb.KernelBenchmarker()
    results = benchmarker.benchmark([vec_basic, vec_optimized], N)
    
    benchmarker.save_results("vector_results.json")
    benchmarker.generate_report("vector_report.txt")


# ============================================================================
# Example 2: Matrix Multiplication with Custom Grid Size
# ============================================================================

def example_matmul_custom_grid():
    """Matrix multiplication with custom grid configuration."""
    print("\n" + "="*70)
    print("Example 2: Matrix Multiplication with Custom Grid")
    print("="*70)
    
    # Custom grid size function for better performance
    def compute_grid_matmul(N, M, K):
        block_x, block_y = 16, 16
        grid_x = (M + block_x - 1) // block_x
        grid_y = (N + block_y - 1) // block_y
        return (grid_x, grid_y, 1)
    
    # Load with custom configuration
    matmul = kb.load_kernel(
        "matmul_kernels.cu", 
        "matmul_optimized",
        block_size=(16, 16, 1),
        grid_size_func=compute_grid_matmul
    )
    
    # Test different matrix sizes
    N = [512, 1024, 2048]
    M = [512, 1024, 2048]
    K = [512, 1024, 2048]
    
    benchmarker = kb.KernelBenchmarker()
    results = benchmarker.benchmark([matmul], N, M, K, iterations=50)
    
    # Print speedup analysis
    for result in results:
        flops = 2 * result.parameters['N'] * result.parameters['M'] * result.parameters['K']
        gflops = (flops / result.execution_time_ms) / 1e6  # GFLOPS
        print(f"N={result.parameters['N']}, M={result.parameters['M']}, K={result.parameters['K']}: "
              f"{gflops:.2f} GFLOPS")


# ============================================================================
# Example 3: Comparing Many Implementations
# ============================================================================

def example_comprehensive_comparison():
    """Compare multiple optimization strategies."""
    print("\n" + "="*70)
    print("Example 3: Comprehensive Kernel Comparison")
    print("="*70)
    
    # Load all variants
    kernels = [
        kb.load_kernel("matmul_kernels.cu", "matmul_naive"),
        kb.load_kernel("matmul_kernels.cu", "matmul_optimized"),
        kb.load_kernel("matmul_kernels.cu", "matmul_coalesced"),
    ]
    
    # Test on a few key sizes
    sizes = [256, 512, 1024, 2048]
    
    benchmarker = kb.KernelBenchmarker()
    results = benchmarker.benchmark(
        kernels, 
        N=sizes, 
        M=sizes, 
        K=sizes,
        iterations=100,
        profile=True  # Include Nsight profiling
    )
    
    # Analyze results
    print("\n" + "="*70)
    print("Performance Analysis")
    print("="*70)
    
    # Group by size
    by_size = {}
    for r in results:
        size = r.parameters['N']
        if size not in by_size:
            by_size[size] = []
        by_size[size].append(r)
    
    for size in sorted(by_size.keys()):
        print(f"\nMatrix size: {size}x{size}")
        print("-" * 70)
        
        group = by_size[size]
        baseline = max(group, key=lambda x: x.execution_time_ms)
        
        for r in sorted(group, key=lambda x: x.execution_time_ms):
            speedup = baseline.execution_time_ms / r.execution_time_ms
            flops = 2 * size * size * size
            gflops = (flops / r.execution_time_ms) / 1e6
            
            print(f"{r.kernel_name:25s}: {r.execution_time_ms:8.3f} ms | "
                  f"{speedup:5.2f}x | {gflops:7.1f} GFLOPS | "
                  f"Mem: {r.memory_throughput_pct:5.1f}%")
    
    benchmarker.save_results("comprehensive_results.json")
    benchmarker.generate_report("comprehensive_report.txt")


# ============================================================================
# Example 4: Using PyCUDA for Faster Benchmarking
# ============================================================================

def example_pycuda_mode():
    """Demonstrate PyCUDA mode for direct kernel execution."""
    print("\n" + "="*70)
    print("Example 4: PyCUDA Mode (Faster)")
    print("="*70)
    
    # Initialize with PyCUDA
    benchmarker = kb.KernelBenchmarker(use_pycuda=True)
    
    # Load kernels
    matmul = benchmarker.load_kernel("matmul_kernels.cu", "matmul_naive")
    
    # Quick benchmark
    results = benchmarker.benchmark(
        [matmul],
        N=[512, 1024],
        M=[512, 1024],
        K=[512, 1024],
        iterations=100,
        profile=False  # Skip profiling for speed
    )
    
    for r in results:
        print(f"N={r.parameters['N']}: {r.execution_time_ms:.3f} ms")


# ============================================================================
# Example 5: Parameter Sweep for Optimization
# ============================================================================

def example_parameter_sweep():
    """Find optimal block size through parameter sweep."""
    print("\n" + "="*70)
    print("Example 5: Block Size Optimization")
    print("="*70)
    
    # Test different block sizes
    block_sizes = [(8, 8, 1), (16, 16, 1), (32, 32, 1)]
    
    all_results = []
    
    for block_size in block_sizes:
        kernel = kb.load_kernel(
            "matmul_kernels.cu",
            "matmul_optimized",
            block_size=block_size
        )
        
        benchmarker = kb.KernelBenchmarker()
        results = benchmarker.benchmark(
            [kernel],
            N=[1024],
            M=[1024],
            K=[1024],
            iterations=50
        )
        
        avg_time = results[0].execution_time_ms
        print(f"Block size {block_size}: {avg_time:.3f} ms")
        all_results.extend(results)
    
    # Find best
    best = min(all_results, key=lambda x: x.execution_time_ms)
    print(f"\nBest configuration: {best.execution_time_ms:.3f} ms")


# ============================================================================
# Example 6: Dictionary-style Parameters
# ============================================================================

def example_dict_params():
    """Use dictionary for parameters (alternative API)."""
    print("\n" + "="*70)
    print("Example 6: Dictionary-style Parameters")
    print("="*70)
    
    kernel = kb.load_kernel("matmul_kernels.cu", "matmul_naive")
    
    benchmarker = kb.KernelBenchmarker()
    
    # Can also pass parameters as dictionary
    params = {
        'N': [512, 1024],
        'M': [512, 1024],
        'K': [512, 1024]
    }
    
    results = benchmarker.benchmark([kernel], params)
    
    for r in results:
        print(f"{r.parameters}: {r.execution_time_ms:.3f} ms")


# ============================================================================
# Run Examples
# ============================================================================

if __name__ == "__main__":
    import sys
    
    examples = {
        '1': example_vector_add,
        '2': example_matmul_custom_grid,
        '3': example_comprehensive_comparison,
        '4': example_pycuda_mode,
        '5': example_parameter_sweep,
        '6': example_dict_params,
    }
    
    if len(sys.argv) > 1:
        example_num = sys.argv[1]
        if example_num in examples:
            examples[example_num]()
        else:
            print(f"Example {example_num} not found")
    else:
        print("Available examples:")
        print("  1. Vector Addition")
        print("  2. Matrix Multiplication with Custom Grid")
        print("  3. Comprehensive Kernel Comparison")
        print("  4. PyCUDA Mode")
        print("  5. Block Size Optimization")
        print("  6. Dictionary-style Parameters")
        print("\nUsage: python advanced_examples.py <example_number>")
        print("   Or: python advanced_examples.py 3")
