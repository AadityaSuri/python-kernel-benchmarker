"""
KernelBenchmarker - Python Library for CUDA Kernel Benchmarking

A high-level Python interface for benchmarking CUDA kernels with automatic
parameter sweeping and comprehensive profiling.

Example:
    >>> import KernelBenchmarker as kb
    >>> kernel = kb.load_kernel("matmul.cu", "matmul_naive")
    >>> N = [128, 256, 512, 1024]
    >>> M = [128, 256, 512, 1024]
    >>> results = kb.benchmark([kernel], N, M)
"""

import subprocess
import json
import os
import tempfile
import time
from pathlib import Path
from typing import List, Dict, Any, Tuple, Callable, Optional
import numpy as np
from dataclasses import dataclass, field
import itertools


@dataclass
class KernelConfig:
    """Configuration for a CUDA kernel."""
    name: str
    source_file: str
    kernel_name: str
    block_size: Tuple[int, int, int] = (16, 16, 1)
    grid_size_func: Optional[Callable] = None
    
    def __repr__(self):
        return f"KernelConfig({self.kernel_name})"


@dataclass
class BenchmarkResult:
    """Results from benchmarking a kernel."""
    kernel_name: str
    parameters: Dict[str, Any]
    execution_time_ms: float
    memory_throughput_pct: float = 0.0
    compute_throughput_pct: float = 0.0
    occupancy_pct: float = 0.0
    memory_efficiency_pct: float = 0.0
    sm_efficiency_pct: float = 0.0
    warp_efficiency_pct: float = 0.0
    metrics: Dict[str, Any] = field(default_factory=dict)
    
    def __repr__(self):
        return (f"BenchmarkResult(kernel={self.kernel_name}, "
                f"params={self.parameters}, time={self.execution_time_ms:.4f}ms)")


class KernelBenchmarker:
    """Main benchmarking class for CUDA kernels."""
    
    def __init__(self, use_pycuda: bool = True, compute_capability: str = "sm_86"):
        """
        Initialize the benchmarker.
        
        Args:
            use_pycuda: Use PyCUDA for direct kernel execution (faster)
                       If False, uses subprocess compilation (more compatible)
            compute_capability: Target GPU compute capability (e.g., sm_86)
        """
        self.use_pycuda = use_pycuda
        self.compute_capability = compute_capability
        self.temp_dir = Path(tempfile.mkdtemp(prefix="kernel_benchmark_"))
        self.results = []
        
        if use_pycuda:
            try:
                import pycuda.autoinit
                import pycuda.driver as cuda
                from pycuda.compiler import SourceModule
                self.pycuda = cuda
                self.SourceModule = SourceModule
                print("✓ Using PyCUDA for direct kernel execution")
            except ImportError:
                print("⚠ PyCUDA not available, falling back to subprocess mode")
                print("  Install with: pip install pycuda")
                self.use_pycuda = False
    
    def load_kernel(self, source_file: str, kernel_name: str, 
                   block_size: Tuple[int, int, int] = (16, 16, 1),
                   grid_size_func: Optional[Callable] = None) -> KernelConfig:
        """
        Load a CUDA kernel from source file.
        
        Args:
            source_file: Path to .cu file containing the kernel
            kernel_name: Name of the __global__ kernel function
            block_size: Thread block dimensions (x, y, z)
            grid_size_func: Optional function to compute grid size from parameters
                           Signature: func(N, M, ...) -> (grid_x, grid_y, grid_z)
        
        Returns:
            KernelConfig object
        """
        source_path = Path(source_file)
        if not source_path.exists():
            raise FileNotFoundError(f"Kernel source file not found: {source_file}")
        
        return KernelConfig(
            name=source_path.stem,
            source_file=str(source_path),
            kernel_name=kernel_name,
            block_size=block_size,
            grid_size_func=grid_size_func
        )
    
    def _compute_grid_size(self, kernel: KernelConfig, *args) -> Tuple[int, int, int]:
        """Compute grid size based on problem dimensions."""
        if kernel.grid_size_func:
            return kernel.grid_size_func(*args)
        
        # Default: assume first two args are N, M (matrix dimensions)
        if len(args) >= 2:
            N, M = args[0], args[1]
            grid_x = (M + kernel.block_size[0] - 1) // kernel.block_size[0]
            grid_y = (N + kernel.block_size[1] - 1) // kernel.block_size[1]
            return (grid_x, grid_y, 1)
        
        # Fallback for 1D problems
        N = args[0] if args else 1024
        grid_x = (N + kernel.block_size[0] - 1) // kernel.block_size[0]
        return (grid_x, 1, 1)
    
    def _prepare_data_matmul(self, N: int, M: int, K: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Prepare input/output matrices for matrix multiplication."""
        a = np.random.randn(N, K).astype(np.float32)
        b = np.random.randn(K, M).astype(np.float32)
        c = np.zeros((N, M), dtype=np.float32)
        return a, b, c
    
    def _benchmark_pycuda(self, kernel: KernelConfig, params: Dict[str, Any], 
                         iterations: int = 100) -> BenchmarkResult:
        """Benchmark using PyCUDA (direct execution)."""
        import pycuda.driver as cuda
        from pycuda.compiler import SourceModule
        
        # Read kernel source
        with open(kernel.source_file, 'r') as f:
            kernel_source = f.read()
        
        # Compile kernel
        mod = SourceModule(kernel_source, arch=self.compute_capability)
        kernel_func = mod.get_function(kernel.kernel_name)
        
        # Prepare data based on kernel type (infer from parameters)
        N = params.get('N', 1024)
        M = params.get('M', N)
        K = params.get('K', N)
        
        a, b, c = self._prepare_data_matmul(N, M, K)
        
        # Allocate GPU memory
        a_gpu = cuda.mem_alloc(a.nbytes)
        b_gpu = cuda.mem_alloc(b.nbytes)
        c_gpu = cuda.mem_alloc(c.nbytes)
        
        cuda.memcpy_htod(a_gpu, a)
        cuda.memcpy_htod(b_gpu, b)
        
        # Compute grid size
        grid_size = self._compute_grid_size(kernel, N, M, K)
        
        # Warmup
        for _ in range(5):
            kernel_func(a_gpu, b_gpu, c_gpu, np.int32(N), np.int32(M), np.int32(K),
                       block=kernel.block_size, grid=grid_size)
        cuda.Context.synchronize()
        
        # Benchmark
        times = []
        start_event = cuda.Event()
        end_event = cuda.Event()
        
        for _ in range(iterations):
            start_event.record()
            kernel_func(a_gpu, b_gpu, c_gpu, np.int32(N), np.int32(M), np.int32(K),
                       block=kernel.block_size, grid=grid_size)
            end_event.record()
            end_event.synchronize()
            
            elapsed = start_event.time_till(end_event)  # milliseconds
            times.append(elapsed)
        
        avg_time = np.mean(times)
        
        # Cleanup
        a_gpu.free()
        b_gpu.free()
        c_gpu.free()
        
        return BenchmarkResult(
            kernel_name=kernel.kernel_name,
            parameters=params,
            execution_time_ms=avg_time
        )
    
    def _benchmark_subprocess(self, kernel: KernelConfig, params: Dict[str, Any],
                             iterations: int = 100) -> BenchmarkResult:
        """Benchmark using compiled binary and subprocess execution."""
        # Generate wrapper C++ code
        wrapper_code = self._generate_wrapper_code(kernel, params, iterations)
        wrapper_file = self.temp_dir / f"{kernel.kernel_name}_wrapper.cu"
        binary_file = self.temp_dir / f"{kernel.kernel_name}_bin"
        
        with open(wrapper_file, 'w') as f:
            f.write(wrapper_code)
        
        # Compile
        compile_cmd = [
            "nvcc",
            str(wrapper_file),
            "-o", str(binary_file),
            "-O3",
            f"-arch={self.compute_capability}",
            "-I", str(Path(kernel.source_file).parent)
        ]
        
        try:
            subprocess.run(compile_cmd, check=True, capture_output=True)
        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"Compilation failed: {e.stderr.decode()}")
        
        # Run benchmark
        result = subprocess.run([str(binary_file)], capture_output=True, text=True)
        
        # Parse output
        avg_time = self._parse_timing_output(result.stdout)
        
        return BenchmarkResult(
            kernel_name=kernel.kernel_name,
            parameters=params,
            execution_time_ms=avg_time
        )
    
    def _generate_wrapper_code(self, kernel: KernelConfig, params: Dict[str, Any],
                               iterations: int) -> str:
        """Generate C++ wrapper code for benchmarking."""
        N = params.get('N', 1024)
        M = params.get('M', N)
        K = params.get('K', N)
        
        # Read original kernel
        with open(kernel.source_file, 'r') as f:
            kernel_code = f.read()
        
        wrapper = f"""
{kernel_code}

#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>

int main() {{
    const int N = {N};
    const int M = {M};
    const int K = {K};
    const int iterations = {iterations};
    
    // Allocate host memory
    float *h_a = (float*)malloc(N * K * sizeof(float));
    float *h_b = (float*)malloc(K * M * sizeof(float));
    float *h_c = (float*)malloc(N * M * sizeof(float));
    
    // Initialize data
    for (int i = 0; i < N * K; i++) h_a[i] = (float)rand() / RAND_MAX;
    for (int i = 0; i < K * M; i++) h_b[i] = (float)rand() / RAND_MAX;
    
    // Allocate device memory
    float *d_a, *d_b, *d_c;
    cudaMalloc(&d_a, N * K * sizeof(float));
    cudaMalloc(&d_b, K * M * sizeof(float));
    cudaMalloc(&d_c, N * M * sizeof(float));
    
    cudaMemcpy(d_a, h_a, N * K * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, K * M * sizeof(float), cudaMemcpyHostToDevice);
    
    // Configure kernel
    dim3 block({kernel.block_size[0]}, {kernel.block_size[1]}, {kernel.block_size[2]});
    dim3 grid(
        (M + block.x - 1) / block.x,
        (N + block.y - 1) / block.y,
        1
    );
    
    // Warmup
    for (int i = 0; i < 5; i++) {{
        {kernel.kernel_name}<<<grid, block>>>(d_a, d_b, d_c, N, M, K);
    }}
    cudaDeviceSynchronize();
    
    // Benchmark
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    float total_time = 0.0f;
    for (int i = 0; i < iterations; i++) {{
        cudaEventRecord(start);
        {kernel.kernel_name}<<<grid, block>>>(d_a, d_b, d_c, N, M, K);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        
        float milliseconds = 0;
        cudaEventElapsedTime(&milliseconds, start, stop);
        total_time += milliseconds;
    }}
    
    float avg_time = total_time / iterations;
    printf("BENCHMARK_TIME: %.6f\\n", avg_time);
    
    // Cleanup
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
    free(h_a);
    free(h_b);
    free(h_c);
    
    return 0;
}}
"""
        return wrapper
    
    def _parse_timing_output(self, output: str) -> float:
        """Parse timing from benchmark output."""
        for line in output.split('\n'):
            if 'BENCHMARK_TIME:' in line:
                return float(line.split(':')[1].strip())
        return 0.0
    
    def _profile_with_ncu(self, kernel: KernelConfig, params: Dict[str, Any]) -> Dict[str, Any]:
        """Profile kernel using Nsight Compute."""
        # Generate and compile wrapper (same as benchmark)
        wrapper_code = self._generate_wrapper_code(kernel, params, 10)
        wrapper_file = self.temp_dir / f"{kernel.kernel_name}_profile.cu"
        binary_file = self.temp_dir / f"{kernel.kernel_name}_profile_bin"
        
        with open(wrapper_file, 'w') as f:
            f.write(wrapper_code)
        
        # Compile
        compile_cmd = [
            "nvcc", str(wrapper_file), "-o", str(binary_file),
            "-O3", f"-arch={self.compute_capability}",
            "--generate-line-info"
        ]
        subprocess.run(compile_cmd, check=True, capture_output=True)
        
        # Profile with NCU
        metrics = [
            "sm__cycles_elapsed.avg",
            "dram__throughput.avg.pct_of_peak_sustained_elapsed",
            "gpu__compute_memory_throughput.avg.pct_of_peak_sustained_elapsed",
            "sm__throughput.avg.pct_of_peak_sustained_elapsed",
            "sm__warps_active.avg.pct_of_peak_sustained_active",
        ]
        
        ncu_cmd = [
            "ncu",
            "--metrics", ",".join(metrics),
            "--csv",
            str(binary_file)
        ]
        
        try:
            result = subprocess.run(ncu_cmd, capture_output=True, text=True, timeout=60)
            return self._parse_ncu_csv(result.stdout)
        except (subprocess.TimeoutExpired, FileNotFoundError):
            return {}
    
    def _parse_ncu_csv(self, csv_output: str) -> Dict[str, Any]:
        """Parse NCU CSV output."""
        metrics = {}
        for line in csv_output.split('\n'):
            if not line or line.startswith('"ID"'):
                continue
            parts = line.strip().split(',')
            if len(parts) >= 2:
                key = parts[0].strip('"')
                try:
                    value = float(parts[1].strip('"'))
                    metrics[key] = value
                except ValueError:
                    pass
        return metrics
    
    def benchmark(self, kernels: List[KernelConfig], *param_ranges,
                 iterations: int = 100, profile: bool = True,
                 zip_params: bool = False) -> List[BenchmarkResult]:
        """
        Benchmark kernels across parameter ranges.

        Args:
            kernels: List of KernelConfig objects
            *param_ranges: Variable number of parameter ranges
                          E.g., N=[128, 256], M=[128, 256]
            iterations: Number of iterations per benchmark
            profile: Whether to run Nsight Compute profiling
            zip_params: If False (default), generates all combinations using
                       itertools.product (e.g., N=[1,2], M=[1,2] → 4 combos).
                       If True, zips parameters together (e.g., N=[1,2], M=[1,2]
                       → 2 combos: (1,1), (2,2)). Useful for square matrices.

        Returns:
            List of BenchmarkResult objects

        Example:
            >>> # All combinations (default): 9 benchmarks
            >>> results = benchmarker.benchmark(
            ...     [kernel1, kernel2],
            ...     N=[128, 256, 512],
            ...     M=[128, 256, 512]
            ... )
            >>> # Zipped parameters: 3 benchmarks (128x128, 256x256, 512x512)
            >>> results = benchmarker.benchmark(
            ...     [kernel1, kernel2],
            ...     N=[128, 256, 512],
            ...     M=[128, 256, 512],
            ...     zip_params=True
            ... )
        """
        # Parse parameter ranges
        param_names = []
        param_values = []
        
        # Handle both positional and keyword arguments
        if param_ranges and isinstance(param_ranges[0], dict):
            # Called as benchmark(kernels, {'N': [...], 'M': [...]})
            params_dict = param_ranges[0]
            param_names = list(params_dict.keys())
            param_values = list(params_dict.values())
        else:
            # Assume positional: N, M, K, etc.
            param_names = ['N', 'M', 'K'][:len(param_ranges)]
            param_values = list(param_ranges)
        
        # Generate parameter combinations
        if zip_params:
            # Zip parameters together (e.g., for square matrices where N=M=K)
            if len(set(len(v) for v in param_values)) != 1:
                raise ValueError("All parameter lists must have the same length when zip_params=True")
            param_combinations = list(zip(*param_values))
        else:
            # Cartesian product of all parameters (default)
            param_combinations = list(itertools.product(*param_values))
        
        print(f"\n{'='*70}")
        print(f"CUDA Kernel Benchmarking")
        print(f"{'='*70}")
        print(f"Kernels: {len(kernels)}")
        print(f"Parameter combinations: {len(param_combinations)}")
        print(f"Total benchmarks: {len(kernels) * len(param_combinations)}")
        print(f"{'='*70}\n")
        
        results = []
        
        for kernel in kernels:
            print(f"\nBenchmarking: {kernel.kernel_name}")
            print("-" * 70)
            
            for param_combo in param_combinations:
                params = dict(zip(param_names, param_combo))
                param_str = ", ".join(f"{k}={v}" for k, v in params.items())
                print(f"  Parameters: {param_str}... ", end='', flush=True)
                
                try:
                    if self.use_pycuda:
                        result = self._benchmark_pycuda(kernel, params, iterations)
                    else:
                        result = self._benchmark_subprocess(kernel, params, iterations)
                    
                    # Add profiling if requested
                    if profile:
                        metrics = self._profile_with_ncu(kernel, params)
                        result.memory_throughput_pct = metrics.get(
                            'dram__throughput.avg.pct_of_peak_sustained_elapsed', 0.0)
                        result.compute_throughput_pct = metrics.get(
                            'gpu__compute_memory_throughput.avg.pct_of_peak_sustained_elapsed', 0.0)
                        result.sm_efficiency_pct = metrics.get(
                            'sm__throughput.avg.pct_of_peak_sustained_elapsed', 0.0)
                        result.occupancy_pct = metrics.get(
                            'sm__warps_active.avg.pct_of_peak_sustained_active', 0.0)
                        result.metrics = metrics
                    
                    results.append(result)
                    print(f"✓ {result.execution_time_ms:.4f} ms")
                    
                except Exception as e:
                    print(f"✗ Failed: {e}")
        
        self.results = results
        return results
    
    def save_results(self, output_file: str = "benchmark_results.json"):
        """Save benchmark results to JSON file."""
        output_path = Path(output_file)
        
        results_dict = {
            'benchmarks': [
                {
                    'kernel': r.kernel_name,
                    'parameters': r.parameters,
                    'execution_time_ms': r.execution_time_ms,
                    'memory_throughput_pct': r.memory_throughput_pct,
                    'compute_throughput_pct': r.compute_throughput_pct,
                    'occupancy_pct': r.occupancy_pct,
                    'sm_efficiency_pct': r.sm_efficiency_pct,
                    'metrics': r.metrics
                }
                for r in self.results
            ]
        }
        
        with open(output_path, 'w') as f:
            json.dump(results_dict, f, indent=2)
        
        print(f"\n✓ Results saved to {output_path}")
    
    def generate_report(self, output_file: str = "benchmark_report.txt"):
        """Generate human-readable comparison report."""
        output_path = Path(output_file)
        
        with open(output_path, 'w') as f:
            f.write("=" * 80 + "\n")
            f.write("CUDA KERNEL BENCHMARK REPORT\n")
            f.write("=" * 80 + "\n\n")
            
            # Group by parameters
            param_groups = {}
            for result in self.results:
                key = tuple(sorted(result.parameters.items()))
                if key not in param_groups:
                    param_groups[key] = []
                param_groups[key].append(result)
            
            for params, group_results in param_groups.items():
                params_dict = dict(params)
                param_str = ", ".join(f"{k}={v}" for k, v in params_dict.items())
                
                f.write(f"\nParameters: {param_str}\n")
                f.write("-" * 80 + "\n")
                f.write(f"{'Kernel':<30} {'Time (ms)':<15} {'Mem %':<10} {'Compute %':<12}\n")
                f.write("-" * 80 + "\n")
                
                for result in sorted(group_results, key=lambda r: r.execution_time_ms):
                    f.write(f"{result.kernel_name:<30} "
                           f"{result.execution_time_ms:<15.4f} "
                           f"{result.memory_throughput_pct:<10.1f} "
                           f"{result.compute_throughput_pct:<12.1f}\n")
                
                # Calculate speedup
                if len(group_results) > 1:
                    baseline = max(group_results, key=lambda r: r.execution_time_ms)
                    f.write("\nSpeedup vs slowest:\n")
                    for result in group_results:
                        speedup = baseline.execution_time_ms / result.execution_time_ms
                        f.write(f"  {result.kernel_name}: {speedup:.2f}x\n")
                
                f.write("\n")
        
        print(f"✓ Report saved to {output_path}")
    
    def cleanup(self):
        """Clean up temporary files."""
        import shutil
        if self.temp_dir.exists():
            shutil.rmtree(self.temp_dir)


# Convenience functions for the API
def load_kernel(source_file: str, kernel_name: str, **kwargs) -> KernelConfig:
    """Load a CUDA kernel (convenience function)."""
    benchmarker = KernelBenchmarker()
    return benchmarker.load_kernel(source_file, kernel_name, **kwargs)


def benchmark(kernels: List[KernelConfig], *args, **kwargs) -> List[BenchmarkResult]:
    """Benchmark kernels (convenience function)."""
    benchmarker = KernelBenchmarker()
    results = benchmarker.benchmark(kernels, *args, **kwargs)
    benchmarker.save_results()
    benchmarker.generate_report()
    return results
