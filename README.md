# KernelBenchmarker - Python API for CUDA Kernel Benchmarking

A high-level Python library for benchmarking CUDA kernels with automatic parameter sweeping and comprehensive profiling.

## ✨ Features

- **Clean Python API** - Load kernels from `.cu` files and benchmark directly
- **Automatic Parameter Sweeping** - Test all combinations of parameters automatically
- **Nsight Integration** - Collect detailed profiling metrics
- **Dual Backend** - Use PyCUDA (fast) or subprocess compilation (compatible)
- **Rich Visualizations** - Generate heatmaps, comparison charts, speedup graphs
- **Flexible** - Works with any kernel signature as long as inputs are consistent

## 🚀 Quick Start

### Installation

```bash
# Install dependencies
pip install numpy matplotlib seaborn

# Optional: Install PyCUDA for faster execution
pip install pycuda

# CUDA toolkit must be installed separately
```

### Basic Usage

```python
import KernelBenchmarker as kb

# Load your kernels
kernel1 = kb.load_kernel("matmul_kernels.cu", "matmul_naive")
kernel2 = kb.load_kernel("matmul_kernels.cu", "matmul_optimized")

# Define parameter ranges
N = [128, 256, 512, 1024, 2048]
M = [128, 256, 512, 1024, 2048]
K = [128, 256, 512, 1024, 2048]

# Benchmark all combinations
kernels = [kernel1, kernel2]
results = kb.benchmark(kernels, N, M, K)

# Results automatically saved to:
# - benchmark_results.json
# - benchmark_report.txt
```

That's it! The exact API you requested.

## 📖 Complete Example

### 1. Create Your Kernels (matmul_kernels.cu)

```cuda
__global__ void matmul_naive(const float *a, const float *b, float *res, 
                             int N, int M, int K) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row < N && col < M) {
        float value = 0;
        for (int k = 0; k < K; k++) {
            value += a[row * K + k] * b[k * M + col];
        }
        res[row * M + col] = value;
    }
}

__global__ void matmul_optimized(const float *a, const float *b, float *res,
                                 int N, int M, int K) {
    // Your optimized implementation
    __shared__ float tile_a[16][16];
    __shared__ float tile_b[16][16];
    // ... shared memory tiling ...
}
```

### 2. Benchmark Them (benchmark.py)

```python
import KernelBenchmarker as kb

# Load kernels
naive = kb.load_kernel("matmul_kernels.cu", "matmul_naive")
optimized = kb.load_kernel("matmul_kernels.cu", "matmul_optimized")

# Define test sizes
sizes = [2**i for i in range(7, 12)]  # 128, 256, ..., 2048

# Benchmark
results = kb.benchmark(
    [naive, optimized],
    N=sizes,
    M=sizes,
    K=sizes
)

# View results
print("Check benchmark_report.txt for detailed comparison")
```

### 3. Visualize Results

```python
import visualize

# Generate all visualizations
visualize.visualize("benchmark_results.json")

# Or create specific plots
from visualize import BenchmarkVisualizer

viz = BenchmarkVisualizer("benchmark_results.json")
viz.plot_kernel_comparison(param_name='N', fixed_params={'M': 1024, 'K': 1024})
viz.plot_speedup_comparison(baseline_kernel='matmul_naive', param_name='N')
```

## 🎯 Advanced Usage

### Custom Block/Grid Configuration

```python
# Define custom grid size function
def my_grid_func(N, M, K):
    block_x, block_y = 32, 32
    grid_x = (M + block_x - 1) // block_x
    grid_y = (N + block_y - 1) // block_y
    return (grid_x, grid_y, 1)

kernel = kb.load_kernel(
    "kernels.cu",
    "my_kernel",
    block_size=(32, 32, 1),
    grid_size_func=my_grid_func
)
```

### Using PyCUDA Mode (Faster)

```python
benchmarker = kb.KernelBenchmarker(use_pycuda=True)

kernel = benchmarker.load_kernel("kernels.cu", "my_kernel")

results = benchmarker.benchmark(
    [kernel],
    N=[1024, 2048],
    M=[1024, 2048],
    iterations=100,
    profile=True  # Include Nsight profiling
)

benchmarker.save_results()
benchmarker.generate_report()
```

### Analyzing Results

```python
results = kb.benchmark([kernel1, kernel2], N=sizes, M=sizes)

for r in results:
    print(f"Kernel: {r.kernel_name}")
    print(f"  Params: {r.parameters}")
    print(f"  Time: {r.execution_time_ms:.3f} ms")
    print(f"  Memory Throughput: {r.memory_throughput_pct:.1f}%")
    print(f"  Occupancy: {r.occupancy_pct:.1f}%")
    
    # Calculate GFLOPS
    N, M, K = r.parameters['N'], r.parameters['M'], r.parameters['K']
    flops = 2 * N * M * K
    gflops = (flops / r.execution_time_ms) / 1e6
    print(f"  Performance: {gflops:.2f} GFLOPS")
```

### Parameter Sweeps for Optimization

```python
# Find optimal block size
block_sizes = [(8, 8, 1), (16, 16, 1), (32, 32, 1)]
results_all = []

for block_size in block_sizes:
    kernel = kb.load_kernel("kernels.cu", "my_kernel", block_size=block_size)
    results = kb.benchmark([kernel], N=[1024], M=[1024])
    results_all.extend(results)

best = min(results_all, key=lambda x: x.execution_time_ms)
print(f"Best block size: {best.execution_time_ms:.3f} ms")
```

## 📊 Output Files

After benchmarking, you get:

### benchmark_results.json
Machine-readable JSON with all metrics:
```json
{
  "benchmarks": [
    {
      "kernel": "matmul_naive",
      "parameters": {"N": 1024, "M": 1024, "K": 1024},
      "execution_time_ms": 45.234,
      "memory_throughput_pct": 76.5,
      "occupancy_pct": 62.3,
      "metrics": { ... }
    }
  ]
}
```

### benchmark_report.txt
Human-readable comparison:
```
===============================================================================
CUDA KERNEL BENCHMARK REPORT
===============================================================================

Parameters: N=1024, M=1024, K=1024
-------------------------------------------------------------------------------
Kernel                         Time (ms)       Mem %     Compute %
-------------------------------------------------------------------------------
matmul_optimized               12.345          85.2      78.9
matmul_naive                   45.234          65.3      45.2

Speedup vs slowest:
  matmul_optimized: 3.67x
  matmul_naive: 1.00x
```

### Visualizations (via visualize.py)
- `heatmap.png` - Execution time across parameter space
- `kernel_comparison.png` - Performance comparison chart
- `speedup.png` - Speedup relative to baseline
- `metrics_<kernel>.png` - Detailed metrics for each kernel

## 🔧 API Reference

### load_kernel()
```python
kernel = kb.load_kernel(
    source_file: str,              # Path to .cu file
    kernel_name: str,              # Name of __global__ function
    block_size: Tuple = (16,16,1), # Thread block dimensions
    grid_size_func: Callable = None # Custom grid size function
) -> KernelConfig
```

### benchmark()
```python
results = kb.benchmark(
    kernels: List[KernelConfig],  # List of kernels to benchmark
    *param_ranges,                # Parameter ranges (N, M, K, ...)
    iterations: int = 100,        # Iterations per benchmark
    profile: bool = True          # Run Nsight profiling
) -> List[BenchmarkResult]
```

### KernelBenchmarker Class
```python
benchmarker = kb.KernelBenchmarker(
    use_pycuda: bool = True,      # Use PyCUDA for direct execution
    compute_capability: str = "sm_86"  # Target GPU architecture
)

# Methods:
benchmarker.load_kernel(...)      # Load a kernel
benchmarker.benchmark(...)        # Run benchmarks
benchmarker.save_results(file)    # Save to JSON
benchmarker.generate_report(file) # Generate text report
benchmarker.cleanup()             # Clean up temp files
```

### BenchmarkResult Object
```python
result.kernel_name              # str: Kernel name
result.parameters               # dict: Parameters used
result.execution_time_ms        # float: Execution time
result.memory_throughput_pct    # float: Memory throughput %
result.compute_throughput_pct   # float: Compute throughput %
result.occupancy_pct            # float: Occupancy %
result.sm_efficiency_pct        # float: SM efficiency %
result.metrics                  # dict: All Nsight metrics
```

## 🎨 Visualization API

```python
from visualize import BenchmarkVisualizer

viz = BenchmarkVisualizer("benchmark_results.json")

# Create specific plots
viz.plot_execution_time_heatmap(param1='N', param2='M', kernel_name='matmul_naive')
viz.plot_kernel_comparison(param_name='N', fixed_params={'M': 1024})
viz.plot_speedup_comparison(baseline_kernel='matmul_naive', param_name='N')
viz.plot_performance_metrics(kernel_name='matmul_optimized', param_name='N')

# Or generate complete dashboard
viz.create_dashboard(output_dir='visualizations')
```

## 🔍 How It Works

### Two Operating Modes:

**1. PyCUDA Mode (Default, Faster)**
- Compiles kernels directly in Python using PyCUDA
- Executes kernels in-process
- Faster iteration time
- Requires PyCUDA installation

**2. Subprocess Mode (Fallback)**
- Generates C++ wrapper code
- Compiles with nvcc
- Executes as separate process
- More compatible, no PyCUDA needed

### Workflow:

1. **Load Kernels**: Parse .cu files and extract kernel code
2. **Compile**: Compile using PyCUDA or nvcc
3. **Prepare Data**: Allocate and initialize input arrays
4. **Benchmark**: Run multiple iterations with CUDA events
5. **Profile**: Optionally run Nsight Compute for metrics
6. **Aggregate**: Collect results across all parameter combinations
7. **Report**: Generate JSON and text reports
8. **Visualize**: Create comparison charts

## 📝 Kernel Requirements

Your kernels must:
1. Be valid CUDA `__global__` functions
2. Have consistent parameter signatures across implementations
3. Accept integer parameters (N, M, K, ...) for problem dimensions

Example valid signatures:
```cuda
__global__ void my_kernel(float *in, float *out, int N)
__global__ void my_kernel(float *a, float *b, float *c, int N, int M)
__global__ void my_kernel(float *a, float *b, float *c, int N, int M, int K)
```

## 🚧 Limitations

- Currently optimized for matrix operations (matmul, etc.)
- Assumes float32 data types (can be extended)
- Grid size computation assumes 2D/3D matrix layouts
- Profiling requires NVIDIA Nsight Compute

## 🤝 Contributing

This is a flexible framework that can be extended:
- Add support for other data types (int, double, etc.)
- Implement automatic kernel correctness verification
- Add more visualization types
- Support for tensor core operations
- Integration with other profiling tools

## 📚 Examples

See the included example files:
- `example_usage.py` - Basic usage matching your requested API
- `advanced_examples.py` - Advanced features and use cases
- `matmul_kernels.cu` - Example kernel implementations

## 📄 License

MIT License - Free to use and modify

## 🙏 Acknowledgments

- Built on PyCUDA for Python-CUDA integration
- Uses NVIDIA Nsight Compute for profiling
- Visualization powered by matplotlib and seaborn

---

**Made for efficient CUDA kernel development and optimization** 🚀
