#!/usr/bin/env python3
"""
Example usage of KernelBenchmarker - Matrix Multiplication

This demonstrates the exact API you requested!
"""

import KernelBenchmarker as kb

# Load kernels from .cu file
matmul_naive = kb.load_kernel("matmul_kernels.cu", "matmul_naive")
matmul_optimized = kb.load_kernel("matmul_kernels.cu", "matmul_optimized")
matmul_coalesced = kb.load_kernel("matmul_kernels.cu", "matmul_coalesced")

# Define parameter ranges
N = [2**i for i in range(7, 12)]  # [128, 256, 512, 1024, 2048]
M = [2**i for i in range(7, 12)]  # Same as N for square matrices
K = [2**i for i in range(7, 12)]  # Same as N

# Or use different ranges for rectangular matrices
# M = [i for i in range(100, 1001, 100)]

# Benchmark all kernels for square matrices (N=M=K)
# Note: For NCU profiling to work, you need sudo with PATH preserved:
#   sudo env PATH=$PATH python example_usage.py
kernels = [matmul_naive, matmul_optimized, matmul_coalesced]
results = kb.benchmark(kernels, N, M, K, zip_params=True, profile=True)

# Results are automatically saved to:
# - benchmark_results.json (machine-readable)
# - benchmark_report.txt (human-readable)

print("\nBenchmarking complete!")
print("Check benchmark_report.txt for detailed results")
