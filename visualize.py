#!/usr/bin/env python3
"""
Visualization module for KernelBenchmarker results.

Automatically generates comparison charts from benchmark results.
"""

import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from typing import List, Dict, Any
import seaborn as sns


class BenchmarkVisualizer:
    """Visualize benchmark results with various chart types."""
    
    def __init__(self, results_file: str = "benchmark_results.json"):
        """Load results from JSON file."""
        with open(results_file, 'r') as f:
            data = json.load(f)
            self.results = data['benchmarks']
        
        # Set style
        sns.set_style("whitegrid")
        plt.rcParams['figure.figsize'] = (12, 6)
    
    def plot_execution_time_heatmap(self, param1: str = 'N', param2: str = 'M', 
                                    kernel_name: str = None, output_file: str = None):
        """
        Create heatmap of execution times across two parameters.
        
        Args:
            param1: First parameter name (e.g., 'N')
            param2: Second parameter name (e.g., 'M')
            kernel_name: Specific kernel to plot (None = first kernel)
            output_file: Where to save the plot
        """
        # Filter results
        if kernel_name:
            data = [r for r in self.results if r['kernel'] == kernel_name]
        else:
            kernel_name = self.results[0]['kernel']
            data = [r for r in self.results if r['kernel'] == kernel_name]
        
        # Extract parameter values
        param1_values = sorted(list(set(r['parameters'][param1] for r in data)))
        param2_values = sorted(list(set(r['parameters'][param2] for r in data)))
        
        # Create matrix
        matrix = np.zeros((len(param1_values), len(param2_values)))
        
        for r in data:
            i = param1_values.index(r['parameters'][param1])
            j = param2_values.index(r['parameters'][param2])
            matrix[i, j] = r['execution_time_ms']
        
        # Plot
        fig, ax = plt.subplots(figsize=(10, 8))
        
        im = ax.imshow(matrix, cmap='YlOrRd', aspect='auto')
        
        # Set ticks
        ax.set_xticks(np.arange(len(param2_values)))
        ax.set_yticks(np.arange(len(param1_values)))
        ax.set_xticklabels(param2_values)
        ax.set_yticklabels(param1_values)
        
        # Labels
        ax.set_xlabel(param2, fontsize=12, fontweight='bold')
        ax.set_ylabel(param1, fontsize=12, fontweight='bold')
        ax.set_title(f'Execution Time Heatmap - {kernel_name}', 
                    fontsize=14, fontweight='bold', pad=20)
        
        # Colorbar
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Execution Time (ms)', rotation=270, labelpad=20)
        
        # Add text annotations
        for i in range(len(param1_values)):
            for j in range(len(param2_values)):
                text = ax.text(j, i, f'{matrix[i, j]:.2f}',
                             ha="center", va="center", color="black", fontsize=8)
        
        plt.tight_layout()
        
        if output_file:
            plt.savefig(output_file, dpi=300, bbox_inches='tight')
            print(f"✓ Saved heatmap to {output_file}")
        else:
            plt.show()
        
        plt.close()
    
    def plot_kernel_comparison(self, param_name: str = 'N', fixed_params: Dict = None,
                              output_file: str = None):
        """
        Compare kernels across one varying parameter.
        
        Args:
            param_name: Parameter to vary on x-axis
            fixed_params: Dictionary of fixed parameters (e.g., {'M': 1024})
            output_file: Where to save the plot
        """
        # Filter by fixed parameters
        data = self.results
        if fixed_params:
            for key, value in fixed_params.items():
                data = [r for r in data if r['parameters'].get(key) == value]
        
        # Group by kernel
        kernels = {}
        for r in data:
            kernel = r['kernel']
            if kernel not in kernels:
                kernels[kernel] = []
            kernels[kernel].append(r)
        
        # Plot
        fig, ax = plt.subplots(figsize=(12, 6))
        
        colors = sns.color_palette("husl", len(kernels))
        
        for idx, (kernel_name, kernel_data) in enumerate(kernels.items()):
            # Sort by parameter
            kernel_data = sorted(kernel_data, key=lambda x: x['parameters'][param_name])
            
            x = [r['parameters'][param_name] for r in kernel_data]
            y = [r['execution_time_ms'] for r in kernel_data]
            
            ax.plot(x, y, marker='o', linewidth=2, markersize=8,
                   label=kernel_name, color=colors[idx])
        
        ax.set_xlabel(param_name, fontsize=12, fontweight='bold')
        ax.set_ylabel('Execution Time (ms)', fontsize=12, fontweight='bold')
        
        title = f'Kernel Performance Comparison'
        if fixed_params:
            param_str = ', '.join(f'{k}={v}' for k, v in fixed_params.items())
            title += f' ({param_str})'
        ax.set_title(title, fontsize=14, fontweight='bold')
        
        ax.legend(fontsize=10, framealpha=0.9)
        ax.grid(True, alpha=0.3)
        
        # Use log scale if values span multiple orders of magnitude
        if max(y) / min(y) > 10:
            ax.set_yscale('log')
        
        plt.tight_layout()
        
        if output_file:
            plt.savefig(output_file, dpi=300, bbox_inches='tight')
            print(f"✓ Saved comparison to {output_file}")
        else:
            plt.show()
        
        plt.close()
    
    def plot_speedup_comparison(self, baseline_kernel: str, param_name: str = 'N',
                               fixed_params: Dict = None, output_file: str = None):
        """
        Plot speedup relative to baseline kernel.
        
        Args:
            baseline_kernel: Name of baseline kernel
            param_name: Parameter to vary on x-axis
            fixed_params: Dictionary of fixed parameters
            output_file: Where to save the plot
        """
        # Filter by fixed parameters
        data = self.results
        if fixed_params:
            for key, value in fixed_params.items():
                data = [r for r in data if r['parameters'].get(key) == value]
        
        # Group by kernel and parameter value
        by_param = {}
        for r in data:
            param_val = r['parameters'][param_name]
            if param_val not in by_param:
                by_param[param_val] = {}
            by_param[param_val][r['kernel']] = r['execution_time_ms']
        
        # Calculate speedups
        param_values = sorted(by_param.keys())
        kernels = set()
        for param_data in by_param.values():
            kernels.update(param_data.keys())
        kernels = sorted(kernels)
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        colors = sns.color_palette("husl", len(kernels))
        
        for idx, kernel_name in enumerate(kernels):
            if kernel_name == baseline_kernel:
                continue
            
            speedups = []
            x_vals = []
            
            for param_val in param_values:
                if (param_val in by_param and 
                    baseline_kernel in by_param[param_val] and
                    kernel_name in by_param[param_val]):
                    
                    baseline_time = by_param[param_val][baseline_kernel]
                    kernel_time = by_param[param_val][kernel_name]
                    speedup = baseline_time / kernel_time
                    
                    x_vals.append(param_val)
                    speedups.append(speedup)
            
            if speedups:
                ax.plot(x_vals, speedups, marker='o', linewidth=2, markersize=8,
                       label=kernel_name, color=colors[idx])
        
        # Baseline line
        ax.axhline(y=1.0, color='red', linestyle='--', linewidth=2, 
                  label=f'{baseline_kernel} (baseline)', alpha=0.7)
        
        ax.set_xlabel(param_name, fontsize=12, fontweight='bold')
        ax.set_ylabel(f'Speedup vs {baseline_kernel}', fontsize=12, fontweight='bold')
        
        title = f'Speedup Comparison (Baseline: {baseline_kernel})'
        if fixed_params:
            param_str = ', '.join(f'{k}={v}' for k, v in fixed_params.items())
            title += f'\n({param_str})'
        ax.set_title(title, fontsize=14, fontweight='bold')
        
        ax.legend(fontsize=10, framealpha=0.9)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if output_file:
            plt.savefig(output_file, dpi=300, bbox_inches='tight')
            print(f"✓ Saved speedup chart to {output_file}")
        else:
            plt.show()
        
        plt.close()
    
    def plot_performance_metrics(self, kernel_name: str, param_name: str = 'N',
                                fixed_params: Dict = None, output_file: str = None):
        """
        Plot multiple performance metrics for a kernel.
        
        Args:
            kernel_name: Kernel to analyze
            param_name: Parameter to vary on x-axis
            fixed_params: Dictionary of fixed parameters
            output_file: Where to save the plot
        """
        # Filter data
        data = [r for r in self.results if r['kernel'] == kernel_name]
        if fixed_params:
            for key, value in fixed_params.items():
                data = [r for r in data if r['parameters'].get(key) == value]
        
        # Sort by parameter
        data = sorted(data, key=lambda x: x['parameters'][param_name])
        
        x = [r['parameters'][param_name] for r in data]
        
        # Create subplots
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle(f'Performance Metrics - {kernel_name}', 
                    fontsize=16, fontweight='bold')
        
        # Execution Time
        y = [r['execution_time_ms'] for r in data]
        axes[0, 0].plot(x, y, marker='o', linewidth=2, markersize=8, color='steelblue')
        axes[0, 0].set_ylabel('Time (ms)', fontweight='bold')
        axes[0, 0].set_title('Execution Time')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Memory Throughput
        y = [r.get('memory_throughput_pct', 0) for r in data]
        axes[0, 1].plot(x, y, marker='o', linewidth=2, markersize=8, color='coral')
        axes[0, 1].set_ylabel('Percentage (%)', fontweight='bold')
        axes[0, 1].set_title('Memory Throughput')
        axes[0, 1].set_ylim([0, 100])
        axes[0, 1].grid(True, alpha=0.3)
        
        # Occupancy
        y = [r.get('occupancy_pct', 0) for r in data]
        axes[1, 0].plot(x, y, marker='o', linewidth=2, markersize=8, color='mediumseagreen')
        axes[1, 0].set_xlabel(param_name, fontweight='bold')
        axes[1, 0].set_ylabel('Percentage (%)', fontweight='bold')
        axes[1, 0].set_title('Occupancy')
        axes[1, 0].set_ylim([0, 100])
        axes[1, 0].grid(True, alpha=0.3)
        
        # SM Efficiency
        y = [r.get('sm_efficiency_pct', 0) for r in data]
        axes[1, 1].plot(x, y, marker='o', linewidth=2, markersize=8, color='mediumpurple')
        axes[1, 1].set_xlabel(param_name, fontweight='bold')
        axes[1, 1].set_ylabel('Percentage (%)', fontweight='bold')
        axes[1, 1].set_title('SM Efficiency')
        axes[1, 1].set_ylim([0, 100])
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if output_file:
            plt.savefig(output_file, dpi=300, bbox_inches='tight')
            print(f"✓ Saved metrics to {output_file}")
        else:
            plt.show()
        
        plt.close()
    
    def create_dashboard(self, output_dir: str = "visualizations"):
        """Create a complete dashboard with all visualizations."""
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        print("\nGenerating visualization dashboard...")
        print("="*70)
        
        # Get unique kernels and parameters
        kernels = list(set(r['kernel'] for r in self.results))
        param_keys = list(self.results[0]['parameters'].keys()) if self.results else []
        
        # 1. Heatmap for first kernel
        if len(param_keys) >= 2 and kernels:
            self.plot_execution_time_heatmap(
                param1=param_keys[0],
                param2=param_keys[1],
                kernel_name=kernels[0],
                output_file=str(output_path / "heatmap.png")
            )
        
        # 2. Kernel comparison
        if param_keys:
            self.plot_kernel_comparison(
                param_name=param_keys[0],
                output_file=str(output_path / "kernel_comparison.png")
            )
        
        # 3. Speedup (if multiple kernels)
        if len(kernels) > 1 and param_keys:
            self.plot_speedup_comparison(
                baseline_kernel=kernels[0],
                param_name=param_keys[0],
                output_file=str(output_path / "speedup.png")
            )
        
        # 4. Performance metrics for each kernel
        for kernel in kernels:
            if param_keys:
                safe_name = kernel.replace('/', '_').replace(' ', '_')
                self.plot_performance_metrics(
                    kernel_name=kernel,
                    param_name=param_keys[0],
                    output_file=str(output_path / f"metrics_{safe_name}.png")
                )
        
        print("="*70)
        print(f"\n✓ Dashboard complete! Charts saved to {output_dir}/")


# Convenience function
def visualize(results_file: str = "benchmark_results.json", output_dir: str = "visualizations"):
    """Create visualizations from benchmark results."""
    viz = BenchmarkVisualizer(results_file)
    viz.create_dashboard(output_dir)


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        visualize(sys.argv[1])
    else:
        visualize()
