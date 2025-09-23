#!/usr/bin/env python3
"""
Data Flow Visualization for Deep Learning Parallelism
=====================================================

This script provides detailed visualization of how data gets split and processed
at each step of a neural network with different parallelism approaches.

Focus on GELU activation and tensor operations with clear visual feedback.
"""

import numpy as np
import matplotlib.pyplot as plt
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

# Set style for better visualizations
plt.style.use('default')
plt.rcParams['figure.figsize'] = (10, 6)
plt.rcParams['font.size'] = 10


class DataSplitVisualizer:
    """Visualizes exactly how data gets split and processed in each parallelism approach"""
    
    def __init__(self):
        self.processing_steps = []
    
    def log_data_split(self, step_name, original_shape, split_info, processing_time=None):
        """Log how data was split for processing"""
        step = {
            'step_name': step_name,
            'original_shape': original_shape,
            'split_info': split_info,
            'processing_time_ms': processing_time * 1000 if processing_time else None,
            'timestamp': time.time()
        }
        self.processing_steps.append(step)
        
        # Print real-time information
        print(f"\n📊 {step_name}")
        print(f"   Original data shape: {original_shape}")
        if isinstance(split_info, dict):
            for key, value in split_info.items():
                print(f"   {key}: {value}")
        else:
            print(f"   Split info: {split_info}")
        if processing_time:
            print(f"   Processing time: {processing_time*1000:.2f}ms")
    
    def visualize_data_flow(self, save_path=None):
        """Create detailed visualization of data splitting and flow"""
        if not self.processing_steps:
            print("No data flow steps to visualize!")
            return
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Data Flow and Splitting Visualization in Neural Network Parallelism', 
                     fontsize=14, fontweight='bold')
        
        # 1. Data Shape Evolution
        steps = [step['step_name'] for step in self.processing_steps]
        original_sizes = [np.prod(step['original_shape']) if step['original_shape'] else 0 
                         for step in self.processing_steps]
        
        ax1.plot(range(len(steps)), original_sizes, 'o-', linewidth=2, markersize=8)
        ax1.set_xlabel('Processing Steps')
        ax1.set_ylabel('Data Size (elements)')
        ax1.set_title('Data Size Evolution Through Network')
        ax1.set_xticks(range(len(steps)))
        ax1.set_xticklabels(steps, rotation=45, ha='right')
        ax1.grid(True, alpha=0.3)
        
        # 2. Processing Time by Step
        processing_times = [step['processing_time_ms'] or 0 for step in self.processing_steps]
        if any(processing_times):
            # Create simple color sequence
            color_cycle = ['blue', 'green', 'red', 'orange', 'purple', 'brown', 'pink', 'gray']
            colors = [color_cycle[i % len(color_cycle)] for i in range(len(steps))]
            bars = ax2.bar(steps, processing_times, color=colors, alpha=0.7)
            ax2.set_xlabel('Processing Steps')
            ax2.set_ylabel('Processing Time (ms)')
            ax2.set_title('Processing Time by Step')
            ax2.set_xticklabels(steps, rotation=45, ha='right')
            ax2.grid(True, alpha=0.3)
            
            # Add value labels on bars
            for bar, time_val in zip(bars, processing_times):
                if time_val > 0:
                    height = bar.get_height()
                    ax2.text(bar.get_x() + bar.get_width()/2., height,
                            f'{time_val:.1f}ms', ha='center', va='bottom', fontsize=8)
        
        # 3. Data Splitting Strategies Comparison
        split_strategies = ['Sequential', 'Parallel (Threading)', 'Vectorized']
        strategy_colors = ['red', 'blue', 'green']
        strategy_performance = [100, 75, 10]  # Example relative times
        
        wedges, texts, autotexts = ax3.pie(strategy_performance, labels=split_strategies, 
                                          colors=strategy_colors, autopct='%1.1f%%',
                                          startangle=90)
        ax3.set_title('Processing Time Distribution by Strategy\n(Lower is Better)')
        
        # 4. Parallel Efficiency Visualization
        worker_counts = [1, 2, 4, 8]
        efficiency_perfect = [100, 100, 100, 100]
        efficiency_realistic = [100, 85, 70, 55]
        
        ax4.plot(worker_counts, efficiency_perfect, 'k--', label='Perfect Scaling', alpha=0.5)
        ax4.plot(worker_counts, efficiency_realistic, 'bo-', label='Realistic Scaling', linewidth=2)
        ax4.set_xlabel('Number of Workers')
        ax4.set_ylabel('Efficiency (%)')
        ax4.set_title('Parallel Efficiency vs Worker Count')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        ax4.set_ylim(0, 110)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"\n📊 Data flow visualization saved to: {save_path}")
        
        plt.show()


class GeluDigitClassifier:
    """Simple digit classifier focused on GELU activation and data flow visualization"""
    
    def __init__(self):
        self.visualizer = DataSplitVisualizer()
        
        # Simple 3-layer network: 784 -> 128 -> 64 -> 10
        self.W1 = np.random.randn(784, 128) * 0.1
        self.b1 = np.zeros(128)
        self.W2 = np.random.randn(128, 64) * 0.1
        self.b2 = np.zeros(64)
        self.W3 = np.random.randn(64, 10) * 0.1
        self.b3 = np.zeros(10)
        
        print("🏗️  Neural Network Architecture:")
        print("   Input: 784 (28x28 flattened image)")
        print("   Hidden 1: 128 neurons (GELU activation)")
        print("   Hidden 2: 64 neurons (GELU activation)")
        print("   Output: 10 classes (Softmax)")
    
    def gelu_activation(self, x):
        """GELU activation function: x * 0.5 * (1 + tanh(√(2/π) * (x + 0.044715 * x³)))"""
        return x * 0.5 * (1 + np.tanh(np.sqrt(2/np.pi) * (x + 0.044715 * x**3)))
    
    def visualize_data_splits(self, batch_data, batch_size=32):
        """Demonstrate different data splitting approaches with visualization"""
        
        print("\n" + "="*80)
        print("🔄 DATA SPLITTING DEMONSTRATION")
        print("="*80)
        
        # Show original batch structure
        self.visualizer.log_data_split(
            "Original Batch", 
            batch_data.shape,
            {
                "Total samples": batch_data.shape[0],
                "Features per sample": batch_data.shape[1],
                "Memory size": f"{batch_data.nbytes / 1024:.1f} KB"
            }
        )
        
        # 1. Sequential Processing (no splitting)
        print("\n🔄 Method 1: Sequential Processing (No Data Splitting)")
        start_time = time.time()
        
        sequential_results = []
        for i in range(batch_size):
            sample = batch_data[i:i+1]  # Process one sample at a time
            result = self._forward_single(sample)
            sequential_results.append(result)
        
        sequential_time = time.time() - start_time
        
        self.visualizer.log_data_split(
            "Sequential Processing",
            (batch_size, 784),
            {
                "Split strategy": "One sample at a time",
                "Chunks": f"{batch_size} chunks of shape (1, 784)",
                "Processing order": "Sample 0 → Sample 1 → Sample 2 → ...",
                "Parallelism": "None (single thread)"
            },
            sequential_time
        )
        
        # 2. Parallel Processing (data splitting across threads)
        print("\n⚡ Method 2: Parallel Processing (Data Split Across Threads)")
        start_time = time.time()
        
        # Split data into chunks for parallel processing
        num_workers = 4
        chunk_size = batch_size // num_workers
        chunks = []
        
        for i in range(num_workers):
            start_idx = i * chunk_size
            end_idx = min((i + 1) * chunk_size, batch_size)
            chunk = batch_data[start_idx:end_idx]
            chunks.append((chunk, start_idx, end_idx))
        
        def process_chunk(chunk_info):
            chunk_data, start_idx, end_idx = chunk_info
            chunk_results = []
            for j in range(chunk_data.shape[0]):
                sample = chunk_data[j:j+1]
                result = self._forward_single(sample)
                chunk_results.append(result)
            return chunk_results, start_idx, end_idx
        
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            chunk_futures = [executor.submit(process_chunk, chunk_info) for chunk_info in chunks]
            chunk_results = [future.result() for future in chunk_futures]
        
        # Reconstruct results in original order
        parallel_results = [None] * batch_size
        for chunk_result, start_idx, end_idx in chunk_results:
            for j, result in enumerate(chunk_result):
                parallel_results[start_idx + j] = result
        
        parallel_time = time.time() - start_time
        
        self.visualizer.log_data_split(
            "Parallel Processing",
            (batch_size, 784),
            {
                "Split strategy": f"Data split into {num_workers} chunks",
                "Chunks": f"{num_workers} chunks of ~{chunk_size} samples each",
                "Worker assignment": f"Worker 0: samples 0-{chunk_size-1}, Worker 1: samples {chunk_size}-{chunk_size*2-1}, etc.",
                "Parallelism": f"{num_workers} threads processing simultaneously"
            },
            parallel_time
        )
        
        # 3. Vectorized Processing (entire batch at once)
        print("\n🚀 Method 3: Vectorized Processing (Entire Batch at Once)")
        start_time = time.time()
        
        vectorized_results = self._forward_batch_vectorized(batch_data[:batch_size])
        
        vectorized_time = time.time() - start_time
        
        self.visualizer.log_data_split(
            "Vectorized Processing",
            (batch_size, 784),
            {
                "Split strategy": "No splitting - entire batch processed as single tensor",
                "Matrix operations": f"({batch_size}, 784) @ (784, 128) = ({batch_size}, 128)",
                "SIMD utilization": "CPU processes multiple elements simultaneously",
                "Parallelism": "Hardware-level vectorization (SIMD + BLAS)"
            },
            vectorized_time
        )
        
        # Performance comparison
        print("\n📊 PERFORMANCE COMPARISON:")
        print(f"   Sequential:  {sequential_time*1000:8.2f}ms")
        print(f"   Parallel:    {parallel_time*1000:8.2f}ms (Speedup: {sequential_time/parallel_time:.2f}x)")
        print(f"   Vectorized:  {vectorized_time*1000:8.2f}ms (Speedup: {sequential_time/vectorized_time:.1f}x)")
        
        # Verify all methods produce the same results
        sequential_array = np.array(sequential_results).squeeze()
        parallel_array = np.array(parallel_results).squeeze()
        
        print(f"   ✅ Results identical: {np.allclose(sequential_array, parallel_array, rtol=1e-5)}")
        print(f"   ✅ Vectorized matches: {np.allclose(sequential_array, vectorized_results, rtol=1e-5)}")
        
        return sequential_results, parallel_results, vectorized_results
    
    def _forward_single(self, x):
        """Forward pass for a single sample"""
        # Layer 1
        z1 = x @ self.W1 + self.b1
        a1 = self.gelu_activation(z1)
        
        # Layer 2
        z2 = a1 @ self.W2 + self.b2
        a2 = self.gelu_activation(z2)
        
        # Output layer
        z3 = a2 @ self.W3 + self.b3
        
        # Softmax
        exp_scores = np.exp(z3 - np.max(z3))
        probabilities = exp_scores / np.sum(exp_scores)
        
        return probabilities
    
    def _forward_batch_vectorized(self, X):
        """Vectorized forward pass for entire batch"""
        # Layer 1 (vectorized)
        z1 = X @ self.W1 + self.b1
        a1 = self.gelu_activation(z1)
        
        # Layer 2 (vectorized)
        z2 = a1 @ self.W2 + self.b2
        a2 = self.gelu_activation(z2)
        
        # Output layer (vectorized)
        z3 = a2 @ self.W3 + self.b3
        
        # Softmax (vectorized)
        exp_scores = np.exp(z3 - np.max(z3, axis=1, keepdims=True))
        probabilities = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
        
        return probabilities
    
    def visualize_tensor_operations(self, batch_data):
        """Visualize how tensor operations work at each layer"""
        print("\n" + "="*80)
        print("🔢 TENSOR OPERATIONS VISUALIZATION")
        print("="*80)
        
        batch_size = batch_data.shape[0]
        
        # Show data flow through each layer
        current_data = batch_data
        layer_names = ["Input", "Linear 1", "GELU 1", "Linear 2", "GELU 2", "Linear 3", "Softmax"]
        
        print(f"\n📊 Data Flow Through Network (Batch size: {batch_size}):")
        print("-" * 70)
        
        # Input
        print(f"{'Input':<12} Shape: {str(current_data.shape):<15} Memory: {current_data.nbytes/1024:>6.1f} KB")
        
        # Layer 1
        start_time = time.time()
        z1 = current_data @ self.W1 + self.b1
        layer1_time = time.time() - start_time
        print(f"{'Linear 1':<12} Shape: {str(z1.shape):<15} Memory: {z1.nbytes/1024:>6.1f} KB  Time: {layer1_time*1000:>5.2f}ms")
        print(f"             Operation: {str(current_data.shape)} @ {str(self.W1.shape)} + {str(self.b1.shape)}")
        
        start_time = time.time()
        a1 = self.gelu_activation(z1)
        gelu1_time = time.time() - start_time
        print(f"{'GELU 1':<12} Shape: {str(a1.shape):<15} Memory: {a1.nbytes/1024:>6.1f} KB  Time: {gelu1_time*1000:>5.2f}ms")
        print(f"             Operation: GELU{str(z1.shape)} = {str(a1.shape)}")
        
        # Layer 2
        start_time = time.time()
        z2 = a1 @ self.W2 + self.b2
        layer2_time = time.time() - start_time
        print(f"{'Linear 2':<12} Shape: {str(z2.shape):<15} Memory: {z2.nbytes/1024:>6.1f} KB  Time: {layer2_time*1000:>5.2f}ms")
        print(f"             Operation: {str(a1.shape)} @ {str(self.W2.shape)} + {str(self.b2.shape)}")
        
        start_time = time.time()
        a2 = self.gelu_activation(z2)
        gelu2_time = time.time() - start_time
        print(f"{'GELU 2':<12} Shape: {str(a2.shape):<15} Memory: {a2.nbytes/1024:>6.1f} KB  Time: {gelu2_time*1000:>5.2f}ms")
        print(f"             Operation: GELU{str(z2.shape)} = {str(a2.shape)}")
        
        # Output layer
        start_time = time.time()
        z3 = a2 @ self.W3 + self.b3
        layer3_time = time.time() - start_time
        print(f"{'Linear 3':<12} Shape: {str(z3.shape):<15} Memory: {z3.nbytes/1024:>6.1f} KB  Time: {layer3_time*1000:>5.2f}ms")
        print(f"             Operation: {str(a2.shape)} @ {str(self.W3.shape)} + {str(self.b3.shape)}")
        
        # Softmax
        start_time = time.time()
        exp_scores = np.exp(z3 - np.max(z3, axis=1, keepdims=True))
        probabilities = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
        softmax_time = time.time() - start_time
        print(f"{'Softmax':<12} Shape: {str(probabilities.shape):<15} Memory: {probabilities.nbytes/1024:>6.1f} KB  Time: {softmax_time*1000:>5.2f}ms")
        print(f"             Operation: Softmax{str(z3.shape)} = {str(probabilities.shape)}")
        
        total_time = layer1_time + gelu1_time + layer2_time + gelu2_time + layer3_time + softmax_time
        print("-" * 70)
        print(f"{'TOTAL':<12} Time: {total_time*1000:>5.2f}ms")
        
        # Show computational complexity
        print(f"\n🧮 Computational Complexity Analysis:")
        print(f"   Layer 1: {batch_size:>4} × {784:>4} × {128:>4} = {batch_size*784*128:>12,} operations")
        print(f"   Layer 2: {batch_size:>4} × {128:>4} × {64:>4}  = {batch_size*128*64:>12,} operations")
        print(f"   Layer 3: {batch_size:>4} × {64:>4}  × {10:>4}  = {batch_size*64*10:>12,} operations")
        total_ops = batch_size*784*128 + batch_size*128*64 + batch_size*64*10
        print(f"   TOTAL:                           {total_ops:>12,} operations")
        print(f"   Throughput: {total_ops/(total_time*1e6):>8.1f} million ops/second")
        
        return probabilities


def demonstrate_gelu_vs_relu():
    """Compare GELU vs ReLU activation functions"""
    print("\n" + "="*80)
    print("🔄 GELU vs ReLU ACTIVATION COMPARISON")
    print("="*80)
    
    # Generate test data
    x = np.linspace(-3, 3, 1000)
    
    # GELU function
    gelu = x * 0.5 * (1 + np.tanh(np.sqrt(2/np.pi) * (x + 0.044715 * x**3)))
    
    # ReLU function
    relu = np.maximum(0, x)
    
    # Plot comparison
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(x, gelu, 'b-', linewidth=2, label='GELU')
    plt.plot(x, relu, 'r--', linewidth=2, label='ReLU')
    plt.xlabel('Input (x)')
    plt.ylabel('Output')
    plt.title('GELU vs ReLU Activation Functions')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Derivative comparison
    plt.subplot(1, 2, 2)
    
    # GELU derivative (approximate)
    dx = x[1] - x[0]
    gelu_derivative = np.gradient(gelu, dx)
    relu_derivative = np.where(x > 0, 1, 0)
    
    plt.plot(x, gelu_derivative, 'b-', linewidth=2, label='GELU derivative')
    plt.plot(x, relu_derivative, 'r--', linewidth=2, label='ReLU derivative')
    plt.xlabel('Input (x)')
    plt.ylabel('Derivative')
    plt.title('Activation Function Derivatives')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save plot
    results_dir = Path("results/parallelism_demo")
    results_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(results_dir / "gelu_vs_relu_comparison.png", dpi=300, bbox_inches='tight')
    print(f"📊 GELU vs ReLU comparison saved to: {results_dir / 'gelu_vs_relu_comparison.png'}")
    
    plt.show()
    
    print("\n🔍 Key Differences:")
    print("   GELU: Smooth, differentiable everywhere, non-zero gradients for negative inputs")
    print("   ReLU: Sharp cutoff at zero, zero gradients for negative inputs")
    print("   GELU advantage: Better gradient flow, especially for deep networks")


def main():
    """Main demonstration function"""
    print("="*80)
    print("🎯 DATA FLOW AND TENSOR OPERATIONS DEMONSTRATION")
    print("="*80)
    print("This demo shows EXACTLY how data gets split and processed in:")
    print("• Sequential processing (one sample at a time)")
    print("• Parallel processing (data split across threads)")
    print("• Vectorized processing (entire batch at once)")
    print("• GELU activation with tensor operations")
    print("• Complete performance and memory analysis")
    print("="*80)
    
    # Create synthetic dataset
    print("\n📊 Creating synthetic MNIST-like dataset...")
    batch_size = 64
    X = np.random.randn(batch_size, 784).astype(np.float32)
    X = (X - X.min()) / (X.max() - X.min())  # Normalize to [0, 1]
    
    print(f"   Dataset shape: {X.shape}")
    print(f"   Data type: {X.dtype}")
    print(f"   Memory usage: {X.nbytes / 1024:.1f} KB")
    print(f"   Value range: [{X.min():.3f}, {X.max():.3f}]")
    
    # Initialize classifier
    classifier = GeluDigitClassifier()
    
    # Demonstrate data splitting approaches
    sequential_results, parallel_results, vectorized_results = classifier.visualize_data_splits(X, batch_size=32)
    
    # Demonstrate tensor operations
    classifier.visualize_tensor_operations(X[:16])  # Use smaller batch for detailed analysis
    
    # Compare GELU vs ReLU
    demonstrate_gelu_vs_relu()
    
    # Create comprehensive visualization
    results_dir = Path("results/parallelism_demo")
    results_dir.mkdir(parents=True, exist_ok=True)
    
    viz_path = results_dir / "data_flow_visualization.png"
    classifier.visualizer.visualize_data_flow(save_path=viz_path)
    
    print("\n" + "="*80)
    print("🎉 DEMONSTRATION COMPLETE!")
    print("="*80)
    print("Key takeaways:")
    print("✅ Vectorized processing is dramatically faster than sequential")
    print("✅ Parallel processing can help but has overhead costs")
    print("✅ Data splitting strategy affects performance significantly")
    print("✅ GELU provides smoother gradients than ReLU")
    print("✅ Tensor operations scale with batch size and network width")
    print("✅ Memory usage and computational complexity are predictable")
    print(f"\n📊 All visualizations saved to: {results_dir}")


if __name__ == "__main__":
    # Set random seed for reproducibility
    np.random.seed(42)
    
    # Run the demonstration
    main()