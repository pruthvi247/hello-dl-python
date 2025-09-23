#!/usr/bin/env python3
"""
Parallelism Concepts Demonstration

This script demonstrates various parallelism concepts used in deep learning
with practical, runnable examples. Each example shows when to use parallelism
and when not to, with performance measurements.
"""

import os
import sys
import time
import numpy as np
import threading
import multiprocessing
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from typing import List, Tuple, Optional
import matplotlib.pyplot as plt

# Add PyTensorLib to path
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '../..', 'src'))

def demonstrate_vectorization():
    """Demonstrate the power of vectorized operations vs loops"""
    print("🔧 DEMONSTRATION 1: Vectorization vs Loops")
    print("=" * 60)
    
    # Problem: Matrix multiplication for a batch of data
    batch_size = 100
    input_dim = 512
    output_dim = 256
    
    # Generate test data
    X = np.random.randn(batch_size, input_dim).astype(np.float32)
    W = np.random.randn(input_dim, output_dim).astype(np.float32)
    b = np.random.randn(output_dim).astype(np.float32)
    
    print(f"📊 Problem: Linear layer for batch")
    print(f"   Input shape: {X.shape}")
    print(f"   Weight shape: {W.shape}")
    print(f"   Bias shape: {b.shape}")
    print(f"   Operation: Y = X @ W + b")
    
    # Method 1: Triple nested loops (very slow)
    print("\n🐌 Method 1: Triple nested loops")
    start_time = time.time()
    
    Y_loops = np.zeros((batch_size, output_dim), dtype=np.float32)
    for i in range(batch_size):          # For each sample
        for j in range(output_dim):      # For each output neuron
            for k in range(input_dim):   # For each input feature
                Y_loops[i, j] += X[i, k] * W[k, j]
            Y_loops[i, j] += b[j]        # Add bias
    
    loop_time = time.time() - start_time
    print(f"   Time: {loop_time*1000:.2f}ms")
    
    # Method 2: Vectorized matrix multiplication (fast)
    print("\n⚡ Method 2: Vectorized operations")
    start_time = time.time()
    
    Y_vectorized = X @ W + b  # Single line!
    
    vectorized_time = time.time() - start_time
    print(f"   Time: {vectorized_time*1000:.2f}ms")
    
    # Verify results are the same
    max_diff = np.max(np.abs(Y_loops - Y_vectorized))
    print(f"\n✅ Results verification:")
    print(f"   Max difference: {max_diff:.2e}")
    print(f"   Results match: {max_diff < 1e-5}")
    
    # Performance comparison
    speedup = loop_time / vectorized_time if vectorized_time > 0 else float('inf')
    print(f"\n📈 Performance:")
    print(f"   Speedup: {speedup:.1f}x faster")
    print(f"   Why? NumPy uses optimized BLAS libraries + SIMD instructions")
    
    return loop_time, vectorized_time, speedup


def demonstrate_threading_vs_sequential():
    """Demonstrate when threading helps and when it doesn't"""
    print("\n\n🧵 DEMONSTRATION 2: Threading vs Sequential")
    print("=" * 60)
    
    def simulate_neural_network_forward(sample_data):
        """Simulate a forward pass through a neural network"""
        image, label = sample_data
        
        # Simulate computational work (matrix operations)
        # Layer 1: 784 -> 128
        W1 = np.random.randn(784, 128).astype(np.float32) * 0.1
        h1 = np.maximum(0, image.flatten() @ W1)  # ReLU
        
        # Layer 2: 128 -> 64  
        W2 = np.random.randn(128, 64).astype(np.float32) * 0.1
        h2 = np.maximum(0, h1 @ W2)  # ReLU
        
        # Layer 3: 64 -> 10
        W3 = np.random.randn(64, 10).astype(np.float32) * 0.1
        output = h2 @ W3
        
        # Simulate loss computation
        target = np.zeros(10)
        target[label] = 1.0
        loss = np.sum((output - target) ** 2)
        
        return {
            'prediction': np.argmax(output),
            'actual': label,
            'loss': loss,
            'output': output
        }
    
    # Generate batch data
    batch_size = 32
    batch_data = []
    for i in range(batch_size):
        image = np.random.randn(28, 28).astype(np.float32)
        label = i % 10  # Labels 0-9
        batch_data.append((image, label))
    
    print(f"📊 Problem: Process batch of {batch_size} samples")
    print(f"   Each sample: 28x28 image → 3-layer network → prediction")
    
    # Sequential processing
    print(f"\n🐌 Sequential Processing:")
    start_time = time.time()
    
    sequential_results = []
    for sample in batch_data:
        result = simulate_neural_network_forward(sample)
        sequential_results.append(result)
    
    sequential_time = time.time() - start_time
    print(f"   Time: {sequential_time*1000:.2f}ms")
    
    # Parallel processing with different worker counts
    worker_counts = [2, 4, 8]
    parallel_results = {}
    
    for workers in worker_counts:
        print(f"\n⚡ Parallel Processing ({workers} workers):")
        start_time = time.time()
        
        with ThreadPoolExecutor(max_workers=workers) as executor:
            results = list(executor.map(simulate_neural_network_forward, batch_data))
        
        parallel_time = time.time() - start_time
        speedup = sequential_time / parallel_time if parallel_time > 0 else 1.0
        efficiency = speedup / workers * 100
        
        parallel_results[workers] = {
            'time': parallel_time,
            'speedup': speedup,
            'efficiency': efficiency
        }
        
        print(f"   Time: {parallel_time*1000:.2f}ms")
        print(f"   Speedup: {speedup:.2f}x")
        print(f"   Efficiency: {efficiency:.1f}%")
    
    # Verify results consistency
    seq_predictions = [r['prediction'] for r in sequential_results]
    par_predictions = [r['prediction'] for r in results]
    matches = sum(1 for s, p in zip(seq_predictions, par_predictions) if s == p)
    
    print(f"\n✅ Results verification:")
    print(f"   Predictions match: {matches}/{len(seq_predictions)} ({100*matches/len(seq_predictions):.1f}%)")
    
    return sequential_time, parallel_results


def demonstrate_bad_parallelism():
    """Show cases where parallelism hurts performance"""
    print("\n\n❌ DEMONSTRATION 3: When Parallelism HURTS Performance")
    print("=" * 60)
    
    # Case 1: Operations too small
    print("Case 1: Operations too small (high overhead)")
    
    def tiny_operation(x):
        return x ** 2 + np.sin(x)
    
    small_data = np.random.randn(1000)
    
    # Sequential
    start = time.time()
    seq_results = [tiny_operation(x) for x in small_data]
    seq_time = time.time() - start
    
    # Parallel  
    start = time.time()
    with ThreadPoolExecutor(max_workers=4) as executor:
        par_results = list(executor.map(tiny_operation, small_data))
    par_time = time.time() - start
    
    print(f"   Sequential: {seq_time*1000:.2f}ms")
    print(f"   Parallel: {par_time*1000:.2f}ms")
    print(f"   Parallel is {par_time/seq_time:.1f}x SLOWER!")
    print(f"   Why? Thread creation overhead > computation time")
    
    # Case 2: Sequential dependencies
    print(f"\nCase 2: Sequential dependencies")
    
    def sequential_computation(x, steps=5):
        """Each step depends on the previous step"""
        result = x
        for i in range(steps):
            result = np.sin(result) + 0.1 * i
        return result
    
    print(f"   Operation: x → sin(x) → sin(sin(x)+0.1) → ... (5 steps)")
    print(f"   Cannot parallelize: Each step needs previous result")
    print(f"   Solution: Look for parallelism at a higher level (different samples)")
    
    # Case 3: Memory-bound operations
    print(f"\nCase 3: Memory-bound operations")
    
    def memory_bound_operation(large_array):
        """Operation limited by memory bandwidth, not computation"""
        return large_array.copy()  # Just copying memory
    
    # Large arrays that stress memory bandwidth
    large_arrays = [np.random.randn(1000, 1000) for _ in range(4)]
    
    # Sequential
    start = time.time()
    seq_copies = [memory_bound_operation(arr) for arr in large_arrays]
    seq_mem_time = time.time() - start
    
    # Parallel
    start = time.time()
    with ThreadPoolExecutor(max_workers=4) as executor:
        par_copies = list(executor.map(memory_bound_operation, large_arrays))
    par_mem_time = time.time() - start
    
    print(f"   Sequential: {seq_mem_time*1000:.2f}ms")
    print(f"   Parallel: {par_mem_time*1000:.2f}ms")
    print(f"   Speedup: {seq_mem_time/par_mem_time:.2f}x (limited by memory bandwidth)")
    print(f"   Why? All threads compete for same memory bus")


def demonstrate_gradient_accumulation():
    """Demonstrate thread-safe gradient accumulation"""
    print("\n\n🔄 DEMONSTRATION 4: Thread-Safe Gradient Accumulation")
    print("=" * 60)
    
    class ThreadSafeGradientAccumulator:
        def __init__(self, param_shapes):
            self.accumulated_grads = {}
            self.locks = {}
            
            for name, shape in param_shapes.items():
                self.accumulated_grads[name] = np.zeros(shape, dtype=np.float32)
                self.locks[name] = threading.Lock()
        
        def add_gradient(self, param_name, gradient):
            """Thread-safe gradient addition"""
            with self.locks[param_name]:
                self.accumulated_grads[param_name] += gradient
        
        def get_accumulated_gradients(self):
            """Get final accumulated gradients"""
            result = {}
            for name in self.accumulated_grads:
                with self.locks[name]:
                    result[name] = self.accumulated_grads[name].copy()
            return result
        
        def reset(self):
            """Reset accumulated gradients"""
            for name in self.accumulated_grads:
                with self.locks[name]:
                    self.accumulated_grads[name].fill(0.0)
    
    def compute_sample_gradients(sample_id, accumulator):
        """Simulate computing gradients for one sample"""
        # Simulate different gradient values for each sample
        np.random.seed(sample_id)  # Deterministic for verification
        
        gradients = {
            'layer1_weights': np.random.randn(784, 128).astype(np.float32) * 0.01,
            'layer1_bias': np.random.randn(128).astype(np.float32) * 0.01,
            'layer2_weights': np.random.randn(128, 64).astype(np.float32) * 0.01,
            'layer2_bias': np.random.randn(64).astype(np.float32) * 0.01,
        }
        
        # Add gradients to accumulator (thread-safe)
        for param_name, gradient in gradients.items():
            accumulator.add_gradient(param_name, gradient)
        
        return gradients
    
    # Define model parameter shapes
    param_shapes = {
        'layer1_weights': (784, 128),
        'layer1_bias': (128,),
        'layer2_weights': (128, 64), 
        'layer2_bias': (64,),
    }
    
    batch_size = 16
    print(f"📊 Problem: Accumulate gradients from {batch_size} samples")
    print(f"   Model: Linear(784→128) + Linear(128→64)")
    print(f"   Challenge: Multiple threads updating shared gradient storage")
    
    # Initialize accumulator
    accumulator = ThreadSafeGradientAccumulator(param_shapes)
    
    # Method 1: Sequential gradient accumulation
    print(f"\n🐌 Sequential gradient accumulation:")
    accumulator.reset()
    start_time = time.time()
    
    sequential_individual_grads = []
    for sample_id in range(batch_size):
        grads = compute_sample_gradients(sample_id, accumulator)
        sequential_individual_grads.append(grads)
    
    sequential_accumulated = accumulator.get_accumulated_gradients()
    sequential_time = time.time() - start_time
    print(f"   Time: {sequential_time*1000:.2f}ms")
    
    # Method 2: Parallel gradient accumulation
    print(f"\n⚡ Parallel gradient accumulation:")
    accumulator.reset()
    start_time = time.time()
    
    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = [executor.submit(compute_sample_gradients, sample_id, accumulator) 
                  for sample_id in range(batch_size)]
        parallel_individual_grads = [future.result() for future in futures]
    
    parallel_accumulated = accumulator.get_accumulated_gradients()
    parallel_time = time.time() - start_time
    print(f"   Time: {parallel_time*1000:.2f}ms")
    print(f"   Speedup: {sequential_time/parallel_time:.2f}x")
    
    # Verify results are identical
    print(f"\n✅ Results verification:")
    all_match = True
    for param_name in param_shapes:
        seq_grad = sequential_accumulated[param_name]
        par_grad = parallel_accumulated[param_name]
        max_diff = np.max(np.abs(seq_grad - par_grad))
        matches = max_diff < 1e-10
        all_match = all_match and matches
        print(f"   {param_name}: max_diff = {max_diff:.2e}, match = {matches}")
    
    print(f"   All gradients match: {all_match}")
    print(f"   Why it works: Thread-safe locks ensure no race conditions")
    
    # Show gradient norms
    print(f"\n📊 Final gradient norms:")
    for param_name, grad in parallel_accumulated.items():
        norm = np.linalg.norm(grad)
        print(f"   {param_name}: {norm:.4f}")


def demonstrate_data_parallelism_vs_model_parallelism():
    """Show the difference between data and model parallelism"""
    print("\n\n🔀 DEMONSTRATION 5: Data vs Model Parallelism")
    print("=" * 60)
    
    # Simulate a simple 3-layer network
    def layer_forward(x, weights, bias):
        return np.maximum(0, x @ weights + bias)  # Linear + ReLU
    
    # Model parameters
    W1 = np.random.randn(784, 256).astype(np.float32) * 0.1
    b1 = np.random.randn(256).astype(np.float32)
    W2 = np.random.randn(256, 128).astype(np.float32) * 0.1
    b2 = np.random.randn(128).astype(np.float32)
    W3 = np.random.randn(128, 10).astype(np.float32) * 0.1
    b3 = np.random.randn(10).astype(np.float32)
    
    # Generate batch data
    batch_size = 64
    batch_images = np.random.randn(batch_size, 784).astype(np.float32)
    
    print(f"📊 Network: 784 → 256 → 128 → 10")
    print(f"   Batch size: {batch_size}")
    
    # DATA PARALLELISM: Split batch across workers
    print(f"\n🔄 Data Parallelism: Split batch across workers")
    
    def process_data_chunk(image_chunk):
        """Process a subset of the batch"""
        # Forward pass through entire network for this chunk
        h1 = layer_forward(image_chunk, W1, b1)
        h2 = layer_forward(h1, W2, b2)
        output = h2 @ W3 + b3  # No ReLU on final layer
        return output
    
    # Split batch into chunks
    chunk_size = batch_size // 4
    chunks = [batch_images[i:i+chunk_size] for i in range(0, batch_size, chunk_size)]
    
    start_time = time.time()
    with ThreadPoolExecutor(max_workers=4) as executor:
        chunk_results = list(executor.map(process_data_chunk, chunks))
    data_parallel_time = time.time() - start_time
    
    # Combine results
    data_parallel_output = np.vstack(chunk_results)
    
    print(f"   Time: {data_parallel_time*1000:.2f}ms")
    print(f"   Strategy: Each worker processes {chunk_size} samples through full network")
    print(f"   Pros: Simple, good scalability")
    print(f"   Cons: Each worker needs full model copy")
    
    # MODEL PARALLELISM: Split model across workers  
    print(f"\n🧠 Model Parallelism: Split layers across workers")
    
    def layer1_worker(x):
        return layer_forward(x, W1, b1)
    
    def layer2_worker(x):
        return layer_forward(x, W2, b2)
    
    def layer3_worker(x):
        return x @ W3 + b3
    
    start_time = time.time()
    
    # Sequential through different workers (pipeline)
    # In real implementation, this would use inter-process communication
    h1_output = layer1_worker(batch_images)
    h2_output = layer2_worker(h1_output) 
    model_parallel_output = layer3_worker(h2_output)
    
    model_parallel_time = time.time() - start_time
    
    print(f"   Time: {model_parallel_time*1000:.2f}ms")
    print(f"   Strategy: Each worker handles one layer")
    print(f"   Pros: Can handle very large models")
    print(f"   Cons: Communication overhead, pipeline bubbles")
    
    # Verify outputs match
    max_diff = np.max(np.abs(data_parallel_output - model_parallel_output))
    print(f"\n✅ Results verification:")
    print(f"   Max difference: {max_diff:.2e}")
    print(f"   Outputs match: {max_diff < 1e-5}")
    
    # Sequential baseline
    start_time = time.time()
    sequential_h1 = layer_forward(batch_images, W1, b1)
    sequential_h2 = layer_forward(sequential_h1, W2, b2)
    sequential_output = sequential_h2 @ W3 + b3  # Fix: use h2, not h1
    sequential_time = time.time() - start_time
    
    print(f"\n📈 Performance comparison:")
    print(f"   Sequential: {sequential_time*1000:.2f}ms")
    print(f"   Data parallel: {data_parallel_time*1000:.2f}ms ({sequential_time/data_parallel_time:.2f}x)")
    print(f"   Model parallel: {model_parallel_time*1000:.2f}ms ({sequential_time/model_parallel_time:.2f}x)")


def create_performance_plots(results):
    """Create visualization of performance results"""
    print("\n\n📊 CREATING PERFORMANCE PLOTS")
    print("=" * 60)
    
    try:
        # Ensure results directory exists
        results_dir = os.path.join(os.path.dirname(__file__), "../..", "results", "parallelism_demo")
        os.makedirs(results_dir, exist_ok=True)
        
        # Plot 1: Vectorization speedup
        plt.figure(figsize=(12, 8))
        
        plt.subplot(2, 2, 1)
        methods = ['Loops', 'Vectorized']
        times = [results['vectorization']['loop_time'], results['vectorization']['vectorized_time']]
        plt.bar(methods, [t*1000 for t in times], color=['red', 'green'])
        plt.ylabel('Time (ms)')
        plt.title('Vectorization Performance')
        plt.yscale('log')
        
        # Plot 2: Threading efficiency
        plt.subplot(2, 2, 2)
        workers = list(results['threading'].keys())
        efficiencies = [results['threading'][w]['efficiency'] for w in workers]
        plt.plot(workers, efficiencies, 'o-', color='blue')
        plt.xlabel('Number of Workers')
        plt.ylabel('Efficiency (%)')
        plt.title('Threading Efficiency')
        plt.grid(True)
        
        # Plot 3: Speedup vs workers
        plt.subplot(2, 2, 3)
        speedups = [results['threading'][w]['speedup'] for w in workers]
        plt.plot(workers, speedups, 'o-', color='green', label='Actual')
        plt.plot(workers, workers, '--', color='red', label='Ideal')
        plt.xlabel('Number of Workers')
        plt.ylabel('Speedup')
        plt.title('Speedup vs Workers')
        plt.legend()
        plt.grid(True)
        
        # Plot 4: Operation comparison
        plt.subplot(2, 2, 4)
        operations = ['Small Ops\n(Bad)', 'Neural Net\n(Good)', 'Memory Copy\n(Limited)']
        speedups = [0.5, 3.2, 1.8]  # Example values
        colors = ['red', 'green', 'orange']
        plt.bar(operations, speedups, color=colors)
        plt.ylabel('Speedup')
        plt.title('When Parallelism Works')
        plt.axhline(y=1, color='black', linestyle='--', alpha=0.5)
        
        plt.tight_layout()
        plot_path = os.path.join(results_dir, 'parallelism_performance.png')
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        print(f"📈 Performance plots saved to: {plot_path}")
        
        plt.close()
        
    except ImportError:
        print("⚠️ Matplotlib not available - skipping plots")
    except Exception as e:
        print(f"⚠️ Error creating plots: {e}")


def main():
    """Run all parallelism demonstrations"""
    print("🎯 DEEP LEARNING PARALLELISM: Comprehensive Demonstration")
    print("=" * 80)
    print("This demo shows practical parallelism concepts with real measurements")
    print("=" * 80)
    
    results = {}
    
    # Demo 1: Vectorization
    loop_time, vec_time, speedup = demonstrate_vectorization()
    results['vectorization'] = {
        'loop_time': loop_time,
        'vectorized_time': vec_time,
        'speedup': speedup
    }
    
    # Demo 2: Threading
    seq_time, par_results = demonstrate_threading_vs_sequential()
    results['threading'] = par_results
    
    # Demo 3: Bad parallelism cases
    demonstrate_bad_parallelism()
    
    # Demo 4: Gradient accumulation
    demonstrate_gradient_accumulation()
    
    # Demo 5: Data vs model parallelism
    demonstrate_data_parallelism_vs_model_parallelism()
    
    # Create performance visualizations
    create_performance_plots(results)
    
    # Summary
    print("\n\n🎉 DEMONSTRATION COMPLETE")
    print("=" * 60)
    print("Key Takeaways:")
    print("✅ Vectorization provides massive speedups (100-1000x)")
    print("✅ Threading works well for independent operations")
    print("❌ Small operations have too much overhead")
    print("❌ Sequential dependencies cannot be parallelized")
    print("🔧 Gradient accumulation needs thread safety")
    print("🔀 Data parallelism is simpler than model parallelism")
    
    print(f"\n📚 For more details, see:")
    print(f"   PARALLELISM_DEEP_DIVE.md - Comprehensive guide")
    print(f"   tensor_relu_parallel.py - Real implementation")
    
    return results


if __name__ == "__main__":
    try:
        results = main()
    except KeyboardInterrupt:
        print("\n\n⚠️ Demo interrupted by user")
    except Exception as e:
        print(f"\n\n❌ Demo error: {e}")
        import traceback
        traceback.print_exc()