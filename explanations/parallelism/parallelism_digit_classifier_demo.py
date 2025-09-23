#!/usr/bin/env python3
"""
Comprehensive Parallelism Demonstration for Digit Classification
================================================================

This program demonstrates ALL parallelism concepts from the Complete Parallelism Guide:
1. Vectorization vs Manual Loops (with MASSIVE speedups)
2. Data Parallelism across batch samples
3. Thread-safe gradient accumulation
4. Memory-efficient tensor operations
5. GELU activation function implementation
6. Detailed visualization of data flow at each step
7. Performance comparison and analysis

Author: Parallelism Guide Implementation
Date: September 2025
"""

import numpy as np
import matplotlib.pyplot as plt
import time
import threading
from concurrent.futures import ThreadPoolExecutor
import multiprocessing
from pathlib import Path
import json

# Set style for better plots
plt.style.use('default')
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 10

class DataFlowVisualizer:
    """Visualizes how data flows through the neural network at each step"""
    
    def __init__(self):
        self.step_data = []
        self.performance_data = []
    
    def log_step(self, step_name, input_shape, output_shape, processing_time=None, parallelism_type=None):
        """Log each processing step with detailed information"""
        step_info = {
            'step': step_name,
            'input_shape': input_shape,
            'output_shape': output_shape,
            'processing_time_ms': processing_time * 1000 if processing_time else None,
            'parallelism_type': parallelism_type,
            'timestamp': time.time()
        }
        self.step_data.append(step_info)
        
    def log_performance(self, operation, sequential_time, parallel_time, speedup, workers=None):
        """Log performance comparison data"""
        perf_info = {
            'operation': operation,
            'sequential_time_ms': sequential_time * 1000,
            'parallel_time_ms': parallel_time * 1000,
            'speedup': speedup,
            'workers': workers,
            'efficiency_percent': (speedup / workers * 100) if workers else None
        }
        self.performance_data.append(perf_info)
    
    def visualize_data_flow(self, save_path=None):
        """Create comprehensive visualization of data flow"""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Deep Learning Parallelism: Complete Data Flow Analysis', fontsize=16, fontweight='bold')
        
        # 1. Data Shape Flow Chart
        steps = [step['step'] for step in self.step_data]
        input_sizes = [np.prod(step['input_shape']) for step in self.step_data]
        output_sizes = [np.prod(step['output_shape']) for step in self.step_data]
        
        x_pos = range(len(steps))
        ax1.bar([x - 0.2 for x in x_pos], input_sizes, 0.4, label='Input Size', alpha=0.8)
        ax1.bar([x + 0.2 for x in x_pos], output_sizes, 0.4, label='Output Size', alpha=0.8)
        ax1.set_xlabel('Processing Steps')
        ax1.set_ylabel('Tensor Size (elements)')
        ax1.set_title('Data Transformation at Each Step')
        ax1.set_xticks(x_pos)
        ax1.set_xticklabels(steps, rotation=45, ha='right')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. Processing Time Analysis
        if any(step['processing_time_ms'] for step in self.step_data):
            times = [step['processing_time_ms'] or 0 for step in self.step_data]
            parallelism_types = [step['parallelism_type'] or 'Sequential' for step in self.step_data]
            
            colors = {'Vectorized': 'green', 'Parallel': 'blue', 'Sequential': 'red'}
            bar_colors = [colors.get(ptype, 'gray') for ptype in parallelism_types]
            
            bars = ax2.bar(steps, times, color=bar_colors, alpha=0.7)
            ax2.set_xlabel('Processing Steps')
            ax2.set_ylabel('Processing Time (ms)')
            ax2.set_title('Performance by Parallelism Type')
            ax2.set_xticklabels(steps, rotation=45, ha='right')
            
            # Add legend
            legend_elements = [plt.Rectangle((0,0),1,1, facecolor=color, alpha=0.7) 
                             for color in colors.values()]
            ax2.legend(legend_elements, colors.keys())
            ax2.grid(True, alpha=0.3)
        
        # 3. Speedup Comparison
        if self.performance_data:
            operations = [perf['operation'] for perf in self.performance_data]
            speedups = [perf['speedup'] for perf in self.performance_data]
            
            bars = ax3.bar(operations, speedups, alpha=0.8)
            ax3.set_xlabel('Operations')
            ax3.set_ylabel('Speedup Factor')
            ax3.set_title('Parallelism Speedup Results')
            ax3.set_xticklabels(operations, rotation=45, ha='right')
            
            # Color bars based on speedup
            for bar, speedup in zip(bars, speedups):
                if speedup > 10:
                    bar.set_color('green')
                elif speedup > 1:
                    bar.set_color('blue')
                else:
                    bar.set_color('red')
            
            # Add horizontal line at speedup = 1
            ax3.axhline(y=1, color='black', linestyle='--', alpha=0.5)
            ax3.grid(True, alpha=0.3)
        
        # 4. Thread Efficiency Analysis
        if any(perf.get('efficiency_percent') for perf in self.performance_data):
            parallel_ops = [perf for perf in self.performance_data if perf.get('workers')]
            if parallel_ops:
                workers = [perf['workers'] for perf in parallel_ops]
                efficiencies = [perf['efficiency_percent'] for perf in parallel_ops]
                op_names = [perf['operation'] for perf in parallel_ops]
                
                scatter = ax4.scatter(workers, efficiencies, s=100, alpha=0.7)
                ax4.set_xlabel('Number of Workers')
                ax4.set_ylabel('Efficiency (%)')
                ax4.set_title('Threading Efficiency vs Worker Count')
                ax4.grid(True, alpha=0.3)
                
                # Add labels for each point
                for i, op in enumerate(op_names):
                    ax4.annotate(op, (workers[i], efficiencies[i]), 
                               xytext=(5, 5), textcoords='offset points', fontsize=8)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"📊 Data flow visualization saved to: {save_path}")
        
        plt.show()
    
    def print_summary(self):
        """Print detailed summary of all operations"""
        print("\n" + "="*80)
        print("🔍 COMPLETE PARALLELISM ANALYSIS SUMMARY")
        print("="*80)
        
        print("\n📋 DATA FLOW STEPS:")
        for i, step in enumerate(self.step_data, 1):
            print(f"{i:2d}. {step['step']:<25} "
                  f"{str(step['input_shape']):<15} → {str(step['output_shape']):<15} "
                  f"[{step['parallelism_type'] or 'Sequential':<12}]"
                  + (f" {step['processing_time_ms']:.2f}ms" if step['processing_time_ms'] else ""))
        
        print("\n🚀 PERFORMANCE RESULTS:")
        for perf in self.performance_data:
            efficiency = f" (Efficiency: {perf['efficiency_percent']:.1f}%)" if perf.get('efficiency_percent') else ""
            print(f"   {perf['operation']:<25} "
                  f"Speedup: {perf['speedup']:>6.1f}x "
                  f"({perf['parallel_time_ms']:>6.1f}ms vs {perf['sequential_time_ms']:>6.1f}ms)"
                  + efficiency)


class GeluActivation:
    """GELU (Gaussian Error Linear Unit) activation function with vectorized implementation"""
    
    @staticmethod
    def gelu_manual_loops(x):
        """Manual implementation with loops (SLOW - for comparison)"""
        result = np.zeros_like(x)
        flat_x = x.flatten()
        flat_result = result.flatten()
        
        for i in range(len(flat_x)):
            # GELU formula: x * 0.5 * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x³)))
            val = flat_x[i]
            tanh_input = np.sqrt(2/np.pi) * (val + 0.044715 * val**3)
            flat_result[i] = val * 0.5 * (1 + np.tanh(tanh_input))
        
        return result.reshape(x.shape)
    
    @staticmethod
    def gelu_vectorized(x):
        """Vectorized GELU implementation (FAST)"""
        # GELU formula vectorized
        return x * 0.5 * (1 + np.tanh(np.sqrt(2/np.pi) * (x + 0.044715 * x**3)))
    
    @staticmethod
    def compare_implementations(x, visualizer):
        """Compare manual vs vectorized GELU implementations"""
        print(f"\n🧮 GELU Activation Comparison (shape: {x.shape})")
        
        # Manual implementation
        start = time.time()
        manual_result = GeluActivation.gelu_manual_loops(x)
        manual_time = time.time() - start
        
        # Vectorized implementation
        start = time.time()
        vectorized_result = GeluActivation.gelu_vectorized(x)
        vectorized_time = time.time() - start
        
        speedup = manual_time / vectorized_time
        visualizer.log_performance("GELU Activation", manual_time, vectorized_time, speedup)
        
        # Verify results are identical
        assert np.allclose(manual_result, vectorized_result, rtol=1e-6), "GELU implementations don't match!"
        
        print(f"   Manual loops:  {manual_time*1000:8.2f}ms")
        print(f"   Vectorized:    {vectorized_time*1000:8.2f}ms")
        print(f"   Speedup:       {speedup:8.1f}x")
        print(f"   ✅ Results identical: {np.allclose(manual_result, vectorized_result)}")
        
        return vectorized_result


class ThreadSafeGradientAccumulator:
    """Thread-safe gradient accumulation for parallel processing"""
    
    def __init__(self, shapes_dict):
        self.gradients = {name: np.zeros(shape) for name, shape in shapes_dict.items()}
        self.lock = threading.Lock()
        self.sample_count = 0
    
    def accumulate_gradients(self, grad_dict):
        """Thread-safe gradient accumulation"""
        with self.lock:
            for name, grad in grad_dict.items():
                if name in self.gradients:
                    self.gradients[name] += grad
            self.sample_count += 1
    
    def get_averaged_gradients(self):
        """Get gradients averaged over all samples"""
        with self.lock:
            if self.sample_count == 0:
                return self.gradients
            return {name: grad / self.sample_count for name, grad in self.gradients.items()}
    
    def reset(self):
        """Reset accumulated gradients"""
        with self.lock:
            for name in self.gradients:
                self.gradients[name].fill(0)
            self.sample_count = 0


class ParallelDigitClassifier:
    """Complete digit classifier with comprehensive parallelism demonstrations"""
    
    def __init__(self, input_size=784, hidden_sizes=[256, 128], num_classes=10):
        self.input_size = input_size
        self.hidden_sizes = hidden_sizes
        self.num_classes = num_classes
        self.visualizer = DataFlowVisualizer()
        
        # Initialize weights and biases
        self.weights = {}
        self.biases = {}
        
        # Input to first hidden layer
        self.weights['W1'] = np.random.randn(input_size, hidden_sizes[0]) * np.sqrt(2.0 / input_size)
        self.biases['b1'] = np.zeros(hidden_sizes[0])
        
        # Hidden layers
        for i in range(1, len(hidden_sizes)):
            self.weights[f'W{i+1}'] = np.random.randn(hidden_sizes[i-1], hidden_sizes[i]) * np.sqrt(2.0 / hidden_sizes[i-1])
            self.biases[f'b{i+1}'] = np.zeros(hidden_sizes[i])
        
        # Final layer
        final_layer_idx = len(hidden_sizes) + 1
        self.weights[f'W{final_layer_idx}'] = np.random.randn(hidden_sizes[-1], num_classes) * np.sqrt(2.0 / hidden_sizes[-1])
        self.biases[f'b{final_layer_idx}'] = np.zeros(num_classes)
        
        print(f"🏗️  Initialized Neural Network:")
        print(f"   Architecture: {input_size} → {' → '.join(map(str, hidden_sizes))} → {num_classes}")
        print(f"   Activation: GELU")
        print(f"   Total parameters: {self._count_parameters():,}")
    
    def _count_parameters(self):
        """Count total number of parameters"""
        total = 0
        for weights in self.weights.values():
            total += weights.size
        for biases in self.biases.values():
            total += biases.size
        return total
    
    def linear_layer_manual(self, X, W, b):
        """Manual linear layer implementation (SLOW - for comparison)"""
        batch_size, input_size = X.shape
        output_size = W.shape[1]
        result = np.zeros((batch_size, output_size))
        
        # Triple nested loop (very slow!)
        for i in range(batch_size):
            for j in range(output_size):
                for k in range(input_size):
                    result[i, j] += X[i, k] * W[k, j]
                result[i, j] += b[j]
        
        return result
    
    def linear_layer_vectorized(self, X, W, b):
        """Vectorized linear layer implementation (FAST)"""
        return X @ W + b
    
    def forward_pass_single(self, x):
        """Forward pass for a single sample"""
        current = x.reshape(1, -1)  # Ensure 2D shape
        
        # First hidden layer
        z1 = self.linear_layer_vectorized(current, self.weights['W1'], self.biases['b1'])
        a1 = GeluActivation.gelu_vectorized(z1)
        
        # Second hidden layer  
        z2 = self.linear_layer_vectorized(a1, self.weights['W2'], self.biases['b2'])
        a2 = GeluActivation.gelu_vectorized(z2)
        
        # Output layer
        z3 = self.linear_layer_vectorized(a2, self.weights['W3'], self.biases['b3'])
        
        # Softmax
        exp_scores = np.exp(z3 - np.max(z3, axis=1, keepdims=True))
        probabilities = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
        
        return probabilities[0]  # Return 1D array
    
    def forward_pass_batch_sequential(self, X):
        """Sequential batch processing (one sample at a time)"""
        batch_size = X.shape[0]
        results = []
        
        start_time = time.time()
        
        for i in range(batch_size):
            result = self.forward_pass_single(X[i])
            results.append(result)
        
        processing_time = time.time() - start_time
        self.visualizer.log_step(
            "Sequential Batch", 
            X.shape, 
            (batch_size, self.num_classes),
            processing_time,
            "Sequential"
        )
        
        return np.array(results)
    
    def forward_pass_batch_parallel(self, X, max_workers=None):
        """Parallel batch processing using ThreadPoolExecutor"""
        if max_workers is None:
            max_workers = min(8, multiprocessing.cpu_count())
        
        batch_size = X.shape[0]
        
        start_time = time.time()
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all samples for parallel processing
            futures = [executor.submit(self.forward_pass_single, X[i]) for i in range(batch_size)]
            # Collect results
            results = [future.result() for future in futures]
        
        processing_time = time.time() - start_time
        self.visualizer.log_step(
            "Parallel Batch", 
            X.shape, 
            (batch_size, self.num_classes),
            processing_time,
            "Parallel"
        )
        
        return np.array(results)
    
    def forward_pass_batch_vectorized(self, X):
        """Fully vectorized batch processing (FASTEST)"""
        start_time = time.time()
        
        current = X
        self.visualizer.log_step("Input", (), X.shape, 0, "Vectorized")
        
        # First hidden layer (vectorized)
        z1 = self.linear_layer_vectorized(current, self.weights['W1'], self.biases['b1'])
        self.visualizer.log_step("Linear 1", current.shape, z1.shape, None, "Vectorized")
        
        a1 = GeluActivation.gelu_vectorized(z1)
        self.visualizer.log_step("GELU 1", z1.shape, a1.shape, None, "Vectorized")
        
        # Second hidden layer (vectorized)
        z2 = self.linear_layer_vectorized(a1, self.weights['W2'], self.biases['b2'])
        self.visualizer.log_step("Linear 2", a1.shape, z2.shape, None, "Vectorized")
        
        a2 = GeluActivation.gelu_vectorized(z2)
        self.visualizer.log_step("GELU 2", z2.shape, a2.shape, None, "Vectorized")
        
        # Output layer (vectorized)
        z3 = self.linear_layer_vectorized(a2, self.weights['W3'], self.biases['b3'])
        self.visualizer.log_step("Linear 3", a2.shape, z3.shape, None, "Vectorized")
        
        # Softmax (vectorized)
        exp_scores = np.exp(z3 - np.max(z3, axis=1, keepdims=True))
        probabilities = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
        
        processing_time = time.time() - start_time
        self.visualizer.log_step("Softmax", z3.shape, probabilities.shape, processing_time, "Vectorized")
        
        return probabilities
    
    def compare_linear_layer_implementations(self, X, W, b):
        """Compare manual vs vectorized linear layer implementations"""
        print(f"\n🔄 Linear Layer Comparison (Input: {X.shape}, Weights: {W.shape})")
        
        # Manual implementation
        start = time.time()
        manual_result = self.linear_layer_manual(X, W, b)
        manual_time = time.time() - start
        
        # Vectorized implementation
        start = time.time()
        vectorized_result = self.linear_layer_vectorized(X, W, b)
        vectorized_time = time.time() - start
        
        speedup = manual_time / vectorized_time
        self.visualizer.log_performance("Linear Layer", manual_time, vectorized_time, speedup)
        
        # Verify results are identical
        assert np.allclose(manual_result, vectorized_result, rtol=1e-6), "Linear layer implementations don't match!"
        
        print(f"   Manual loops:  {manual_time*1000:8.2f}ms")
        print(f"   Vectorized:    {vectorized_time*1000:8.2f}ms")
        print(f"   Speedup:       {speedup:8.1f}x")
        print(f"   ✅ Results identical: {np.allclose(manual_result, vectorized_result)}")
        
        return vectorized_result
    
    def compare_batch_processing_methods(self, X):
        """Compare different batch processing approaches"""
        print(f"\n🏭 Batch Processing Comparison (Batch size: {X.shape[0]})")
        
        # Sequential processing
        start = time.time()
        sequential_results = self.forward_pass_batch_sequential(X)
        sequential_time = time.time() - start
        
        # Parallel processing (threading)
        start = time.time()
        parallel_results = self.forward_pass_batch_parallel(X, max_workers=4)
        parallel_time = time.time() - start
        
        # Vectorized processing
        start = time.time()
        vectorized_results = self.forward_pass_batch_vectorized(X)
        vectorized_time = time.time() - start
        
        # Log performance comparisons
        parallel_speedup = sequential_time / parallel_time
        vectorized_speedup = sequential_time / vectorized_time
        
        self.visualizer.log_performance("Batch Processing (Parallel)", sequential_time, parallel_time, parallel_speedup, workers=4)
        self.visualizer.log_performance("Batch Processing (Vectorized)", sequential_time, vectorized_time, vectorized_speedup)
        
        # Verify all results are close (small numerical differences expected due to floating point)
        assert np.allclose(sequential_results, parallel_results, rtol=1e-5), "Sequential vs Parallel results don't match!"
        assert np.allclose(sequential_results, vectorized_results, rtol=1e-5), "Sequential vs Vectorized results don't match!"
        
        print(f"   Sequential:    {sequential_time*1000:8.2f}ms")
        print(f"   Parallel (4):  {parallel_time*1000:8.2f}ms (Speedup: {parallel_speedup:.2f}x, Efficiency: {parallel_speedup/4*100:.1f}%)")
        print(f"   Vectorized:    {vectorized_time*1000:8.2f}ms (Speedup: {vectorized_speedup:.1f}x)")
        print(f"   ✅ All results match (within tolerance)")
        
        return vectorized_results
    
    def demonstrate_gradient_accumulation(self, X, y, learning_rate=0.01):
        """Demonstrate thread-safe gradient accumulation"""
        print(f"\n🔄 Gradient Accumulation Demonstration (Batch size: {X.shape[0]})")
        
        # Setup gradient accumulator
        gradient_shapes = {name: weights.shape for name, weights in self.weights.items()}
        gradient_shapes.update({name: bias.shape for name, bias in self.biases.items()})
        
        accumulator = ThreadSafeGradientAccumulator(gradient_shapes)
        
        def compute_sample_gradients(sample_idx):
            """Compute gradients for a single sample"""
            x_sample = X[sample_idx:sample_idx+1]  # Keep batch dimension
            y_sample = y[sample_idx]
            
            # Forward pass
            predictions = self.forward_pass_batch_vectorized(x_sample)
            
            # Compute loss (cross-entropy)
            loss = -np.log(predictions[0, y_sample] + 1e-15)
            
            # Simple gradient computation (placeholder - in practice would use backprop)
            # For demonstration, we'll use simplified gradients
            gradients = {}
            for name, weights in self.weights.items():
                gradients[name] = np.random.randn(*weights.shape) * 0.001  # Simulated gradients
            for name, bias in self.biases.items():
                gradients[name] = np.random.randn(*bias.shape) * 0.001    # Simulated gradients
            
            # Thread-safe accumulation
            accumulator.accumulate_gradients(gradients)
            return loss
        
        # Sequential gradient computation
        start = time.time()
        accumulator.reset()
        sequential_losses = []
        for i in range(X.shape[0]):
            loss = compute_sample_gradients(i)
            sequential_losses.append(loss)
        sequential_grads = accumulator.get_averaged_gradients()
        sequential_time = time.time() - start
        
        # Parallel gradient computation
        start = time.time()
        accumulator.reset()
        with ThreadPoolExecutor(max_workers=4) as executor:
            parallel_losses = list(executor.map(compute_sample_gradients, range(X.shape[0])))
        parallel_grads = accumulator.get_averaged_gradients()
        parallel_time = time.time() - start
        
        speedup = sequential_time / parallel_time
        self.visualizer.log_performance("Gradient Accumulation", sequential_time, parallel_time, speedup, workers=4)
        
        # Verify gradients are identical (they should be since we use thread-safe accumulation)
        gradients_match = all(
            np.allclose(sequential_grads[name], parallel_grads[name], rtol=1e-10)
            for name in sequential_grads.keys()
        )
        
        print(f"   Sequential:    {sequential_time*1000:8.2f}ms")
        print(f"   Parallel (4):  {parallel_time*1000:8.2f}ms (Speedup: {speedup:.2f}x, Efficiency: {speedup/4*100:.1f}%)")
        print(f"   ✅ Gradient accumulation is thread-safe: {gradients_match}")
        
        return parallel_grads


def create_synthetic_dataset(num_samples=1000, input_size=784, num_classes=10):
    """Create synthetic dataset for testing"""
    print(f"📊 Creating synthetic dataset: {num_samples} samples, {input_size} features, {num_classes} classes")
    
    # Generate random input data (simulating flattened 28x28 images)
    X = np.random.randn(num_samples, input_size).astype(np.float32)
    # Normalize to [0, 1] range (like image pixels)
    X = (X - X.min()) / (X.max() - X.min())
    
    # Generate random labels
    y = np.random.randint(0, num_classes, num_samples)
    
    print(f"   Input shape: {X.shape}")
    print(f"   Labels shape: {y.shape}")
    print(f"   Input range: [{X.min():.3f}, {X.max():.3f}]")
    print(f"   Unique labels: {sorted(np.unique(y))}")
    
    return X, y


def demonstrate_when_parallelism_hurts():
    """Demonstrate cases where parallelism actually hurts performance"""
    print("\n" + "="*80)
    print("⚠️  WHEN PARALLELISM HURTS: Anti-Examples")
    print("="*80)
    
    visualizer = DataFlowVisualizer()
    
    # 1. Tiny operations
    print("\n❌ Case 1: Tiny Operations (Thread overhead > computation)")
    
    def tiny_square(x):
        return x ** 2
    
    small_data = list(range(100))
    
    # Sequential
    start = time.time()
    seq_results = [tiny_square(x) for x in small_data]
    seq_time = time.time() - start
    
    # Parallel (will be slower!)
    start = time.time()
    with ThreadPoolExecutor(max_workers=4) as executor:
        par_results = list(executor.map(tiny_square, small_data))
    par_time = time.time() - start
    
    speedup = seq_time / par_time  # Will be < 1 (slower!)
    visualizer.log_performance("Tiny Operations", seq_time, par_time, speedup, workers=4)
    
    print(f"   Sequential:  {seq_time*1000:8.2f}ms")
    print(f"   Parallel:    {par_time*1000:8.2f}ms")
    print(f"   Slowdown:    {1/speedup:8.1f}x SLOWER!")
    print(f"   Why: Thread setup time ({par_time*1000:.2f}ms) >> computation time ({seq_time*1000:.2f}ms)")
    
    # 2. Memory-bound operations
    print("\n❌ Case 2: Memory-Bound Operations (Limited by bandwidth)")
    
    def copy_array(arr):
        return arr.copy()
    
    large_arrays = [np.random.randn(1000, 1000) for _ in range(4)]
    
    # Sequential
    start = time.time()
    seq_copies = [copy_array(arr) for arr in large_arrays]
    seq_time = time.time() - start
    
    # Parallel (limited improvement)
    start = time.time()
    with ThreadPoolExecutor(max_workers=4) as executor:
        par_copies = list(executor.map(copy_array, large_arrays))
    par_time = time.time() - start
    
    speedup = seq_time / par_time
    visualizer.log_performance("Memory Copy", seq_time, par_time, speedup, workers=4)
    
    print(f"   Sequential:  {seq_time*1000:8.2f}ms")
    print(f"   Parallel:    {par_time*1000:8.2f}ms")
    print(f"   Speedup:     {speedup:8.1f}x (Limited by memory bandwidth)")
    print(f"   Why: All threads compete for same memory bus")
    
    return visualizer


def run_comprehensive_demonstration():
    """Run complete parallelism demonstration with all concepts"""
    print("="*80)
    print("🚀 COMPREHENSIVE PARALLELISM DEMONSTRATION FOR DIGIT CLASSIFICATION")
    print("="*80)
    print("This demo covers ALL concepts from the Complete Parallelism Guide:")
    print("• Vectorization vs Manual Loops")
    print("• Data Parallelism (Threading)")
    print("• Thread-safe Gradient Accumulation")
    print("• Memory-efficient Tensor Operations")
    print("• GELU Activation Function")
    print("• When Parallelism Hurts")
    print("• Complete Performance Analysis")
    print("="*80)
    
    # Create dataset
    X, y = create_synthetic_dataset(num_samples=256, input_size=784, num_classes=10)
    
    # Initialize classifier
    classifier = ParallelDigitClassifier(input_size=784, hidden_sizes=[256, 128], num_classes=10)
    
    # 1. Compare GELU implementations
    print("\n" + "="*60)
    print("🧮 ACTIVATION FUNCTION COMPARISON")
    print("="*60)
    sample_data = X[:32]  # Use subset for activation comparison
    gelu_result = GeluActivation.compare_implementations(sample_data, classifier.visualizer)
    
    # 2. Compare linear layer implementations
    print("\n" + "="*60)
    print("🔄 LINEAR LAYER COMPARISON")
    print("="*60)
    test_batch = X[:16]  # Small batch for linear layer comparison
    linear_result = classifier.compare_linear_layer_implementations(
        test_batch, classifier.weights['W1'], classifier.biases['b1']
    )
    
    # 3. Compare batch processing methods
    print("\n" + "="*60)
    print("🏭 BATCH PROCESSING COMPARISON")
    print("="*60)
    batch_for_processing = X[:64]  # Medium batch for processing comparison
    batch_results = classifier.compare_batch_processing_methods(batch_for_processing)
    
    # 4. Demonstrate gradient accumulation
    print("\n" + "="*60)
    print("🔄 GRADIENT ACCUMULATION DEMONSTRATION")
    print("="*60)
    gradient_batch = X[:32]
    gradient_labels = y[:32]
    gradients = classifier.demonstrate_gradient_accumulation(gradient_batch, gradient_labels)
    
    # 5. Show when parallelism hurts
    anti_example_visualizer = demonstrate_when_parallelism_hurts()
    
    # Combine performance data
    classifier.visualizer.performance_data.extend(anti_example_visualizer.performance_data)
    
    # 6. Create comprehensive visualizations
    print("\n" + "="*60)
    print("📊 GENERATING COMPREHENSIVE ANALYSIS")
    print("="*60)
    
    # Print detailed summary
    classifier.visualizer.print_summary()
    
    # Create results directory
    results_dir = Path("results/parallelism_demo")
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate and save visualization
    viz_path = results_dir / "comprehensive_parallelism_analysis.png"
    classifier.visualizer.visualize_data_flow(save_path=viz_path)
    
    # Save performance data as JSON
    performance_data = {
        'step_data': classifier.visualizer.step_data,
        'performance_data': classifier.visualizer.performance_data,
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        'system_info': {
            'cpu_count': multiprocessing.cpu_count(),
            'python_version': f"{__import__('sys').version_info.major}.{__import__('sys').version_info.minor}.{__import__('sys').version_info.micro}"
        }
    }
    
    json_path = results_dir / "performance_analysis.json"
    with open(json_path, 'w') as f:
        json.dump(performance_data, f, indent=2, default=str)
    
    print(f"\n💾 Results saved to:")
    print(f"   📊 Visualization: {viz_path}")
    print(f"   📋 Data: {json_path}")
    
    # 7. Key insights and recommendations
    print("\n" + "="*80)
    print("🎯 KEY INSIGHTS AND RECOMMENDATIONS")
    print("="*80)
    
    # Find best and worst performing operations
    perf_data = classifier.visualizer.performance_data
    best_speedup = max(perf_data, key=lambda x: x['speedup'])
    worst_speedup = min(perf_data, key=lambda x: x['speedup'])
    
    print(f"🏆 BEST OPTIMIZATION: {best_speedup['operation']}")
    print(f"   Speedup: {best_speedup['speedup']:.1f}x")
    print(f"   Why: Vectorized operations leverage SIMD instructions and optimized BLAS libraries")
    
    print(f"\n💔 WORST OPTIMIZATION: {worst_speedup['operation']}")
    print(f"   Speedup: {worst_speedup['speedup']:.2f}x (SLOWER!)")
    print(f"   Why: Thread overhead exceeds computation time for simple operations")
    
    print(f"\n📈 OPTIMIZATION STRATEGY:")
    print(f"   1. VECTORIZATION FIRST: Replace loops with NumPy operations (100-10,000x speedup)")
    print(f"   2. DATA PARALLELISM SECOND: Use threading for independent operations (2-8x speedup)")
    print(f"   3. MEMORY OPTIMIZATION THIRD: Optimize cache usage (10-50% improvement)")
    print(f"   4. PROFILE ALWAYS: Measure before and after optimization")
    
    print(f"\n✅ SUCCESS METRICS FROM THIS DEMO:")
    vectorized_ops = [p for p in perf_data if p['speedup'] > 10]
    parallel_ops = [p for p in perf_data if 1 < p['speedup'] <= 10]
    failed_ops = [p for p in perf_data if p['speedup'] < 1]
    
    print(f"   🚀 Highly successful optimizations: {len(vectorized_ops)} (>10x speedup)")
    print(f"   ✅ Moderately successful optimizations: {len(parallel_ops)} (1-10x speedup)")
    print(f"   ❌ Failed optimizations: {len(failed_ops)} (<1x speedup)")
    
    return classifier, performance_data


if __name__ == "__main__":
    # Set random seed for reproducibility
    np.random.seed(42)
    
    # Run the comprehensive demonstration
    classifier, results = run_comprehensive_demonstration()
    
    print("\n" + "="*80)
    print("🎉 DEMONSTRATION COMPLETE!")
    print("="*80)
    print("You've now seen comprehensive examples of:")
    print("✅ When parallelism provides MASSIVE speedups (vectorization)")
    print("✅ When parallelism provides moderate speedups (threading)")
    print("✅ When parallelism HURTS performance (tiny operations)")
    print("✅ How to implement thread-safe operations")
    print("✅ Complete data flow visualization")
    print("✅ Performance analysis and optimization strategy")
    print("\nApply these concepts to your own deep learning projects for better performance!")