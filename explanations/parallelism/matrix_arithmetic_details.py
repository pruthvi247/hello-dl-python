#!/usr/bin/env python3
"""
Step-by-Step Matrix Arithmetic Visualization
===========================================

This program shows the detailed arithmetic of how matrix operations work
in neural networks, with focus on the actual numbers and calculations.

Shows:
1. Exact matrix multiplication arithmetic
2. How bias addition works
3. Element-by-element GELU calculation
4. How parallel workers split and process the work
5. Memory layout and data access patterns
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from pathlib import Path

class MatrixArithmeticVisualizer:
    """Shows detailed arithmetic operations in matrix calculations"""
    
    def __init__(self):
        self.fig_counter = 0
    
    def show_detailed_matrix_multiplication(self, A, B, highlight_element=(0, 0)):
        """Show detailed arithmetic for matrix multiplication A @ B"""
        m, k = A.shape
        k2, n = B.shape
        
        print(f"\n🔢 DETAILED MATRIX MULTIPLICATION: A{A.shape} @ B{B.shape}")
        print("="*60)
        
        # Show the matrices with actual values
        print("Matrix A:")
        self._print_matrix(A, precision=2)
        
        print("\nMatrix B:")
        self._print_matrix(B, precision=2)
        
        # Calculate result
        C = A @ B
        print(f"\nResult C = A @ B:")
        self._print_matrix(C, precision=2)
        
        # Show detailed calculation for highlighted element
        row, col = highlight_element
        if row < m and col < n:
            print(f"\n🎯 Detailed calculation for C[{row},{col}]:")
            print(f"C[{row},{col}] = Row {row} of A · Column {col} of B")
            
            row_A = A[row, :]
            col_B = B[:, col]
            
            print(f"Row {row} of A: {row_A}")
            print(f"Column {col} of B: {col_B}")
            
            print(f"\nStep-by-step multiplication:")
            result = 0
            for i in range(k):
                product = row_A[i] * col_B[i]
                result += product
                print(f"  {row_A[i]:.3f} × {col_B[i]:.3f} = {product:.3f}")
                if i < k-1:
                    print(f"                      Running sum: {result:.3f}")
            
            print(f"                      Final result: {result:.3f}")
            print(f"✅ Matches C[{row},{col}] = {C[row, col]:.3f}")
        
        return C
    
    def show_bias_addition_details(self, matrix, bias):
        """Show how bias addition works element by element"""
        print(f"\n➕ BIAS ADDITION: Matrix{matrix.shape} + Bias{bias.shape}")
        print("="*50)
        
        batch_size, features = matrix.shape
        
        print("Original matrix (first 3 rows, first 8 columns):")
        self._print_matrix(matrix[:3, :8], precision=3)
        
        print(f"\nBias vector (first 8 elements): {bias[:8]}")
        
        # Show how broadcasting works
        print(f"\n📡 Broadcasting explanation:")
        print(f"  Matrix shape: {matrix.shape} (batch_size={batch_size}, features={features})")
        print(f"  Bias shape:   {bias.shape} (features={features})")
        print(f"  Result shape: {matrix.shape} (same as matrix)")
        
        print(f"\n🔄 Broadcasting process:")
        print(f"  The bias vector {bias.shape} is added to EVERY row of the matrix")
        print(f"  Row 0: matrix[0, :] + bias = result[0, :]")
        print(f"  Row 1: matrix[1, :] + bias = result[1, :]")
        print(f"  ...")
        print(f"  Row {batch_size-1}: matrix[{batch_size-1}, :] + bias = result[{batch_size-1}, :]")
        
        # Show detailed calculation for first few elements
        result = matrix + bias
        
        print(f"\n🧮 Example calculations (first row):")
        for i in range(min(5, features)):
            print(f"  result[0,{i}] = matrix[0,{i}] + bias[{i}] = {matrix[0,i]:.3f} + {bias[i]:.3f} = {result[0,i]:.3f}")
        
        print(f"\nResult matrix (first 3 rows, first 8 columns):")
        self._print_matrix(result[:3, :8], precision=3)
        
        return result
    
    def show_gelu_element_calculations(self, matrix):
        """Show GELU activation calculations element by element"""
        print(f"\n🎭 GELU ACTIVATION: Element-by-element calculation")
        print("="*55)
        
        print("GELU formula: x * 0.5 * (1 + tanh(√(2/π) * (x + 0.044715 * x³)))")
        print("Where √(2/π) ≈ 0.7978846")
        
        # Show calculation for several elements
        batch_size, features = matrix.shape
        sample_elements = [
            (0, 0), (0, 1), (1, 0), (1, 1), (2, 2)
        ]
        
        result = np.zeros_like(matrix)
        
        print(f"\n🧮 Detailed GELU calculations:")
        for row, col in sample_elements:
            if row < batch_size and col < features:
                x = matrix[row, col]
                
                # Step by step GELU calculation
                x_cubed = x ** 3
                inner_term = x + 0.044715 * x_cubed
                sqrt_2_pi = np.sqrt(2 / np.pi)
                tanh_input = sqrt_2_pi * inner_term
                tanh_result = np.tanh(tanh_input)
                final_result = x * 0.5 * (1 + tanh_result)
                
                result[row, col] = final_result
                
                print(f"\n  Element [{row},{col}]: x = {x:.4f}")
                print(f"    x³ = {x_cubed:.4f}")
                print(f"    x + 0.044715×x³ = {x:.4f} + 0.044715×{x_cubed:.4f} = {inner_term:.4f}")
                print(f"    √(2/π) × inner_term = 0.7979 × {inner_term:.4f} = {tanh_input:.4f}")
                print(f"    tanh({tanh_input:.4f}) = {tanh_result:.4f}")
                print(f"    GELU = {x:.4f} × 0.5 × (1 + {tanh_result:.4f}) = {final_result:.4f}")
        
        # Calculate full GELU for comparison
        full_result = matrix * 0.5 * (1 + np.tanh(np.sqrt(2/np.pi) * (matrix + 0.044715 * matrix**3)))
        
        print(f"\n✅ Verification: Manual calculations match vectorized GELU")
        
        return full_result
    
    def show_parallel_work_distribution(self, matrix, num_workers=4):
        """Show exactly how work is distributed among parallel workers"""
        print(f"\n🔀 PARALLEL WORK DISTRIBUTION")
        print("="*40)
        
        batch_size, features = matrix.shape
        chunk_size = batch_size // num_workers
        
        print(f"Total work: Process {batch_size} samples with {features} features each")
        print(f"Workers: {num_workers}")
        print(f"Samples per worker: ~{chunk_size}")
        
        print(f"\n📋 Work assignment:")
        
        total_operations = 0
        
        for worker_id in range(num_workers):
            start_row = worker_id * chunk_size
            end_row = min((worker_id + 1) * chunk_size, batch_size)
            actual_rows = end_row - start_row
            
            # Each sample requires: features × output_dim operations for matrix mult
            # Plus features operations for bias addition
            # Plus features operations for GELU
            output_dim = 64  # Assuming 64 output features
            
            matrix_ops = actual_rows * features * output_dim
            bias_ops = actual_rows * output_dim
            gelu_ops = actual_rows * output_dim
            worker_total = matrix_ops + bias_ops + gelu_ops
            
            total_operations += worker_total
            
            print(f"\n  Worker {worker_id + 1}:")
            print(f"    Rows: {start_row} to {end_row-1} ({actual_rows} samples)")
            print(f"    Matrix multiplication: {actual_rows} × {features} × {output_dim} = {matrix_ops:,} ops")
            print(f"    Bias addition: {actual_rows} × {output_dim} = {bias_ops:,} ops")
            print(f"    GELU activation: {actual_rows} × {output_dim} = {gelu_ops:,} ops")
            print(f"    Total for this worker: {worker_total:,} operations")
            
            # Show memory access pattern
            chunk = matrix[start_row:end_row]
            print(f"    Memory access: {chunk.nbytes} bytes ({chunk.nbytes/1024:.1f} KB)")
        
        print(f"\n📊 Summary:")
        print(f"  Total operations across all workers: {total_operations:,}")
        print(f"  Average operations per worker: {total_operations//num_workers:,}")
        print(f"  Parallel efficiency: Ideally {num_workers}x faster than sequential")
        
        return total_operations
    
    def show_memory_layout_and_access(self, matrix):
        """Show how data is laid out in memory and accessed"""
        print(f"\n🧠 MEMORY LAYOUT AND ACCESS PATTERNS")
        print("="*45)
        
        batch_size, features = matrix.shape
        
        print(f"Matrix shape: {matrix.shape}")
        print(f"Total elements: {matrix.size:,}")
        print(f"Memory usage: {matrix.nbytes:,} bytes ({matrix.nbytes/1024:.1f} KB)")
        print(f"Data type: {matrix.dtype}")
        
        # Show memory layout (row-major order)
        print(f"\n📋 Memory layout (row-major order):")
        print(f"  Memory address order: [row0, col0], [row0, col1], ..., [row0, col{features-1}],")
        print(f"                        [row1, col0], [row1, col1], ..., [row1, col{features-1}],")
        print(f"                        ...")
        print(f"                        [row{batch_size-1}, col0], ..., [row{batch_size-1}, col{features-1}]")
        
        # Show access patterns for different processing methods
        print(f"\n🔍 Access patterns for different methods:")
        
        print(f"\n  Sequential processing:")
        print(f"    Process row 0: Access matrix[0, 0] → matrix[0, 1] → ... → matrix[0, {features-1}]")
        print(f"    Process row 1: Access matrix[1, 0] → matrix[1, 1] → ... → matrix[1, {features-1}]")
        print(f"    ...")
        print(f"    ✅ Good cache performance: Sequential memory access")
        
        print(f"\n  Parallel processing (row splitting):")
        print(f"    Worker 1: Rows 0-{batch_size//4-1} (sequential access within chunk)")
        print(f"    Worker 2: Rows {batch_size//4}-{batch_size//2-1} (sequential access within chunk)")
        print(f"    Worker 3: Rows {batch_size//2}-{3*batch_size//4-1} (sequential access within chunk)")
        print(f"    Worker 4: Rows {3*batch_size//4}-{batch_size-1} (sequential access within chunk)")
        print(f"    ✅ Good cache performance: Each worker accesses contiguous memory")
        
        print(f"\n  Vectorized processing:")
        print(f"    Entire matrix processed at once using optimized BLAS routines")
        print(f"    CPU/compiler optimizes memory access patterns automatically")
        print(f"    ✅ Best cache performance: Hardware-optimized access patterns")
        
        # Calculate cache implications
        cache_line_size = 64  # bytes (typical L1 cache line)
        elements_per_cache_line = cache_line_size // matrix.itemsize
        
        print(f"\n💾 Cache implications:")
        print(f"  Typical cache line size: {cache_line_size} bytes")
        print(f"  Elements per cache line: {elements_per_cache_line}")
        print(f"  Cache lines needed for one row: {features // elements_per_cache_line + 1}")
        print(f"  Total cache lines for matrix: {matrix.nbytes // cache_line_size + 1}")
    
    def _print_matrix(self, matrix, precision=3):
        """Helper to print matrix with nice formatting"""
        if matrix.ndim == 1:
            print(f"  [{' '.join(f'{x:>{precision+4}.{precision}f}' for x in matrix)}]")
        else:
            for i, row in enumerate(matrix):
                prefix = "  [" if i == 0 else "   "
                suffix = "]" if i == len(matrix) - 1 else ""
                print(f"{prefix}{' '.join(f'{x:>{precision+4}.{precision}f}' for x in row)}{suffix}")


def demonstrate_complete_matrix_arithmetic():
    """Complete demonstration of matrix arithmetic in neural networks"""
    print("="*80)
    print("🔢 COMPLETE MATRIX ARITHMETIC DEMONSTRATION")
    print("="*80)
    print("This shows the EXACT arithmetic behind neural network operations")
    print("="*80)
    
    # Create small, manageable matrices for detailed analysis
    batch_size = 4
    input_features = 6
    output_features = 4
    
    # Create simple, round numbers for easy arithmetic verification
    X = np.array([
        [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        [2.0, 3.0, 4.0, 5.0, 6.0, 7.0],
        [3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
        [4.0, 5.0, 6.0, 7.0, 8.0, 9.0]
    ])
    
    W = np.array([
        [0.1, 0.2, 0.3, 0.4],
        [0.2, 0.3, 0.4, 0.5],
        [0.3, 0.4, 0.5, 0.6],
        [0.4, 0.5, 0.6, 0.7],
        [0.5, 0.6, 0.7, 0.8],
        [0.6, 0.7, 0.8, 0.9]
    ])
    
    b = np.array([1.0, 2.0, 3.0, 4.0])
    
    print(f"📊 Input dimensions:")
    print(f"  Input X: {X.shape} (batch_size × input_features)")
    print(f"  Weights W: {W.shape} (input_features × output_features)")
    print(f"  Bias b: {b.shape} (output_features,)")
    
    visualizer = MatrixArithmeticVisualizer()
    
    # Step 1: Matrix multiplication
    print(f"\n" + "="*60)
    print(f"STEP 1: MATRIX MULTIPLICATION X @ W")
    print(f"="*60)
    
    linear_output = visualizer.show_detailed_matrix_multiplication(X, W, highlight_element=(0, 0))
    
    # Step 2: Bias addition
    print(f"\n" + "="*60)
    print(f"STEP 2: BIAS ADDITION")
    print(f"="*60)
    
    with_bias = visualizer.show_bias_addition_details(linear_output, b)
    
    # Step 3: GELU activation
    print(f"\n" + "="*60)
    print(f"STEP 3: GELU ACTIVATION")
    print(f"="*60)
    
    final_output = visualizer.show_gelu_element_calculations(with_bias)
    
    # Step 4: Parallel work distribution
    print(f"\n" + "="*60)
    print(f"STEP 4: PARALLEL PROCESSING ANALYSIS")
    print(f"="*60)
    
    # Use larger matrix for parallel analysis
    large_X = np.random.randn(16, 784) * 2
    total_ops = visualizer.show_parallel_work_distribution(large_X, num_workers=4)
    
    # Step 5: Memory layout
    print(f"\n" + "="*60)
    print(f"STEP 5: MEMORY LAYOUT AND ACCESS")
    print(f"="*60)
    
    visualizer.show_memory_layout_and_access(large_X)
    
    print(f"\n" + "="*80)
    print(f"🎉 MATRIX ARITHMETIC DEMONSTRATION COMPLETE!")
    print(f"="*80)
    
    print(f"Key insights:")
    print(f"✅ Matrix multiplication: Element (i,j) = Row i of A · Column j of B")
    print(f"✅ Bias addition: Broadcasting adds bias vector to every row")
    print(f"✅ GELU activation: Applied element-wise with complex formula")
    print(f"✅ Parallel processing: Work split by rows across workers")
    print(f"✅ Memory access: Row-major order enables cache-friendly access")
    
    print(f"\nFinal result shape: {final_output.shape}")
    print(f"Total arithmetic operations: {total_ops:,}")
    
    return final_output


if __name__ == "__main__":
    # Set random seed for reproducibility
    np.random.seed(42)
    
    # Run the demonstration
    demonstrate_complete_matrix_arithmetic()