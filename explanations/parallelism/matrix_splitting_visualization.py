#!/usr/bin/env python3
"""
Matrix Operations Visualization for Deep Learning Parallelism
============================================================

This program visualizes EXACTLY how matrices are split, processed, and combined
in different parallelism approaches. Focus is on understanding the mechanics
of matrix operations rather than performance comparisons.

Key visualizations:
1. How input matrices are split for parallel processing
2. Step-by-step matrix multiplication visualization
3. How results are combined back together
4. GELU activation applied to matrix chunks
5. Memory layout and data flow patterns
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from concurrent.futures import ThreadPoolExecutor
import time
from pathlib import Path

# Set up plotting
plt.rcParams['figure.figsize'] = (14, 10)
plt.rcParams['font.size'] = 10


class MatrixVisualizationHelper:
    """Helper class for creating detailed matrix operation visualizations"""
    
    def __init__(self):
        self.step_counter = 0
    
    def visualize_matrix_split(self, matrix, split_type="rows", num_chunks=4, title="Matrix Splitting"):
        """Visualize how a matrix is split for parallel processing"""
        rows, cols = matrix.shape
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        
        # Original matrix
        im1 = ax1.imshow(matrix, cmap='viridis', aspect='auto')
        ax1.set_title(f'Original Matrix: {matrix.shape}')
        ax1.set_xlabel('Columns')
        ax1.set_ylabel('Rows')
        
        # Add grid lines to show structure
        for i in range(rows + 1):
            ax1.axhline(i - 0.5, color='white', linewidth=0.5, alpha=0.3)
        for j in range(cols + 1):
            ax1.axvline(j - 0.5, color='white', linewidth=0.5, alpha=0.3)
        
        plt.colorbar(im1, ax=ax1, shrink=0.6)
        
        # Split matrix visualization
        if split_type == "rows":
            chunk_size = rows // num_chunks
            colors = plt.cm.Set3(np.linspace(0, 1, num_chunks))
            
            # Create split visualization
            split_matrix = np.zeros_like(matrix)
            for i in range(num_chunks):
                start_row = i * chunk_size
                end_row = min((i + 1) * chunk_size, rows)
                split_matrix[start_row:end_row, :] = i + 1
            
            im2 = ax2.imshow(split_matrix, cmap='Set3', aspect='auto', vmin=0, vmax=num_chunks)
            ax2.set_title(f'Split into {num_chunks} Row Chunks')
            
            # Add chunk labels and boundaries
            for i in range(num_chunks):
                start_row = i * chunk_size
                end_row = min((i + 1) * chunk_size, rows)
                mid_row = (start_row + end_row) / 2
                ax2.text(cols/2, mid_row, f'Chunk {i+1}\n{end_row-start_row}×{cols}', 
                        ha='center', va='center', fontweight='bold', 
                        bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.8))
                
                # Add boundary lines
                if i > 0:
                    ax2.axhline(start_row - 0.5, color='red', linewidth=2)
        
        elif split_type == "cols":
            chunk_size = cols // num_chunks
            
            # Create split visualization
            split_matrix = np.zeros_like(matrix)
            for i in range(num_chunks):
                start_col = i * chunk_size
                end_col = min((i + 1) * chunk_size, cols)
                split_matrix[:, start_col:end_col] = i + 1
            
            im2 = ax2.imshow(split_matrix, cmap='Set3', aspect='auto', vmin=0, vmax=num_chunks)
            ax2.set_title(f'Split into {num_chunks} Column Chunks')
            
            # Add chunk labels and boundaries
            for i in range(num_chunks):
                start_col = i * chunk_size
                end_col = min((i + 1) * chunk_size, cols)
                mid_col = (start_col + end_col) / 2
                ax2.text(mid_col, rows/2, f'Chunk {i+1}\n{rows}×{end_col-start_col}', 
                        ha='center', va='center', fontweight='bold',
                        bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.8))
                
                # Add boundary lines
                if i > 0:
                    ax2.axvline(start_col - 0.5, color='red', linewidth=2)
        
        ax2.set_xlabel('Columns')
        ax2.set_ylabel('Rows')
        
        plt.tight_layout()
        return fig
    
    def visualize_matrix_multiplication_steps(self, A, B, title="Matrix Multiplication A @ B"):
        """Show step-by-step matrix multiplication process"""
        m, k = A.shape
        k2, n = B.shape
        assert k == k2, f"Inner dimensions must match: {k} != {k2}"
        
        # Create figure with subplots
        fig = plt.figure(figsize=(16, 10))
        
        # Layout: [A] [B] [=] [Result]
        gs = fig.add_gridspec(3, 4, height_ratios=[1, 1, 0.3], width_ratios=[1, 1, 0.2, 1])
        
        # Matrix A
        ax_A = fig.add_subplot(gs[0, 0])
        im_A = ax_A.imshow(A, cmap='Blues', aspect='auto')
        ax_A.set_title(f'Matrix A: {A.shape}')
        ax_A.set_ylabel('Rows')
        plt.colorbar(im_A, ax=ax_A, shrink=0.6)
        
        # Matrix B
        ax_B = fig.add_subplot(gs[0, 1])
        im_B = ax_B.imshow(B, cmap='Reds', aspect='auto')
        ax_B.set_title(f'Matrix B: {B.shape}')
        plt.colorbar(im_B, ax=ax_B, shrink=0.6)
        
        # Equals sign
        ax_eq = fig.add_subplot(gs[0, 2])
        ax_eq.text(0.5, 0.5, '@', ha='center', va='center', fontsize=30, fontweight='bold')
        ax_eq.set_xlim(0, 1)
        ax_eq.set_ylim(0, 1)
        ax_eq.axis('off')
        
        # Result matrix
        C = A @ B
        ax_C = fig.add_subplot(gs[0, 3])
        im_C = ax_C.imshow(C, cmap='Greens', aspect='auto')
        ax_C.set_title(f'Result C: {C.shape}')
        plt.colorbar(im_C, ax=ax_C, shrink=0.6)
        
        # Show detailed computation for one element
        ax_detail = fig.add_subplot(gs[1, :])
        ax_detail.set_xlim(0, 10)
        ax_detail.set_ylim(0, 3)
        ax_detail.axis('off')
        
        # Pick element (0,0) for detailed view
        row_idx, col_idx = 0, 0
        ax_detail.text(5, 2.5, f'Computing C[{row_idx},{col_idx}] = Row {row_idx} of A · Column {col_idx} of B', 
                      ha='center', fontsize=14, fontweight='bold')
        
        # Show the computation
        row_A = A[row_idx, :]
        col_B = B[:, col_idx]
        
        computation_text = ""
        for i in range(len(row_A)):
            if i > 0:
                computation_text += " + "
            computation_text += f"{row_A[i]:.2f} × {col_B[i]:.2f}"
            if i > 3 and len(row_A) > 6:  # Truncate if too long
                computation_text += " + ..."
                break
        
        result_value = np.dot(row_A, col_B)
        computation_text += f" = {result_value:.2f}"
        
        ax_detail.text(5, 1.5, computation_text, ha='center', fontsize=12,
                      bbox=dict(boxstyle="round,pad=0.5", facecolor='lightblue', alpha=0.8))
        
        ax_detail.text(5, 0.5, f'This process repeats for all {m} × {n} = {m*n} elements in the result matrix',
                      ha='center', fontsize=11, style='italic')
        
        plt.suptitle(title, fontsize=16, fontweight='bold')
        plt.tight_layout()
        return fig


class ParallelMatrixProcessor:
    """Demonstrates different ways to process matrices in parallel"""
    
    def __init__(self):
        self.visualizer = MatrixVisualizationHelper()
    
    def gelu_activation(self, x):
        """GELU activation function"""
        return x * 0.5 * (1 + np.tanh(np.sqrt(2/np.pi) * (x + 0.044715 * x**3)))
    
    def demonstrate_sequential_processing(self, batch_data, weights, bias):
        """Show how sequential processing works sample by sample"""
        print("\n" + "="*70)
        print("🔄 SEQUENTIAL MATRIX PROCESSING")
        print("="*70)
        
        batch_size, input_dim = batch_data.shape
        output_dim = weights.shape[1]
        
        print(f"Processing {batch_size} samples sequentially...")
        print(f"Input: {batch_data.shape}, Weights: {weights.shape}, Bias: {bias.shape}")
        
        results = []
        
        for i in range(batch_size):
            sample = batch_data[i:i+1]  # Keep 2D shape
            
            print(f"\nStep {i+1}: Processing sample {i}")
            print(f"  Sample shape: {sample.shape}")
            
            # Linear transformation
            linear_output = sample @ weights + bias
            print(f"  Linear output: {sample.shape} @ {weights.shape} + {bias.shape} = {linear_output.shape}")
            
            # GELU activation
            activated = self.gelu_activation(linear_output)
            print(f"  After GELU: {activated.shape}")
            
            results.append(activated)
        
        final_result = np.vstack(results)
        print(f"\nFinal result: {final_result.shape}")
        
        return final_result
    
    def demonstrate_parallel_row_splitting(self, batch_data, weights, bias, num_workers=4):
        """Show how to split input matrix by rows for parallel processing"""
        print("\n" + "="*70)
        print("🔀 PARALLEL PROCESSING: ROW SPLITTING")
        print("="*70)
        
        batch_size, input_dim = batch_data.shape
        output_dim = weights.shape[1]
        
        print(f"Splitting {batch_size} samples across {num_workers} workers...")
        
        # Calculate chunk sizes
        chunk_size = batch_size // num_workers
        chunks = []
        
        for i in range(num_workers):
            start_idx = i * chunk_size
            end_idx = min((i + 1) * chunk_size, batch_size)
            chunk = batch_data[start_idx:end_idx]
            chunks.append((chunk, start_idx, end_idx))
            
            print(f"Worker {i+1}: Rows {start_idx}-{end_idx-1} (shape: {chunk.shape})")
        
        # Visualize the splitting
        fig = self.visualizer.visualize_matrix_split(
            batch_data, split_type="rows", num_chunks=num_workers,
            title="Input Matrix Split by Rows for Parallel Processing"
        )
        
        # Save visualization
        results_dir = Path("results/matrix_visualization")
        results_dir.mkdir(parents=True, exist_ok=True)
        fig.savefig(results_dir / "row_splitting.png", dpi=300, bbox_inches='tight')
        plt.close(fig)
        
        def process_chunk(chunk_info):
            """Process one chunk of data"""
            chunk_data, start_idx, end_idx = chunk_info
            
            print(f"  Worker processing rows {start_idx}-{end_idx-1}:")
            print(f"    Input chunk: {chunk_data.shape}")
            
            # Linear transformation
            linear_output = chunk_data @ weights + bias
            print(f"    Linear: {chunk_data.shape} @ {weights.shape} + {bias.shape} = {linear_output.shape}")
            
            # GELU activation
            activated = self.gelu_activation(linear_output)
            print(f"    After GELU: {activated.shape}")
            
            return activated, start_idx, end_idx
        
        # Process chunks in parallel
        print("\n🔄 Processing chunks in parallel...")
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            chunk_results = list(executor.map(process_chunk, chunks))
        
        # Combine results
        print("\n🔗 Combining results...")
        final_result = np.zeros((batch_size, output_dim))
        
        for chunk_result, start_idx, end_idx in chunk_results:
            final_result[start_idx:end_idx] = chunk_result
            print(f"  Placed chunk result at rows {start_idx}-{end_idx-1}")
        
        print(f"\nFinal combined result: {final_result.shape}")
        
        return final_result
    
    def demonstrate_vectorized_processing(self, batch_data, weights, bias):
        """Show how vectorized processing works on entire matrices"""
        print("\n" + "="*70)
        print("🚀 VECTORIZED MATRIX PROCESSING")
        print("="*70)
        
        batch_size, input_dim = batch_data.shape
        output_dim = weights.shape[1]
        
        print(f"Processing entire batch as single matrix operation...")
        print(f"Input: {batch_data.shape}, Weights: {weights.shape}, Bias: {bias.shape}")
        
        # Show the matrix multiplication structure
        fig = self.visualizer.visualize_matrix_multiplication_steps(
            batch_data, weights, 
            title="Vectorized Batch Processing: X @ W + b"
        )
        
        # Save visualization
        results_dir = Path("results/matrix_visualization")
        results_dir.mkdir(parents=True, exist_ok=True)
        fig.savefig(results_dir / "vectorized_multiplication.png", dpi=300, bbox_inches='tight')
        plt.close(fig)
        
        # Perform the actual computation
        print("\n🧮 Matrix operations:")
        
        # Linear transformation (vectorized)
        print(f"  Step 1: Linear transformation")
        print(f"    {batch_data.shape} @ {weights.shape} = {(batch_size, output_dim)}")
        linear_output = batch_data @ weights
        
        print(f"    Adding bias: {(batch_size, output_dim)} + {bias.shape} = {(batch_size, output_dim)}")
        linear_output += bias
        
        # GELU activation (vectorized)
        print(f"  Step 2: GELU activation")
        print(f"    GELU({linear_output.shape}) = {linear_output.shape}")
        activated = self.gelu_activation(linear_output)
        
        print(f"\nResult: {activated.shape}")
        print(f"✅ Single vectorized operation processes all {batch_size} samples simultaneously")
        
        return activated
    
    def visualize_gelu_activation_on_matrices(self, input_matrix):
        """Visualize how GELU activation is applied element-wise to matrices"""
        print("\n" + "="*70)
        print("🎭 GELU ACTIVATION VISUALIZATION")
        print("="*70)
        
        # Apply GELU
        output_matrix = self.gelu_activation(input_matrix)
        
        # Create visualization
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 10))
        
        # Original matrix
        im1 = ax1.imshow(input_matrix, cmap='RdBu_r', aspect='auto')
        ax1.set_title(f'Input Matrix: {input_matrix.shape}')
        ax1.set_ylabel('Samples (rows)')
        ax1.set_xlabel('Features (cols)')
        plt.colorbar(im1, ax=ax1, shrink=0.8)
        
        # GELU output
        im2 = ax2.imshow(output_matrix, cmap='RdBu_r', aspect='auto')
        ax2.set_title(f'After GELU: {output_matrix.shape}')
        ax2.set_ylabel('Samples (rows)')
        ax2.set_xlabel('Features (cols)')
        plt.colorbar(im2, ax=ax2, shrink=0.8)
        
        # GELU function plot
        x_range = np.linspace(-3, 3, 1000)
        gelu_y = self.gelu_activation(x_range)
        ax3.plot(x_range, gelu_y, 'b-', linewidth=2)
        ax3.set_xlabel('Input value')
        ax3.set_ylabel('GELU output')
        ax3.set_title('GELU Function: x * 0.5 * (1 + tanh(√(2/π) * (x + 0.044715*x³)))')
        ax3.grid(True, alpha=0.3)
        
        # Element-wise transformation visualization
        # Show how some specific elements transform
        sample_indices = [(0, 0), (1, 5), (2, 10), (3, 15)]
        
        ax4.set_xlim(0, 10)
        ax4.set_ylim(0, len(sample_indices) + 1)
        ax4.set_title('Element-wise Transformation Examples')
        
        for i, (row, col) in enumerate(sample_indices):
            if row < input_matrix.shape[0] and col < input_matrix.shape[1]:
                input_val = input_matrix[row, col]
                output_val = output_matrix[row, col]
                
                y_pos = len(sample_indices) - i
                ax4.text(1, y_pos, f'[{row},{col}]:', ha='right', va='center', fontweight='bold')
                ax4.text(2, y_pos, f'{input_val:.3f}', ha='center', va='center',
                        bbox=dict(boxstyle="round,pad=0.3", facecolor='lightblue'))
                ax4.text(4, y_pos, '→', ha='center', va='center', fontsize=16)
                ax4.text(6, y_pos, f'{output_val:.3f}', ha='center', va='center',
                        bbox=dict(boxstyle="round,pad=0.3", facecolor='lightgreen'))
        
        ax4.axis('off')
        
        plt.tight_layout()
        
        # Save visualization
        results_dir = Path("results/matrix_visualization")
        results_dir.mkdir(parents=True, exist_ok=True)
        fig.savefig(results_dir / "gelu_activation_matrix.png", dpi=300, bbox_inches='tight')
        plt.close(fig)
        
        print(f"📊 GELU visualization saved")
        
        return output_matrix
    
    def demonstrate_complete_neural_layer(self, batch_data):
        """Complete demonstration of a neural network layer with all approaches"""
        print("\n" + "="*80)
        print("🧠 COMPLETE NEURAL LAYER DEMONSTRATION")
        print("="*80)
        
        # Create weights and bias
        input_dim = batch_data.shape[1]
        output_dim = 64  # Hidden layer size
        
        weights = np.random.randn(input_dim, output_dim) * 0.1
        bias = np.zeros(output_dim)
        
        print(f"Neural Layer: {input_dim} → {output_dim}")
        print(f"Batch data: {batch_data.shape}")
        print(f"Weights: {weights.shape}")
        print(f"Bias: {bias.shape}")
        
        # Method 1: Sequential processing
        sequential_result = self.demonstrate_sequential_processing(batch_data, weights, bias)
        
        # Method 2: Parallel processing (row splitting)
        parallel_result = self.demonstrate_parallel_row_splitting(batch_data, weights, bias, num_workers=4)
        
        # Method 3: Vectorized processing
        vectorized_result = self.demonstrate_vectorized_processing(batch_data, weights, bias)
        
        # Verify all methods give the same result
        print("\n" + "="*70)
        print("✅ VERIFICATION")
        print("="*70)
        
        seq_vs_par = np.allclose(sequential_result, parallel_result, rtol=1e-10)
        seq_vs_vec = np.allclose(sequential_result, vectorized_result, rtol=1e-10)
        
        print(f"Sequential vs Parallel: {'✅ Identical' if seq_vs_par else '❌ Different'}")
        print(f"Sequential vs Vectorized: {'✅ Identical' if seq_vs_vec else '❌ Different'}")
        
        if seq_vs_par and seq_vs_vec:
            print("🎉 All three methods produce identical results!")
        
        # Visualize GELU activation
        self.visualize_gelu_activation_on_matrices(vectorized_result)
        
        return vectorized_result


def create_sample_data():
    """Create sample data for demonstration"""
    print("📊 Creating sample data...")
    
    # Simulate a batch of flattened 28x28 images (like MNIST)
    batch_size = 16
    input_features = 784  # 28*28
    
    # Create structured data that's easy to visualize
    data = np.random.randn(batch_size, input_features) * 2
    
    # Add some patterns to make visualization more interesting
    for i in range(batch_size):
        # Add different patterns to different samples
        if i % 4 == 0:
            data[i, :100] += 3  # Bright region
        elif i % 4 == 1:
            data[i, 200:300] -= 2  # Dark region
        elif i % 4 == 2:
            data[i, 400:600] += np.sin(np.arange(200) * 0.1) * 2  # Sinusoidal pattern
        else:
            data[i, 600:700] += np.random.randn(100) * 3  # Random noise
    
    print(f"  Batch shape: {data.shape}")
    print(f"  Data range: [{data.min():.2f}, {data.max():.2f}]")
    print(f"  Memory usage: {data.nbytes / 1024:.1f} KB")
    
    return data


def main():
    """Main demonstration function"""
    print("="*80)
    print("🔍 MATRIX OPERATIONS AND SPLITTING VISUALIZATION")
    print("="*80)
    print("This demo shows EXACTLY how matrices are:")
    print("• Split for parallel processing")
    print("• Multiplied step by step")
    print("• Combined back together")
    print("• Processed with GELU activation")
    print("• Handled in sequential vs parallel vs vectorized approaches")
    print("="*80)
    
    # Create sample data
    batch_data = create_sample_data()
    
    # Initialize processor
    processor = ParallelMatrixProcessor()
    
    # Show initial data visualization
    print("\n📊 Visualizing input data structure...")
    fig = processor.visualizer.visualize_matrix_split(
        batch_data[:, :100],  # Show first 100 features for clarity
        split_type="rows", 
        num_chunks=4,
        title="Input Batch Data (first 100 features shown)"
    )
    
    results_dir = Path("results/matrix_visualization")
    results_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(results_dir / "input_data_structure.png", dpi=300, bbox_inches='tight')
    plt.close(fig)
    
    # Run complete demonstration
    final_result = processor.demonstrate_complete_neural_layer(batch_data)
    
    print("\n" + "="*80)
    print("🎉 MATRIX VISUALIZATION COMPLETE!")
    print("="*80)
    print("Key insights from matrix operations:")
    print("✅ Sequential: Processes one sample at a time (easy to understand)")
    print("✅ Parallel: Splits input matrix by rows across workers")
    print("✅ Vectorized: Processes entire matrix in single operation (fastest)")
    print("✅ GELU: Applied element-wise to entire matrices")
    print("✅ All methods produce identical numerical results")
    print()
    print("📊 Matrix visualizations saved to:")
    print(f"   • Input data structure: {results_dir / 'input_data_structure.png'}")
    print(f"   • Row splitting demo: {results_dir / 'row_splitting.png'}")
    print(f"   • Vectorized multiplication: {results_dir / 'vectorized_multiplication.png'}")
    print(f"   • GELU activation: {results_dir / 'gelu_activation_matrix.png'}")
    print()
    print("🧮 Matrix operation mechanics:")
    print(f"   • Input: {batch_data.shape} (batch_size × features)")
    print(f"   • Weights: ({batch_data.shape[1]}, 64) (features × output_dim)")
    print(f"   • Result: {final_result.shape} (batch_size × output_dim)")
    print(f"   • Total operations: {batch_data.shape[0] * batch_data.shape[1] * 64:,}")


if __name__ == "__main__":
    # Set random seed for reproducibility
    np.random.seed(42)
    
    # Run the demonstration
    main()