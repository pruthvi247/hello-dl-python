#!/usr/bin/env python3
"""
Memory Layout and Data Movement Visualization for CNN Parallelism
===============================================================

This script creates detailed visualizations showing:
1. How 28×28 image data is laid out in memory
2. How workers access different memory regions
3. How results are gathered and consolidated
4. Memory access patterns and cache efficiency
5. Synchronization points and data dependencies

Focus: Memory-centric view of parallel CNN processing
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.colors import ListedColormap
from pathlib import Path
import os

# Set style for better visualizations
plt.style.use('default')

class MemoryLayoutVisualizer:
    """Visualize memory layout and access patterns in CNN parallelism"""
    
    def __init__(self):
        self.fig_counter = 0
        
        # Create results directory
        self.results_dir = Path("explanations/parallelism/memory_visualizations")
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
    def save_figure(self, filename: str):
        """Save figure with proper formatting"""
        filepath = self.results_dir / f"{self.fig_counter:02d}_{filename}.png"
        plt.savefig(filepath, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"   💾 Saved: {filepath}")
        self.fig_counter += 1
        
    def visualize_image_memory_layout(self):
        """Show how 28×28 image is stored in memory"""
        
        print("\n🧠 MEMORY LAYOUT VISUALIZATION")
        print("="*50)
        
        # Create sample image
        np.random.seed(42)
        image = np.random.rand(28, 28)
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('28×28 Image Memory Layout and Access Patterns', fontsize=16, fontweight='bold')
        
        # 1. Original image as 2D array
        ax1 = axes[0, 0]
        im1 = ax1.imshow(image, cmap='viridis', aspect='equal')
        ax1.set_title('A) Original 28×28 Image\n(2D Memory View)')
        ax1.set_xlabel('Width (j)')
        ax1.set_ylabel('Height (i)')
        
        # Add grid lines every 4 pixels to show memory blocks
        for i in range(0, 29, 4):
            ax1.axhline(i-0.5, color='white', linewidth=0.5, alpha=0.7)
            ax1.axvline(i-0.5, color='white', linewidth=0.5, alpha=0.7)
        
        plt.colorbar(im1, ax=ax1, shrink=0.7)
        
        # 2. Flattened memory layout (row-major order)
        ax2 = axes[0, 1]
        flattened = image.flatten()
        
        # Reshape for visualization (784 elements → 28×28 for display)
        memory_view = flattened.reshape(28, 28)
        im2 = ax2.imshow(memory_view, cmap='viridis', aspect='equal')
        ax2.set_title('B) Row-Major Memory Layout\n(Sequential Access Pattern)')
        ax2.set_xlabel('Memory Address (sequential)')
        ax2.set_ylabel('Memory Banks')
        
        # Show memory access order with arrows
        for i in range(0, 28, 7):  # Show every 7th row to avoid clutter
            ax2.annotate('', xy=(27, i), xytext=(0, i),
                        arrowprops=dict(arrowstyle='->', color='red', lw=2, alpha=0.7))
        
        plt.colorbar(im2, ax=ax2, shrink=0.7)
        
        # 3. Parallel worker memory regions
        ax3 = axes[1, 0]
        
        # Create worker assignment visualization
        worker_image = np.zeros((28, 28))
        colors = [0, 1, 2, 3]  # 4 workers
        
        # Assign pixels to workers (spatial tiling)
        tile_size = 14  # 28/2 = 14
        for i in range(28):
            for j in range(28):
                worker_i = i // tile_size
                worker_j = j // tile_size
                worker_id = worker_i * 2 + worker_j
                worker_image[i, j] = worker_id
        
        cmap = ListedColormap(['red', 'blue', 'green', 'orange'])
        im3 = ax3.imshow(worker_image, cmap=cmap, aspect='equal')
        ax3.set_title('C) Spatial Parallel Assignment\n(4 Workers, 14×14 tiles each)')
        ax3.set_xlabel('Width')
        ax3.set_ylabel('Height')
        
        # Add worker boundaries
        ax3.axhline(13.5, color='black', linewidth=3)
        ax3.axvline(13.5, color='black', linewidth=3)
        
        # Label workers
        ax3.text(7, 7, 'Worker 0', ha='center', va='center', fontweight='bold', color='white')
        ax3.text(21, 7, 'Worker 1', ha='center', va='center', fontweight='bold', color='white')
        ax3.text(7, 21, 'Worker 2', ha='center', va='center', fontweight='bold', color='white')
        ax3.text(21, 21, 'Worker 3', ha='center', va='center', fontweight='bold', color='white')
        
        plt.colorbar(im3, ax=ax3, ticks=[0, 1, 2, 3], shrink=0.7)
        
        # 4. Memory access pattern timeline
        ax4 = axes[1, 1]
        
        # Simulate memory access timeline
        time_steps = 100
        workers = 4
        access_pattern = np.zeros((workers, time_steps))
        
        # Each worker accesses different memory regions over time
        for t in range(time_steps):
            for w in range(workers):
                # Simulate stride access pattern
                base_addr = w * (784 // workers)
                access_addr = (base_addr + t * 8) % 784
                access_pattern[w, t] = access_addr
        
        im4 = ax4.imshow(access_pattern, cmap='plasma', aspect='auto')
        ax4.set_title('D) Memory Access Timeline\n(Each worker accesses different regions)')
        ax4.set_xlabel('Time Steps')
        ax4.set_ylabel('Worker ID')
        ax4.set_yticks(range(workers))
        ax4.set_yticklabels([f'Worker {i}' for i in range(workers)])
        
        plt.colorbar(im4, ax=ax4, label='Memory Address', shrink=0.7)
        
        plt.tight_layout()
        self.save_figure('image_memory_layout')
        plt.show()
        
    def visualize_convolution_data_flow(self):
        """Show how convolution data flows through parallel workers"""
        
        print("\n🔄 CONVOLUTION DATA FLOW")
        print("="*40)
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Convolution Layer: Data Distribution and Consolidation', fontsize=16, fontweight='bold')
        
        # Input dimensions
        input_h, input_w = 28, 28
        kernel_size = 3
        output_h, output_w = 26, 26
        num_channels = 32
        num_workers = 4
        
        # 1. Input data distribution
        ax1 = axes[0, 0]
        input_data = np.random.rand(input_h, input_w)
        im1 = ax1.imshow(input_data, cmap='Blues', aspect='equal')
        ax1.set_title('Step 1: Input 28×28 Image\n(Shared across all workers)')
        ax1.set_xlabel('Width')
        ax1.set_ylabel('Height')
        
        # Show 3×3 kernel regions
        for i in range(0, output_h, 8):
            for j in range(0, output_w, 8):
                rect = patches.Rectangle((j, i), kernel_size, kernel_size, 
                                       linewidth=2, edgecolor='red', facecolor='none', alpha=0.7)
                ax1.add_patch(rect)
        
        plt.colorbar(im1, ax=ax1, shrink=0.7)
        
        # 2. Worker channel assignment
        ax2 = axes[0, 1]
        
        # Show which channels each worker processes
        channels_per_worker = num_channels // num_workers
        worker_assignment = np.zeros((4, 8))  # 4 workers × 8 channels each
        
        for w in range(num_workers):
            start_ch = w * channels_per_worker
            end_ch = start_ch + channels_per_worker
            worker_assignment[w, :] = range(start_ch, end_ch)
        
        im2 = ax2.imshow(worker_assignment, cmap='Set3', aspect='auto')
        ax2.set_title('Step 2: Channel Assignment\n(8 channels per worker)')
        ax2.set_xlabel('Channel Index (within worker)')
        ax2.set_ylabel('Worker ID')
        ax2.set_yticks(range(4))
        ax2.set_yticklabels([f'Worker {i}' for i in range(4)])
        
        # Add channel numbers as text
        for w in range(4):
            for c in range(8):
                channel_num = int(worker_assignment[w, c])
                ax2.text(c, w, str(channel_num), ha='center', va='center', fontweight='bold')
        
        plt.colorbar(im2, ax=ax2, shrink=0.7)
        
        # 3. Parallel computation visualization
        ax3 = axes[0, 2]
        
        # Show computation progress over time
        time_steps = 50
        worker_progress = np.zeros((num_workers, time_steps))
        
        # Simulate different completion rates
        completion_rates = [1.0, 0.9, 1.1, 0.95]  # Different worker speeds
        
        for t in range(time_steps):
            for w in range(num_workers):
                progress = min(100, t * completion_rates[w] * 2)
                worker_progress[w, t] = progress
        
        im3 = ax3.imshow(worker_progress, cmap='RdYlGn', aspect='auto', vmin=0, vmax=100)
        ax3.set_title('Step 3: Parallel Computation\n(Progress over time)')
        ax3.set_xlabel('Time Steps')
        ax3.set_ylabel('Worker ID')
        ax3.set_yticks(range(4))
        ax3.set_yticklabels([f'Worker {i}' for i in range(4)])
        
        plt.colorbar(im3, ax=ax3, label='Completion %', shrink=0.7)
        
        # 4. Output channel consolidation
        ax4 = axes[1, 0]
        
        # Show how output channels are assembled
        output_channels = np.zeros((4, 8, 26, 26))  # 4 workers × 8 channels × 26×26
        
        for w in range(4):
            for c in range(8):
                # Simulate output feature map
                output_channels[w, c] = np.random.rand(26, 26) * (w + 1) / 4
        
        # Show combined result (mean across workers for visualization)
        combined_output = np.mean(output_channels, axis=(0, 1))
        im4 = ax4.imshow(combined_output, cmap='viridis', aspect='equal')
        ax4.set_title('Step 4: Consolidated Output\n(26×26×32 feature maps)')
        ax4.set_xlabel('Width')
        ax4.set_ylabel('Height')
        
        plt.colorbar(im4, ax=ax4, shrink=0.7)
        
        # 5. Memory bandwidth utilization
        ax5 = axes[1, 1]
        
        # Simulate memory bandwidth usage
        bandwidth_time = np.arange(0, 100, 1)
        total_bandwidth = 100  # GB/s
        
        # Each worker's bandwidth usage
        worker_bandwidth = np.zeros((num_workers, len(bandwidth_time)))
        for w in range(num_workers):
            # Simulate varying bandwidth usage
            peak_time = 20 + w * 15
            for t, time_val in enumerate(bandwidth_time):
                usage = total_bandwidth * 0.25 * np.exp(-((time_val - peak_time) / 10) ** 2)
                worker_bandwidth[w, t] = max(0, usage)
        
        # Plot bandwidth usage
        colors = ['red', 'blue', 'green', 'orange']
        for w in range(num_workers):
            ax5.plot(bandwidth_time, worker_bandwidth[w], 
                    color=colors[w], label=f'Worker {w}', linewidth=2)
        
        # Total bandwidth
        total_usage = np.sum(worker_bandwidth, axis=0)
        ax5.plot(bandwidth_time, total_usage, 'k--', linewidth=3, label='Total Usage')
        ax5.axhline(total_bandwidth, color='gray', linestyle=':', label='Max Bandwidth')
        
        ax5.set_title('Step 5: Memory Bandwidth Usage\n(Over computation time)')
        ax5.set_xlabel('Time (arbitrary units)')
        ax5.set_ylabel('Bandwidth (GB/s)')
        ax5.legend()
        ax5.grid(True, alpha=0.3)
        
        # 6. Synchronization and results gathering
        ax6 = axes[1, 2]
        
        # Show synchronization points
        sync_data = np.zeros((5, 4))  # 5 stages × 4 workers
        stages = ['Start', 'Load Data', 'Compute', 'Write Results', 'Sync']
        
        # Simulate completion times (normalized)
        completion_times = [
            [0, 0, 0, 0],      # Start (all start together)
            [1, 1, 1, 1],      # Load Data (all load together)
            [8, 7, 9, 8],      # Compute (different completion times)
            [9, 8, 10, 9],     # Write Results
            [10, 10, 10, 10]   # Sync (all must finish)
        ]
        
        for stage_idx, times in enumerate(completion_times):
            sync_data[stage_idx] = times
        
        im6 = ax6.imshow(sync_data, cmap='RdYlBu', aspect='auto')
        ax6.set_title('Step 6: Synchronization Timeline\n(All workers must complete)')
        ax6.set_xlabel('Worker ID')
        ax6.set_ylabel('Processing Stage')
        ax6.set_xticks(range(4))
        ax6.set_xticklabels([f'Worker {i}' for i in range(4)])
        ax6.set_yticks(range(5))
        ax6.set_yticklabels(stages)
        
        # Add completion time text
        for i in range(5):
            for j in range(4):
                ax6.text(j, i, f'{sync_data[i, j]:.0f}', ha='center', va='center', 
                        fontweight='bold', color='white')
        
        plt.colorbar(im6, ax=ax6, label='Completion Time', shrink=0.7)
        
        plt.tight_layout()
        self.save_figure('convolution_data_flow')
        plt.show()
        
    def visualize_linear_layer_parallelism(self):
        """Show linear layer matrix multiplication parallelism"""
        
        print("\n🔗 LINEAR LAYER PARALLELISM")
        print("="*35)
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Linear Layer: Matrix Multiplication Parallelism', fontsize=16, fontweight='bold')
        
        # Matrix dimensions
        input_size = 128
        output_size = 512
        num_workers = 4
        
        # 1. Input vector and weight matrix
        ax1 = axes[0, 0]
        
        # Create weight matrix visualization
        weights = np.random.randn(input_size, output_size) * 0.1
        im1 = ax1.imshow(weights, cmap='RdBu', aspect='auto', vmin=-0.3, vmax=0.3)
        ax1.set_title('A) Weight Matrix (128×512)\nInput Features × Output Features')
        ax1.set_xlabel('Output Features (512)')
        ax1.set_ylabel('Input Features (128)')
        
        # Show worker divisions
        features_per_worker = output_size // num_workers
        for i in range(1, num_workers):
            ax1.axvline(i * features_per_worker - 0.5, color='white', linewidth=3)
        
        # Label worker regions
        for w in range(num_workers):
            start_feat = w * features_per_worker
            end_feat = start_feat + features_per_worker
            ax1.text((start_feat + end_feat) / 2, input_size / 2, f'Worker {w}', 
                    ha='center', va='center', fontweight='bold', 
                    bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7))
        
        plt.colorbar(im1, ax=ax1, shrink=0.7)
        
        # 2. Input vector replication
        ax2 = axes[0, 1]
        
        input_vector = np.random.randn(input_size)
        
        # Show input vector replicated to all workers
        replicated_input = np.tile(input_vector.reshape(-1, 1), (1, num_workers))
        im2 = ax2.imshow(replicated_input, cmap='Greens', aspect='auto')
        ax2.set_title('B) Input Vector Replication\n(Shared across all workers)')
        ax2.set_xlabel('Worker ID')
        ax2.set_ylabel('Input Features (128)')
        ax2.set_xticks(range(num_workers))
        ax2.set_xticklabels([f'Worker {i}' for i in range(num_workers)])
        
        plt.colorbar(im2, ax=ax2, shrink=0.7)
        
        # 3. Parallel computation
        ax3 = axes[1, 0]
        
        # Show computation progress for each worker
        computation_steps = 64  # Simulate computation steps
        worker_computation = np.zeros((num_workers, computation_steps))
        
        for step in range(computation_steps):
            for w in range(num_workers):
                # Simulate different computation loads
                if step < computation_steps * 0.8:  # Normal computation
                    worker_computation[w, step] = 50 + 30 * np.sin(step * 0.1 + w)
                else:  # Finishing up
                    worker_computation[w, step] = 20 + 10 * np.exp(-(step - computation_steps * 0.8))
        
        im3 = ax3.imshow(worker_computation, cmap='hot', aspect='auto')
        ax3.set_title('C) Parallel Matrix Multiplication\n(Computation intensity over time)')
        ax3.set_xlabel('Computation Steps')
        ax3.set_ylabel('Worker ID')
        ax3.set_yticks(range(num_workers))
        ax3.set_yticklabels([f'Worker {i}' for i in range(num_workers)])
        
        plt.colorbar(im3, ax=ax3, label='CPU Usage %', shrink=0.7)
        
        # 4. Result consolidation
        ax4 = axes[1, 1]
        
        # Show how results are combined
        partial_results = np.zeros((num_workers, features_per_worker))
        for w in range(num_workers):
            # Simulate different output ranges for each worker
            partial_results[w] = np.random.randn(features_per_worker) * (0.5 + w * 0.2)
        
        im4 = ax4.imshow(partial_results, cmap='viridis', aspect='auto')
        ax4.set_title('D) Result Consolidation\n(Partial outputs combined)')
        ax4.set_xlabel('Output Features (per worker)')
        ax4.set_ylabel('Worker ID')
        ax4.set_yticks(range(num_workers))
        ax4.set_yticklabels([f'Worker {i}' for i in range(num_workers)])
        
        # Show how results are concatenated
        ax4.annotate('Concatenate →', xy=(features_per_worker + 10, 2), xytext=(features_per_worker + 30, 2),
                    arrowprops=dict(arrowstyle='->', lw=2, color='red'))
        
        plt.colorbar(im4, ax=ax4, shrink=0.7)
        
        plt.tight_layout()
        self.save_figure('linear_layer_parallelism')
        plt.show()
        
    def create_complete_pipeline_visualization(self):
        """Create overview of complete CNN pipeline data flow"""
        
        print("\n🚀 COMPLETE CNN PIPELINE")
        print("="*30)
        
        fig, ax = plt.subplots(1, 1, figsize=(20, 12))
        fig.suptitle('Complete CNN Pipeline: Data Flow and Parallelism Overview', fontsize=18, fontweight='bold')
        
        # Define pipeline stages
        stages = [
            {'name': 'Input\n28×28×1', 'x': 1, 'y': 6, 'width': 1.5, 'height': 2, 'color': 'lightblue'},
            {'name': 'Conv1\n26×26×32', 'x': 3.5, 'y': 5, 'width': 2, 'height': 4, 'color': 'lightgreen'},
            {'name': 'MaxPool1\n13×13×32', 'x': 6.5, 'y': 5.5, 'width': 1.5, 'height': 3, 'color': 'lightcoral'},
            {'name': 'GELU1\n13×13×32', 'x': 9, 'y': 6, 'width': 1.5, 'height': 2, 'color': 'lightyellow'},
            {'name': 'Conv2\n11×11×64', 'x': 11.5, 'y': 4.5, 'width': 2, 'height': 5, 'color': 'lightgreen'},
            {'name': 'MaxPool2\n5×5×64', 'x': 14.5, 'y': 5.5, 'width': 1.5, 'height': 3, 'color': 'lightcoral'},
            {'name': 'GELU2\n5×5×64', 'x': 17, 'y': 6, 'width': 1.5, 'height': 2, 'color': 'lightyellow'},
            {'name': 'Flatten\n1600→128', 'x': 19.5, 'y': 6.5, 'width': 1, 'height': 1, 'color': 'lightgray'},
            {'name': 'Linear1\n128→512', 'x': 21.5, 'y': 4, 'width': 2, 'height': 6, 'color': 'lightpink'},
            {'name': 'Output\n512', 'x': 24.5, 'y': 6, 'width': 1.5, 'height': 2, 'color': 'lightblue'}
        ]
        
        # Draw stages
        for i, stage in enumerate(stages):
            rect = patches.Rectangle((stage['x'], stage['y']), stage['width'], stage['height'],
                                   linewidth=2, edgecolor='black', facecolor=stage['color'], alpha=0.7)
            ax.add_patch(rect)
            
            # Add stage name
            ax.text(stage['x'] + stage['width']/2, stage['y'] + stage['height']/2, stage['name'],
                   ha='center', va='center', fontweight='bold', fontsize=10)
            
            # Add arrows between stages
            if i < len(stages) - 1:
                next_stage = stages[i + 1]
                arrow_start_x = stage['x'] + stage['width']
                arrow_end_x = next_stage['x']
                arrow_y = stage['y'] + stage['height']/2
                
                ax.annotate('', xy=(arrow_end_x, arrow_y), xytext=(arrow_start_x, arrow_y),
                           arrowprops=dict(arrowstyle='->', lw=2, color='blue'))
        
        # Add parallelism annotations
        parallel_annotations = [
            {'stage': 1, 'text': 'Channel\nParallelism\n(32 filters)', 'offset': -2},
            {'stage': 2, 'text': 'Spatial\nParallelism\n(pool regions)', 'offset': -2},
            {'stage': 3, 'text': 'Element\nParallelism\n(5408 elements)', 'offset': -2},
            {'stage': 4, 'text': 'Channel\nParallelism\n(64 filters)', 'offset': -2},
            {'stage': 8, 'text': 'Feature\nParallelism\n(512 outputs)', 'offset': -3}
        ]
        
        for ann in parallel_annotations:
            stage = stages[ann['stage']]
            ax.text(stage['x'] + stage['width']/2, stage['y'] + ann['offset'], ann['text'],
                   ha='center', va='center', fontsize=9, color='red', fontweight='bold',
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='red', alpha=0.8))
        
        # Add worker distribution information
        worker_info = [
            "🔄 Data Distribution Strategies:",
            "",
            "Conv Layers: 4 workers × 8 channels each",
            "MaxPool: 4 workers × 8 channels each", 
            "GELU: 4 workers × ~1350 elements each",
            "Linear: 4 workers × 128 features each",
            "",
            "📊 Memory Access Patterns:",
            "• Sequential: Good cache locality",
            "• Parallel: Non-overlapping regions",
            "• Shared: Read-only input data",
            "• Exclusive: Write-only output data",
            "",
            "⚡ Synchronization Points:",
            "• After each layer completion",
            "• Before next layer starts",
            "• No mid-layer synchronization needed"
        ]
        
        # Add text box with worker information
        ax.text(0.5, 2, '\n'.join(worker_info), fontsize=10, fontfamily='monospace',
               bbox=dict(boxstyle='round,pad=0.5', facecolor='lightgray', alpha=0.8),
               verticalalignment='top')
        
        # Set limits and labels
        ax.set_xlim(0, 27)
        ax.set_ylim(0, 12)
        ax.set_aspect('equal')
        ax.axis('off')
        
        # Add title for data flow
        ax.text(13, 11, 'Data Flow Direction →', ha='center', va='center', 
               fontsize=14, fontweight='bold', color='blue')
        
        plt.tight_layout()
        self.save_figure('complete_cnn_pipeline')
        plt.show()

def main():
    """Generate all memory layout and data flow visualizations"""
    
    print("🎨 CNN Memory Layout and Data Flow Visualizations")
    print("="*55)
    print("Generating detailed visualizations of parallel processing...")
    
    visualizer = MemoryLayoutVisualizer()
    
    # Generate all visualizations
    visualizer.visualize_image_memory_layout()
    visualizer.visualize_convolution_data_flow()
    visualizer.visualize_linear_layer_parallelism()
    visualizer.create_complete_pipeline_visualization()
    
    print(f"\n✅ All visualizations generated!")
    print(f"📁 Saved to: {visualizer.results_dir}")
    print(f"📊 Total figures: {visualizer.fig_counter}")
    
    # Summary of what was visualized
    print(f"\n📋 Generated Visualizations:")
    print(f"   1. Image Memory Layout - How 28×28 data is stored and accessed")
    print(f"   2. Convolution Data Flow - Channel parallelism and consolidation")
    print(f"   3. Linear Layer Parallelism - Matrix multiplication distribution")
    print(f"   4. Complete CNN Pipeline - End-to-end data flow overview")

if __name__ == "__main__":
    main()