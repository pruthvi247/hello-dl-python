#!/usr/bin/env python3
"""
Step-by-Step CNN Parallelism: 28×28 Image Matrix Distribution Example
===================================================================

This example shows EXACTLY how a 28×28 image matrix gets distributed across 
processes and how results are consolidated at each step of the CNN pipeline.

We'll use a concrete example with actual numbers to demonstrate:
1. How data is split across workers
2. What each worker computes
3. How results are gathered and combined
4. Memory layout and access patterns
5. Synchronization points and data flow

Architecture: 28×28×1 → Conv(3×3,32) → Pool(2×2) → GELU → ... → 26 classes
"""

import numpy as np
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Tuple, Dict, Any
import queue
from dataclasses import dataclass, field
import json

# Set seed for reproducible results
np.random.seed(42)

@dataclass
class WorkerState:
    """Track what each worker is processing"""
    worker_id: int
    task_description: str
    input_data_shape: tuple
    output_data_shape: tuple
    processing_time: float = 0.0
    memory_used: int = 0
    operations_count: int = 0

@dataclass 
class LayerDistribution:
    """Track how a layer's computation is distributed"""
    layer_name: str
    input_shape: tuple
    output_shape: tuple
    total_operations: int
    num_workers: int
    worker_states: List[WorkerState] = field(default_factory=list)
    consolidation_time: float = 0.0
    
class DetailedCNNParallelismDemo:
    """Demonstrate step-by-step parallelism with concrete data"""
    
    def __init__(self, num_workers: int = 4):
        self.num_workers = num_workers
        self.layer_distributions = []
        
    def create_sample_image(self) -> np.ndarray:
        """Create a realistic 28×28 image (simulating handwritten letter)"""
        # Create a sample letter 'A' pattern
        image = np.zeros((28, 28), dtype=np.float32)
        
        # Draw an 'A' pattern with some noise
        for i in range(28):
            for j in range(28):
                # Diagonal lines for 'A'
                if (abs(i - 2*j + 14) < 2) or (abs(i + 2*j - 42) < 2) or (12 <= i <= 14 and 8 <= j <= 20):
                    image[i, j] = 0.8 + np.random.normal(0, 0.1)
                else:
                    image[i, j] = np.random.normal(0, 0.05)
        
        # Normalize to [0, 1]
        image = np.clip(image, 0, 1)
        return image
    
    def print_image_ascii(self, image: np.ndarray, title: str = "Image"):
        """Print image as ASCII art for visualization"""
        print(f"\n📸 {title}:")
        print("   " + "".join([f"{j%10}" for j in range(min(28, image.shape[1]))]))
        for i in range(min(10, image.shape[0])):  # Show first 10 rows
            row_str = f"{i:2} "
            for j in range(min(28, image.shape[1])):
                val = image[i, j] if image.ndim == 2 else image[i, j, 0] if image.ndim == 3 else image[0, 0, i, j]
                if val > 0.5:
                    row_str += "█"
                elif val > 0.3:
                    row_str += "▓"
                elif val > 0.1:
                    row_str += "░"
                else:
                    row_str += " "
            print(row_str)
        if image.shape[0] > 10:
            print("   ... (showing first 10 rows)")
    
    def demonstrate_step1_convolution(self, image: np.ndarray):
        """Step 1: Convolution 28×28×1 → 26×26×32 with detailed parallelism"""
        
        print("\n" + "="*80)
        print("🔸 STEP 1: CONVOLUTION LAYER - 28×28×1 → 26×26×32")
        print("="*80)
        
        # Show input
        self.print_image_ascii(image, "Input 28×28 Image")
        
        # Create 32 convolution kernels (3×3 each)
        kernels = np.random.randn(32, 1, 3, 3) * 0.1
        
        print(f"\n📊 Convolution Setup:")
        print(f"   Input shape: {image.shape}")
        print(f"   Kernels shape: {kernels.shape}")
        print(f"   Output shape will be: (26, 26, 32)")
        print(f"   Workers: {self.num_workers}")

        """
        Given Convolution Setup:
        Input shape: (28, 28) — This means the input image is 28 pixels high and 28 pixels wide (grayscale, so single channel assumed).

        Kernels shape: (32, 1, 3, 3)

        32: Number of filters (output channels)
        1: Number of input channels (grayscale image)
        3: Kernel height
        3: Kernel width

        Output shape: (26, 26, 32) — Output height 26, width 26, and 32 channels (one per kernel)."""

        
        # Distribute work: Each worker handles different output channels
        channels_per_worker = 32 // self.num_workers
        remainder_channels = 32 % self.num_workers
        
        output = np.zeros((26, 26, 32))
        worker_states = []
        
        print(f"\n🔄 Work Distribution:")
        
        def process_channels(worker_id: int, start_channel: int, end_channel: int):
            """Each worker processes a subset of output channels"""
            
            worker_start = time.time()
            worker_ops = 0
            worker_channels = end_channel - start_channel
            
            print(f"   Worker {worker_id}: Processing channels {start_channel}-{end_channel-1} ({worker_channels} channels)")
            
            # Process each assigned channel
            for out_ch in range(start_channel, end_channel):
                kernel = kernels[out_ch, 0]  # 3×3 kernel for this channel
                
                # Convolution for this channel
                for i in range(26):  # Output height
                    for j in range(26):  # Output width
                        # Extract 3×3 patch from input
                        patch = image[i:i+3, j:j+3]                        
                        # Convolution operation: element-wise multiply and sum
                        conv_result = np.sum(patch * kernel)
                        output[i, j, out_ch] = conv_result
                        worker_ops += 9  # 3×3=9 operations per convolution
            
            worker_time = time.time() - worker_start
            
            # Create worker state
            state = WorkerState(
                worker_id=worker_id,
                task_description=f"Convolution channels {start_channel}-{end_channel-1}",
                input_data_shape=image.shape,
                output_data_shape=(26, 26, worker_channels),
                processing_time=worker_time * 1000,  # Convert to ms
                memory_used=26 * 26 * worker_channels * 4,  # float32
                operations_count=worker_ops
            )
            
            print(f"   Worker {worker_id}: {worker_ops:,} operations in {worker_time*1000:.1f}ms")
            
            return state
        
        # Execute convolution in parallel
        start_time = time.time()
        
        with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
            futures = []
            
            for worker_id in range(self.num_workers):
                start_ch = worker_id * channels_per_worker
                end_ch = start_ch + channels_per_worker
                if worker_id == self.num_workers - 1:  # Last worker gets remainder
                    end_ch += remainder_channels
                
                future = executor.submit(process_channels, worker_id, start_ch, end_ch)
                futures.append(future)
            
            # Collect results
            for future in as_completed(futures):
                worker_state = future.result()
                worker_states.append(worker_state)
        
        total_time = time.time() - start_time
        
        # Results consolidation (no explicit gather needed - shared memory)
        print(f"\n📋 Results Consolidation:")
        print(f"   All workers wrote to shared output array")
        print(f"   No explicit gather operation needed")
        print(f"   Total parallel time: {total_time*1000:.1f}ms")
        
        # Show output sample
        print(f"\n📊 Output Analysis:")
        print(f"   Output shape: {output.shape}")
        print(f"   Sample values (channel 0, top-left 5×5):")
        for i in range(5):
            row = "      " + " ".join([f"{output[i,j,0]:6.3f}" for j in range(5)])
            print(row)
        
        # Track layer distribution
        #  Run explanations/parallelism/print_layer_distribution.py to see example output
        layer_dist = LayerDistribution(
            layer_name="Conv1",
            input_shape=image.shape,
            output_shape=output.shape,
            total_operations=sum(ws.operations_count for ws in worker_states),
            num_workers=self.num_workers,
            worker_states=worker_states,
            consolidation_time=0  # No explicit consolidation needed
        )
        self.layer_distributions.append(layer_dist)
        
        return output
    
    def demonstrate_step2_maxpooling(self, conv_output: np.ndarray):
        """Step 2: MaxPooling 26×26×32 → 13×13×32 with detailed parallelism"""
        
        print("\n" + "="*80)
        print("🔹 STEP 2: MAX POOLING LAYER - 26×26×32 → 13×13×32")
        print("="*80)
        
        print(f"📊 MaxPool Setup:")
        print(f"   Input shape: {conv_output.shape}")
        print(f"   Pool size: 2×2")
        print(f"   Stride: 2")
        print(f"   Output shape will be: (13, 13, 32)")
        
        # Distribute work: Each worker handles different channels
        channels_per_worker = 32 // self.num_workers
        remainder_channels = 32 % self.num_workers
        
        output = np.zeros((13, 13, 32))
        worker_states = []
        
        print(f"\n🔄 Work Distribution (by channels):")
        
        def process_pooling_channels(worker_id: int, start_channel: int, end_channel: int):
            """Each worker handles pooling for a subset of channels"""
            
            worker_start = time.time()
            worker_ops = 0
            worker_channels = end_channel - start_channel
            
            print(f"   Worker {worker_id}: Pooling channels {start_channel}-{end_channel-1}")
            
            for ch in range(start_channel, end_channel):
                # Process each 2×2 region
                for i in range(13):  # Output height
                    for j in range(13):  # Output width
                        # Extract 2×2 region from input
                        i_start, i_end = i*2, i*2+2
                        j_start, j_end = j*2, j*2+2
                        
                        region = conv_output[i_start:i_end, j_start:j_end, ch]
                        
                        # Max pooling operation
                        max_val = np.max(region)
                        output[i, j, ch] = max_val
                        worker_ops += 4  # Compare 4 values to find max
            
            worker_time = time.time() - worker_start
            
            state = WorkerState(
                worker_id=worker_id,
                task_description=f"MaxPool channels {start_channel}-{end_channel-1}",
                input_data_shape=(26, 26, worker_channels),
                output_data_shape=(13, 13, worker_channels),
                processing_time=worker_time * 1000,
                memory_used=13 * 13 * worker_channels * 4,
                operations_count=worker_ops
            )
            
            print(f"   Worker {worker_id}: {worker_ops:,} operations in {worker_time*1000:.1f}ms")
            return state
        
        # Execute pooling in parallel
        start_time = time.time()
        
        with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
            futures = []
            
            for worker_id in range(self.num_workers):
                start_ch = worker_id * channels_per_worker
                end_ch = start_ch + channels_per_worker
                if worker_id == self.num_workers - 1:
                    end_ch += remainder_channels
                
                future = executor.submit(process_pooling_channels, worker_id, start_ch, end_ch)
                futures.append(future)
            
            for future in as_completed(futures):
                worker_state = future.result()
                worker_states.append(worker_state)
        
        total_time = time.time() - start_time
        
        print(f"\n📋 Results Consolidation:")
        print(f"   Each worker wrote to non-overlapping channels")
        print(f"   No conflicts or explicit synchronization needed")
        print(f"   Total parallel time: {total_time*1000:.1f}ms")
        
        # Show reduction in spatial dimensions
        print(f"\n📊 Spatial Reduction Analysis:")
        print(f"   Input spatial size: 26×26 = 676 pixels per channel")
        print(f"   Output spatial size: 13×13 = 169 pixels per channel") 
        print(f"   Reduction factor: {676/169:.1f}x smaller")
        
        # Show sample before/after for one channel
        print(f"\n📸 Channel 0 Comparison (top-left 6×6 → 3×3):")
        print("   Before pooling:")
        for i in range(6):
            row = "      " + " ".join([f"{conv_output[i,j,0]:5.2f}" for j in range(6)])
            print(row)
        print("   After pooling:")
        for i in range(3):
            row = "      " + " ".join([f"{output[i,j,0]:5.2f}" for j in range(3)])
            print(row)
        
        layer_dist = LayerDistribution(
            layer_name="MaxPool1",
            input_shape=conv_output.shape,
            output_shape=output.shape,
            total_operations=sum(ws.operations_count for ws in worker_states),
            num_workers=self.num_workers,
            worker_states=worker_states,
            consolidation_time=0
        )
        self.layer_distributions.append(layer_dist)
        
        return output
    
    def demonstrate_step3_gelu(self, pool_output: np.ndarray):
        """Step 3: GELU Activation with element-wise parallelism"""
        
        print("\n" + "="*80)
        print("🔸 STEP 3: GELU ACTIVATION - Element-wise on 13×13×32")
        print("="*80)
        
        total_elements = np.prod(pool_output.shape)
        print(f"📊 GELU Setup:")
        print(f"   Input shape: {pool_output.shape}")
        print(f"   Total elements: {total_elements:,}")
        print(f"   Operation: x * 0.5 * (1 + tanh(√(2/π) * (x + 0.044715 * x³)))")
        
        # Distribute work: Each worker handles different elements
        elements_per_worker = total_elements // self.num_workers
        remainder_elements = total_elements % self.num_workers
        
        # Flatten for easier distribution
        flat_input = pool_output.flatten()
        flat_output = np.zeros_like(flat_input)
        worker_states = []
        
        print(f"\n🔄 Work Distribution (by elements):")
        
        def process_gelu_elements(worker_id: int, start_idx: int, end_idx: int):
            """Each worker processes a chunk of elements"""
            
            worker_start = time.time()
            worker_elements = end_idx - start_idx
            
            print(f"   Worker {worker_id}: Elements {start_idx:,}-{end_idx-1:,} ({worker_elements:,} elements)")
            
            # GELU formula breakdown for demonstration
            chunk = flat_input[start_idx:end_idx]
            
            # Step by step GELU computation (usually vectorized)
            x_cubed = chunk ** 3
            inner_term = chunk + 0.044715 * x_cubed
            sqrt_2_pi = np.sqrt(2 / np.pi)
            tanh_input = sqrt_2_pi * inner_term
            tanh_result = np.tanh(tanh_input)
            gelu_result = chunk * 0.5 * (1 + tanh_result)
            
            flat_output[start_idx:end_idx] = gelu_result
            
            worker_time = time.time() - worker_start
            worker_ops = worker_elements * 7  # Approximate ops per GELU
            
            state = WorkerState(
                worker_id=worker_id,
                task_description=f"GELU elements {start_idx:,}-{end_idx-1:,}",
                input_data_shape=(worker_elements,),
                output_data_shape=(worker_elements,),
                processing_time=worker_time * 1000,
                memory_used=worker_elements * 8,  # input + output
                operations_count=worker_ops
            )
            
            print(f"   Worker {worker_id}: {worker_ops:,} operations in {worker_time*1000:.1f}ms")
            return state
        
        # Execute GELU in parallel
        start_time = time.time()
        
        with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
            futures = []
            
            for worker_id in range(self.num_workers):
                start_idx = worker_id * elements_per_worker
                end_idx = start_idx + elements_per_worker
                if worker_id == self.num_workers - 1:
                    end_idx += remainder_elements
                
                future = executor.submit(process_gelu_elements, worker_id, start_idx, end_idx)
                futures.append(future)
            
            for future in as_completed(futures):
                worker_state = future.result()
                worker_states.append(worker_state)
        
        total_time = time.time() - start_time
        
        # Reshape back to original shape
        output = flat_output.reshape(pool_output.shape)
        
        print(f"\n📋 Results Consolidation:")
        print(f"   Each worker processed non-overlapping elements")
        print(f"   Flattened array reassembled to original shape")
        print(f"   Total parallel time: {total_time*1000:.1f}ms")
        
        # Show GELU effect on sample values
        print(f"\n📊 GELU Transformation (channel 0, position [0,0]):")
        sample_input = pool_output[0, 0, 0]
        sample_output = output[0, 0, 0]
        print(f"   Input value: {sample_input:.6f}")
        print(f"   GELU output: {sample_output:.6f}")
        print(f"   Transformation ratio: {sample_output/sample_input:.3f}")
        
        layer_dist = LayerDistribution(
            layer_name="GELU1",
            input_shape=pool_output.shape,
            output_shape=output.shape,
            total_operations=sum(ws.operations_count for ws in worker_states),
            num_workers=self.num_workers,
            worker_states=worker_states,
            consolidation_time=0
        )
        self.layer_distributions.append(layer_dist)
        
        return output
    
    def demonstrate_step4_linear(self, flattened_input: np.ndarray):
        """Step 4: Linear layer 128→512 with matrix parallelism"""
        
        print("\n" + "="*80)
        print("🔗 STEP 4: LINEAR LAYER - 128 → 512")
        print("="*80)
        
        # Create weight matrix and bias
        weights = np.random.randn(128, 512) * 0.1
        bias = np.random.randn(512) * 0.01
        
        print(f"📊 Linear Layer Setup:")
        print(f"   Input shape: {flattened_input.shape}")
        print(f"   Weights shape: {weights.shape}")
        print(f"   Bias shape: {bias.shape}")
        print(f"   Output shape will be: (512,)")
        print(f"   Operation: output = input @ weights + bias")
        
        # Strategy 1: Parallel by output features
        features_per_worker = 512 // self.num_workers
        remainder_features = 512 % self.num_workers
        
        output = np.zeros(512)
        worker_states = []
        
        print(f"\n🔄 Work Distribution (by output features):")
        
        def process_linear_features(worker_id: int, start_feat: int, end_feat: int):
            """Each worker computes a subset of output features"""
            
            worker_start = time.time()
            worker_features = end_feat - start_feat
            
            print(f"   Worker {worker_id}: Features {start_feat}-{end_feat-1} ({worker_features} features)")
            
            # Extract relevant weights and bias for this worker
            worker_weights = weights[:, start_feat:end_feat]  # 128 × worker_features
            worker_bias = bias[start_feat:end_feat]
            
            # Matrix multiplication for this subset
            worker_output = flattened_input @ worker_weights + worker_bias
            
            # Store results
            output[start_feat:end_feat] = worker_output
            
            worker_time = time.time() - worker_start
            worker_ops = 128 * worker_features  # Multiply-accumulate operations
            
            state = WorkerState(
                worker_id=worker_id,
                task_description=f"Linear features {start_feat}-{end_feat-1}",
                input_data_shape=flattened_input.shape,
                output_data_shape=(worker_features,),
                processing_time=worker_time * 1000,
                memory_used=(128 * worker_features + worker_features) * 4,
                operations_count=worker_ops
            )
            
            print(f"   Worker {worker_id}: {worker_ops:,} operations in {worker_time*1000:.1f}ms")
            return state, worker_output
        
        # Execute linear layer in parallel
        start_time = time.time()
        
        with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
            futures = []
            
            for worker_id in range(self.num_workers):
                start_feat = worker_id * features_per_worker
                end_feat = start_feat + features_per_worker
                if worker_id == self.num_workers - 1:
                    end_feat += remainder_features
                
                future = executor.submit(process_linear_features, worker_id, start_feat, end_feat)
                futures.append(future)
            
            # Collect results and combine
            partial_results = []
            for future in as_completed(futures):
                worker_state, partial_output = future.result()
                worker_states.append(worker_state)
                partial_results.append((worker_state.task_description, partial_output))
        
        total_time = time.time() - start_time
        
        print(f"\n📋 Results Consolidation:")
        print(f"   Each worker computed different output features")
        print(f"   Results concatenated to form complete output vector")
        print(f"   Total parallel time: {total_time*1000:.1f}ms")
        
        # Show matrix multiplication breakdown
        print(f"\n🔢 Matrix Operation Breakdown:")
        print(f"   Input vector: 128 elements")
        print(f"   Weight matrix: 128×512 = 65,536 parameters")
        print(f"   Bias vector: 512 elements")
        print(f"   Total operations: 128×512 = 65,536 multiply-adds")
        
        # Show sample outputs
        print(f"\n📊 Sample Results:")
        print(f"   Input sample (first 5): {flattened_input[:5]}")
        print(f"   Output sample (first 5): {output[:5]}")
        print(f"   Output range: [{np.min(output):.3f}, {np.max(output):.3f}]")
        
        layer_dist = LayerDistribution(
            layer_name="Linear1",
            input_shape=flattened_input.shape,
            output_shape=output.shape,
            total_operations=sum(ws.operations_count for ws in worker_states),
            num_workers=self.num_workers,
            worker_states=worker_states,
            consolidation_time=0
        )
        self.layer_distributions.append(layer_dist)
        
        return output
    
    def show_complete_data_flow_summary(self):
        """Show complete summary of data distribution and consolidation"""
        
        print("\n" + "="*80)
        print("📊 COMPLETE DATA FLOW SUMMARY")
        print("="*80)
        
        total_ops = 0
        total_time = 0
        
        for i, layer_dist in enumerate(self.layer_distributions):
            print(f"\n{i+1}. {layer_dist.layer_name}:")
            print(f"   Input: {layer_dist.input_shape} → Output: {layer_dist.output_shape}")
            print(f"   Workers: {layer_dist.num_workers}")
            print(f"   Total operations: {layer_dist.total_operations:,}")
            
            # Worker breakdown
            for ws in layer_dist.worker_states:
                print(f"     Worker {ws.worker_id}: {ws.operations_count:,} ops, {ws.processing_time:.1f}ms")
            
            layer_time = sum(ws.processing_time for ws in layer_dist.worker_states) / len(layer_dist.worker_states)
            total_ops += layer_dist.total_operations
            total_time += layer_time
        
        print(f"\n🎯 Overall Summary:")
        print(f"   Total operations across all layers: {total_ops:,}")
        print(f"   Total processing time: {total_time:.1f}ms")
        print(f"   Average operations per ms: {total_ops/total_time:,.0f}")
        
        print(f"\n🔄 Data Distribution Strategies Used:")
        print(f"   ✅ Channel parallelism (Convolution)")
        print(f"   ✅ Spatial parallelism (MaxPooling)")
        print(f"   ✅ Element parallelism (GELU)")
        print(f"   ✅ Feature parallelism (Linear)")
        
        print(f"\n📋 Consolidation Methods:")
        print(f"   ✅ Shared memory arrays (no explicit gather)")
        print(f"   ✅ Non-overlapping writes (no conflicts)")
        print(f"   ✅ Direct assignment to output regions")
        print(f"   ✅ Automatic synchronization via thread completion")

def main():
    """Main demonstration of step-by-step CNN parallelism"""
    
    print("🧠 Step-by-Step CNN Parallelism: 28×28 Image Distribution Example")
    print("="*80)
    print("This example shows EXACTLY how data flows through parallel CNN processing")
    print("="*80)
    
    # Create demo instance
    demo = DetailedCNNParallelismDemo(num_workers=4)
    
    # Create sample 28×28 image
    print("\n📸 Creating sample 28×28 image (letter 'A')...")
    image = demo.create_sample_image()
    
    print(f"\n📊 Initial Data:")
    print(f"   Image shape: {image.shape}")
    print(f"   Image memory: {image.nbytes} bytes")
    print(f"   Data type: {image.dtype}")
    
    # Step-by-step processing
    conv_output = demo.demonstrate_step1_convolution(image)
    pool_output = demo.demonstrate_step2_maxpooling(conv_output)
    gelu_output = demo.demonstrate_step3_gelu(pool_output)
    
    # Flatten for linear layer (simulating end of conv layers)
    flattened = gelu_output.flatten()[:128]  # Take first 128 elements
    print(f"\n📏 Flattening for linear layers: {gelu_output.shape} → {flattened.shape}")
    
    linear_output = demo.demonstrate_step4_linear(flattened)
    
    # Complete summary
    demo.show_complete_data_flow_summary()
    
    print(f"\n✅ Demonstration Complete!")
    print(f"   Final output shape: {linear_output.shape}")
    print(f"   Ready for next layer or final classification")

if __name__ == "__main__":
    main()