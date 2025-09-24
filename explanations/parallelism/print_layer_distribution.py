#!/usr/bin/env python3
"""
Print Sample Layer Distribution Content
====================================

This script demonstrates the layer distribution data structure and content
from the CNN parallelism example.
"""

import sys
import os

# Add the current directory to path so we can import the demo
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from step_by_step_cnn_parallelism import DetailedCNNParallelismDemo, LayerDistribution, WorkerState
import numpy as np
import json

def print_layer_distribution_structure():
    """Print the structure and sample content of LayerDistribution"""
    
    print("📋 LayerDistribution Data Structure")
    print("="*50)
    
    # Create a sample layer distribution
    sample_worker_states = [
        WorkerState(
            worker_id=0,
            task_description="Conv channels 0-7",
            input_data_shape=(28, 28, 1),
            output_data_shape=(26, 26, 8),
            processing_time=12.5,
            memory_used=21632,
            operations_count=48672
        ),
        WorkerState(
            worker_id=1,
            task_description="Conv channels 8-15", 
            input_data_shape=(28, 28, 1),
            output_data_shape=(26, 26, 8),
            processing_time=11.8,
            memory_used=21632,
            operations_count=48672
        ),
        WorkerState(
            worker_id=2,
            task_description="Conv channels 16-23",
            input_data_shape=(28, 28, 1),
            output_data_shape=(26, 26, 8),
            processing_time=13.2,
            memory_used=21632,
            operations_count=48672
        ),
        WorkerState(
            worker_id=3,
            task_description="Conv channels 24-31",
            input_data_shape=(28, 28, 1),
            output_data_shape=(26, 26, 8),
            processing_time=12.1,
            memory_used=21632,
            operations_count=48672
        )
    ]
    
    sample_layer_dist = LayerDistribution(
        layer_name="Convolution_1",
        input_shape=(28, 28, 1),
        output_shape=(26, 26, 32),
        total_operations=194688,  # 48672 × 4 workers
        num_workers=4,
        worker_states=sample_worker_states,
        consolidation_time=0.5
    )
    
    print(f"Layer Name: {sample_layer_dist.layer_name}")
    print(f"Input Shape: {sample_layer_dist.input_shape}")
    print(f"Output Shape: {sample_layer_dist.output_shape}")
    print(f"Total Operations: {sample_layer_dist.total_operations:,}")
    print(f"Number of Workers: {sample_layer_dist.num_workers}")
    print(f"Consolidation Time: {sample_layer_dist.consolidation_time}ms")
    
    print(f"\n📊 Worker States:")
    print(f"   {'Worker':<8} {'Task':<20} {'Input Shape':<15} {'Output Shape':<15} {'Time(ms)':<10} {'Memory':<10} {'Operations':<12}")
    print(f"   {'-'*8} {'-'*20} {'-'*15} {'-'*15} {'-'*10} {'-'*10} {'-'*12}")
    
    for ws in sample_layer_dist.worker_states:
        print(f"   {ws.worker_id:<8} {ws.task_description:<20} {str(ws.input_data_shape):<15} {str(ws.output_data_shape):<15} {ws.processing_time:<10.1f} {ws.memory_used:<10,} {ws.operations_count:<12,}")
    
    print(f"\n🔍 Individual Worker State Details:")
    for i, ws in enumerate(sample_layer_dist.worker_states):
        print(f"\n   Worker {ws.worker_id}:")
        print(f"     Task: {ws.task_description}")
        print(f"     Input Shape: {ws.input_data_shape}")
        print(f"     Output Shape: {ws.output_data_shape}")
        print(f"     Processing Time: {ws.processing_time:.1f} ms")
        print(f"     Memory Used: {ws.memory_used:,} bytes ({ws.memory_used/1024:.1f} KB)")
        print(f"     Operations Count: {ws.operations_count:,}")
        print(f"     Operations/ms: {ws.operations_count/ws.processing_time:,.0f}")

def print_complete_pipeline_distribution():
    """Show the complete pipeline distribution"""
    
    print(f"\n🚀 Complete CNN Pipeline Distribution")
    print("="*50)
    
    # Sample complete pipeline
    layers = [
        {
            'name': 'Convolution_1',
            'input_shape': (28, 28, 1),
            'output_shape': (26, 26, 32),
            'strategy': 'Channel Parallelism',
            'workers_distribution': '4 workers × 8 channels each',
            'operations_per_worker': 48672,
            'total_operations': 194688,
            'avg_time_ms': 12.4
        },
        {
            'name': 'MaxPooling_1', 
            'input_shape': (26, 26, 32),
            'output_shape': (13, 13, 32),
            'strategy': 'Channel Parallelism',
            'workers_distribution': '4 workers × 8 channels each',
            'operations_per_worker': 5408,
            'total_operations': 21632,
            'avg_time_ms': 3.2
        },
        {
            'name': 'GELU_Activation',
            'input_shape': (13, 13, 32),
            'output_shape': (13, 13, 32),
            'strategy': 'Element Parallelism',
            'workers_distribution': '4 workers × 1352 elements each',
            'operations_per_worker': 9464,  # Including exp, tanh calculations
            'total_operations': 37856,
            'avg_time_ms': 0.9
        },
        {
            'name': 'Linear_Layer',
            'input_shape': (5408,),  # Flattened
            'output_shape': (128,),
            'strategy': 'Feature Parallelism',
            'workers_distribution': '4 workers × 32 output features each',
            'operations_per_worker': 16384,  # 5408 × 32 multiply-adds
            'total_operations': 65536,
            'avg_time_ms': 1.5
        }
    ]
    
    print(f"   {'Layer':<16} {'Strategy':<18} {'Distribution':<28} {'Ops/Worker':<12} {'Total Ops':<12} {'Time(ms)':<10}")
    print(f"   {'-'*16} {'-'*18} {'-'*28} {'-'*12} {'-'*12} {'-'*10}")
    
    total_ops = 0
    total_time = 0
    
    for layer in layers:
        print(f"   {layer['name']:<16} {layer['strategy']:<18} {layer['workers_distribution']:<28} {layer['operations_per_worker']:<12,} {layer['total_operations']:<12,} {layer['avg_time_ms']:<10.1f}")
        total_ops += layer['total_operations']
        total_time += layer['avg_time_ms']
    
    print(f"   {'-'*16} {'-'*18} {'-'*28} {'-'*12} {'-'*12} {'-'*10}")
    print(f"   {'TOTALS':<16} {'All Strategies':<18} {'4 workers total':<28} {'':<12} {total_ops:<12,} {total_time:<10.1f}")
    
    print(f"\n📈 Performance Metrics:")
    print(f"   Total Operations: {total_ops:,}")
    print(f"   Total Time: {total_time:.1f} ms")
    print(f"   Average Throughput: {total_ops/total_time:,.0f} operations/ms")
    print(f"   Parallel Efficiency: ~75% (estimated)")

def run_actual_demo_and_show_distribution():
    """Run the actual demo and show real distribution data"""
    
    print(f"\n🎯 Running Actual Demo to Show Real Distribution")
    print("="*55)
    print("(This will create and process a real 28×28 image)")
    
    # Create demo instance
    demo = DetailedCNNParallelismDemo(num_workers=4)
    
    # Create and process image
    image = demo.create_sample_image()
    print(f"\n📸 Created {image.shape} image")
    
    # Process first layer only to show real distribution
    conv_output = demo.demonstrate_step1_convolution(image)
    
    # Access the layer distribution data
    if demo.layer_distributions:
        layer_dist = demo.layer_distributions[0]  # First layer
        
        print(f"\n📊 Real Layer Distribution Data:")
        print(f"   Layer: {layer_dist.layer_name}")
        print(f"   Input: {layer_dist.input_shape} → Output: {layer_dist.output_shape}")
        print(f"   Total Operations: {layer_dist.total_operations:,}")
        print(f"   Workers: {layer_dist.num_workers}")
        
        print(f"\n🔍 Real Worker Performance:")
        for ws in layer_dist.worker_states:
            efficiency = ws.operations_count / ws.processing_time if ws.processing_time > 0 else 0
            print(f"   Worker {ws.worker_id}: {ws.operations_count:,} ops in {ws.processing_time:.1f}ms ({efficiency:,.0f} ops/ms)")
        
        return layer_dist
    else:
        print("   No distribution data available")
        return None

def main():
    """Print all layer distribution examples"""
    
    print("🗂️  CNN Layer Distribution Content Examples")
    print("="*60)
    
    # 1. Show the data structure
    print_layer_distribution_structure()
    
    # 2. Show complete pipeline
    print_complete_pipeline_distribution()
    
    # 3. Run actual demo (optional - comment out if you just want structure)
    print(f"\n❓ Would you like to run the actual demo? (Takes ~2 seconds)")
    print("   Uncomment the line below in the script to run it:")
    print("   # run_actual_demo_and_show_distribution()")
    
    # Uncomment to run actual demo:
    # run_actual_demo_and_show_distribution()

if __name__ == "__main__":
    main()