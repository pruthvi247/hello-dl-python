#!/usr/bin/env python3
"""
Translation Invariance Visual Demo

This demo shows how shifting (translating) an input image affects different stages
of CNN processing:
1. Raw image (sensitive to translation)
2. Convolution output (slightly less sensitive)
3. Edge detection (detects features regardless of position)
4. Pooled output (most translation invariant)

The demo creates side-by-side visualizations to demonstrate how pooling operations
provide translation invariance - the ability to recognize features regardless of
their position in the image.

Dependencies:
- numpy
- matplotlib 
- PIL (Pillow)
- pytensorlib

Usage:
    python translation_invariance_demo.py
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import sys
import os

# Add the src directory to the path to import pytensorlib
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

try:
    from pytensorlib import Tensor
except ImportError:
    print("Error: Could not import pytensorlib. Make sure you're in the correct environment.")
    print("Run: source pytensor-env/bin/activate")
    sys.exit(1)


class TranslationInvarianceDemo:
    """
    Visual demonstration of translation invariance in CNNs
    """
    
    def __init__(self, image_size=32):
        """
        Initialize the demo
        
        Args:
            image_size (int): Size of the square images to use
        """
        self.image_size = image_size
        self.setup_filters()
    
    def setup_filters(self):
        """Setup convolution filters for edge detection and feature detection"""
        
        # Vertical edge detection filter (Sobel-like)
        self.vertical_edge_filter = np.array([
            [-1, 0, 1],
            [-2, 0, 2],
            [-1, 0, 1]
        ], dtype=np.float32) / 8.0
        
        # Horizontal edge detection filter
        self.horizontal_edge_filter = np.array([
            [-1, -2, -1],
            [ 0,  0,  0],
            [ 1,  2,  1]
        ], dtype=np.float32) / 8.0
        
        # Feature detection filter (corner/blob detector)
        self.feature_filter = np.array([
            [-1, -1, -1],
            [-1,  8, -1],
            [-1, -1, -1]
        ], dtype=np.float32) / 8.0
    
    def create_test_image(self, pattern_type='cross'):
        """
        Create a test image with a clear pattern
        
        Args:
            pattern_type (str): Type of pattern ('cross', 'square', 'L', 'circle')
            
        Returns:
            np.ndarray: Test image
        """
        img = np.zeros((self.image_size, self.image_size), dtype=np.float32)
        center = self.image_size // 2
        
        if pattern_type == 'cross':
            # Create a cross pattern
            img[center-1:center+2, center-8:center+9] = 1.0  # Horizontal bar
            img[center-8:center+9, center-1:center+2] = 1.0  # Vertical bar
            
        elif pattern_type == 'square':
            # Create a square pattern
            size = 8
            start = center - size // 2
            end = center + size // 2
            img[start:end, start:end] = 1.0
            img[start+2:end-2, start+2:end-2] = 0.0  # Hollow square
            
        elif pattern_type == 'L':
            # Create an L-shaped pattern
            img[center-6:center+2, center-2:center+1] = 1.0  # Vertical part
            img[center-1:center+2, center-2:center+6] = 1.0  # Horizontal part
            
        elif pattern_type == 'circle':
            # Create a circle pattern
            y, x = np.ogrid[:self.image_size, :self.image_size]
            distance = np.sqrt((x - center)**2 + (y - center)**2)
            img[(distance >= 6) & (distance <= 8)] = 1.0
        
        return img
    
    def translate_image(self, image, dx, dy):
        """
        Translate (shift) an image by dx, dy pixels
        
        Args:
            image (np.ndarray): Input image
            dx (int): Horizontal shift
            dy (int): Vertical shift
            
        Returns:
            np.ndarray: Translated image
        """
        translated = np.zeros_like(image)
        h, w = image.shape
        
        # Calculate valid regions
        src_y_start = max(0, -dy)
        src_y_end = min(h, h - dy)
        src_x_start = max(0, -dx)
        src_x_end = min(w, w - dx)
        
        dst_y_start = max(0, dy)
        dst_y_end = dst_y_start + (src_y_end - src_y_start)
        dst_x_start = max(0, dx)
        dst_x_end = dst_x_start + (src_x_end - src_x_start)
        
        if src_y_end > src_y_start and src_x_end > src_x_start:
            translated[dst_y_start:dst_y_end, dst_x_start:dst_x_end] = \
                image[src_y_start:src_y_end, src_x_start:src_x_end]
        
        return translated
    
    def convolve_2d(self, image, kernel):
        """
        Perform 2D convolution
        
        Args:
            image (np.ndarray): Input image
            kernel (np.ndarray): Convolution kernel
            
        Returns:
            np.ndarray: Convolved image
        """
        img_h, img_w = image.shape
        kernel_h, kernel_w = kernel.shape
        
        # Calculate output size
        out_h = img_h - kernel_h + 1
        out_w = img_w - kernel_w + 1
        
        output = np.zeros((out_h, out_w), dtype=np.float32)
        
        for i in range(out_h):
            for j in range(out_w):
                output[i, j] = np.sum(image[i:i+kernel_h, j:j+kernel_w] * kernel)
        
        return output
    
    def max_pool_2d(self, image, pool_size=2, stride=2):
        """
        Perform 2D max pooling
        
        Args:
            image (np.ndarray): Input image
            pool_size (int): Size of pooling window
            stride (int): Stride for pooling
            
        Returns:
            np.ndarray: Pooled image
        """
        img_h, img_w = image.shape
        
        # Calculate output size
        out_h = (img_h - pool_size) // stride + 1
        out_w = (img_w - pool_size) // stride + 1
        
        output = np.zeros((out_h, out_w), dtype=np.float32)
        
        for i in range(out_h):
            for j in range(out_w):
                y_start = i * stride
                y_end = y_start + pool_size
                x_start = j * stride
                x_end = x_start + pool_size
                
                output[i, j] = np.max(image[y_start:y_end, x_start:x_end])
        
        return output
    
    def compute_feature_similarity(self, original_features, shifted_features):
        """
        Compute similarity between feature maps
        
        Args:
            original_features (np.ndarray): Original feature map
            shifted_features (np.ndarray): Shifted feature map
            
        Returns:
            float: Cosine similarity score
        """
        # Flatten the arrays
        orig_flat = original_features.flatten()
        shift_flat = shifted_features.flatten()
        
        # Compute cosine similarity
        dot_product = np.dot(orig_flat, shift_flat)
        norm_product = np.linalg.norm(orig_flat) * np.linalg.norm(shift_flat)
        
        if norm_product == 0:
            return 0.0
        
        return dot_product / norm_product
    
    def run_demo(self, pattern_type='cross', shifts=[(0, 0), (3, 3), (6, 6), (-4, 2)]):
        """
        Run the complete translation invariance demo
        
        Args:
            pattern_type (str): Type of pattern to use
            shifts (list): List of (dx, dy) translation values
        """
        print(f"Translation Invariance Demo - {pattern_type.upper()} Pattern")
        print("=" * 60)
        
        # Create original image
        original_image = self.create_test_image(pattern_type)
        
        # Setup the figure
        n_shifts = len(shifts)
        fig, axes = plt.subplots(4, n_shifts, figsize=(4*n_shifts, 16))
        fig.suptitle(f'Translation Invariance Demo - {pattern_type.upper()} Pattern', 
                    fontsize=16, fontweight='bold')
        
        # Store results for similarity analysis
        results = {
            'raw_images': [],
            'conv_outputs': [],
            'edge_outputs': [],
            'pooled_outputs': [],
            'similarities': {'conv': [], 'edge': [], 'pooled': []}
        }
        
        for col, (dx, dy) in enumerate(shifts):
            # Translate the image
            if dx == 0 and dy == 0:
                shifted_image = original_image.copy()
                title_suffix = "(Original)"
            else:
                shifted_image = self.translate_image(original_image, dx, dy)
                title_suffix = f"(Shift: {dx:+d}, {dy:+d})"
            
            # 1. Raw image
            axes[0, col].imshow(shifted_image, cmap='viridis', vmin=0, vmax=1)
            axes[0, col].set_title(f'Raw Image {title_suffix}')
            axes[0, col].axis('off')
            
            # 2. Convolution output (feature detection)
            conv_output = self.convolve_2d(shifted_image, self.feature_filter)
            axes[1, col].imshow(conv_output, cmap='RdBu', vmin=-1, vmax=1)
            axes[1, col].set_title(f'Convolution Output {title_suffix}')
            axes[1, col].axis('off')
            
            # 3. Edge detection
            edge_output = np.abs(self.convolve_2d(shifted_image, self.vertical_edge_filter)) + \
                         np.abs(self.convolve_2d(shifted_image, self.horizontal_edge_filter))
            axes[2, col].imshow(edge_output, cmap='hot', vmin=0, vmax=np.max(edge_output))
            axes[2, col].set_title(f'Edge Detection {title_suffix}')
            axes[2, col].axis('off')
            
            # 4. Pooled output
            pooled_output = self.max_pool_2d(edge_output, pool_size=3, stride=2)
            axes[3, col].imshow(pooled_output, cmap='hot', vmin=0, vmax=np.max(pooled_output))
            axes[3, col].set_title(f'Max Pooled {title_suffix}')
            axes[3, col].axis('off')
            
            # Store results
            results['raw_images'].append(shifted_image)
            results['conv_outputs'].append(conv_output)
            results['edge_outputs'].append(edge_output)
            results['pooled_outputs'].append(pooled_output)
            
            # Compute similarities with original (first image)
            if col > 0:
                conv_sim = self.compute_feature_similarity(results['conv_outputs'][0], conv_output)
                edge_sim = self.compute_feature_similarity(results['edge_outputs'][0], edge_output)
                pooled_sim = self.compute_feature_similarity(results['pooled_outputs'][0], pooled_output)
                
                results['similarities']['conv'].append(conv_sim)
                results['similarities']['edge'].append(edge_sim)
                results['similarities']['pooled'].append(pooled_sim)
                
                print(f"Shift ({dx:+2d}, {dy:+2d}) - Conv: {conv_sim:.3f}, Edge: {edge_sim:.3f}, Pooled: {pooled_sim:.3f}")
        
        # Add row labels
        row_labels = ['Raw Image\n(Translation Sensitive)', 
                     'Convolution\n(Feature Detection)',
                     'Edge Detection\n(Less Sensitive)', 
                     'Max Pooling\n(Translation Invariant)']
        
        for i, label in enumerate(row_labels):
            axes[i, 0].set_ylabel(label, fontsize=12, fontweight='bold', rotation=0, 
                                 labelpad=100, ha='right', va='center')
        
        plt.tight_layout()
        plt.subplots_adjust(top=0.93, left=0.15)
        
        # Show similarity analysis
        if len(results['similarities']['conv']) > 0:
            self.plot_similarity_analysis(results['similarities'], shifts[1:])
        
        plt.show()
        
        return results
    
    def plot_similarity_analysis(self, similarities, shifts):
        """
        Plot similarity analysis showing translation invariance
        
        Args:
            similarities (dict): Similarity scores for each processing stage
            shifts (list): List of shift amounts
        """
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))
        
        shift_labels = [f"({dx:+d}, {dy:+d})" for dx, dy in shifts]
        x_pos = np.arange(len(shift_labels))
        
        width = 0.25
        
        conv_bars = ax.bar(x_pos - width, similarities['conv'], width, 
                          label='Convolution', alpha=0.8, color='blue')
        edge_bars = ax.bar(x_pos, similarities['edge'], width, 
                          label='Edge Detection', alpha=0.8, color='green')
        pooled_bars = ax.bar(x_pos + width, similarities['pooled'], width, 
                            label='Max Pooling', alpha=0.8, color='red')
        
        ax.set_xlabel('Translation Amount (dx, dy)')
        ax.set_ylabel('Cosine Similarity with Original')
        ax.set_title('Translation Invariance Analysis\n(Higher = More Invariant to Translation)')
        ax.set_xticks(x_pos)
        ax.set_xticklabels(shift_labels)
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 1)
        
        # Add value labels on bars
        for bars in [conv_bars, edge_bars, pooled_bars]:
            for bar in bars:
                height = bar.get_height()
                ax.annotate(f'{height:.2f}',
                           xy=(bar.get_x() + bar.get_width() / 2, height),
                           xytext=(0, 3),  # 3 points vertical offset
                           textcoords="offset points",
                           ha='center', va='bottom', fontsize=9)
        
        plt.tight_layout()
        plt.show()
    
    def run_multiple_patterns_demo(self):
        """Run demo with multiple patterns to show generalization"""
        patterns = ['cross', 'square', 'L', 'circle']
        shifts = [(0, 0), (4, 4), (8, -3)]
        
        for pattern in patterns:
            print(f"\n{'-'*60}")
            results = self.run_demo(pattern_type=pattern, shifts=shifts)
            
            # Brief analysis
            if len(results['similarities']['pooled']) > 0:
                avg_pooled_sim = np.mean(results['similarities']['pooled'])
                avg_conv_sim = np.mean(results['similarities']['conv'])
                print(f"Average similarity - Convolution: {avg_conv_sim:.3f}, Pooling: {avg_pooled_sim:.3f}")
                print(f"Translation invariance improvement: {avg_pooled_sim - avg_conv_sim:.3f}")


def main():
    """Main function to run the demo"""
    print("Translation Invariance Visual Demo")
    print("=" * 60)
    print("This demo shows how CNN operations handle image translations:")
    print("1. Raw images are sensitive to position")
    print("2. Convolution detects features but position matters")
    print("3. Edge detection is less position-sensitive")
    print("4. Pooling provides translation invariance")
    print("=" * 60)
    
    # Create demo instance
    demo = TranslationInvarianceDemo(image_size=32)
    
    # Run single pattern demo
    print("\n--- Single Pattern Demo (Cross) ---")
    demo.run_demo(pattern_type='cross', shifts=[(0, 0), (3, 3), (6, 6), (-4, 2)])
    
    # Ask user if they want to see more patterns
    try:
        response = input("\nWould you like to see demos for all patterns? (y/n): ").strip().lower()
        if response in ['y', 'yes']:
            print("\n--- Multiple Patterns Demo ---")
            demo.run_multiple_patterns_demo()
    except KeyboardInterrupt:
        print("\nDemo interrupted by user.")
    
    print("\n" + "="*60)
    print("Key Observations:")
    print("• Raw images: High sensitivity to translation")
    print("• Convolution: Detects features but position-dependent")
    print("• Edge detection: Less sensitive to exact position")
    print("• Max pooling: Provides translation invariance")
    print("• Higher similarity scores = better translation invariance")
    print("="*60)


if __name__ == "__main__":
    main()