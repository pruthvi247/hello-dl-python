#!/usr/bin/env python3
"""
Translation Invariance Demo - Quick Test

A simplified version of the translation invariance demo that can be run
without interactive input. This version generates a single visualization
showing how translation affects different CNN processing stages.

Usage:
    python translation_invariance_quick_test.py
"""

import sys
import os

# Add the src directory to the path to import pytensorlib
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from translation_invariance_demo import TranslationInvarianceDemo


def quick_test():
    """Run a quick test of the translation invariance demo"""
    print("Translation Invariance Quick Test")
    print("=" * 50)
    
    # Create demo instance
    demo = TranslationInvarianceDemo(image_size=24)
    
    # Run with a cross pattern and moderate shifts
    shifts = [(0, 0), (3, 3), (6, 6)]
    results = demo.run_demo(pattern_type='cross', shifts=shifts)
    
    # Print summary
    print("\nSummary of Translation Invariance:")
    print("-" * 40)
    
    if len(results['similarities']['pooled']) > 0:
        avg_conv_sim = sum(results['similarities']['conv']) / len(results['similarities']['conv'])
        avg_edge_sim = sum(results['similarities']['edge']) / len(results['similarities']['edge'])
        avg_pooled_sim = sum(results['similarities']['pooled']) / len(results['similarities']['pooled'])
        
        print(f"Average Convolution Similarity: {avg_conv_sim:.3f}")
        print(f"Average Edge Detection Similarity: {avg_edge_sim:.3f}")
        print(f"Average Pooling Similarity: {avg_pooled_sim:.3f}")
        print(f"Invariance Improvement (Pooling vs Conv): {avg_pooled_sim - avg_conv_sim:.3f}")
        
        if avg_pooled_sim > avg_conv_sim:
            print("✅ Pooling provides better translation invariance!")
        else:
            print("❌ Unexpected result - check implementation")
    
    print("\nTest completed successfully!")
    return True


if __name__ == "__main__":
    try:
        quick_test()
    except Exception as e:
        print(f"Error running quick test: {e}")
        sys.exit(1)