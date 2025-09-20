#!/usr/bin/env python3
"""
Basic Tensor Operations Example

This script demonstrates basic tensor operations and automatic differentiation
with PyTensorLib.
"""

import sys
import os

# Add src directory to path for development
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from pytensorlib import Tensor, relu, sigmoid, tanh, square
import numpy as np


def basic_operations_demo():
    """Demonstrate basic tensor operations"""
    print("🔢 Basic Tensor Operations")
    print("=" * 40)
    
    # Create tensors
    print("1. Creating tensors:")
    a = Tensor(2, 2)
    a.constant(1.0)
    a[0, 1] = 2.0
    a[1, 0] = 3.0
    a[1, 1] = 4.0
    
    b = Tensor(2, 2)
    b.constant(2.0)
    b[1, 0] = 1.0
    
    print(f"   a = 2x2 matrix with values")
    print(f"   b = 2x2 matrix with values")
    
    # Basic arithmetic
    print("\n2. Arithmetic operations:")
    c = a + b
    print(f"   a + b = sum of matrices")
    
    d = a * b
    print(f"   a * b = element-wise multiplication")
    
    # Activation functions
    print("\n3. Activation functions:")
    f = relu(a)
    print(f"   relu(a) = applied to matrix")
    
    g = sigmoid(a)
    print(f"   sigmoid(a) = applied to matrix")
    
    return a, b, c, d, f, g


def gradient_demo():
    """Demonstrate automatic differentiation"""
    print("\n🎯 Automatic Differentiation")
    print("=" * 40)
    
    # Simple gradient example
    print("1. Simple gradient computation:")
    x = Tensor(1, 1)
    x.constant(2.0)
    y = square(x)  # y = x²
    
    print(f"   x = {x[0,0]}")
    print(f"   y = x² = {y[0,0]}")
    
    y.backward()
    grad = x.raw()  # Get gradient
    print(f"   dy/dx = {grad[0][0]} (expected: {2*x[0,0]})")
    
    # Multi-variable function
    print("\n2. Multi-variable gradients:")
    a = Tensor(1, 1)
    a.constant(1.0)
    b = Tensor(1, 1) 
    b.constant(2.0)
    
    # Create a simple function: z = a² + b²
    a_sq = square(a)
    b_sq = square(b)
    z = a_sq + b_sq
    
    print(f"   a = {a[0,0]}, b = {b[0,0]}")
    print(f"   z = a² + b² = {z[0,0]}")
    
    z.backward()
    grad_a = a.raw()
    grad_b = b.raw()
    print(f"   ∂z/∂a = {grad_a[0][0]} (expected: {2*a[0,0]})")
    print(f"   ∂z/∂b = {grad_b[0][0]} (expected: {2*b[0,0]})")


def matrix_operations_demo():
    """Demonstrate matrix operations with gradients"""
    print("\n📊 Matrix Operations with Gradients")
    print("=" * 40)
    
    # Matrix operations
    print("1. Matrix operations:")
    A = Tensor(2, 2)
    A.constant(1.0)
    A[0, 1] = 2.0
    A[1, 0] = 3.0
    A[1, 1] = 4.0
    
    B = Tensor(2, 2) 
    B.constant(2.0)
    B[1, 0] = 1.0
    B[1, 1] = 3.0
    
    C = A.dot(B)  # Matrix multiplication
    
    print(f"   A = 2x2 matrix")
    print(f"   B = 2x2 matrix") 
    print(f"   C = A • B (dot product)")
    
    # Sum for scalar output
    sum_c = C.sum()
    print(f"   sum(C) = {sum_c[0,0]}")
    
    sum_c.backward()
    print(f"   Gradients computed successfully ✓")


def activation_comparison():
    """Compare different activation functions"""
    print("\n⚡ Activation Functions Comparison")
    print("=" * 40)
    
    # Test different values
    test_values = [-2.0, -1.0, 0.0, 1.0, 2.0]
    
    print("Testing activation functions:")
    print(f"{'Value':<8} {'ReLU':<8} {'Sigmoid':<8} {'Tanh':<8}")
    print("-" * 32)
    
    for val in test_values:
        x = Tensor(1, 1)
        x.constant(val)
        
        relu_out = relu(x)
        sigmoid_out = sigmoid(x) 
        tanh_out = tanh(x)
        
        print(f"{val:<8.1f} {relu_out[0,0]:<8.3f} {sigmoid_out[0,0]:<8.3f} {tanh_out[0,0]:<8.3f}")
    
    print("\n✓ Activation functions working correctly")


def optimization_demo():
    """Demonstrate simple optimization"""
    print("\n🎯 Simple Optimization Example")
    print("=" * 40)
    
    print("Minimizing f(x) = (x - 3)²")
    print("Expected minimum at x = 3")
    print()
    
    # Starting point
    x = Tensor(1, 1)
    x.constant(0.0)
    learning_rate = 0.1
    
    print("Step | x      | f(x)   ")
    print("-" * 20)
    
    for step in range(15):
        # Compute function value: f(x) = (x - 3)²
        x_minus_3 = Tensor(1, 1)
        x_minus_3.constant(x[0,0] - 3.0)
        f = square(x_minus_3)
        
        # Print status
        if step % 3 == 0:
            print(f"{step:4d} | {x[0,0]:6.3f} | {f[0,0]:6.3f}")
        
        # Compute gradient manually for this simple case
        # f'(x) = 2(x - 3), so gradient = 2 * (x[0,0] - 3)
        gradient = 2 * (x[0,0] - 3.0)
        
        # Update x
        new_val = x[0,0] - learning_rate * gradient
        x.constant(new_val)
    
    print(f"\nFinal result: x = {x[0,0]:.6f}")
    
    # Final function value
    x_minus_3 = Tensor(1, 1)
    x_minus_3.constant(x[0,0] - 3.0)
    final_f = square(x_minus_3)
    print(f"Function value: f(x) = {final_f[0,0]:.6f}")


def main():
    """Run all demonstrations"""
    print("🧠 PyTensorLib Basic Examples")
    print("=" * 50)
    
    try:
        # Run demonstrations
        basic_operations_demo()
        gradient_demo()
        matrix_operations_demo()
        activation_comparison()
        optimization_demo()
        
        print("\n🎉 All examples completed successfully!")
        print("\nNext steps:")
        print("- Try the neural network example (examples/neural_network.py)")
        print("- Explore MNIST utilities (examples/mnist_demo.py)")
        print("- Check out the full API in the README")
        
    except Exception as e:
        print(f"\n❌ Error running examples: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)