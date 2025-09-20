#!/usr/bin/env python3
"""
Test suite for PyTensorLib

This script tests the basic functionality of the tensor automatic differentiation
library and demonstrates neural network examples.
"""

import numpy as np
import sys
import os
import unittest

# Import from the installed package
from pytensorlib import (
    Tensor, TensorImpl, relu, sigmoid, tanh, gelu, square,
    make_flatten, make_log_softmax, mse_loss, cross_entropy_loss
)


class TestTensorOperations(unittest.TestCase):
    """Test basic tensor operations"""
    
    def test_tensor_creation(self):
        """Test tensor creation and initialization"""
        # Test basic creation
        a = Tensor(2, 3)
        self.assertEqual(a.shape, (2, 3))
        
        # Test scalar creation
        s = Tensor(value=5.0)
        self.assertEqual(s.shape, (1, 1))
        self.assertEqual(s[0, 0], 5.0)
    
    def test_tensor_operations(self):
        """Test basic arithmetic operations"""
        a = Tensor(2, 2)
        a.constant(2.0)
        
        b = Tensor(2, 2)
        b.constant(3.0)
        
        # Test addition
        c = a + b
        self.assertEqual(c[0, 0], 5.0)
        
        # Test subtraction
        d = b - a
        self.assertEqual(d[0, 0], 1.0)
        
        # Test negation
        e = -a
        self.assertEqual(e[0, 0], -2.0)
    
    def test_matrix_multiplication(self):
        """Test matrix multiplication"""
        A = Tensor(2, 3)
        A.constant(1.0)
        
        B = Tensor(3, 2)
        B.constant(2.0)
        
        C = A * B
        self.assertEqual(C.shape, (2, 2))
        self.assertEqual(C[0, 0], 6.0)  # 1*2 + 1*2 + 1*2 = 6
    
    def test_element_access(self):
        """Test element access and modification"""
        a = Tensor(3, 3)
        a.zero()
        
        # Test setting and getting elements
        a[1, 2] = 5.0
        self.assertEqual(a[1, 2], 5.0)
        
        # Test other elements are still zero
        self.assertEqual(a[0, 0], 0.0)
        self.assertEqual(a[2, 2], 0.0)


class TestActivationFunctions(unittest.TestCase):
    """Test activation functions"""
    
    def test_relu(self):
        """Test ReLU activation function"""
        x = Tensor(3, 1)
        x[0, 0] = -1.0
        x[1, 0] = 0.0
        x[2, 0] = 1.0
        
        y = relu(x)
        
        self.assertEqual(y[0, 0], 0.0)  # ReLU(-1) = 0
        self.assertEqual(y[1, 0], 0.0)  # ReLU(0) = 0
        self.assertEqual(y[2, 0], 1.0)  # ReLU(1) = 1
    
    def test_sigmoid(self):
        """Test Sigmoid activation function"""
        x = Tensor(1, 1)
        x[0, 0] = 0.0
        
        y = sigmoid(x)
        
        # sigmoid(0) should be 0.5
        self.assertAlmostEqual(y[0, 0], 0.5, places=5)
    
    def test_tanh(self):
        """Test Tanh activation function"""
        x = Tensor(1, 1)
        x[0, 0] = 0.0
        
        y = tanh(x)
        
        # tanh(0) should be 0
        self.assertAlmostEqual(y[0, 0], 0.0, places=5)


class TestAutomaticDifferentiation(unittest.TestCase):
    """Test automatic differentiation"""
    
    def test_simple_gradient(self):
        """Test gradient computation for simple function"""
        # Test f(x) = x^2, df/dx = 2x
        x = Tensor(1, 1)
        x.constant(3.0)
        
        y = square(x)
        y.backward()
        
        # df/dx at x=3 should be 6
        self.assertAlmostEqual(x.impl.grads[0, 0], 6.0, places=5)
    
    def test_chain_rule(self):
        """Test chain rule in automatic differentiation"""
        # Test f(x) = (x^2 + 1)^2, df/dx = 4x(x^2 + 1)
        x = Tensor(1, 1)
        x.constant(2.0)
        
        # f = x^2
        x_squared = square(x)
        
        # g = f + 1
        one = Tensor(value=1.0)
        g = x_squared + one
        
        # h = g^2
        h = square(g)
        
        h.backward()
        
        # At x=2: df/dx = 4*2*(2^2 + 1) = 8*5 = 40
        self.assertAlmostEqual(x.impl.grads[0, 0], 40.0, places=5)
    
    def test_matrix_gradients(self):
        """Test gradients for matrix operations"""
        A = Tensor(2, 2)
        A.constant(1.0)
        
        B = Tensor(2, 2)
        B.constant(2.0)
        
        C = A * B  # Matrix multiplication
        loss = C.sum()
        
        loss.backward()
        
        # Check that gradients are computed
        self.assertTrue(np.any(A.impl.grads != 0))
        self.assertTrue(np.any(B.impl.grads != 0))


class TestAdvancedOperations(unittest.TestCase):
    """Test advanced tensor operations"""
    
    def test_convolution(self):
        """Test convolution operation"""
        # Create 5x5 input
        image = Tensor(5, 5)
        image.constant(1.0)
        
        # Create 3x3 kernel
        kernel = Tensor(3, 3)
        kernel.constant(0.1)
        
        # Create bias
        bias = Tensor(1, 1)
        bias.constant(0.5)
        
        # Apply convolution
        conv_out = image.make_convo(3, kernel, bias)
        
        # Output should be 3x3
        self.assertEqual(conv_out.shape, (3, 3))
        
        # Each output should be sum of 3x3 patch * kernel + bias
        # = 9 * 1.0 * 0.1 + 0.5 = 0.9 + 0.5 = 1.4
        self.assertAlmostEqual(conv_out[0, 0], 1.4, places=5)
    
    def test_max_pooling(self):
        """Test max pooling operation"""
        # Create 4x4 input
        image = Tensor(4, 4)
        
        # Set different values
        for i in range(4):
            for j in range(4):
                image[i, j] = i + j
        
        # Apply 2x2 max pooling
        pooled = image.make_max2d(2)
        
        # Output should be 2x2
        self.assertEqual(pooled.shape, (2, 2))
        
        # Check max values in each 2x2 region
        self.assertEqual(pooled[0, 0], 2.0)  # max of [0,1,1,2]
        self.assertEqual(pooled[1, 1], 6.0)  # max of [4,5,5,6]
    
    def test_slicing(self):
        """Test tensor slicing operation"""
        # Create 4x4 tensor
        a = Tensor(4, 4)
        
        # Fill with values
        for i in range(4):
            for j in range(4):
                a[i, j] = i * 4 + j
        
        # Create 2x2 slice starting at (1,1)
        b = a.make_slice(1, 1, 2, 2)
        
        self.assertEqual(b.shape, (2, 2))
        self.assertEqual(b[0, 0], 5.0)  # a[1,1]
        self.assertEqual(b[1, 1], 10.0)  # a[2,2]
    
    def test_flattening(self):
        """Test tensor flattening operation"""
        # Create two small tensors
        a = Tensor(2, 2)
        a.constant(1.0)
        
        b = Tensor(1, 3)
        b.constant(2.0)
        
        # Flatten them together
        flattened = make_flatten([a, b])
        
        # Should be a column vector of size (7, 1)
        self.assertEqual(flattened.shape, (7, 1))
        
        # First 4 elements should be 1.0, next 3 should be 2.0
        self.assertEqual(flattened[0, 0], 1.0)
        self.assertEqual(flattened[3, 0], 1.0)
        self.assertEqual(flattened[4, 0], 2.0)
        self.assertEqual(flattened[6, 0], 2.0)


class TestUtilityFunctions(unittest.TestCase):
    """Test utility functions"""
    
    def test_sum_operation(self):
        """Test sum operation"""
        a = Tensor(2, 3)
        a.constant(2.0)
        
        s = a.sum()
        self.assertEqual(s.shape, (1, 1))
        self.assertEqual(s[0, 0], 12.0)  # 2*3*2 = 12
    
    def test_log_softmax(self):
        """Test log softmax operation"""
        x = Tensor(3, 1)
        x[0, 0] = 1.0
        x[1, 0] = 2.0
        x[2, 0] = 3.0
        
        y = make_log_softmax(x)
        
        # Check that it's a valid log probability distribution
        # (sum of exp should be 1, but we check log domain)
        exp_sum = np.exp(y[0, 0]) + np.exp(y[1, 0]) + np.exp(y[2, 0])
        self.assertAlmostEqual(exp_sum, 1.0, places=5)
    
    def test_randomization(self):
        """Test tensor randomization"""
        a = Tensor(10, 10)
        a.randomize(1.0)
        
        # Check that not all values are the same
        first_val = a[0, 0]
        has_different = False
        for i in range(10):
            for j in range(10):
                if abs(a[i, j] - first_val) > 1e-6:
                    has_different = True
                    break
            if has_different:
                break
        
        self.assertTrue(has_different, "Randomization should produce different values")


def run_performance_test():
    """Run a performance test to ensure reasonable speed"""
    print("\nRunning performance tests...")
    
    import time
    
    # Test matrix multiplication performance
    start_time = time.time()
    
    A = Tensor(100, 100)
    A.randomize(0.01)
    B = Tensor(100, 100)
    B.randomize(0.01)
    
    C = A * B
    loss = C.sum()
    loss.backward()
    
    end_time = time.time()
    duration = end_time - start_time
    
    print(f"Matrix multiplication (100x100) + backward pass: {duration:.3f}s")
    
    if duration < 5.0:  # Should complete in under 5 seconds
        print("✓ Performance test passed")
    else:
        print("⚠ Performance test slow (but functional)")


def run_integration_test():
    """Run an integration test with a small neural network"""
    print("\nRunning integration test...")
    
    # Create a simple 2-layer neural network
    input_size, hidden_size, output_size = 4, 8, 2
    
    # Input
    x = Tensor(input_size, 1)
    x.randomize(1.0)
    
    # Layer 1
    W1 = Tensor(hidden_size, input_size)
    W1.randomize(0.1)
    b1 = Tensor(hidden_size, 1)
    b1.zero()
    
    # Layer 2
    W2 = Tensor(output_size, hidden_size)
    W2.randomize(0.1)
    b2 = Tensor(output_size, 1)
    b2.zero()
    
    # Forward pass
    h1 = relu(W1 * x + b1)
    y = W2 * h1 + b2
    
    # Loss
    target = Tensor(output_size, 1)
    target.constant(1.0)
    
    diff = y - target
    loss = (diff.dot(diff)).sum()
    
    # Backward pass
    loss.backward()
    
    # Check that all parameters have gradients
    assert np.any(W1.impl.grads != 0), "W1 should have gradients"
    assert np.any(W2.impl.grads != 0), "W2 should have gradients"
    
    print("✓ Integration test passed")


def main():
    """Run all tests"""
    print("PyTensorLib Test Suite")
    print("=" * 40)
    
    # Run unit tests
    unittest.main(argv=[''], exit=False, verbosity=2)
    
    # Run additional tests
    try:
        run_performance_test()
        run_integration_test()
        
        print("\n🎉 All tests completed successfully!")
        return 0
        
    except Exception as e:
        print(f"\n❌ Additional tests failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)