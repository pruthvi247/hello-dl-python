"""
Test class to verify all documentation examples from tensor_lib.py

This test suite validates that all the sample inputs and outputs provided
in the comprehensive documentation for the tensor_lib module are accurate 
and executable. This specific test file focuses on tensor_lib.py - future
test files can be created for other modules like mnist_utils.py.
"""

import unittest
import numpy as np
import sys
import os

# Add src to path to import pytensorlib
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from pytensorlib import Tensor, relu, gelu, sigmoid, tanh, square, mse_loss, cross_entropy_loss
from pytensorlib.tensor_lib import TensorImpl, TMode


class TestDocumentationExamples(unittest.TestCase):
    """Test class for verifying tensor_lib.py documentation examples"""
    
    def setUp(self):
        """Set up test fixtures before each test method"""
        pass
    
    def test_tensor_creation_examples(self):
        """Test Tensor.__init__() documentation examples"""
        print("Testing Tensor creation examples...")
        
        # Matrix tensor creation
        tensor = Tensor(2, 3)
        self.assertEqual(tensor.shape, (2, 3))
        self.assertEqual(tensor[0, 0], 0.0)  # initialized to zero
        self.assertEqual(tensor.impl.mode, TMode.PARAMETER)
        
        # Scalar tensor creation
        scalar = Tensor(value=42.0)
        self.assertEqual(scalar.shape, (1, 1))
        self.assertEqual(scalar[0, 0], 42.0)
        self.assertIsNotNone(scalar.impl.val)
        
        # Operation tensor (created by operations)
        result = Tensor()
        self.assertEqual(result.impl.mode, TMode.UNASSIGNED)
        self.assertIsNone(result.impl.val)
        
        print("✓ Tensor creation examples passed")
    
    def test_tensor_indexing_examples(self):
        """Test Tensor.__getitem__() and __setitem__() documentation examples"""
        print("Testing Tensor indexing examples...")
        
        # Matrix indexing
        tensor = Tensor(3, 3)
        tensor.constant(5.0)
        self.assertEqual(tensor[1, 2], 5.0)
        self.assertEqual(tensor[0, 0], 5.0)
        
        # Matrix element setting
        tensor = Tensor(3, 3)
        tensor[0, 0] = 1.0
        tensor[1, 2] = 2.5
        self.assertEqual(tensor[0, 0], 1.0)
        self.assertEqual(tensor[1, 2], 2.5)
        self.assertEqual(tensor[2, 2], 0.0)  # unchanged
        
        # Building a matrix
        m = Tensor(2, 2)
        m[0, 0] = 1.0
        m[0, 1] = 2.0
        m[1, 0] = 3.0
        m[1, 1] = 4.0
        self.assertEqual(m[0, 0], 1.0)
        self.assertEqual(m[0, 1], 2.0)
        self.assertEqual(m[1, 0], 3.0)
        self.assertEqual(m[1, 1], 4.0)
        
        print("✓ Tensor indexing examples passed")
    
    def test_constant_examples(self):
        """Test Tensor.constant() documentation examples"""
        print("Testing constant() examples...")
        
        # Fill 3x3 matrix with constant
        tensor = Tensor(3, 3)
        tensor.constant(5.0)
        self.assertEqual(tensor[0, 0], 5.0)
        self.assertEqual(tensor[1, 2], 5.0)
        self.assertEqual(tensor[2, 1], 5.0)
        self.assertEqual(tensor.impl.mode, TMode.PARAMETER)
        
        # Common initialization patterns
        zeros = Tensor(2, 2)
        zeros.constant(0.0)
        self.assertEqual(zeros[0, 0], 0.0)
        self.assertEqual(zeros[1, 1], 0.0)
        
        ones = Tensor(2, 2)
        ones.constant(1.0)
        self.assertEqual(ones[0, 0], 1.0)
        self.assertEqual(ones[1, 1], 1.0)
        
        weights = Tensor(3, 4)
        weights.constant(0.01)
        self.assertAlmostEqual(float(weights[0, 0]), 0.01, places=5)
        self.assertAlmostEqual(float(weights[2, 3]), 0.01, places=5)
        
        # Works with scalars too
        scalar = Tensor(value=0.0)
        scalar.constant(42.0)
        self.assertEqual(scalar[0, 0], 42.0)
        
        print("✓ constant() examples passed")
    
    def test_shape_examples(self):
        """Test Tensor.shape property documentation examples"""
        print("Testing shape property examples...")
        
        # Parameter tensor shape
        tensor = Tensor(3, 4)
        self.assertEqual(tensor.shape, (3, 4))
        
        # Scalar tensor shape
        scalar = Tensor(value=5.0)
        self.assertEqual(scalar.shape, (1, 1))
        
        # Operation result shape
        a = Tensor(2, 3)
        b = Tensor(2, 3)
        result = a + b
        self.assertEqual(result.shape, (2, 3))
        
        # Matrix multiplication shape
        m1 = Tensor(3, 4)
        m2 = Tensor(4, 2)
        product = m1 * m2
        self.assertEqual(product.shape, (3, 2))
        
        print("✓ Shape property examples passed")
    
    def test_addition_examples(self):
        """Test Tensor.__add__() documentation examples"""
        print("Testing addition examples...")
        
        # Matrix addition
        a = Tensor(2, 2)
        a.constant(3.0)
        b = Tensor(2, 2)
        b.constant(2.0)
        result = a + b
        self.assertEqual(result[0, 0], 5.0)
        self.assertEqual(result[1, 1], 5.0)
        
        # Scalar addition
        scalar1 = Tensor(value=10.0)
        scalar2 = Tensor(value=5.0)
        sum_result = scalar1 + scalar2
        self.assertEqual(sum_result[0, 0], 15.0)
        
        print("✓ Addition examples passed")
    
    def test_multiplication_examples(self):
        """Test Tensor.__mul__() documentation examples"""
        print("Testing multiplication examples...")
        
        # Standard matrix multiplication
        a = Tensor(2, 3)
        a[0, 0] = 1.0; a[0, 1] = 2.0; a[0, 2] = 3.0
        a[1, 0] = 4.0; a[1, 1] = 5.0; a[1, 2] = 6.0
        # a = [[1, 2, 3], [4, 5, 6]]
        
        b = Tensor(3, 2)
        b[0, 0] = 1.0; b[0, 1] = 2.0
        b[1, 0] = 3.0; b[1, 1] = 4.0
        b[2, 0] = 5.0; b[2, 1] = 6.0
        # b = [[1, 2], [3, 4], [5, 6]]
        
        result = a * b  # (2x3) * (3x2) = (2x2)
        self.assertEqual(result[0, 0], 22.0)  # 1*1 + 2*3 + 3*5
        self.assertEqual(result[0, 1], 28.0)  # 1*2 + 2*4 + 3*6
        self.assertEqual(result[1, 0], 49.0)  # 4*1 + 5*3 + 6*5
        self.assertEqual(result[1, 1], 64.0)  # 4*2 + 5*4 + 6*6
        
        # Vector multiplication
        vec1 = Tensor(1, 3)
        vec1[0, 0] = 1.0; vec1[0, 1] = 2.0; vec1[0, 2] = 3.0
        vec2 = Tensor(3, 1)
        vec2[0, 0] = 4.0; vec2[1, 0] = 5.0; vec2[2, 0] = 6.0
        dot_product = vec1 * vec2  # (1x3) * (3x1) = (1x1)
        self.assertEqual(dot_product[0, 0], 32.0)  # 1*4 + 2*5 + 3*6
        
        print("✓ Multiplication examples passed")
    
    def test_one_hot_examples(self):
        """Test one_hot_column() and one_hot_row() documentation examples"""
        print("Testing one-hot encoding examples...")
        
        # One-hot column for class 2 out of 5 classes
        classes = Tensor(1, 5)
        classes.one_hot_column(2)
        self.assertEqual(classes[0, 0], 0.0)
        self.assertEqual(classes[0, 1], 0.0)
        self.assertEqual(classes[0, 2], 1.0)  # hot
        self.assertEqual(classes[0, 3], 0.0)
        self.assertEqual(classes[0, 4], 0.0)
        
        # Verify sum is 1.0
        total = sum(classes[0, i] for i in range(5))
        self.assertEqual(total, 1.0)
        
        # One-hot row for class 3 out of 6 classes
        classes_row = Tensor(6, 1)
        classes_row.one_hot_row(3)
        self.assertEqual(classes_row[0, 0], 0.0)
        self.assertEqual(classes_row[1, 0], 0.0)
        self.assertEqual(classes_row[2, 0], 0.0)
        self.assertEqual(classes_row[3, 0], 1.0)  # hot
        self.assertEqual(classes_row[4, 0], 0.0)
        self.assertEqual(classes_row[5, 0], 0.0)
        
        print("✓ One-hot encoding examples passed")
    
    def test_dot_product_examples(self):
        """Test Tensor.dot() documentation examples"""
        print("Testing dot product (Hadamard) examples...")
        
        # Element-wise multiplication of matrices
        a = Tensor(2, 2)
        a[0, 0] = 2.0; a[0, 1] = 3.0
        a[1, 0] = 4.0; a[1, 1] = 5.0
        # a = [[2, 3], [4, 5]]
        
        b = Tensor(2, 2)
        b[0, 0] = 1.0; b[0, 1] = 2.0
        b[1, 0] = 3.0; b[1, 1] = 4.0
        # b = [[1, 2], [3, 4]]
        
        result = a.dot(b)
        self.assertEqual(result[0, 0], 2.0)   # 2*1
        self.assertEqual(result[0, 1], 6.0)   # 3*2
        self.assertEqual(result[1, 0], 12.0)  # 4*3
        self.assertEqual(result[1, 1], 20.0)  # 5*4
        
        # Neural network weight masking
        weights = Tensor(3, 3)
        weights.constant(0.5)
        mask = Tensor(3, 3)
        mask.constant(0.8)  # 80% of weights kept
        masked_weights = weights.dot(mask)
        self.assertAlmostEqual(float(masked_weights[0, 0]), 0.4, places=5)  # 0.5 * 0.8
        
        print("✓ Dot product examples passed")
    
    def test_activation_function_examples(self):
        """Test activation function documentation examples"""
        print("Testing activation function examples...")
        
        # ReLU activation
        x = Tensor(2, 2)
        x[0, 0] = -1.0; x[0, 1] = 2.0
        x[1, 0] = -0.5; x[1, 1] = 3.0
        
        relu_result = relu(x)
        self.assertEqual(relu_result[0, 0], 0.0)  # max(0, -1.0)
        self.assertEqual(relu_result[0, 1], 2.0)  # max(0, 2.0)
        self.assertEqual(relu_result[1, 0], 0.0)  # max(0, -0.5)
        self.assertEqual(relu_result[1, 1], 3.0)  # max(0, 3.0)
        
        # Sigmoid activation (approximate test since sigmoid is computed)
        x_small = Tensor(value=0.0)
        sig_result = sigmoid(x_small)
        # sigmoid(0) ≈ 0.5
        self.assertAlmostEqual(float(sig_result[0, 0]), 0.5, places=1)
        
        # Square activation
        x_square = Tensor(2, 2)
        x_square[0, 0] = 2.0; x_square[0, 1] = -3.0
        x_square[1, 0] = 4.0; x_square[1, 1] = -1.0
        
        square_result = square(x_square)
        self.assertEqual(square_result[0, 0], 4.0)   # 2^2
        self.assertEqual(square_result[0, 1], 9.0)   # (-3)^2
        self.assertEqual(square_result[1, 0], 16.0)  # 4^2
        self.assertEqual(square_result[1, 1], 1.0)   # (-1)^2
        
        print("✓ Activation function examples passed")
    
    def test_loss_function_examples(self):
        """Test loss function documentation examples"""
        print("Testing loss function examples...")
        
        # MSE Loss - perfect prediction
        predictions = Tensor(3, 1)
        predictions[0, 0] = 2.0; predictions[1, 0] = 4.0; predictions[2, 0] = 6.0
        targets = Tensor(3, 1)
        targets[0, 0] = 2.0; targets[1, 0] = 4.0; targets[2, 0] = 6.0
        
        mse_perfect = mse_loss(predictions, targets)
        self.assertAlmostEqual(float(mse_perfect[0, 0]), 0.0, places=5)  # Perfect prediction
        
        # MSE Loss - with error
        predictions_error = Tensor(3, 1)
        predictions_error[0, 0] = 2.0; predictions_error[1, 0] = 5.0; predictions_error[2, 0] = 7.0
        # Errors: [0, 1, 1], Squared: [0, 1, 1], Sum: 2.0
        mse_error = mse_loss(predictions_error, targets)
        expected_sum = 2.0  # Sum of squared errors, not mean
        self.assertAlmostEqual(float(mse_error[0, 0]), expected_sum, places=2)
        
        print("✓ Loss function examples passed")
    
    def test_tensor_arithmetic_operations(self):
        """Test additional arithmetic operations"""
        print("Testing additional arithmetic operations...")
        
        # Subtraction
        a = Tensor(2, 2)
        a.constant(5.0)
        b = Tensor(2, 2)
        b.constant(3.0)
        result = a - b
        self.assertEqual(result[0, 0], 2.0)
        self.assertEqual(result[1, 1], 2.0)
        
        # Negation
        x = Tensor(2, 2)
        x[0, 0] = 3.0; x[0, 1] = -2.0
        x[1, 0] = -1.0; x[1, 1] = 4.0
        neg_x = -x
        self.assertEqual(neg_x[0, 0], -3.0)
        self.assertEqual(neg_x[0, 1], 2.0)
        self.assertEqual(neg_x[1, 0], 1.0)
        self.assertEqual(neg_x[1, 1], -4.0)
        
        # Division by scalar
        dividend = Tensor(2, 2)
        dividend.constant(10.0)
        divisor = Tensor(value=2.0)
        quotient = dividend / divisor
        self.assertEqual(quotient[0, 0], 5.0)
        self.assertEqual(quotient[1, 1], 5.0)
        
        print("✓ Additional arithmetic operations passed")
    
    def test_edge_cases(self):
        """Test edge cases and error conditions"""
        print("Testing edge cases...")
        
        # Empty tensor operations
        empty = Tensor()
        self.assertEqual(empty.impl.mode, TMode.UNASSIGNED)
        
        # Zero tensor operations
        zero_tensor = Tensor(2, 2)
        zero_tensor.constant(0.0)
        self.assertEqual(zero_tensor[0, 0], 0.0)
        self.assertEqual(zero_tensor[1, 1], 0.0)
        
        # Single element tensor
        single = Tensor(1, 1)
        single[0, 0] = 42.0
        self.assertEqual(single[0, 0], 42.0)
        self.assertEqual(single.shape, (1, 1))
        
        print("✓ Edge cases passed")


def run_documentation_tests():
    """Run all tensor_lib.py documentation example tests"""
    print("=" * 60)
    print("RUNNING TENSOR_LIB DOCUMENTATION EXAMPLE TESTS")
    print("=" * 60)
    
    # Create test suite
    suite = unittest.TestLoader().loadTestsFromTestCase(TestDocumentationExamples)
    
    # Run tests with detailed output
    runner = unittest.TextTestRunner(verbosity=2, stream=sys.stdout)
    result = runner.run(suite)
    
    print("\n" + "=" * 60)
    if result.wasSuccessful():
        print("🎉 ALL DOCUMENTATION EXAMPLES PASSED!")
        print(f"✓ {result.testsRun} tests completed successfully")
    else:
        print("❌ SOME DOCUMENTATION EXAMPLES FAILED")
        print(f"Failed: {len(result.failures)}, Errors: {len(result.errors)}")
        
        if result.failures:
            print("\nFailures:")
            for test, failure in result.failures:
                print(f"  - {test}: {failure}")
        
        if result.errors:
            print("\nErrors:")
            for test, error in result.errors:
                print(f"  - {test}: {error}")
    
    print("=" * 60)
    return result.wasSuccessful()


if __name__ == "__main__":
    # Configure environment
    os.environ.setdefault('PYTHONPATH', 
                         os.path.join(os.path.dirname(__file__), '..', 'src'))
    
    # Run the tests
    success = run_documentation_tests()
    
    # Exit with appropriate code
    sys.exit(0 if success else 1)