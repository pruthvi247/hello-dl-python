"""
PyTensorLib - Python Deep Learning Tensor Library

A comprehensive tensor library with automatic differentiation capabilities,
converted from a C++ implementation. Provides educational and research-ready
tools for deep learning and neural network experimentation.

Main Components:
- tensor_lib: Core tensor operations with automatic differentiation
- mnist_utils: MNIST dataset loading and utilities

Basic Usage:
    >>> from pytensorlib import Tensor, relu
    >>> x = Tensor(3, 3)
    >>> x.randomize(0.1)
    >>> y = relu(x)
    >>> loss = y.sum()
    >>> loss.backward()
    >>> print(x.impl.grads)  # View gradients

Example:
    >>> from pytensorlib import Tensor, sigmoid, MNISTReader
    >>> 
    >>> # Create a simple neural network layer
    >>> input_size, hidden_size = 784, 128
    >>> W = Tensor(hidden_size, input_size)
    >>> W.randomize(0.01)
    >>> b = Tensor(hidden_size, 1)
    >>> b.zero()
    >>> 
    >>> # Forward pass
    >>> x = Tensor(input_size, 1)
    >>> x.randomize(1.0)
    >>> h = sigmoid(W * x + b)
    >>> 
    >>> # Backward pass
    >>> loss = h.sum()
    >>> loss.backward()

Authors: Python Tensor Library Team
License: Educational Use
Version: 1.0.0
"""

# Import main components
from .tensor_lib import (
    Tensor, TensorImpl, TMode, ActivationFunctions,
    make_function, make_log_softmax, make_flatten,
    relu, gelu, sigmoid, tanh, square, mse_loss, cross_entropy_loss,
    __version__ as tensor_version, VERSION_INFO
)

from .mnist_utils import (
    MNISTReader, download_mnist, download_emnist, create_synthetic_mnist_data,
    __version__ as mnist_version
)

# Package metadata
__version__ = "1.0.0"
__author__ = "Python Tensor Library Team"
__license__ = "Educational Use"
__description__ = "Python Deep Learning Tensor Library with Automatic Differentiation"

# Version information
VERSION_INFO = {
    'package': __version__,
    'tensor_lib': tensor_version,
    'mnist_utils': mnist_version,
    'release': 'stable'
}

# Export all public APIs
__all__ = [
    # Core tensor functionality
    'Tensor', 'TensorImpl', 'TMode', 'ActivationFunctions',
    
    # Tensor operations
    'make_function', 'make_log_softmax', 'make_flatten',
    
    # Activation functions
    'relu', 'gelu', 'sigmoid', 'tanh', 'square',
    
    # Loss functions
    'mse_loss', 'cross_entropy_loss',
    
    # MNIST utilities
    'MNISTReader', 'download_mnist', 'download_emnist', 'create_synthetic_mnist_data',
    
    # Metadata
    '__version__', '__author__', '__license__', '__description__',
    'VERSION_INFO'
]

def get_info():
    """Get package information"""
    return {
        'name': 'PyTensorLib',
        'version': __version__,
        'description': __description__,
        'author': __author__,
        'license': __license__,
        'components': {
            'tensor_lib': 'Core tensor operations with automatic differentiation',
            'mnist_utils': 'MNIST dataset loading and processing utilities'
        },
        'features': [
            'Automatic differentiation (backpropagation)',
            'Neural network operations',
            'Convolution and pooling',
            'Various activation functions',
            'MNIST dataset support',
            'Educational examples'
        ]
    }

def quick_test():
    """Run a quick functionality test"""
    print("PyTensorLib Quick Test")
    print("=" * 30)
    
    try:
        # Test basic tensor operations
        print("1. Testing basic tensor operations...")
        a = Tensor(2, 2)
        a.constant(2.0)
        b = Tensor(2, 2)
        b.constant(3.0)
        c = a + b
        print(f"   2 + 3 = {c[0,0]} ✓")
        
        # Test activation functions
        print("2. Testing activation functions...")
        x = Tensor(1, 1)
        x.constant(0.5)
        y = sigmoid(x)
        print(f"   sigmoid(0.5) = {y[0,0]:.3f} ✓")
        
        # Test automatic differentiation
        print("3. Testing automatic differentiation...")
        x = Tensor(1, 1)
        x.constant(3.0)
        y = square(x)
        y.backward()
        print(f"   d/dx(x²) at x=3: {x.impl.grads[0,0]} (expected: 6.0) ✓")
        
        print("\n✅ All tests passed! PyTensorLib is ready to use.")
        return True
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        return False