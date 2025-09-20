"""
PyTensorLib: A Python Deep Learning Tensor Library with Automatic Differentiation

This module provides a complete tensor library with automatic differentiation capabilities,
inspired by and converted from a C++ implementation. It includes support for:

- Automatic differentiation (backpropagation)
- Neural network operations
- Convolution and pooling
- Various activation functions
- Matrix operations with gradient computation

Author: Converted from C++ hello-dl tensor library
License: Educational use
"""

import numpy as np
import math
from typing import List, Optional, Callable, Union, Any
from enum import IntEnum
import random

__version__ = "1.0.0"
__author__ = "Python Tensor Library Team"

# Version info
VERSION_INFO = {
    'major': 1,
    'minor': 0,
    'patch': 0,
    'release': 'stable'
}


class TMode(IntEnum):
    """Tensor operation modes - equivalent to C++ enum"""
    UNASSIGNED = 0
    PARAMETER = 1
    ADDITION = 2
    MULT = 3
    DIV = 4
    FUNC = 5
    MAX = 6
    SUM = 7
    SLICE = 8
    FLATTEN = 9
    DOT_PROD = 10
    LOG_SOFTMAX = 11
    NEG = 12
    CONVO = 13
    MAX2D = 14
    DROPOUT = 15


class ActivationFunctions:
    """Collection of activation functions and their derivatives"""
    
    class ReLU:
        @staticmethod
        def func(x: np.ndarray) -> np.ndarray:
            return np.maximum(0.0, x)
        
        @staticmethod
        def deriv(x: np.ndarray) -> np.ndarray:
            return (x > 0).astype(np.float32)
    
    class GELU:
        @staticmethod
        def func(x: np.ndarray) -> np.ndarray:
            # GELU(x) = 0.5 * x * (1 + erf(x / sqrt(2)))
            invsqrt2 = 0.70710678118654752440  # 1/sqrt(2)
            return 0.5 * x * (1 + np.vectorize(math.erf)(x * invsqrt2))
        
        @staticmethod
        def deriv(x: np.ndarray) -> np.ndarray:
            invsqrt2 = 0.70710678118654752440
            invsqrt2pi = 0.3989422804014327  # 1/sqrt(2*pi)
            erf_term = np.vectorize(math.erf)(x * invsqrt2)
            exp_term = np.exp(-0.5 * x * x)
            return (1 + erf_term) / 2 + x * exp_term * invsqrt2pi
    
    class Square:
        @staticmethod
        def func(x: np.ndarray) -> np.ndarray:
            return x * x
        
        @staticmethod
        def deriv(x: np.ndarray) -> np.ndarray:
            return 2 * x
    
    class Tanh:
        @staticmethod
        def func(x: np.ndarray) -> np.ndarray:
            return np.tanh(x)
        
        @staticmethod
        def deriv(x: np.ndarray) -> np.ndarray:
            t = np.tanh(x)
            return 1 - t * t
    
    class Sigmoid:
        @staticmethod
        def func(x: np.ndarray) -> np.ndarray:
            return 1.0 / (1.0 + np.exp(-x))
        
        @staticmethod
        def deriv(x: np.ndarray) -> np.ndarray:
            sigma = 1.0 / (1.0 + np.exp(-x))
            return sigma * (1.0 - sigma)


class TensorImpl:
    """Implementation class for tensor operations with automatic differentiation"""
    
    def __init__(self, rows: int = 0, cols: int = 0, 
                 lhs: Optional['TensorImpl'] = None, 
                 rhs: Optional['TensorImpl'] = None, 
                 mode: TMode = TMode.UNASSIGNED):
        
        self.mode = mode
        self.lhs = lhs
        self.rhs = rhs
        self.have_val = False
        self.no_grad = False
        
        # Initialize matrices
        if rows > 0 and cols > 0:
            self.val = np.zeros((rows, cols), dtype=np.float32)
            self.grads = np.zeros((rows, cols), dtype=np.float32)
            self.accum_grads = np.zeros((rows, cols), dtype=np.float32)
            self.prev_accum_grads = np.zeros((rows, cols), dtype=np.float32)
            self.have_val = True
            self.mode = TMode.PARAMETER
        else:
            self.val = None
            self.grads = None
            self.accum_grads = None
            self.prev_accum_grads = None
        
        # Function pointers for activation functions
        self.func: Optional[Callable] = None
        self.deriv: Optional[Callable] = None
        
        # Parameters for different operations
        self.slice_params = {'r': 0, 'c': 0, 'h': 0, 'w': 0}
        self.max2d_params = {'kernel': 0}
        self.convo_params = {'kernel': 0, 'bias': None}
        self.flatten_params = {'members': []}
        self.random_params = {'rate': 0.0}
        self.adam_vals = {'m': None, 'v': None}
    
    def assure_value(self):
        """Compute the forward pass value if not already computed"""
        if self.have_val or self.mode == TMode.PARAMETER:
            return
        
        if self.mode == TMode.SUM:
            self.lhs.assure_value()
            self.val = np.array([[self.lhs.val.sum()]], dtype=np.float32)
            
        elif self.mode == TMode.ADDITION:
            self.lhs.assure_value()
            self.rhs.assure_value()
            self.val = self.lhs.val + self.rhs.val
            
        elif self.mode == TMode.MULT:
            self.lhs.assure_value()
            self.rhs.assure_value()
            self.val = self.lhs.val @ self.rhs.val  # Matrix multiplication
            
        elif self.mode == TMode.NEG:
            self.lhs.assure_value()
            self.val = -self.lhs.val
            
        elif self.mode == TMode.DIV:
            self.lhs.assure_value()
            self.rhs.assure_value()
            # Special case: RHS must be a single number
            assert self.rhs.val.shape == (1, 1), "Division only supports scalar divisor"
            self.val = self.lhs.val / self.rhs.val[0, 0]
            
        elif self.mode == TMode.DOT_PROD:
            self.lhs.assure_value()
            self.rhs.assure_value()
            self.val = self.lhs.val * self.rhs.val  # Element-wise multiplication
            
        elif self.mode == TMode.DROPOUT:
            self.lhs.assure_value()
            rate = self.random_params['rate']
            # Create dropout mask
            mask = np.random.random(self.lhs.val.shape) > rate
            # PyTorch-style scaling
            scale = 1.0 / (1 - rate) if rate < 1.0 else 0.0
            dropout_mask = mask.astype(np.float32) * scale
            self.val = self.lhs.val * dropout_mask
            # Store mask for backward pass
            self.dropout_mask = dropout_mask
            
        elif self.mode == TMode.SLICE:
            self.lhs.assure_value()
            r, c, h, w = self.slice_params['r'], self.slice_params['c'], \
                        self.slice_params['h'], self.slice_params['w']
            self.val = self.lhs.val[r:r+h, c:c+w].copy()
            
        elif self.mode == TMode.FLATTEN:
            total_size = 0
            for member in self.flatten_params['members']:
                member.assure_value()
                total_size += member.val.size
            
            self.val = np.zeros((total_size, 1), dtype=np.float32)
            pos = 0
            for member in self.flatten_params['members']:
                flat = member.val.flatten('F')  # Fortran order to match C++
                self.val[pos:pos+len(flat), 0] = flat
                pos += len(flat)
                
        elif self.mode == TMode.FUNC:
            self.lhs.assure_value()
            if self.func is not None:
                self.val = self.func(self.lhs.val)
            else:
                raise ValueError("Function not set for FUNC mode")
                
        elif self.mode == TMode.LOG_SOFTMAX:
            self.lhs.assure_value()
            # Numerically stable log softmax
            max_val = np.max(self.lhs.val)
            exp_vals = np.exp(self.lhs.val - max_val)
            sum_exp = np.sum(exp_vals)
            self.val = self.lhs.val - max_val - np.log(sum_exp)
            
        elif self.mode == TMode.CONVO:
            self.lhs.assure_value()  # input
            self.rhs.assure_value()  # weights
            self.convo_params['bias'].assure_value()  # bias
            
            input_val = self.lhs.val
            weights = self.rhs.val
            bias = self.convo_params['bias'].val[0, 0]
            kernel = self.convo_params['kernel']
            
            # Output dimensions
            out_rows = input_val.shape[0] - kernel + 1
            out_cols = input_val.shape[1] - kernel + 1
            self.val = np.zeros((out_rows, out_cols), dtype=np.float32)
            
            # Convolution operation
            for r in range(out_rows):
                for c in range(out_cols):
                    patch = input_val[r:r+kernel, c:c+kernel]
                    self.val[r, c] = np.sum(patch * weights) + bias
                    
        elif self.mode == TMode.MAX2D:
            self.lhs.assure_value()
            kernel = self.max2d_params['kernel']
            input_val = self.lhs.val
            
            # Output dimensions with potential padding
            out_rows = (input_val.shape[0] + kernel - 1) // kernel
            out_cols = (input_val.shape[1] + kernel - 1) // kernel
            self.val = np.zeros((out_rows, out_cols), dtype=np.float32)
            
            # Max pooling
            for r in range(0, input_val.shape[0], kernel):
                for c in range(0, input_val.shape[1], kernel):
                    # Handle padding
                    eff_height = min(r + kernel, input_val.shape[0]) - r
                    eff_width = min(c + kernel, input_val.shape[1]) - c
                    patch = input_val[r:r+eff_height, c:c+eff_width]
                    self.val[r//kernel, c//kernel] = np.max(patch)
        else:
            raise ValueError(f"Unknown mode: {self.mode}")
        
        # Initialize gradients with same shape as value
        self.grads = np.zeros_like(self.val)
        self.accum_grads = np.zeros_like(self.val)
        self.have_val = True
    
    def build_topo(self, visited: set, topo: List['TensorImpl']):
        """Build topological ordering for backpropagation"""
        if self in visited:
            return
        visited.add(self)
        
        if self.lhs:
            self.lhs.build_topo(visited, topo)
        if self.rhs:
            self.rhs.build_topo(visited, topo)
        
        if self.mode == TMode.FLATTEN:
            for member in self.flatten_params['members']:
                member.build_topo(visited, topo)
        elif self.mode == TMode.CONVO:
            self.convo_params['bias'].build_topo(visited, topo)
        
        topo.append(self)
    
    def do_grad(self):
        """Compute gradients (backward pass)"""
        if self.mode == TMode.PARAMETER:
            return
            
        elif self.mode == TMode.FLATTEN:
            grad_pos = 0
            for member in self.flatten_params['members']:
                flat_size = member.grads.size
                member_grads_flat = self.grads[grad_pos:grad_pos+flat_size, 0]
                member.grads += member_grads_flat.reshape(member.grads.shape, order='F')
                grad_pos += flat_size
                
        elif self.mode == TMode.ADDITION:
            self.lhs.grads += self.grads
            self.rhs.grads += self.grads
            
        elif self.mode == TMode.NEG:
            self.lhs.grads -= self.grads
            
        elif self.mode == TMode.MULT:
            # Matrix multiplication gradients
            self.lhs.grads += self.grads @ self.rhs.val.T
            self.rhs.grads += self.lhs.val.T @ self.grads
            
        elif self.mode == TMode.DIV:
            self.lhs.grads += self.grads / self.rhs.val[0, 0]
            # Note: RHS gradient computation omitted as noted in C++ code
            
        elif self.mode == TMode.DOT_PROD:
            self.lhs.grads += self.grads * self.rhs.val
            self.rhs.grads += self.grads * self.lhs.val
            
        elif self.mode == TMode.DROPOUT:
            self.lhs.grads += self.grads * self.dropout_mask
            
        elif self.mode == TMode.SLICE:
            r, c, h, w = self.slice_params['r'], self.slice_params['c'], \
                        self.slice_params['h'], self.slice_params['w']
            self.lhs.grads[r:r+h, c:c+w] += self.grads
            
        elif self.mode == TMode.SUM:
            self.lhs.grads += self.grads[0, 0]
            
        elif self.mode == TMode.LOG_SOFTMAX:
            # Log softmax gradient
            exp_vals = np.exp(self.val)
            self.lhs.grads += self.grads - exp_vals * np.sum(self.grads)
            
        elif self.mode == TMode.FUNC:
            if self.deriv is not None:
                self.lhs.grads += self.grads * self.deriv(self.lhs.val)
            else:
                raise ValueError("Derivative function not set for FUNC mode")
                
        elif self.mode == TMode.CONVO:
            # Convolution backward pass
            kernel = self.convo_params['kernel']
            
            # Gradients to input
            if not self.lhs.no_grad:
                for r in range(self.val.shape[0]):
                    for c in range(self.val.shape[1]):
                        self.lhs.grads[r:r+kernel, c:c+kernel] += \
                            self.rhs.val * self.grads[r, c]
            
            # Gradients to weights
            for r in range(self.rhs.val.shape[0]):
                for c in range(self.rhs.val.shape[1]):
                    grad_sum = 0.0
                    for out_r in range(self.val.shape[0]):
                        for out_c in range(self.val.shape[1]):
                            grad_sum += self.lhs.val[r+out_r, c+out_c] * self.grads[out_r, out_c]
                    self.rhs.grads[r, c] += grad_sum
            
            # Normalize weight gradients
            self.rhs.grads /= np.sqrt(self.grads.shape[0] * self.grads.shape[1])
            
            # Gradients to bias
            self.convo_params['bias'].grads[0, 0] += np.sum(self.grads)
            
        elif self.mode == TMode.MAX2D:
            # Max pooling backward pass
            kernel = self.max2d_params['kernel']
            for r in range(0, self.lhs.val.shape[0], kernel):
                for c in range(0, self.lhs.val.shape[1], kernel):
                    eff_height = min(r + kernel, self.lhs.val.shape[0]) - r
                    eff_width = min(c + kernel, self.lhs.val.shape[1]) - c
                    patch = self.lhs.val[r:r+eff_height, c:c+eff_width]
                    
                    # Find location of maximum
                    max_pos = np.unravel_index(np.argmax(patch), patch.shape)
                    max_r, max_c = max_pos[0] + r, max_pos[1] + c
                    
                    # Add gradient only to the maximum location
                    self.lhs.grads[max_r, max_c] += self.grads[r//kernel, c//kernel]


class Tensor:
    """Main Tensor class - user interface"""
    
    def __init__(self, rows: int = 0, cols: int = 0, value: Optional[float] = None):
        if value is not None:
            # Create scalar tensor
            self.impl = TensorImpl(1, 1)
            self.impl.val[0, 0] = value
        else:
            self.impl = TensorImpl(rows, cols)
    
    def __getitem__(self, key):
        """Get element by index"""
        self.impl.assure_value()
        if isinstance(key, tuple):
            return self.impl.val[key[0], key[1]]
        else:
            return self.impl.val[key]
    
    def __setitem__(self, key, value):
        """Set element by index"""
        if isinstance(key, tuple):
            self.impl.val[key[0], key[1]] = value
        else:
            self.impl.val[key] = value
    
    def raw(self) -> np.ndarray:
        """Get raw numpy array (only for parameter tensors)"""
        assert self.impl.mode == TMode.PARAMETER, "raw() only available for parameter tensors"
        return self.impl.val
    
    def sum(self) -> 'Tensor':
        """Sum all elements"""
        result = Tensor()
        result.impl = TensorImpl(lhs=self.impl, mode=TMode.SUM)
        return result
    
    def get_topo(self) -> List[TensorImpl]:
        """Get topological ordering for backpropagation"""
        topo = []
        visited = set()
        self.impl.build_topo(visited, topo)
        return topo
    
    def backward(self, topo: Optional[List[TensorImpl]] = None):
        """Perform backpropagation"""
        if topo is None:
            topo = self.get_topo()
        
        self.impl.assure_value()
        self.impl.grads = np.ones_like(self.impl.val)
        
        for tensor_impl in reversed(topo):
            tensor_impl.do_grad()
    
    def zero_grad(self, topo: List[TensorImpl]):
        """Zero all gradients"""
        for tensor_impl in reversed(topo):
            tensor_impl.grads = np.zeros_like(tensor_impl.grads)
            
            if tensor_impl.mode != TMode.PARAMETER:
                tensor_impl.have_val = False
            
            # Handle special cases
            if tensor_impl.mode == TMode.CONVO:
                tensor_impl.convo_params['bias'].grads.fill(0)
            
            if tensor_impl.mode == TMode.FLATTEN:
                for member in tensor_impl.flatten_params['members']:
                    member.grads.fill(0)
    
    def randomize(self, factor: float = 1.0):
        """Initialize with random values"""
        self.impl.mode = TMode.PARAMETER
        self.impl.val = np.random.uniform(-1, 1, self.impl.val.shape).astype(np.float32)
        self.impl.val *= factor
    
    def zero(self):
        """Fill with zeros"""
        self.constant(0.0)
    
    def constant(self, value: float):
        """Fill with constant value"""
        self.impl.mode = TMode.PARAMETER
        self.impl.val.fill(value)
    
    def one_hot_column(self, c: int):
        """Set one-hot encoding for column"""
        self.zero()
        self.impl.val[0, c] = 1.0
    
    def one_hot_row(self, r: int):
        """Set one-hot encoding for row"""
        self.zero()
        self.impl.val[r, 0] = 1.0
    
    def dot(self, other: 'Tensor') -> 'Tensor':
        """Element-wise multiplication (Hadamard product)"""
        result = Tensor()
        result.impl = TensorImpl(lhs=self.impl, rhs=other.impl, mode=TMode.DOT_PROD)
        return result
    
    def make_slice(self, r: int, c: int, h: int, w: int = -1) -> 'Tensor':
        """Create a slice of the tensor"""
        if w <= 0:
            w = h
        result = Tensor()
        result.impl = TensorImpl(lhs=self.impl, mode=TMode.SLICE)
        result.impl.slice_params = {'r': r, 'c': c, 'h': h, 'w': w}
        return result
    
    def make_convo(self, kernel: int, weights: 'Tensor', bias: 'Tensor') -> 'Tensor':
        """Create convolution operation"""
        result = Tensor()
        result.impl = TensorImpl(lhs=self.impl, rhs=weights.impl, mode=TMode.CONVO)
        result.impl.convo_params = {'kernel': kernel, 'bias': bias.impl}
        return result
    
    def make_max2d(self, kernel: int) -> 'Tensor':
        """Create 2D max pooling operation"""
        result = Tensor()
        result.impl = TensorImpl(lhs=self.impl, mode=TMode.MAX2D)
        result.impl.max2d_params = {'kernel': kernel}
        return result
    
    def make_dropout(self, rate: float) -> 'Tensor':
        """Create dropout operation"""
        result = Tensor()
        rnd = Tensor()
        rnd.impl.mode = TMode.PARAMETER
        result.impl = TensorImpl(lhs=self.impl, rhs=rnd.impl, mode=TMode.DROPOUT)
        result.impl.random_params = {'rate': rate}
        return result
    
    @property
    def shape(self):
        """Get tensor shape"""
        self.impl.assure_value()
        return self.impl.val.shape
    
    def __str__(self):
        """String representation"""
        self.impl.assure_value()
        return str(self.impl.val)
    
    def __repr__(self):
        return f"Tensor(shape={self.shape})"
    
    # Arithmetic operators
    def __add__(self, other: 'Tensor') -> 'Tensor':
        result = Tensor()
        result.impl = TensorImpl(lhs=self.impl, rhs=other.impl, mode=TMode.ADDITION)
        return result
    
    def __sub__(self, other: 'Tensor') -> 'Tensor':
        neg = Tensor()
        neg.impl = TensorImpl(lhs=other.impl, mode=TMode.NEG)
        return self + neg
    
    def __neg__(self) -> 'Tensor':
        result = Tensor()
        result.impl = TensorImpl(lhs=self.impl, mode=TMode.NEG)
        return result
    
    def __mul__(self, other: 'Tensor') -> 'Tensor':
        """Matrix multiplication"""
        result = Tensor()
        result.impl = TensorImpl(lhs=self.impl, rhs=other.impl, mode=TMode.MULT)
        return result
    
    def __truediv__(self, other: 'Tensor') -> 'Tensor':
        """Division (only by scalars)"""
        assert other.shape == (1, 1), "Division only supports scalar divisor"
        result = Tensor()
        result.impl = TensorImpl(lhs=self.impl, rhs=other.impl, mode=TMode.DIV)
        return result


# Utility functions for creating special tensors and operations

def make_function(tensor: Tensor, activation_class) -> Tensor:
    """Apply activation function to tensor"""
    result = Tensor()
    result.impl = TensorImpl(lhs=tensor.impl, mode=TMode.FUNC)
    result.impl.func = activation_class.func
    result.impl.deriv = activation_class.deriv
    return result

def make_log_softmax(tensor: Tensor) -> Tensor:
    """Apply log softmax to tensor"""
    result = Tensor()
    result.impl = TensorImpl(lhs=tensor.impl, mode=TMode.LOG_SOFTMAX)
    return result

def make_flatten(tensors: List[Tensor]) -> Tensor:
    """Flatten multiple tensors into a single column vector"""
    result = Tensor()
    result.impl = TensorImpl(mode=TMode.FLATTEN)
    result.impl.flatten_params = {'members': [t.impl for t in tensors]}
    return result

# Convenience functions for common activation functions
def relu(tensor: Tensor) -> Tensor:
    return make_function(tensor, ActivationFunctions.ReLU)

def gelu(tensor: Tensor) -> Tensor:
    return make_function(tensor, ActivationFunctions.GELU)

def sigmoid(tensor: Tensor) -> Tensor:
    return make_function(tensor, ActivationFunctions.Sigmoid)

def tanh(tensor: Tensor) -> Tensor:
    return make_function(tensor, ActivationFunctions.Tanh)

def square(tensor: Tensor) -> Tensor:
    return make_function(tensor, ActivationFunctions.Square)


# Loss functions
def mse_loss(predictions: Tensor, targets: Tensor) -> Tensor:
    """
    Mean Squared Error loss function
    
    Args:
        predictions: Predicted values tensor
        targets: Target values tensor
        
    Returns:
        Scalar tensor containing the MSE loss
    """
    diff = predictions - targets
    squared_diff = square(diff)
    return squared_diff.sum()


def cross_entropy_loss(predictions: Tensor, targets: Tensor) -> Tensor:
    """
    Cross Entropy loss function
    
    Args:
        predictions: Predicted probabilities tensor (typically output of softmax)
        targets: Target values tensor (one-hot encoded or class indices)
        
    Returns:
        Scalar tensor containing the cross-entropy loss
    """
    # Apply log softmax to predictions for numerical stability
    log_predictions = make_log_softmax(predictions)
    
    # Compute cross-entropy: -sum(targets * log_predictions)
    product = targets * log_predictions
    return -product.sum()


# Export all public functions and classes
__all__ = [
    'Tensor', 'TensorImpl', 'TMode', 'ActivationFunctions',
    'make_function', 'make_log_softmax', 'make_flatten',
    'relu', 'gelu', 'sigmoid', 'tanh', 'square', 'mse_loss', 'cross_entropy_loss',
    '__version__', 'VERSION_INFO'
]