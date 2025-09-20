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
    """
    Implementation class for tensor operations with automatic differentiation
    
    This class represents nodes in a computational graph where each node can be:
    1. A parameter tensor (leaf node) - stores actual data
    2. An operation tensor (internal node) - represents a computation
    
    The class implements lazy evaluation and automatic differentiation using
    the chain rule for gradient computation.
    
    Examples:
        >>> # Create a parameter tensor (leaf node)
        >>> param = TensorImpl(2, 3)  # 2x3 matrix
        >>> param.val.shape  # Output: (2, 3)
        
        >>> # Create an operation tensor (internal node)  
        >>> add_op = TensorImpl(lhs=param1, rhs=param2, mode=TMode.ADDITION)
        >>> add_op.assure_value()  # Computes param1 + param2
    """
    
    def __init__(self, rows: int = 0, cols: int = 0, 
                 lhs: Optional['TensorImpl'] = None, 
                 rhs: Optional['TensorImpl'] = None, 
                 mode: TMode = TMode.UNASSIGNED):
        """
        Initialize a TensorImpl object
        
        Args:
            rows (int): Number of rows (for parameter tensors)
            cols (int): Number of columns (for parameter tensors)  
            lhs (TensorImpl, optional): Left operand (for operation tensors)
            rhs (TensorImpl, optional): Right operand (for operation tensors)
            mode (TMode): Type of operation this tensor represents
            
        Sample Input/Output:
            # Parameter tensor creation:
            >>> impl = TensorImpl(2, 3)
            >>> impl.val.shape          # Output: (2, 3)
            >>> impl.mode              # Output: TMode.PARAMETER
            >>> impl.have_val          # Output: True
            >>> impl.val               # Output: [[0. 0. 0.], [0. 0. 0.]]
            
            # Operation tensor creation:
            >>> add_impl = TensorImpl(lhs=impl1, rhs=impl2, mode=TMode.ADDITION)
            >>> add_impl.val           # Output: None (computed lazily)
            >>> add_impl.have_val      # Output: False
            >>> add_impl.lhs           # Output: impl1
            >>> add_impl.rhs           # Output: impl2
        """
        
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
        """
        Compute the forward pass value if not already computed (lazy evaluation)
        
        This method implements the forward pass of automatic differentiation.
        It recursively computes values by traversing the computation graph
        depth-first, ensuring all dependencies are computed before the current node.
        
        Process:
        1. Check if value already computed or if it's a parameter tensor
        2. Based on operation mode, recursively compute operand values
        3. Apply the specific operation (addition, multiplication, etc.)
        4. Store result and mark as computed
        
        Sample Input/Output Examples:
        
        >>> # Example 1: SUM operation
        >>> param = TensorImpl(2, 2)
        >>> param.val = np.array([[1, 2], [3, 4]], dtype=np.float32)
        >>> sum_op = TensorImpl(lhs=param, mode=TMode.SUM)
        >>> sum_op.assure_value()
        >>> sum_op.val                # Output: [[10.0]] (1+2+3+4=10)
        
        >>> # Example 2: ADDITION operation  
        >>> a = TensorImpl(2, 2)
        >>> a.val = np.array([[1, 2], [3, 4]], dtype=np.float32)
        >>> b = TensorImpl(2, 2) 
        >>> b.val = np.array([[5, 6], [7, 8]], dtype=np.float32)
        >>> add_op = TensorImpl(lhs=a, rhs=b, mode=TMode.ADDITION)
        >>> add_op.assure_value()
        >>> add_op.val               # Output: [[6, 8], [10, 12]]
        
        >>> # Example 3: MATRIX MULTIPLICATION
        >>> a = TensorImpl(2, 3)
        >>> a.val = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32)
        >>> b = TensorImpl(3, 2)
        >>> b.val = np.array([[7, 8], [9, 10], [11, 12]], dtype=np.float32)
        >>> mult_op = TensorImpl(lhs=a, rhs=b, mode=TMode.MULT)
        >>> mult_op.assure_value()
        >>> mult_op.val             # Output: [[58, 64], [139, 154]]
        
        >>> # Example 4: ReLU activation function
        >>> x = TensorImpl(2, 2)
        >>> x.val = np.array([[-1, 2], [-3, 4]], dtype=np.float32)
        >>> relu_op = TensorImpl(lhs=x, mode=TMode.FUNC)
        >>> relu_op.func = ActivationFunctions.ReLU.func
        >>> relu_op.assure_value()
        >>> relu_op.val            # Output: [[0, 2], [0, 4]] (negative values become 0)
        
        Edge Cases Handled:
        - None operands: Returns zero tensor [[0.0]]
        - Already computed: Returns immediately
        - Parameter tensors: No computation needed
        """
        if self.have_val or self.mode == TMode.PARAMETER:
            return
        
        if self.mode == TMode.SUM:
            if self.lhs is not None:
                self.lhs.assure_value()
                if self.lhs.val is not None:
                    self.val = np.array([[self.lhs.val.sum()]], dtype=np.float32)
                else:
                    self.val = np.array([[0.0]], dtype=np.float32)
            else:
                self.val = np.array([[0.0]], dtype=np.float32)
            
        elif self.mode == TMode.ADDITION:
            if self.lhs is not None and self.rhs is not None:
                self.lhs.assure_value()
                self.rhs.assure_value()
                if self.lhs.val is not None and self.rhs.val is not None:
                    self.val = self.lhs.val + self.rhs.val
                else:
                    # Handle cases where one or both values are None
                    lhs_val = self.lhs.val if self.lhs.val is not None else np.zeros_like(self.rhs.val) if self.rhs.val is not None else np.array([[0.0]], dtype=np.float32)
                    rhs_val = self.rhs.val if self.rhs.val is not None else np.zeros_like(self.lhs.val) if self.lhs.val is not None else np.array([[0.0]], dtype=np.float32)
                    self.val = lhs_val + rhs_val
            else:
                self.val = np.array([[0.0]], dtype=np.float32)
            
        elif self.mode == TMode.MULT:
            if self.lhs is not None and self.rhs is not None:
                self.lhs.assure_value()
                self.rhs.assure_value()
                if self.lhs.val is not None and self.rhs.val is not None:
                    self.val = self.lhs.val @ self.rhs.val  # Matrix multiplication
                else:
                    # Default to zero matrix if either operand is None
                    self.val = np.array([[0.0]], dtype=np.float32)
            else:
                self.val = np.array([[0.0]], dtype=np.float32)
            
        elif self.mode == TMode.NEG:
            if self.lhs is not None:
                self.lhs.assure_value()
                if self.lhs.val is not None:
                    self.val = -self.lhs.val
                else:
                    self.val = np.array([[0.0]], dtype=np.float32)
            else:
                self.val = np.array([[0.0]], dtype=np.float32)
            
        elif self.mode == TMode.DIV:
            if self.lhs is not None and self.rhs is not None:
                self.lhs.assure_value()
                self.rhs.assure_value()
                if self.lhs.val is not None and self.rhs.val is not None:
                    # Special case: RHS must be a single number
                    assert self.rhs.val.shape == (1, 1), "Division only supports scalar divisor"
                    self.val = self.lhs.val / self.rhs.val[0, 0]
                else:
                    self.val = np.array([[0.0]], dtype=np.float32)
            else:
                self.val = np.array([[0.0]], dtype=np.float32)
            
        elif self.mode == TMode.DOT_PROD:
            if self.lhs is not None and self.rhs is not None:
                self.lhs.assure_value()
                self.rhs.assure_value()
                if self.lhs.val is not None and self.rhs.val is not None:
                    self.val = self.lhs.val * self.rhs.val  # Element-wise multiplication
                else:
                    self.val = np.array([[0.0]], dtype=np.float32)
            else:
                self.val = np.array([[0.0]], dtype=np.float32)
            
        elif self.mode == TMode.DROPOUT:
            if self.lhs is not None:
                self.lhs.assure_value()
                if self.lhs.val is not None:
                    rate = self.random_params['rate']
                    # Create dropout mask
                    mask = np.random.random(self.lhs.val.shape) > rate
                    # PyTorch-style scaling
                    scale = 1.0 / (1 - rate) if rate < 1.0 else 0.0
                    dropout_mask = mask.astype(np.float32) * scale
                    self.val = self.lhs.val * dropout_mask
                    # Store mask for backward pass
                    self.dropout_mask = dropout_mask
                else:
                    self.val = np.array([[0.0]], dtype=np.float32)
            else:
                self.val = np.array([[0.0]], dtype=np.float32)
            
        elif self.mode == TMode.SLICE:
            if self.lhs is not None:
                self.lhs.assure_value()
                if self.lhs.val is not None:
                    r, c, h, w = self.slice_params['r'], self.slice_params['c'], \
                                self.slice_params['h'], self.slice_params['w']
                    self.val = self.lhs.val[r:r+h, c:c+w].copy()
                else:
                    self.val = np.array([[0.0]], dtype=np.float32)
            else:
                self.val = np.array([[0.0]], dtype=np.float32)
            
        elif self.mode == TMode.FLATTEN:
            total_size = 0
            for member in self.flatten_params['members']:
                if member is not None:
                    member.assure_value()
                    if member.val is not None:
                        total_size += member.val.size
            
            self.val = np.zeros((total_size, 1), dtype=np.float32)
            pos = 0
            for member in self.flatten_params['members']:
                if member is not None and member.val is not None:
                    flat = member.val.flatten('F')  # Fortran order to match C++
                    self.val[pos:pos+len(flat), 0] = flat
                    pos += len(flat)
                
        elif self.mode == TMode.FUNC:
            if self.lhs is not None:
                self.lhs.assure_value()
                if self.func is not None and self.lhs.val is not None:
                    self.val = self.func(self.lhs.val)
                else:
                    self.val = np.array([[0.0]], dtype=np.float32)
            else:
                self.val = np.array([[0.0]], dtype=np.float32)
                
        elif self.mode == TMode.LOG_SOFTMAX:
            if self.lhs is not None:
                self.lhs.assure_value()
                if self.lhs.val is not None:
                    # Numerically stable log softmax
                    max_val = np.max(self.lhs.val)
                    exp_vals = np.exp(self.lhs.val - max_val)
                    sum_exp = np.sum(exp_vals)
                    self.val = self.lhs.val - max_val - np.log(sum_exp)
                else:
                    self.val = np.array([[0.0]], dtype=np.float32)
            else:
                self.val = np.array([[0.0]], dtype=np.float32)
            
        elif self.mode == TMode.CONVO:
            if self.lhs is not None and self.rhs is not None and self.convo_params['bias'] is not None:
                self.lhs.assure_value()  # input
                self.rhs.assure_value()  # weights
                self.convo_params['bias'].assure_value()  # bias
                
                if (self.lhs.val is not None and self.rhs.val is not None and 
                    self.convo_params['bias'].val is not None):
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
                else:
                    self.val = np.array([[0.0]], dtype=np.float32)
            else:
                self.val = np.array([[0.0]], dtype=np.float32)
                    
        elif self.mode == TMode.MAX2D:
            if self.lhs is not None:
                self.lhs.assure_value()
                if self.lhs.val is not None:
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
                    self.val = np.array([[0.0]], dtype=np.float32)
            else:
                self.val = np.array([[0.0]], dtype=np.float32)
        else:
            raise ValueError(f"Unknown mode: {self.mode}")
        
        # Initialize gradients with same shape as value
        if self.val is not None:
            self.grads = np.zeros_like(self.val)
            self.accum_grads = np.zeros_like(self.val)
        else:
            self.grads = np.array([[0.0]], dtype=np.float32)
            self.accum_grads = np.array([[0.0]], dtype=np.float32)
        self.have_val = True
    
    def build_topo(self, visited: set, topo: List['TensorImpl']):
        """
        Build topological ordering for backpropagation using depth-first search
        
        Creates a topologically sorted list of tensors in the computation graph,
        ensuring that dependencies are processed before dependents during backward pass.
        This is crucial for automatic differentiation to work correctly.
        
        Algorithm:
        1. Mark current node as visited (cycle detection)
        2. Recursively visit all dependencies (left, right operands)
        3. Add current node to topological order after all dependencies
        4. Result: Dependencies appear before dependents in the list
        
        Args:
            visited (set): Set to track visited nodes (prevents cycles/duplicates)
            topo (List[TensorImpl]): List to store topological ordering
            
        Sample Input/Output:
        
        >>> # Example: Build topo for expression (A + B) * C
        >>> # Graph structure:
        >>> #     A   B
        >>> #      \\ /
        >>> #     ADD    C  
        >>> #       \\   /
        >>> #       MULT
        >>> #        |
        >>> #     Result
        
        >>> A = TensorImpl(2, 2)           # Parameter tensor
        >>> B = TensorImpl(2, 2)           # Parameter tensor  
        >>> C = TensorImpl(2, 2)           # Parameter tensor
        >>> ADD = TensorImpl(lhs=A, rhs=B, mode=TMode.ADDITION)
        >>> MULT = TensorImpl(lhs=ADD, rhs=C, mode=TMode.MULT)
        
        >>> visited = set()
        >>> topo = []
        >>> MULT.build_topo(visited, topo)
        >>> # Output order: [A, B, ADD, C, MULT]
        >>> # This ensures: A,B computed before ADD; ADD,C computed before MULT
        
        >>> # Example: Complex expression (A @ B) + (C * D)
        >>> #     A   B       C   D
        >>> #      \\ /         \\ /
        >>> #      MULT       DOT
        >>> #        \\       /
        >>> #         \\     /
        >>> #          ADD
        >>> #           |
        >>> #        Result
        
        >>> A = TensorImpl(2, 3)
        >>> B = TensorImpl(3, 2) 
        >>> C = TensorImpl(2, 2)
        >>> D = TensorImpl(2, 2)
        >>> MULT = TensorImpl(lhs=A, rhs=B, mode=TMode.MULT)
        >>> DOT = TensorImpl(lhs=C, rhs=D, mode=TMode.DOT_PROD)
        >>> ADD = TensorImpl(lhs=MULT, rhs=DOT, mode=TMode.ADDITION)
        
        >>> visited = set()
        >>> topo = []
        >>> ADD.build_topo(visited, topo)
        >>> # Output order: [A, B, MULT, C, D, DOT, ADD]
        >>> # Guarantees proper dependency order for gradient computation
        
        Why Topological Order Matters:
        - Forward pass: Can compute in any order (lazy evaluation handles dependencies)
        - Backward pass: Must process in reverse topological order
        - Ensures gradients flow correctly through the computation graph
        - Prevents accessing uncomputed gradients
        
        Edge Cases:
        - Cycles: Prevented by visited set (shouldn't occur in valid computation graphs)
        - Multiple paths to same node: Node added only once
        - Special operations: FLATTEN and CONVO handle multiple dependencies
        """
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
        """
        Compute gradients (backward pass) using automatic differentiation
        
        This method implements the backward pass by applying the chain rule to
        propagate gradients from the current node to its operands. Each operation
        type has its own gradient computation rule based on calculus derivatives.
        
        Process:
        1. Check operation mode to determine gradient rules
        2. Compute local gradients using derivative formulas
        3. Apply chain rule: local_grad × incoming_grad
        4. Accumulate gradients in operand tensors
        
        Mathematical Background:
        - Addition: ∂(A + B)/∂A = 1, ∂(A + B)/∂B = 1
        - Multiplication: ∂(A @ B)/∂A = grad @ B^T, ∂(A @ B)/∂B = A^T @ grad
        - ReLU: ∂ReLU(x)/∂x = 1 if x > 0, else 0
        - Sum: ∂sum(A)/∂A = ones_like(A)
        
        Sample Input/Output Examples:
        
        >>> # Example 1: Addition gradient
        >>> # Forward: C = A + B where A=[[1,2]], B=[[3,4]], C=[[4,6]]
        >>> # Backward: If ∂L/∂C = [[1,1]], then:
        >>> a.grads  # Before: [[0,0]]
        >>> b.grads  # Before: [[0,0]]
        >>> c.grads = np.array([[1, 1]], dtype=np.float32)  # Loss gradient
        >>> c.do_grad()  # Apply chain rule
        >>> a.grads  # After: [[1,1]] (∂L/∂A = ∂L/∂C × ∂C/∂A = [[1,1]] × 1)
        >>> b.grads  # After: [[1,1]] (∂L/∂B = ∂L/∂C × ∂C/∂B = [[1,1]] × 1)
        
        >>> # Example 2: Matrix multiplication gradient  
        >>> # Forward: C = A @ B where A=[[1,2],[3,4]], B=[[5,6],[7,8]]
        >>> # C = [[19,22],[43,50]]
        >>> # Backward: If ∂L/∂C = [[1,1],[1,1]], then:
        >>> a.grads  # Before: [[0,0],[0,0]]
        >>> b.grads  # Before: [[0,0],[0,0]]
        >>> c.grads = np.array([[1, 1], [1, 1]], dtype=np.float32)
        >>> c.do_grad()
        >>> a.grads  # After: [[11,15],[11,15]] (∂L/∂C @ B^T)
        >>> b.grads  # After: [[4,4],[6,6]] (A^T @ ∂L/∂C)
        
        >>> # Example 3: ReLU gradient
        >>> # Forward: Y = ReLU(X) where X=[[-1,2],[-3,4]], Y=[[0,2],[0,4]]
        >>> # Backward: If ∂L/∂Y = [[1,1],[1,1]], then:
        >>> x.grads  # Before: [[0,0],[0,0]]
        >>> y.grads = np.array([[1, 1], [1, 1]], dtype=np.float32)
        >>> y.do_grad()
        >>> x.grads  # After: [[0,1],[0,1]] (gradient blocked where X <= 0)
        
        >>> # Example 4: Sum gradient  
        >>> # Forward: Y = sum(X) where X=[[1,2],[3,4]], Y=[[10]]
        >>> # Backward: If ∂L/∂Y = [[1]], then:
        >>> x.grads  # Before: [[0,0],[0,0]]
        >>> y.grads = np.array([[1]], dtype=np.float32)
        >>> y.do_grad()
        >>> x.grads  # After: [[1,1],[1,1]] (gradient broadcast to all elements)
        
        Edge Cases Handled:
        - None gradients: Skip gradient computation
        - Parameter tensors: No further propagation needed
        - Zero gradients: Handled naturally by addition
        """
        if self.mode == TMode.PARAMETER:
            return
            
        elif self.mode == TMode.FLATTEN:
            grad_pos = 0
            for member in self.flatten_params['members']:
                if member is not None and member.grads is not None and self.grads is not None:
                    flat_size = member.grads.size
                    member_grads_flat = self.grads[grad_pos:grad_pos+flat_size, 0]
                    member.grads += member_grads_flat.reshape(member.grads.shape, order='F')
                    grad_pos += flat_size
                
        elif self.mode == TMode.ADDITION:
            if self.lhs is not None and self.lhs.grads is not None and self.grads is not None:
                self.lhs.grads += self.grads
            if self.rhs is not None and self.rhs.grads is not None and self.grads is not None:
                self.rhs.grads += self.grads
            
        elif self.mode == TMode.NEG:
            if self.lhs is not None and self.lhs.grads is not None and self.grads is not None:
                self.lhs.grads -= self.grads
            
        elif self.mode == TMode.MULT:
            # Matrix multiplication gradients
            if (self.lhs is not None and self.lhs.grads is not None and 
                self.rhs is not None and self.rhs.val is not None and self.grads is not None):
                self.lhs.grads += self.grads @ self.rhs.val.T
            if (self.rhs is not None and self.rhs.grads is not None and 
                self.lhs is not None and self.lhs.val is not None and self.grads is not None):
                self.rhs.grads += self.lhs.val.T @ self.grads
            
        elif self.mode == TMode.DIV:
            if (self.lhs is not None and self.lhs.grads is not None and 
                self.rhs is not None and self.rhs.val is not None and self.grads is not None):
                self.lhs.grads += self.grads / self.rhs.val[0, 0]
            # Note: RHS gradient computation omitted as noted in C++ code
            
        elif self.mode == TMode.DOT_PROD:
            if (self.lhs is not None and self.lhs.grads is not None and 
                self.rhs is not None and self.rhs.val is not None and self.grads is not None):
                self.lhs.grads += self.grads * self.rhs.val
            if (self.rhs is not None and self.rhs.grads is not None and 
                self.lhs is not None and self.lhs.val is not None and self.grads is not None):
                self.rhs.grads += self.grads * self.lhs.val
            
        elif self.mode == TMode.DROPOUT:
            if (self.lhs is not None and self.lhs.grads is not None and 
                self.grads is not None and hasattr(self, 'dropout_mask')):
                self.lhs.grads += self.grads * self.dropout_mask
            
        elif self.mode == TMode.SLICE:
            if (self.lhs is not None and self.lhs.grads is not None and self.grads is not None):
                r, c, h, w = self.slice_params['r'], self.slice_params['c'], \
                            self.slice_params['h'], self.slice_params['w']
                self.lhs.grads[r:r+h, c:c+w] += self.grads
            
        elif self.mode == TMode.SUM:
            if (self.lhs is not None and self.lhs.grads is not None and self.grads is not None):
                self.lhs.grads += self.grads[0, 0]
            
        elif self.mode == TMode.LOG_SOFTMAX:
            # Log softmax gradient
            if (self.lhs is not None and self.lhs.grads is not None and 
                self.val is not None and self.grads is not None):
                exp_vals = np.exp(self.val)
                self.lhs.grads += self.grads - exp_vals * np.sum(self.grads)
            
        elif self.mode == TMode.FUNC:
            if (self.lhs is not None and self.lhs.grads is not None and 
                self.lhs.val is not None and self.grads is not None):
                if self.deriv is not None:
                    self.lhs.grads += self.grads * self.deriv(self.lhs.val)
                else:
                    raise ValueError("Derivative function not set for FUNC mode")
                
        elif self.mode == TMode.CONVO:
            # Convolution backward pass
            if (self.lhs is not None and self.rhs is not None and 
                self.convo_params['bias'] is not None and self.val is not None and self.grads is not None):
                kernel = self.convo_params['kernel']
                
                # Gradients to input
                if (not self.lhs.no_grad and self.lhs.grads is not None and 
                    self.rhs.val is not None):
                    for r in range(self.val.shape[0]):
                        for c in range(self.val.shape[1]):
                            self.lhs.grads[r:r+kernel, c:c+kernel] += \
                                self.rhs.val * self.grads[r, c]
                
                # Gradients to weights
                if (self.rhs.grads is not None and self.rhs.val is not None and 
                    self.lhs.val is not None):
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
                if self.convo_params['bias'].grads is not None:
                    self.convo_params['bias'].grads[0, 0] += np.sum(self.grads)
            
        elif self.mode == TMode.MAX2D:
            # Max pooling backward pass
            if (self.lhs is not None and self.lhs.val is not None and 
                self.lhs.grads is not None and self.grads is not None):
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
    """
    Main Tensor class - High-level user interface for tensor operations
    
    This class provides a user-friendly API for creating and manipulating tensors
    with automatic differentiation capabilities. It wraps the TensorImpl class
    and provides convenient methods for common operations.
    
    Key Features:
    - Easy tensor creation and initialization
    - Operator overloading for natural mathematical expressions
    - Automatic differentiation through computation graph building
    - NumPy-like interface for familiar usage
    
    Examples:
        >>> # Create tensors
        >>> a = Tensor(2, 3)           # 2x3 matrix tensor
        >>> b = Tensor(value=5.0)      # Scalar tensor with value 5.0
        >>> c = Tensor()               # Empty tensor (operation result)
        
        >>> # Basic operations
        >>> result = a + b             # Addition
        >>> product = a * b            # Matrix multiplication
        >>> activated = relu(a)        # Apply activation function
    """
    
    def __init__(self, rows: int = 0, cols: int = 0, value: Optional[float] = None):
        """
        Initialize a new Tensor
        
        Creates either a parameter tensor (with specific dimensions) or a scalar tensor.
        Parameter tensors are initialized with zeros and can store gradients.
        
        Args:
            rows (int): Number of rows (default: 0 for operation tensors)
            cols (int): Number of columns (default: 0 for operation tensors)
            value (float, optional): If provided, creates a 1x1 scalar tensor
            
        Sample Input/Output:
            >>> # Matrix tensor creation
            >>> tensor = Tensor(2, 3)
            >>> tensor.shape              # Output: (2, 3)
            >>> tensor[0, 0]              # Output: 0.0 (initialized to zero)
            >>> tensor.impl.mode          # Output: TMode.PARAMETER
            
            >>> # Scalar tensor creation
            >>> scalar = Tensor(value=42.0)
            >>> scalar.shape              # Output: (1, 1)  
            >>> scalar[0, 0]              # Output: 42.0
            >>> scalar.impl.val           # Output: [[42.0]]
            
            >>> # Operation tensor (created by operations)
            >>> result = Tensor()
            >>> result.impl.mode          # Output: TMode.UNASSIGNED
            >>> result.impl.val           # Output: None (computed lazily)
        """
        if value is not None:
            # Create scalar tensor
            self.impl = TensorImpl(1, 1)
            if self.impl.val is not None:
                self.impl.val[0, 0] = value
        else:
            self.impl = TensorImpl(rows, cols)
    
    def __getitem__(self, key):
        """
        Get element by index (supports both single index and tuple indexing)
        
        Automatically computes tensor value if not already computed (lazy evaluation).
        Supports both matrix-style indexing tensor[row, col] and linear indexing.
        
        Args:
            key: Index (int) or tuple of indices (row, col)
            
        Returns:
            float: Element value at specified position
            
        Sample Input/Output:
            >>> # Matrix indexing
            >>> tensor = Tensor(3, 3)
            >>> tensor.constant(5.0)
            >>> tensor[1, 2]              # Output: 5.0
            >>> tensor[0, 0]              # Output: 5.0
            
            >>> # Linear indexing
            >>> tensor[0]                 # Output: 5.0 (first row)
            
            >>> # With computation results
            >>> a = Tensor(2, 2)
            >>> a.constant(2.0)
            >>> b = Tensor(2, 2) 
            >>> b.constant(3.0)
            >>> result = a + b            # Creates computation graph
            >>> result[0, 0]              # Output: 5.0 (triggers computation)
            >>> result[1, 1]              # Output: 5.0
        """
        self.impl.assure_value()
        if self.impl.val is not None:
            if isinstance(key, tuple):
                return self.impl.val[key[0], key[1]]
            else:
                return self.impl.val[key]
        else:
            return 0.0
    
    def __setitem__(self, key, value):
        """
        Set element by index (supports both single index and tuple indexing)
        
        Directly modifies the underlying numpy array. Only works for parameter
        tensors (not operation results). Used to initialize tensor values.
        
        Args:
            key: Index (int) or tuple of indices (row, col)
            value (float): Value to set at specified position
            
        Sample Input/Output:
            >>> # Matrix element setting
            >>> tensor = Tensor(3, 3)
            >>> tensor[0, 0] = 1.0
            >>> tensor[1, 2] = 2.5
            >>> tensor[0, 0]              # Output: 1.0
            >>> tensor[1, 2]              # Output: 2.5
            >>> tensor[2, 2]              # Output: 0.0 (unchanged)
            
            >>> # Linear indexing  
            >>> tensor[1] = 3.0           # Sets first element of row 1
            >>> tensor[1, 0]              # Output: 3.0
            
            >>> # Building a matrix
            >>> m = Tensor(2, 2)
            >>> m[0, 0] = 1.0
            >>> m[0, 1] = 2.0  
            >>> m[1, 0] = 3.0
            >>> m[1, 1] = 4.0
            >>> # Result: [[1.0, 2.0], [3.0, 4.0]]
        """
        if self.impl.val is not None:
            if isinstance(key, tuple):
                self.impl.val[key[0], key[1]] = value
            else:
                self.impl.val[key] = value
    
    def raw(self) -> np.ndarray:
        """Get raw numpy array (only for parameter tensors)"""
        assert self.impl.mode == TMode.PARAMETER, "raw() only available for parameter tensors"
        if self.impl.val is not None:
            return self.impl.val
        else:
            return np.array([[0.0]], dtype=np.float32)
    
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
        if self.impl.val is not None:
            self.impl.val = np.random.uniform(-1, 1, self.impl.val.shape).astype(np.float32)
            self.impl.val *= factor
    
    def zero(self):
        """Fill with zeros"""
        self.constant(0.0)
    
    def constant(self, value: float):
        """
        Fill tensor with constant value and mark as parameter
        
        Initializes all elements of the tensor to the same constant value
        and sets the tensor mode to PARAMETER, making it trainable.
        This is commonly used for weight initialization in neural networks.
        
        Args:
            value (float): Constant value to fill tensor with
            
        Sample Input/Output:
            >>> # Initialize weight matrix with small values
            >>> weights = Tensor(3, 4)
            >>> weights.constant(0.01)
            >>> weights[0, 0]              # Output: 0.01
            >>> weights[2, 3]              # Output: 0.01
            >>> weights.impl.mode          # Output: TMode.PARAMETER
            
            >>> # Initialize bias vector to zeros
            >>> bias = Tensor(4, 1)
            >>> bias.constant(0.0)
            >>> bias[0, 0]                 # Output: 0.0
            >>> bias[3, 0]                 # Output: 0.0
            
            >>> # Initialize identity-like values
            >>> identity_scale = Tensor(2, 2)
            >>> identity_scale.constant(1.0)
            >>> identity_scale[0, 0]       # Output: 1.0
            >>> identity_scale[1, 1]       # Output: 1.0
        """
        self.impl.mode = TMode.PARAMETER
        if self.impl.val is not None:
            self.impl.val.fill(value)
    
    def one_hot_column(self, c: int):
        """
        Set one-hot encoding for specified column
        
        Creates a one-hot vector where only the specified column is 1.0
        and all other elements are 0.0. Commonly used for categorical
        data representation in machine learning.
        
        Args:
            c (int): Column index to set to 1.0 (zero-based)
            
        Sample Input/Output:
            >>> # Create one-hot for class 2 out of 5 classes
            >>> classes = Tensor(1, 5)
            >>> classes.one_hot_column(2)
            >>> classes[0, 0]              # Output: 0.0
            >>> classes[0, 1]              # Output: 0.0
            >>> classes[0, 2]              # Output: 1.0 (hot)
            >>> classes[0, 3]              # Output: 0.0
            >>> classes[0, 4]              # Output: 0.0
            
            >>> # Label encoding for digit classification
            >>> digit_label = Tensor(1, 10)  # 10 digits (0-9)
            >>> digit_label.one_hot_column(7)  # Digit "7"
            >>> digit_label[0, 7]          # Output: 1.0
            >>> sum(digit_label[0, i] for i in range(10))  # Output: 1.0
        """
        self.zero()
        if self.impl.val is not None:
            self.impl.val[0, c] = 1.0
    
    def one_hot_row(self, r: int):
        """
        Set one-hot encoding for specified row
        
        Creates a one-hot vector where only the specified row is 1.0
        and all other elements are 0.0. Used for categorical data when
        data is organized in column format.
        
        Args:
            r (int): Row index to set to 1.0 (zero-based)
            
        Sample Input/Output:
            >>> # Create one-hot for class 3 out of 6 classes (column format)
            >>> classes = Tensor(6, 1)
            >>> classes.one_hot_row(3)
            >>> classes[0, 0]              # Output: 0.0
            >>> classes[1, 0]              # Output: 0.0
            >>> classes[2, 0]              # Output: 0.0
            >>> classes[3, 0]              # Output: 1.0 (hot)
            >>> classes[4, 0]              # Output: 0.0
            >>> classes[5, 0]              # Output: 0.0
            
            >>> # Sentiment classification (positive/negative/neutral)
            >>> sentiment = Tensor(3, 1)
            >>> sentiment.one_hot_row(1)    # Negative sentiment
            >>> sentiment[1, 0]             # Output: 1.0
        """
        self.zero()
        if self.impl.val is not None:
            self.impl.val[r, 0] = 1.0
    
    def dot(self, other: 'Tensor') -> 'Tensor':
        """
        Element-wise multiplication (Hadamard product)
        
        Performs element-wise multiplication between two tensors of the same shape.
        This is different from matrix multiplication (__mul__) - it multiplies
        corresponding elements rather than doing linear algebra operations.
        
        Args:
            other (Tensor): Tensor to multiply element-wise (must have same shape)
            
        Returns:
            Tensor: New tensor with element-wise multiplication result
            
        Sample Input/Output:
            >>> # Element-wise multiplication of matrices
            >>> a = Tensor(2, 2)
            >>> a[0, 0] = 2.0; a[0, 1] = 3.0
            >>> a[1, 0] = 4.0; a[1, 1] = 5.0
            >>> # a = [[2, 3], [4, 5]]
            
            >>> b = Tensor(2, 2)
            >>> b[0, 0] = 1.0; b[0, 1] = 2.0
            >>> b[1, 0] = 3.0; b[1, 1] = 4.0
            >>> # b = [[1, 2], [3, 4]]
            
            >>> result = a.dot(b)
            >>> result[0, 0]               # Output: 2.0 (2*1)
            >>> result[0, 1]               # Output: 6.0 (3*2)
            >>> result[1, 0]               # Output: 12.0 (4*3)
            >>> result[1, 1]               # Output: 20.0 (5*4)
            
            >>> # Neural network weight masking
            >>> weights = Tensor(3, 3)
            >>> weights.constant(0.5)
            >>> mask = Tensor(3, 3)
            >>> mask.constant(0.8)         # 80% of weights kept
            >>> masked_weights = weights.dot(mask)
            >>> masked_weights[0, 0]       # Output: 0.4 (0.5 * 0.8)
        """
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
        """
        Get tensor dimensions as tuple
        
        Returns the shape of the tensor as a (rows, cols) tuple.
        For operation results, ensures value is computed first through lazy evaluation.
        
        Returns:
            tuple: (rows, cols) dimensions of the tensor
            
        Sample Input/Output:
            >>> # Parameter tensor shape
            >>> tensor = Tensor(3, 4)
            >>> tensor.shape              # Output: (3, 4)
            
            >>> # Scalar tensor shape  
            >>> scalar = Tensor(value=5.0)
            >>> scalar.shape              # Output: (1, 1)
            
            >>> # Operation result shape
            >>> a = Tensor(2, 3)
            >>> b = Tensor(2, 3)
            >>> result = a + b
            >>> result.shape              # Output: (2, 3)
            
            >>> # Matrix multiplication shape
            >>> m1 = Tensor(3, 4)
            >>> m2 = Tensor(4, 2)  
            >>> product = m1 * m2
            >>> product.shape             # Output: (3, 2)
            
            >>> # Convolution shape (depends on kernel size)
            >>> input_img = Tensor(28, 28)
            >>> weights = Tensor(3, 3)
            >>> conv = input_img.make_convo(3, weights, bias)
            >>> conv.shape                # Output: (26, 26) for 3x3 kernel
        """
        self.impl.assure_value()
        if self.impl.val is not None:
            return self.impl.val.shape
        else:
            return (1, 1)
    
    def __str__(self):
        """String representation"""
        self.impl.assure_value()
        return str(self.impl.val)
    
    def __repr__(self):
        return f"Tensor(shape={self.shape})"
    
    # Arithmetic operators
    def __add__(self, other: 'Tensor') -> 'Tensor':
        """
        Element-wise addition operator (+)
        
        Performs element-wise addition between two tensors. Creates a new tensor
        in the computation graph representing the addition operation.
        Tensors must have compatible shapes for broadcasting.
        
        Args:
            other (Tensor): Right operand tensor
            
        Returns:
            Tensor: New tensor representing addition result
            
        Sample Input/Output:
            >>> # Matrix addition
            >>> a = Tensor(2, 2)
            >>> a.constant(3.0)
            >>> b = Tensor(2, 2)
            >>> b.constant(2.0)
            >>> result = a + b
            >>> result[0, 0]              # Output: 5.0
            >>> result[1, 1]              # Output: 5.0
            
            >>> # Scalar addition with broadcasting
            >>> matrix = Tensor(2, 2)
            >>> matrix[0, 0] = 1.0; matrix[0, 1] = 2.0
            >>> matrix[1, 0] = 3.0; matrix[1, 1] = 4.0
            >>> scalar = Tensor(value=10.0)
            >>> shifted = matrix + scalar
            >>> shifted[0, 0]             # Output: 11.0
            >>> shifted[1, 1]             # Output: 14.0
            
            >>> # Neural network bias addition
            >>> layer_output = Tensor(1, 3)
            >>> layer_output[0, 0] = 0.5; layer_output[0, 1] = -0.2
            >>> layer_output[0, 2] = 0.8
            >>> bias = Tensor(1, 3)
            >>> bias[0, 0] = 0.1; bias[0, 1] = 0.2; bias[0, 2] = -0.1
            >>> final_output = layer_output + bias
            >>> final_output[0, 0]        # Output: 0.6 (0.5 + 0.1)
            >>> final_output[0, 1]        # Output: 0.0 (-0.2 + 0.2)
            >>> final_output[0, 2]        # Output: 0.7 (0.8 + (-0.1))
        """
        result = Tensor()
        result.impl = TensorImpl(lhs=self.impl, rhs=other.impl, mode=TMode.ADDITION)
        return result
    
    def __sub__(self, other: 'Tensor') -> 'Tensor':
        """
        Subtraction operator (-)
        
        Performs element-wise subtraction by creating negation of the right operand
        and then adding. Equivalent to self + (-other).
        
        Args:
            other (Tensor): Right operand tensor to subtract
            
        Returns:
            Tensor: New tensor representing subtraction result
            
        Sample Input/Output:
            >>> # Matrix subtraction
            >>> a = Tensor(2, 2)
            >>> a.constant(5.0)
            >>> b = Tensor(2, 2)
            >>> b.constant(2.0)
            >>> result = a - b
            >>> result[0, 0]              # Output: 3.0
            >>> result[1, 1]              # Output: 3.0
            
            >>> # Gradient computation (error calculation)
            >>> predicted = Tensor(1, 3)
            >>> predicted[0, 0] = 0.8; predicted[0, 1] = 0.1; predicted[0, 2] = 0.1
            >>> actual = Tensor(1, 3)
            >>> actual[0, 0] = 1.0; actual[0, 1] = 0.0; actual[0, 2] = 0.0
            >>> error = predicted - actual
            >>> error[0, 0]               # Output: -0.2 (0.8 - 1.0)
            >>> error[0, 1]               # Output: 0.1 (0.1 - 0.0)
            >>> error[0, 2]               # Output: 0.1 (0.1 - 0.0)
        """
        neg = Tensor()
        neg.impl = TensorImpl(lhs=other.impl, mode=TMode.NEG)
        return self + neg
    
    def __neg__(self) -> 'Tensor':
        """
        Unary negation operator (-)
        
        Creates a new tensor with all elements negated. Useful for implementing
        subtraction and creating negative weights or biases.
        
        Returns:
            Tensor: New tensor with negated values
            
        Sample Input/Output:
            >>> # Negate a matrix
            >>> a = Tensor(2, 2)
            >>> a[0, 0] = 1.0; a[0, 1] = -2.0
            >>> a[1, 0] = 3.0; a[1, 1] = -4.0
            >>> neg_a = -a
            >>> neg_a[0, 0]               # Output: -1.0
            >>> neg_a[0, 1]               # Output: 2.0
            >>> neg_a[1, 0]               # Output: -3.0
            >>> neg_a[1, 1]               # Output: 4.0
            
            >>> # Gradient descent (opposite direction)
            >>> gradient = Tensor(2, 1)
            >>> gradient[0, 0] = 0.1; gradient[1, 0] = -0.05
            >>> descent_direction = -gradient
            >>> descent_direction[0, 0]   # Output: -0.1
            >>> descent_direction[1, 0]   # Output: 0.05
        """
        result = Tensor()
        result.impl = TensorImpl(lhs=self.impl, mode=TMode.NEG)
        return result
    
    def __mul__(self, other: 'Tensor') -> 'Tensor':
        """
        Matrix multiplication operator (*)
        
        Performs matrix multiplication between two tensors. Creates a new tensor
        in the computation graph representing the multiplication operation.
        For valid multiplication: left.cols must equal right.rows.
        
        Args:
            other (Tensor): Right operand tensor
            
        Returns:
            Tensor: New tensor representing multiplication result
            
        Sample Input/Output:
            >>> # Standard matrix multiplication
            >>> a = Tensor(2, 3)
            >>> a[0, 0] = 1.0; a[0, 1] = 2.0; a[0, 2] = 3.0
            >>> a[1, 0] = 4.0; a[1, 1] = 5.0; a[1, 2] = 6.0
            >>> # a = [[1, 2, 3], [4, 5, 6]]
            
            >>> b = Tensor(3, 2)
            >>> b[0, 0] = 1.0; b[0, 1] = 2.0
            >>> b[1, 0] = 3.0; b[1, 1] = 4.0  
            >>> b[2, 0] = 5.0; b[2, 1] = 6.0
            >>> # b = [[1, 2], [3, 4], [5, 6]]
            
            >>> result = a * b            # (2x3) * (3x2) = (2x2)
            >>> result[0, 0]              # Output: 22.0 (1*1 + 2*3 + 3*5)
            >>> result[0, 1]              # Output: 28.0 (1*2 + 2*4 + 3*6)
            >>> result[1, 0]              # Output: 49.0 (4*1 + 5*3 + 6*5)
            >>> result[1, 1]              # Output: 64.0 (4*2 + 5*4 + 6*6)
            
            >>> # Neural network layer computation
            >>> input_vec = Tensor(1, 4)  # Input features
            >>> input_vec[0, 0] = 1.0; input_vec[0, 1] = 0.5
            >>> input_vec[0, 2] = -0.2; input_vec[0, 3] = 0.8
            >>> weights = Tensor(4, 3)    # Weight matrix (4 inputs, 3 outputs)
            >>> weights.constant(0.1)     # Initialize with small weights
            >>> layer_output = input_vec * weights  # (1x4) * (4x3) = (1x3)
            >>> layer_output.shape        # Output: (1, 3)
        """
        result = Tensor()
        result.impl = TensorImpl(lhs=self.impl, rhs=other.impl, mode=TMode.MULT)
        return result
    
    def __truediv__(self, other: 'Tensor') -> 'Tensor':
        """
        Division operator (/) - only supports scalar divisors
        
        Performs element-wise division by a scalar tensor. All elements of the
        left operand are divided by the single scalar value in the right operand.
        This is commonly used for normalization and scaling operations.
        
        Args:
            other (Tensor): Scalar tensor (1x1) to divide by
            
        Returns:
            Tensor: New tensor with division result
            
        Raises:
            AssertionError: If divisor is not a scalar (1x1) tensor
            
        Sample Input/Output:
            >>> # Scale down a matrix
            >>> matrix = Tensor(2, 2)
            >>> matrix[0, 0] = 10.0; matrix[0, 1] = 20.0
            >>> matrix[1, 0] = 30.0; matrix[1, 1] = 40.0
            >>> divisor = Tensor(value=2.0)
            >>> result = matrix / divisor
            >>> result[0, 0]              # Output: 5.0 (10/2)
            >>> result[0, 1]              # Output: 10.0 (20/2)
            >>> result[1, 0]              # Output: 15.0 (30/2)
            >>> result[1, 1]              # Output: 20.0 (40/2)
            
            >>> # Normalize by sum (simple normalization)
            >>> data = Tensor(1, 4)
            >>> data[0, 0] = 1.0; data[0, 1] = 2.0
            >>> data[0, 2] = 3.0; data[0, 3] = 4.0
            >>> total = Tensor(value=10.0)  # Sum of all elements
            >>> normalized = data / total
            >>> normalized[0, 0]          # Output: 0.1 (1/10)
            >>> normalized[0, 1]          # Output: 0.2 (2/10)
            >>> normalized[0, 2]          # Output: 0.3 (3/10)
            >>> normalized[0, 3]          # Output: 0.4 (4/10)
            
            >>> # Learning rate scaling in optimization
            >>> gradient = Tensor(2, 1)
            >>> gradient[0, 0] = 0.5; gradient[1, 0] = -0.3
            >>> learning_rate = Tensor(value=0.01)
            >>> update = gradient / learning_rate  # Scale by 1/lr
            >>> update[0, 0]              # Output: 50.0 (0.5/0.01)
            >>> update[1, 0]              # Output: -30.0 (-0.3/0.01)
        """
        assert other.shape == (1, 1), "Division only supports scalar divisor"
        result = Tensor()
        result.impl = TensorImpl(lhs=self.impl, rhs=other.impl, mode=TMode.DIV)
        return result


# Utility functions for creating special tensors and operations

def make_function(tensor: Tensor, activation_class) -> Tensor:
    """
    Apply activation function to tensor
    
    Creates a new tensor in the computation graph that applies the specified
    activation function element-wise to the input tensor. This is the core
    mechanism for applying non-linear transformations in neural networks.
    
    Args:
        tensor (Tensor): Input tensor to apply activation to
        activation_class: Activation function class with func and deriv methods
        
    Returns:
        Tensor: New tensor representing the activation result
        
    Sample Input/Output:
        >>> # Apply ReLU activation
        >>> x = Tensor(2, 2)
        >>> x[0, 0] = -1.0; x[0, 1] = 2.0
        >>> x[1, 0] = -3.0; x[1, 1] = 4.0
        >>> relu_x = make_function(x, ActivationFunctions.ReLU)
        >>> relu_x[0, 0]              # Output: 0.0 (max(0, -1))
        >>> relu_x[0, 1]              # Output: 2.0 (max(0, 2))
        >>> relu_x[1, 0]              # Output: 0.0 (max(0, -3))
        >>> relu_x[1, 1]              # Output: 4.0 (max(0, 4))
        
        >>> # Apply Sigmoid activation  
        >>> logits = Tensor(1, 2)
        >>> logits[0, 0] = 0.0; logits[0, 1] = 2.0
        >>> sigmoid_x = make_function(logits, ActivationFunctions.Sigmoid)
        >>> sigmoid_x[0, 0]           # Output: 0.5 (sigmoid(0))
        >>> sigmoid_x[0, 1]           # Output: ~0.88 (sigmoid(2))
    """
    result = Tensor()
    result.impl = TensorImpl(lhs=tensor.impl, mode=TMode.FUNC)
    result.impl.func = activation_class.func
    result.impl.deriv = activation_class.deriv
    return result

def make_log_softmax(tensor: Tensor) -> Tensor:
    """
    Apply log softmax to tensor
    
    Computes the logarithm of the softmax function, which is numerically more
    stable than computing softmax followed by log. Commonly used in classification
    tasks and cross-entropy loss computation.
    
    Mathematical Formula:
        log_softmax(x_i) = log(exp(x_i) / sum(exp(x_j))) = x_i - log(sum(exp(x_j)))
    
    Args:
        tensor (Tensor): Input tensor (typically logits from final layer)
        
    Returns:
        Tensor: New tensor with log softmax applied
        
    Sample Input/Output:
        >>> # Classification logits
        >>> logits = Tensor(1, 3)
        >>> logits[0, 0] = 1.0; logits[0, 1] = 2.0; logits[0, 2] = 3.0
        >>> log_probs = make_log_softmax(logits)
        >>> # Expected: log probabilities summing to 1 when exponentiated
        >>> log_probs[0, 0]           # Output: ~-2.41 (log(exp(1)/sum))
        >>> log_probs[0, 1]           # Output: ~-1.41 (log(exp(2)/sum))
        >>> log_probs[0, 2]           # Output: ~-0.41 (log(exp(3)/sum))
        
        >>> # Verify probabilities sum to 1
        >>> import math
        >>> prob_sum = sum(math.exp(log_probs[0, i]) for i in range(3))
        >>> prob_sum                  # Output: ~1.0
    """
    result = Tensor()
    result.impl = TensorImpl(lhs=tensor.impl, mode=TMode.LOG_SOFTMAX)
    return result

def make_flatten(tensors: List[Tensor]) -> Tensor:
    """
    Flatten multiple tensors into a single column vector
    
    Concatenates multiple tensors by flattening each one and stacking them
    vertically. Useful for converting feature maps or multiple layer outputs
    into a single vector for fully connected layers.
    
    Args:
        tensors (List[Tensor]): List of tensors to flatten and concatenate
        
    Returns:
        Tensor: Single column vector containing flattened data
        
    Sample Input/Output:
        >>> # Flatten multiple feature vectors
        >>> vec1 = Tensor(2, 2)
        >>> vec1[0, 0] = 1.0; vec1[0, 1] = 2.0
        >>> vec1[1, 0] = 3.0; vec1[1, 1] = 4.0
        >>> # vec1 = [[1, 2], [3, 4]]
        
        >>> vec2 = Tensor(1, 3)
        >>> vec2[0, 0] = 5.0; vec2[0, 1] = 6.0; vec2[0, 2] = 7.0
        >>> # vec2 = [[5, 6, 7]]
        
        >>> flattened = make_flatten([vec1, vec2])
        >>> flattened.shape           # Output: (7, 1) - total elements as column
        >>> flattened[0, 0]           # Output: 1.0 (from vec1[0,0])
        >>> flattened[1, 0]           # Output: 2.0 (from vec1[0,1])
        >>> flattened[2, 0]           # Output: 3.0 (from vec1[1,0])
        >>> flattened[3, 0]           # Output: 4.0 (from vec1[1,1])
        >>> flattened[4, 0]           # Output: 5.0 (from vec2[0,0])
        >>> flattened[5, 0]           # Output: 6.0 (from vec2[0,1])
        >>> flattened[6, 0]           # Output: 7.0 (from vec2[0,2])
        
        >>> # CNN to FC layer transition
        >>> conv1_output = Tensor(5, 5)  # 5x5 feature map
        >>> conv2_output = Tensor(3, 3)  # 3x3 feature map
        >>> fc_input = make_flatten([conv1_output, conv2_output])
        >>> fc_input.shape            # Output: (34, 1) - 25+9 features
    """
    result = Tensor()
    result.impl = TensorImpl(mode=TMode.FLATTEN)
    result.impl.flatten_params = {'members': [t.impl for t in tensors]}
    return result

# Convenience functions for common activation functions

def relu(tensor: Tensor) -> Tensor:
    """
    Rectified Linear Unit (ReLU) activation function
    
    Applies ReLU activation: f(x) = max(0, x). This is the most commonly
    used activation function in deep learning due to its simplicity and
    effectiveness in preventing vanishing gradients.
    
    Mathematical Properties:
    - f(x) = max(0, x)
    - f'(x) = 1 if x > 0, else 0
    - Range: [0, +∞)
    - Non-linear but maintains sparsity
    
    Args:
        tensor (Tensor): Input tensor
        
    Returns:
        Tensor: ReLU-activated tensor
        
    Sample Input/Output:
        >>> # Basic ReLU application
        >>> x = Tensor(2, 3)
        >>> x[0, 0] = -2.0; x[0, 1] = 0.0; x[0, 2] = 3.0
        >>> x[1, 0] = -1.0; x[1, 1] = 1.5; x[1, 2] = -0.5
        >>> activated = relu(x)
        >>> activated[0, 0]           # Output: 0.0 (max(0, -2))
        >>> activated[0, 1]           # Output: 0.0 (max(0, 0))
        >>> activated[0, 2]           # Output: 3.0 (max(0, 3))
        >>> activated[1, 0]           # Output: 0.0 (max(0, -1))
        >>> activated[1, 1]           # Output: 1.5 (max(0, 1.5))
        >>> activated[1, 2]           # Output: 0.0 (max(0, -0.5))
        
        >>> # Hidden layer in neural network
        >>> hidden_input = Tensor(1, 4)
        >>> hidden_input[0, 0] = 0.5; hidden_input[0, 1] = -0.3
        >>> hidden_input[0, 2] = 2.1; hidden_input[0, 3] = -1.2
        >>> hidden_output = relu(hidden_input)
        >>> hidden_output[0, 0]       # Output: 0.5 (positive preserved)
        >>> hidden_output[0, 1]       # Output: 0.0 (negative clipped)
        >>> hidden_output[0, 2]       # Output: 2.1 (positive preserved)
        >>> hidden_output[0, 3]       # Output: 0.0 (negative clipped)
    """
    return make_function(tensor, ActivationFunctions.ReLU)

def gelu(tensor: Tensor) -> Tensor:
    """
    Gaussian Error Linear Unit (GELU) activation function
    
    Applies GELU activation which provides a smooth approximation to ReLU.
    GELU is increasingly popular in transformer architectures and modern
    deep learning models due to its smoothness and probabilistic interpretation.
    
    Mathematical Formula:
    - f(x) = x * Φ(x) where Φ is the CDF of standard normal distribution
    - Approximation: f(x) ≈ 0.5 * x * (1 + tanh(√(2/π) * (x + 0.044715 * x³)))
    - Range: (-∞, +∞) but mostly [0, +∞) for positive inputs
    
    Args:
        tensor (Tensor): Input tensor
        
    Returns:
        Tensor: GELU-activated tensor
        
    Sample Input/Output:
        >>> # GELU activation on various inputs
        >>> x = Tensor(1, 5)
        >>> x[0, 0] = -2.0; x[0, 1] = -1.0; x[0, 2] = 0.0
        >>> x[0, 3] = 1.0; x[0, 4] = 2.0
        >>> activated = gelu(x)
        >>> activated[0, 0]           # Output: ~-0.045 (small negative)
        >>> activated[0, 1]           # Output: ~-0.159 (small negative)
        >>> activated[0, 2]           # Output: 0.0 (zero preserved)
        >>> activated[0, 3]           # Output: ~0.841 (most of positive)
        >>> activated[0, 4]           # Output: ~1.954 (nearly all of positive)
        
        >>> # Transformer feed-forward layer
        >>> ff_input = Tensor(1, 3)
        >>> ff_input[0, 0] = 0.5; ff_input[0, 1] = -0.2; ff_input[0, 2] = 1.5
        >>> ff_activated = gelu(ff_input)
        >>> # Smooth transitions, no hard cutoffs like ReLU
    """
    return make_function(tensor, ActivationFunctions.GELU)

def sigmoid(tensor: Tensor) -> Tensor:
    """
    Sigmoid activation function
    
    Applies sigmoid activation: f(x) = 1 / (1 + exp(-x)). Maps any real
    number to range (0, 1), making it useful for binary classification
    and gate mechanisms in LSTM/GRU networks.
    
    Mathematical Properties:
    - f(x) = 1 / (1 + e^(-x))
    - f'(x) = f(x) * (1 - f(x))
    - Range: (0, 1)
    - Symmetric around f(0) = 0.5
    
    Args:
        tensor (Tensor): Input tensor
        
    Returns:
        Tensor: Sigmoid-activated tensor
        
    Sample Input/Output:
        >>> # Binary classification output
        >>> logits = Tensor(1, 4)
        >>> logits[0, 0] = -2.0; logits[0, 1] = 0.0
        >>> logits[0, 2] = 2.0; logits[0, 3] = 5.0
        >>> probabilities = sigmoid(logits)
        >>> probabilities[0, 0]       # Output: ~0.119 (low confidence)
        >>> probabilities[0, 1]       # Output: 0.5 (neutral)
        >>> probabilities[0, 2]       # Output: ~0.881 (high confidence)
        >>> probabilities[0, 3]       # Output: ~0.993 (very high confidence)
        
        >>> # LSTM gate computation
        >>> gate_input = Tensor(1, 3)
        >>> gate_input[0, 0] = 1.2; gate_input[0, 1] = -0.8; gate_input[0, 2] = 0.3
        >>> gate_output = sigmoid(gate_input)
        >>> gate_output[0, 0]         # Output: ~0.768 (mostly open)
        >>> gate_output[0, 1]         # Output: ~0.310 (mostly closed)
        >>> gate_output[0, 2]         # Output: ~0.574 (slightly open)
    """
    return make_function(tensor, ActivationFunctions.Sigmoid)

def tanh(tensor: Tensor) -> Tensor:
    """
    Hyperbolic Tangent (tanh) activation function
    
    Applies tanh activation: f(x) = (e^x - e^(-x)) / (e^x + e^(-x)).
    Similar to sigmoid but maps to range (-1, 1) and is zero-centered,
    which can help with gradient flow in deep networks.
    
    Mathematical Properties:
    - f(x) = (e^x - e^(-x)) / (e^x + e^(-x))
    - f'(x) = 1 - f(x)²
    - Range: (-1, 1)
    - Zero-centered: f(0) = 0
    - Antisymmetric: f(-x) = -f(x)
    
    Args:
        tensor (Tensor): Input tensor
        
    Returns:
        Tensor: Tanh-activated tensor
        
    Sample Input/Output:
        >>> # Zero-centered activation
        >>> x = Tensor(1, 5)
        >>> x[0, 0] = -2.0; x[0, 1] = -1.0; x[0, 2] = 0.0
        >>> x[0, 3] = 1.0; x[0, 4] = 2.0
        >>> activated = tanh(x)
        >>> activated[0, 0]           # Output: ~-0.964 (strong negative)
        >>> activated[0, 1]           # Output: ~-0.762 (moderate negative)
        >>> activated[0, 2]           # Output: 0.0 (zero preserved)
        >>> activated[0, 3]           # Output: ~0.762 (moderate positive)
        >>> activated[0, 4]           # Output: ~0.964 (strong positive)
        
        >>> # RNN hidden state update
        >>> hidden_candidate = Tensor(1, 3)
        >>> hidden_candidate[0, 0] = 0.8; hidden_candidate[0, 1] = -1.2
        >>> hidden_candidate[0, 2] = 2.5
        >>> new_hidden = tanh(hidden_candidate)
        >>> new_hidden[0, 0]          # Output: ~0.664
        >>> new_hidden[0, 1]          # Output: ~-0.834
        >>> new_hidden[0, 2]          # Output: ~0.987
    """
    return make_function(tensor, ActivationFunctions.Tanh)

def square(tensor: Tensor) -> Tensor:
    """
    Square activation function
    
    Applies element-wise squaring: f(x) = x². Used in loss functions
    (mean squared error) and some specialized architectures. Always
    produces non-negative outputs.
    
    Mathematical Properties:
    - f(x) = x²
    - f'(x) = 2x
    - Range: [0, +∞)
    - Even function: f(-x) = f(x)
    
    Args:
        tensor (Tensor): Input tensor
        
    Returns:
        Tensor: Element-wise squared tensor
        
    Sample Input/Output:
        >>> # Basic squaring operation
        >>> x = Tensor(2, 2)
        >>> x[0, 0] = -3.0; x[0, 1] = 2.0
        >>> x[1, 0] = 0.0; x[1, 1] = -1.5
        >>> squared = square(x)
        >>> squared[0, 0]             # Output: 9.0 (-3)²
        >>> squared[0, 1]             # Output: 4.0 (2)²
        >>> squared[1, 0]             # Output: 0.0 (0)²
        >>> squared[1, 1]             # Output: 2.25 (-1.5)²
        
        >>> # Mean squared error computation
        >>> predicted = Tensor(1, 3)
        >>> predicted[0, 0] = 0.8; predicted[0, 1] = 0.2; predicted[0, 2] = 0.0
        >>> actual = Tensor(1, 3)
        >>> actual[0, 0] = 1.0; actual[0, 1] = 0.0; actual[0, 2] = 0.0
        >>> error = predicted - actual
        >>> squared_error = square(error)
        >>> squared_error[0, 0]       # Output: 0.04 (0.2)²
        >>> squared_error[0, 1]       # Output: 0.04 (0.2)²
        >>> squared_error[0, 2]       # Output: 0.0 (0)²
        
        >>> # L2 regularization term
        >>> weights = Tensor(2, 2)
        >>> weights[0, 0] = 0.5; weights[0, 1] = -0.3
        >>> weights[1, 0] = 0.8; weights[1, 1] = -0.2
        >>> l2_penalty = square(weights)  # For summing in regularization
        >>> l2_penalty[0, 0]          # Output: 0.25 (0.5)²
        >>> l2_penalty[1, 0]          # Output: 0.64 (0.8)²
    """
    return make_function(tensor, ActivationFunctions.Square)


# Loss functions

def mse_loss(predictions: Tensor, targets: Tensor) -> Tensor:
    """
    Mean Squared Error (MSE) loss function
    
    Computes the mean squared error between predictions and targets.
    This is the most common loss function for regression tasks.
    Measures the average squared difference between predicted and actual values.
    
    Mathematical Formula:
        MSE = (1/n) * Σ(y_pred - y_true)²
        
    For tensor computation, this returns the sum of squared differences
    (mean can be computed by dividing by number of elements).
    
    Args:
        predictions (Tensor): Predicted values from model
        targets (Tensor): Ground truth target values
        
    Returns:
        Tensor: Scalar tensor containing the MSE loss value
        
    Sample Input/Output:
        >>> # Simple regression example
        >>> predicted = Tensor(1, 3)
        >>> predicted[0, 0] = 2.5; predicted[0, 1] = 1.0; predicted[0, 2] = 3.8
        >>> actual = Tensor(1, 3)
        >>> actual[0, 0] = 2.0; actual[0, 1] = 1.5; actual[0, 2] = 4.0
        >>> loss = mse_loss(predicted, actual)
        >>> # Differences: [0.5, -0.5, -0.2]
        >>> # Squared: [0.25, 0.25, 0.04]
        >>> # Sum: 0.54
        >>> loss[0, 0]                # Output: 0.54
        
        >>> # Housing price prediction
        >>> model_prices = Tensor(1, 4)
        >>> model_prices[0, 0] = 250000; model_prices[0, 1] = 180000
        >>> model_prices[0, 2] = 320000; model_prices[0, 3] = 195000
        >>> true_prices = Tensor(1, 4)
        >>> true_prices[0, 0] = 240000; true_prices[0, 1] = 175000
        >>> true_prices[0, 2] = 310000; true_prices[0, 3] = 200000
        >>> price_loss = mse_loss(model_prices, true_prices)
        >>> # Large loss values due to price scale
        >>> price_loss[0, 0]          # Output: large number (squared dollar differences)
        
        >>> # Neural network training example
        >>> nn_output = Tensor(2, 1)  # Batch of 2 predictions
        >>> nn_output[0, 0] = 0.8; nn_output[1, 0] = 0.3
        >>> ground_truth = Tensor(2, 1)
        >>> ground_truth[0, 0] = 1.0; ground_truth[1, 0] = 0.0
        >>> training_loss = mse_loss(nn_output, ground_truth)
        >>> # Errors: [0.2, 0.3], Squared: [0.04, 0.09], Sum: 0.13
        >>> training_loss[0, 0]       # Output: 0.13
    """
    diff = predictions - targets
    squared_diff = square(diff)
    return squared_diff.sum()

def cross_entropy_loss(predictions: Tensor, targets: Tensor) -> Tensor:
    """
    Cross-Entropy loss function for classification
    
    Computes the cross-entropy loss between predicted probabilities and target labels.
    This is the standard loss function for multi-class classification tasks.
    Measures how far the predicted probability distribution is from the true distribution.
    
    Mathematical Formula:
        CrossEntropy = -Σ(y_true * log(y_pred))
        
    For numerical stability, this function applies log_softmax to predictions
    before computing the cross-entropy, avoiding potential overflow/underflow.
    
    Args:
        predictions (Tensor): Raw logits or probabilities from model (before softmax)
        targets (Tensor): One-hot encoded target labels or class probabilities
        
    Returns:
        Tensor: Scalar tensor containing the cross-entropy loss value
        
    Sample Input/Output:
        >>> # 3-class classification example
        >>> logits = Tensor(1, 3)     # Raw model outputs (logits)
        >>> logits[0, 0] = 2.0; logits[0, 1] = 1.0; logits[0, 2] = 0.1
        >>> # Model predicts class 0 most likely, then class 1, then class 2
        
        >>> targets = Tensor(1, 3)    # One-hot: true class is 0
        >>> targets[0, 0] = 1.0; targets[0, 1] = 0.0; targets[0, 2] = 0.0
        >>> loss = cross_entropy_loss(logits, targets)
        >>> # Since prediction aligns with target (highest logit for true class),
        >>> # loss should be relatively low
        >>> loss[0, 0]                # Output: small positive value (good prediction)
        
        >>> # Wrong prediction example
        >>> wrong_targets = Tensor(1, 3)  # True class is 2 (least likely prediction)
        >>> wrong_targets[0, 0] = 0.0; wrong_targets[0, 1] = 0.0; wrong_targets[0, 2] = 1.0
        >>> wrong_loss = cross_entropy_loss(logits, wrong_targets)
        >>> # Model predicted class 2 with low confidence, but that's the true class
        >>> wrong_loss[0, 0]          # Output: larger positive value (poor prediction)
        
        >>> # MNIST digit classification
        >>> digit_logits = Tensor(1, 10)  # 10 classes for digits 0-9
        >>> digit_logits.constant(0.1)    # Initialize with small equal values
        >>> digit_logits[0, 7] = 3.0      # High confidence for digit 7
        >>> true_digit = Tensor(1, 10)    # True label: digit 7
        >>> true_digit.one_hot_column(7)
        >>> digit_loss = cross_entropy_loss(digit_logits, true_digit)
        >>> # Good prediction since highest logit matches true class
        >>> digit_loss[0, 0]          # Output: small positive value
        
        >>> # Multi-sample batch training
        >>> batch_logits = Tensor(3, 4)  # Batch of 3 samples, 4 classes each
        >>> # Sample 1: [1.5, 0.5, -0.2, 0.8] -> predicts class 0
        >>> batch_logits[0, 0] = 1.5; batch_logits[0, 1] = 0.5
        >>> batch_logits[0, 2] = -0.2; batch_logits[0, 3] = 0.8
        >>> # Sample 2: [0.2, 2.1, 0.3, -0.5] -> predicts class 1  
        >>> batch_logits[1, 0] = 0.2; batch_logits[1, 1] = 2.1
        >>> batch_logits[1, 2] = 0.3; batch_logits[1, 3] = -0.5
        >>> # Sample 3: [-0.1, 0.4, 1.8, 0.2] -> predicts class 2
        >>> batch_logits[2, 0] = -0.1; batch_logits[2, 1] = 0.4
        >>> batch_logits[2, 2] = 1.8; batch_logits[2, 3] = 0.2
        
        >>> batch_targets = Tensor(3, 4)  # True labels
        >>> batch_targets.one_hot_column(0)  # All samples are actually class 0
        >>> batch_loss = cross_entropy_loss(batch_logits, batch_targets)
        >>> # Loss will be higher for samples 2 and 3 since they predict wrong classes
        >>> batch_loss[0, 0]          # Output: sum of individual sample losses
    """
    # Apply log softmax to predictions for numerical stability
    log_predictions = make_log_softmax(predictions)
    
    # Compute cross-entropy: -sum(targets * log_predictions)
    product = targets.dot(log_predictions)  # Element-wise multiplication
    return -product.sum()


# Export all public functions and classes
__all__ = [
    'Tensor', 'TensorImpl', 'TMode', 'ActivationFunctions',
    'make_function', 'make_log_softmax', 'make_flatten',
    'relu', 'gelu', 'sigmoid', 'tanh', 'square', 'mse_loss', 'cross_entropy_loss',
    '__version__', 'VERSION_INFO'
]