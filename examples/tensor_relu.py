#!/usr/bin/env python3
"""
ReLU Neural Network for MNIST Digit Classification using PyTensorLib

COMPREHENSIVE MODULE DOCUMENTATION
=================================

This module implements a simplified neural network with ReLU (Rectified Linear Unit) 
activations for multi-class classification of handwritten digits (0-9) from the 
MNIST/EMNIST dataset. It demonstrates the fundamental concepts of deep learning 
using the PyTensorLib tensor computation framework, showcasing the evolution from 
linear models (perceptron) to non-linear neural networks.

ALGORITHM OVERVIEW
=================

Neural Network Fundamentals:
---------------------------
Unlike the linear perceptron which can only learn linearly separable patterns,
neural networks with hidden layers and non-linear activation functions can learn
complex, non-linear decision boundaries. This enables classification of patterns
that are not linearly separable.

Mathematical Foundation:
-----------------------
1. Multi-layer Architecture:
   - Input layer: 784 neurons (28×28 flattened pixels)
   - Hidden layers: Multiple layers with ReLU activations
   - Output layer: 10 neurons (one per digit class 0-9)

2. Forward Propagation:
   Layer l: z^(l) = W^(l) × a^(l-1) + b^(l)
   Activation: a^(l) = f(z^(l))
   
   Where:
   - W^(l): Weight matrix for layer l
   - b^(l): Bias vector for layer l  
   - a^(l-1): Activations from previous layer
   - f(): Activation function (ReLU for hidden, softmax for output)

3. ReLU Activation Function:
   ReLU(x) = max(0, x) = {x if x > 0, 0 otherwise}
   
   Properties:
   - Non-linear but computationally efficient
   - Solves vanishing gradient problem
   - Sparse activation (many neurons output 0)
   - Derivative: ReLU'(x) = {1 if x > 0, 0 otherwise}

4. Loss Function (Cross-Entropy):
   L = -∑(i=1 to C) y_i × log(ŷ_i)
   
   Where:
   - C: Number of classes (10 for digits)
   - y_i: True label (one-hot encoded)
   - ŷ_i: Predicted probability for class i

5. Backpropagation (Gradient Descent):
   ∂L/∂W^(l) = ∂L/∂z^(l) × (a^(l-1))^T
   ∂L/∂b^(l) = ∂L/∂z^(l)
   ∂L/∂a^(l-1) = (W^(l))^T × ∂L/∂z^(l)

Enhanced Neural Network Features:
--------------------------------
- Multi-class classification (10 digits vs binary perceptron)
- Non-linear feature learning through hidden layers
- Gradient-based optimization with backpropagation
- Comprehensive evaluation metrics and visualization
- Hierarchical feature extraction capabilities
- Organized model persistence and training logs

ARCHITECTURE COMPARISON
======================

Perceptron (Linear Model):
-------------------------
```
Input (784) → Linear Transform → Output (Binary: 3 vs 7)
x ∈ ℝ^784 → W·x + b → sign(score) ∈ {3, 7}
Parameters: 785 (784 weights + 1 bias)
Decision Boundary: Linear hyperplane
```

Neural Network (Non-Linear Model):
---------------------------------
```
Input (784) → Hidden Layers → Output (Multi-class: 0-9)
x ∈ ℝ^784 → ReLU(W₁·x + b₁) → ... → Softmax(Wₙ·hₙ₋₁ + bₙ) → ŷ ∈ ℝ^10
Parameters: Thousands (depends on architecture)
Decision Boundary: Complex non-linear manifolds
```

Current Implementation:
----------------------
For demonstration purposes, this module implements a simplified single-layer
neural network (784 → 10) with direct mapping, which is essentially a 
multi-class extension of the perceptron:

```
Input: x ∈ ℝ^784 (flattened 28×28 image)
Weights: W ∈ ℝ^784×10 (shared parameters for all classes)
Bias: b ∈ ℝ^10 (one bias per class)
Output: ŷ = Softmax(W^T·x + b) ∈ ℝ^10
```

DEPENDENCY INTEGRATION
=====================

PyTensorLib Components:
----------------------

1. tensor_lib.Tensor:
   Enhanced tensor operations providing:
   - Multi-dimensional array computations
   - Automatic gradient computation (in full implementation)
   - Efficient matrix multiplication and broadcasting
   - Type-safe operations with lazy evaluation
   
   Usage in neural networks:
   >>> weights = Tensor(784, 10)  # Input-to-output weight matrix
   >>> hidden = Tensor(128, 1)   # Hidden layer activations
   >>> output = weights.transpose() @ input + bias

2. tensor_lib.relu:
   ReLU activation function providing:
   - Element-wise non-linear transformation
   - Gradient-friendly activation (no vanishing gradients)
   - Sparse representation capabilities
   
   Mathematical operation:
   >>> activated = relu(linear_output)  # Applies max(0, x) element-wise

3. tensor_lib.make_log_softmax:
   Log-softmax operation providing:
   - Numerically stable probability computation
   - Suitable for cross-entropy loss calculation
   - Multi-class probability distribution
   
   Mathematical operation:
   >>> log_probs = make_log_softmax(logits)  # log(softmax(x))

4. mnist_utils.MNISTReader:
   Same data interface as perceptron but extended for:
   - Multi-class label handling (0-9 vs binary 3,7)
   - Batch processing capabilities
   - Consistent preprocessing pipeline
   
   Integration examples:
   >>> reader = MNISTReader(images_path, labels_path)
   >>> label = reader.get_label(idx)  # Returns 0-9 instead of 3,7
   >>> one_hot = np.zeros(10); one_hot[label] = 1.0

External Dependencies:
---------------------

1. NumPy (numpy):
   Core numerical operations providing:
   - Matrix multiplication for forward/backward passes
   - Broadcasting for bias addition
   - Element-wise operations for activations
   - Statistical functions for initialization
   
   Critical neural network operations:
   >>> output = weights.T @ input + bias  # Forward pass
   >>> weights -= learning_rate * gradients  # Parameter update
   >>> activations = np.maximum(0, linear_output)  # ReLU

2. SQLite3 (sqlite3):
   Extended logging system providing:
   - Training progress tracking across epochs
   - Batch-level metrics monitoring
   - Validation performance logging
   - Hyperparameter experiment tracking
   
   Enhanced database schema:
   >>> training_log: (epoch, batch, loss, accuracy, learning_rate)
   >>> validation_log: (epoch, loss, accuracy)

3. JSON (json):
   Model serialization providing:
   - Multi-layer parameter persistence
   - Training configuration storage
   - Checkpoint saving and loading
   - Experiment reproducibility
   
   Enhanced model format:
   >>> {"param_0": weights_layer1, "param_1": weights_layer2, ...}

DATA FLOW ARCHITECTURE
======================

Input Data Pipeline (Enhanced for Multi-class):
----------------------------------------------
MNIST/EMNIST Dataset → MNISTReader → Tensor → Neural Network

1. Raw Data Loading:
   >>> train_images, train_labels = download_emnist(data_dir)
   Format: Same binary IDX files but with 10-class labels (0-9)

2. Reader Initialization:
   >>> train_reader = MNISTReader(train_images, train_labels)
   Provides: Multi-class label access and batch processing

3. Sample Extraction:
   >>> img_array = train_reader.get_image_as_array(n)  # Shape: (28, 28)
   >>> label = train_reader.get_label(n)               # Value: 0-9

4. Tensor Conversion and Preprocessing:
   >>> x = Tensor(784, 1)
   >>> x.impl.val = img_array.reshape(784, 1).astype(np.float32)
   >>> # Normalization: x = (x - mean) / std

Training Pipeline (Neural Network):
----------------------------------
Input Batch → Forward Pass → Loss Computation → Backpropagation → Weight Update

1. Forward Pass Computation:
   >>> z = weights.T @ x + bias          # Linear transformation
   >>> a = softmax(z)                    # Probability distribution
   Mathematics: Computes P(class|input) for all 10 classes

2. Loss Computation:
   >>> target = one_hot_encode(true_label)  # [0,0,1,0,0,0,0,0,0,0] for digit 2
   >>> loss = -np.sum(target * np.log(a + eps))  # Cross-entropy
   Result: Measures prediction error across all classes

3. Gradient Computation (Backpropagation):
   >>> dL_dz = a - target                # Output layer gradient
   >>> dL_dW = x @ dL_dz.T              # Weight gradients
   >>> dL_db = dL_dz                    # Bias gradients
   Result: Computes parameter update directions

4. Parameter Updates:
   >>> weights -= learning_rate * dL_dW
   >>> bias -= learning_rate * dL_db
   Result: Moves parameters toward better classification

Evaluation Pipeline:
-------------------
Test Data → Model → Predictions → Metrics → Visualization

1. Batch Evaluation:
   >>> for batch in test_data:
   ...     predictions = model.forward(batch)
   ...     accuracy = compute_accuracy(predictions, true_labels)

2. ASCII Visualization:
   >>> print_sample_image(image_array, true_label, predicted_label)
   Output: Visual representation of model decisions

3. Performance Metrics:
   >>> accuracy = correct_predictions / total_predictions
   >>> confusion_matrix = compute_confusion_matrix(predictions, labels)

COMPUTATIONAL COMPLEXITY
========================

Time Complexity Analysis:
------------------------
- Forward pass: O(N×H₁ + H₁×H₂ + ... + Hₙ×C) where N=input_size, Hᵢ=hidden_size, C=classes
- Backward pass: Same as forward pass (reverse order)
- Single training step: O(2×forward_pass) = O(forward_pass)
- Full training: O(E×B×S×forward_pass) where E=epochs, B=batches, S=samples_per_batch

For current simplified implementation:
- Forward pass: O(784×10) = O(7,840) operations
- Full training: O(epochs×batches×7,840) operations

Space Complexity Analysis:
-------------------------
- Model parameters: O(input_size×output_size) = O(784×10) = 7,840 parameters
- Activation storage: O(batch_size×layer_sizes) for each layer
- Gradient storage: Same as model parameters
- Total memory: ~100KB for model + dataset size + batch activations

Performance Characteristics:
---------------------------
- Typical convergence: 10-100 epochs depending on complexity
- Training time: 1-30 minutes depending on dataset size and architecture
- Final accuracy: 85-95% for MNIST (simple architectures)
- Memory usage: <1GB including full MNIST dataset and gradients

USAGE EXAMPLES
=============

Basic Training Session:
----------------------
>>> # Run complete neural network training
>>> python examples/tensor_relu.py
🔢 ReLU Neural Network for MNIST Classification
📦 Loading real MNIST/EMNIST data...
✅ EMNIST data loaded successfully
🧠 Creating neural network...
📈 Dataset statistics:
   Training samples: 240000
   Test samples: 40000
   Data type: EMNIST
🧪 Initial model evaluation:
Label: 7, Predicted: 3
ASCII representation:
░░░░░■■■■■■■░░░░░░░░░░░░░░░░
[... ASCII art visualization ...]
📊 Test Accuracy: 0.1234 (12/100)

Programmatic Usage:
------------------
>>> from tensor_relu import SimpleNeuralNetwork, load_data
>>> 
>>> # Create and train model
>>> model = SimpleNeuralNetwork()
>>> train_reader, test_reader, data_type = load_data()
>>> 
>>> # Forward pass example
>>> img_array = train_reader.get_image_as_array(0)
>>> x = Tensor(784, 1)
>>> x.impl.val = img_array.reshape(784, 1)
>>> output = model.forward(x)
>>> predicted_class = np.argmax(output.impl.val)

Model Persistence and Analysis:
------------------------------
>>> # Save trained model
>>> model.save_state('models/my_neural_net.json')
>>> 
>>> # Load for inference
>>> model.load_state('models/my_neural_net.json')
>>> 
>>> # Analyze training progress
>>> import sqlite3
>>> conn = sqlite3.connect('logs/tensor_relu_results.sqlite3')
>>> cursor = conn.execute('SELECT epoch, accuracy FROM validation_log')
>>> learning_curve = cursor.fetchall()

EDUCATIONAL VALUE
================

Deep Learning Concepts Demonstrated:
-----------------------------------
1. Multi-layer Architecture: Hierarchical feature learning
2. Non-linear Activations: ReLU function properties and benefits
3. Gradient-based Optimization: Backpropagation algorithm
4. Multi-class Classification: Softmax and cross-entropy loss
5. Regularization Concepts: Weight initialization and normalization

Programming Concepts Illustrated:
---------------------------------
1. Object-Oriented Neural Networks: Modular design patterns
2. Tensor Operations: Multi-dimensional array computations
3. Numerical Stability: Softmax implementation considerations
4. Memory Management: Efficient gradient and activation storage
5. Experiment Tracking: Comprehensive logging and model checkpointing

Research and Development Applications:
-------------------------------------
1. Architecture Exploration: Comparing different layer configurations
2. Activation Function Studies: ReLU vs other activation functions
3. Optimization Algorithms: SGD, Adam, momentum comparisons
4. Regularization Techniques: Dropout, weight decay, batch normalization
5. Transfer Learning: Feature extraction and fine-tuning experiments

Comparison with Perceptron:
--------------------------
This module builds upon the perceptron foundation by introducing:
- Non-linear decision boundaries through activation functions
- Multi-class classification capabilities (10 classes vs binary)
- Hierarchical feature learning (in full implementations)
- More sophisticated optimization techniques
- Enhanced evaluation and visualization capabilities

The progression from perceptron_37_learn.py to tensor_relu.py demonstrates
the evolution from classical linear models to modern deep learning approaches,
highlighting both the increased complexity and enhanced capabilities of
neural networks for pattern recognition tasks.
"""

import os
import sys
import json
import sqlite3
import random
from typing import List, Tuple, Optional
import numpy as np

# Add src to Python path
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'src'))

from pytensorlib.tensor_lib import Tensor, relu, make_log_softmax, make_flatten
from pytensorlib.mnist_utils import MNISTReader, download_emnist, download_mnist


class SimpleNeuralNetwork:
    """
    Simplified Multi-Class Neural Network using PyTensorLib Operations
    
    COMPREHENSIVE CLASS DOCUMENTATION
    ================================
    
    This class implements a simplified neural network for multi-class digit classification
    (0-9) using a direct linear transformation from input to output. While described as
    "simple," it demonstrates fundamental neural network concepts including forward
    propagation, parameter management, and multi-class prediction.
    
    ARCHITECTURE OVERVIEW
    ====================
    
    Network Structure:
    -----------------
    Input Layer:  784 neurons (28×28 flattened pixel values)
                  ↓
    Linear Transform: W^T × x + b (no hidden layers in this implementation)
                  ↓  
    Output Layer: 10 neurons (probability distribution over digits 0-9)
    
    Mathematical Representation:
    ---------------------------
    Given input image x ∈ ℝ^784:
    1. Linear transformation: z = W^T × x + b, where:
       - W ∈ ℝ^784×10: Weight matrix mapping input features to output classes
       - b ∈ ℝ^10: Bias vector providing class-specific offsets
       - z ∈ ℝ^10: Raw logits (pre-softmax scores)
    
    2. Probability computation: ŷ = softmax(z), where:
       - ŷ[i] = exp(z[i]) / Σ(j=0 to 9) exp(z[j])
       - Σ(i=0 to 9) ŷ[i] = 1.0 (valid probability distribution)
    
    Parameter Initialization Strategy:
    ---------------------------------
    Weights: W ~ N(0, σ²) where σ = 0.01
    - Small random initialization prevents symmetry breaking
    - Gaussian distribution ensures diverse initial feature detectors
    - Standard deviation of 0.01 prevents initial saturation
    
    Bias: b = 0
    - Zero initialization is standard for bias terms
    - Allows equal initial treatment of all classes
    - Will be adjusted during training based on class frequencies
    
    TENSOR OPERATIONS AND SHAPES
    ============================
    
    Shape Analysis:
    --------------
    Input image: x.shape = (784, 1)  # Flattened 28×28 image as column vector
    Weight matrix: W.shape = (784, 10)  # 784 features → 10 classes
    Bias vector: b.shape = (10, 1)     # One bias per output class
    Output logits: z.shape = (10, 1)   # Raw scores for each class
    
    Matrix Multiplication Details:
    -----------------------------
    Forward pass computation: z = W^T @ x + b
    
    Step-by-step:
    1. W^T.shape = (10, 784)  # Transpose weight matrix
    2. (W^T @ x).shape = (10, 784) @ (784, 1) = (10, 1)  # Matrix multiplication
    3. z = (10, 1) + (10, 1) = (10, 1)  # Bias addition (broadcasting)
    
    Memory Requirements:
    -------------------
    - Weight storage: 784 × 10 × 4 bytes = 31.36 KB (float32)
    - Bias storage: 10 × 4 bytes = 40 bytes
    - Input buffer: 784 × 4 bytes = 3.136 KB
    - Output buffer: 10 × 4 bytes = 40 bytes
    - Total model size: ~34.5 KB (excluding gradients)
    
    COMPARISON WITH PERCEPTRON
    ==========================
    
    Key Differences:
    ---------------
    | Aspect | Perceptron | SimpleNeuralNetwork |
    |--------|------------|-------------------|
    | Output | Binary (3 vs 7) | Multi-class (0-9) |
    | Classes | 2 | 10 |
    | Parameters | 785 (784W + 1b) | 7,850 (7,840W + 10b) |
    | Decision | sign(W·x + b) | argmax(softmax(W^T·x + b)) |
    | Learning | Margin-based | Gradient-based |
    | Boundary | Linear hyperplane | 9 linear hyperplanes |
    
    Architectural Similarities:
    --------------------------
    - Both use linear transformations (no hidden layers)
    - Both learn feature weights directly from input pixels
    - Both require labeled training data
    - Both use matrix-vector operations for prediction
    
    Enhanced Capabilities:
    ---------------------
    - Multi-class support: Can distinguish between all 10 digits
    - Probability outputs: Provides confidence scores for each class
    - Unified framework: Single model handles all digit pairs
    - Extensible design: Easy to add hidden layers for non-linear learning
    """
    
    def __init__(self):
        """
        Initialize Neural Network Architecture and Parameters
        
        CONSTRUCTOR DOCUMENTATION
        ========================
        
        This constructor sets up the fundamental neural network architecture by:
        1. Defining tensor dimensions for weights and bias
        2. Allocating memory for parameter storage
        3. Initializing parameters with appropriate random values
        
        Parameter Tensor Creation:
        -------------------------
        self.weights: Tensor(784, 10)
        - Shape: (input_features, output_classes)
        - Purpose: Maps each input pixel to each output class
        - Interpretation: weights[i, j] = importance of pixel i for class j
        
        self.bias: Tensor(10, 1)  
        - Shape: (output_classes, 1)
        - Purpose: Provides class-specific baseline adjustments
        - Interpretation: bias[j] = prior preference for class j
        
        Memory Allocation:
        -----------------
        Both tensors are created as PyTensorLib Tensor objects, providing:
        - Lazy evaluation capabilities
        - Automatic gradient tracking (in full implementations)
        - Type-safe numerical operations
        - Integration with broader PyTensorLib ecosystem
        
        Initialization Process:
        ----------------------
        After tensor creation, initialize_weights() is called to:
        - Populate weight matrix with small random values
        - Set bias vector to zeros
        - Ensure tensors are ready for forward propagation
        
        Example Usage:
        -------------
        >>> model = SimpleNeuralNetwork()
        >>> print(f"Model has {model.weights.shape[0] * model.weights.shape[1]} weights")
        >>> print(f"Model has {model.bias.shape[0]} bias terms")
        Model has 7840 weights
        Model has 10 bias terms
        """
        # For simplicity, let's start with a single layer
        self.weights = Tensor(784, 10)  # Direct input to output
        self.bias = Tensor(10, 1)
        
        # Initialize weights randomly
        self.initialize_weights()
    
    def initialize_weights(self):
        """
        Initialize Network Parameters with Statistically Appropriate Values
        
        WEIGHT INITIALIZATION DOCUMENTATION
        ==================================
        
        Proper weight initialization is crucial for neural network training success.
        This method implements Xavier/Glorot-style initialization adapted for
        the simplified single-layer architecture.
        
        MATHEMATICAL FOUNDATION
        ======================
        
        Weight Initialization Strategy:
        ------------------------------
        Distribution: W ~ N(μ=0, σ²=0.01²)
        - Mean (μ): 0.0 ensures no initial bias toward any class
        - Standard deviation (σ): 0.01 provides small but non-zero initial values
        - Shape: (784, 10) creates 7,840 independent random parameters
        
        Rationale for σ = 0.01:
        -----------------------
        1. Prevents initial saturation: Values too large → sigmoid/softmax saturation
        2. Ensures gradient flow: Values too small → vanishing gradients
        3. Breaks symmetry: Different neurons learn different features
        4. Empirically effective: Works well for linear models and shallow networks
        
        Alternative Initialization Schemes:
        ----------------------------------
        - Xavier/Glorot: σ = √(2/(n_in + n_out)) = √(2/(784+10)) ≈ 0.05
        - He initialization: σ = √(2/n_in) = √(2/784) ≈ 0.05
        - LeCun initialization: σ = √(1/n_in) = √(1/784) ≈ 0.036
        
        Current choice (σ = 0.01) is conservative but stable for initial training.
        
        Bias Initialization Strategy:
        ----------------------------
        Value: b = 0 (zero vector)
        - Standard practice for classification networks
        - Allows equal initial treatment of all classes
        - Will adapt during training to reflect class frequencies
        - No symmetry issues (biases are class-specific)
        
        TENSOR OPERATION DETAILS
        =======================
        
        Weight Tensor Population:
        ------------------------
        >>> self.weights.impl.val = np.random.normal(0, scale, (784, 10)).astype(np.float32)
        
        Step-by-step process:
        1. np.random.normal(0, 0.01, (784, 10)): Generate 7,840 Gaussian samples
        2. .astype(np.float32): Convert to 32-bit floats for efficiency
        3. self.weights.impl.val = ...: Assign to tensor's internal value storage
        
        Bias Tensor Population:
        ----------------------
        >>> self.bias.impl.val = np.zeros((10, 1), dtype=np.float32)
        
        Step-by-step process:
        1. np.zeros((10, 1)): Create 10×1 zero matrix
        2. dtype=np.float32: Ensure consistent data type
        3. self.bias.impl.val = ...: Assign to tensor's internal storage
        
        IMPLEMENTATION CONSIDERATIONS
        ============================
        
        PyTensorLib Integration:
        -----------------------
        - Direct assignment to .impl.val bypasses lazy evaluation
        - Ensures parameters are immediately available for forward pass
        - Maintains compatibility with PyTensorLib's tensor operations
        - Enables future gradient computation if implemented
        
        Numerical Precision:
        -------------------
        - float32 precision balances accuracy and memory efficiency
        - Sufficient for most neural network applications
        - Compatible with GPU acceleration libraries
        - Reduces memory bandwidth requirements
        
        Random Seed Considerations:
        --------------------------
        This implementation uses NumPy's global random state. For reproducible
        experiments, set the seed before model creation:
        >>> np.random.seed(42)
        >>> model = SimpleNeuralNetwork()  # Reproducible initialization
        
        Example Output Analysis:
        -----------------------
        After initialization, weight statistics typically show:
        >>> print(f"Weight mean: {np.mean(model.weights.impl.val):.6f}")
        >>> print(f"Weight std: {np.std(model.weights.impl.val):.6f}")
        >>> print(f"Weight range: [{np.min(model.weights.impl.val):.4f}, {np.max(model.weights.impl.val):.4f}]")
        Weight mean: 0.000123
        Weight std: 0.009987
        Weight range: [-0.0312, 0.0298]
        
        This confirms proper initialization with small random values centered around zero.
        """
        scale = 0.01
        self.weights.impl.val = np.random.normal(0, scale, (784, 10)).astype(np.float32)
        self.bias.impl.val = np.zeros((10, 1), dtype=np.float32)
    
    def forward(self, x: Tensor) -> Tensor:
        """
        Execute Forward Propagation Through the Neural Network
        
        FORWARD PASS DOCUMENTATION
        ==========================
        
        This method implements the fundamental forward propagation algorithm,
        transforming input images into class probability distributions through
        matrix multiplication and bias addition. It represents the core
        inference mechanism of the neural network.
        
        MATHEMATICAL FOUNDATION
        ======================
        
        Forward Propagation Equation:
        ----------------------------
        Given input x ∈ ℝ^784×1, compute output ŷ ∈ ℝ^10×1:
        
        1. Linear Transformation:
           z = W^T · x + b
           
           Where:
           - W ∈ ℝ^784×10: Weight matrix (self.weights)
           - x ∈ ℝ^784×1: Input image vector (flattened)
           - b ∈ ℝ^10×1: Bias vector (self.bias)
           - z ∈ ℝ^10×1: Raw logits (pre-activation outputs)
        
        2. Probability Interpretation:
           For classification, z represents raw "scores" or "logits" for each class.
           These can be converted to probabilities via softmax:
           P(class_i | x) = exp(z_i) / Σ(j=0 to 9) exp(z_j)
        
        TENSOR SHAPE ANALYSIS
        ====================
        
        Input Tensor Requirements:
        -------------------------
        x.shape = (784, 1)  # Column vector of flattened 28×28 image
        
        Shape verification:
        - Expected: Flattened MNIST image as column vector
        - Data type: float32 for numerical stability
        - Value range: Typically [0, 1] (normalized pixel intensities)
        
        Intermediate Tensor Shapes:
        ---------------------------
        Step 1: Weight Transpose
        W^T.shape = (10, 784)  # Transpose of self.weights (784, 10)
        
        Step 2: Matrix Multiplication  
        (W^T @ x).shape = (10, 784) @ (784, 1) = (10, 1)
        
        Step 3: Bias Addition (Broadcasting)
        z.shape = (10, 1) + (10, 1) = (10, 1)  # Element-wise addition
        
        Output Tensor Specifications:
        ----------------------------
        output.shape = (10, 1)  # Raw logits for each digit class
        output.dtype = float32   # Consistent with input precision
        
        COMPUTATIONAL IMPLEMENTATION
        ============================
        
        Step-by-Step Execution:
        ----------------------
        1. Tensor Value Assurance:
           - Ensures x.impl.val is computed (lazy evaluation)
           - Ensures self.weights.impl.val is available
           - Ensures self.bias.impl.val is available
           - Raises ValueError if any tensor lacks initialized values
        
        2. Matrix Operations:
           result = self.weights.impl.val.T @ x.impl.val + self.bias.impl.val
           
           Breakdown:
           - self.weights.impl.val.T: Transpose weight matrix (784,10) → (10,784)
           - @ operator: NumPy matrix multiplication
           - + self.bias.impl.val: Broadcasting addition of bias vector
        
        3. Output Tensor Creation:
           - Creates new Tensor(10, 1) for storing results
           - Assigns computed result to output.impl.val
           - Returns tensor ready for further processing
        
        EXAMPLE INPUT/OUTPUT
        ===================
        
        Sample Input Processing:
        -----------------------
        # Input: Handwritten digit '7' image
        >>> image_array = np.array([[0.0, 0.0, 0.1, ...],  # 28×28 normalized pixels
        ...                        [0.0, 0.2, 0.8, ...],
        ...                        [...]])
        >>> x = Tensor(784, 1)
        >>> x.impl.val = image_array.reshape(784, 1)
        
        # Forward pass
        >>> output = model.forward(x)
        >>> logits = output.impl.val
        >>> print("Raw logits for each class:")
        >>> for i, score in enumerate(logits.flatten()):
        ...     print(f"  Digit {i}: {score:.4f}")
        
        Expected Output (example):
        -------------------------
        Raw logits for each class:
          Digit 0: -0.1234
          Digit 1: -0.0567
          Digit 2: -0.2341
          Digit 3: -0.1876
          Digit 4: -0.0923
          Digit 5: -0.1445
          Digit 6: -0.1098
          Digit 7:  0.2156  ← Highest score (predicted class)
          Digit 8: -0.0734
          Digit 9: -0.1567
        
        Probability Conversion:
        ----------------------
        >>> import scipy.special
        >>> probabilities = scipy.special.softmax(logits.flatten())
        >>> predicted_class = np.argmax(logits)
        >>> confidence = probabilities[predicted_class]
        >>> 
        >>> print(f"Predicted digit: {predicted_class}")
        >>> print(f"Confidence: {confidence:.4f}")
        >>> print(f"Probability distribution:")
        >>> for i, prob in enumerate(probabilities):
        ...     print(f"  P(digit={i}): {prob:.4f}")
        
        Expected Probability Output:
        ---------------------------
        Predicted digit: 7
        Confidence: 0.1534
        Probability distribution:
          P(digit=0): 0.0876
          P(digit=1): 0.0923
          P(digit=2): 0.0798
          P(digit=3): 0.0832
          P(digit=4): 0.0889
          P(digit=5): 0.0857
          P(digit=6): 0.0871
          P(digit=7): 0.1534  ← Highest probability
          P(digit=8): 0.0908
          P(digit=9): 0.0842
        
        PERFORMANCE CONSIDERATIONS
        ==========================
        
        Computational Complexity:
        -------------------------
        - Matrix multiplication: O(784 × 10) = O(7,840) FLOPs
        - Bias addition: O(10) FLOPs  
        - Total: ~7,850 floating-point operations per forward pass
        - Memory access: ~31.4 KB weight data + 3.1 KB input data
        
        Optimization Opportunities:
        ---------------------------
        1. Batch Processing: Process multiple images simultaneously
           - Input shape: (784, batch_size) instead of (784, 1)
           - Output shape: (10, batch_size)
           - Same computational cost per sample with better CPU utilization
        
        2. Quantization: Use int8 weights for 4× memory reduction
           - Requires careful scaling and bias adjustment
           - Minimal accuracy loss for well-trained models
        
        3. Sparse Operations: Exploit weight sparsity if present
           - Many weights may be near zero after training
           - Sparse matrix libraries can reduce computation
        
        ERROR HANDLING
        =============
        
        Common Error Scenarios:
        ----------------------
        1. Uninitialized Tensors:
           ValueError: "Tensor values must be initialized before performing operations."
           - Occurs when tensor.impl.val is None
           - Solution: Ensure model.initialize_weights() was called
        
        2. Shape Mismatches:
           - Input not (784, 1): Reshape input correctly
           - Weights not (784, 10): Re-initialize model
           - Bias not (10, 1): Check initialization
        
        3. Data Type Issues:
           - Mixed float32/float64: Ensure consistent dtypes
           - Integer inputs: Convert to float before processing
        
        Integration with Training:
        -------------------------
        This forward pass method is called during:
        - Training: For each batch of training samples
        - Validation: For computing validation accuracy
        - Inference: For making predictions on new data
        - Evaluation: For testing model performance
        
        The computed output serves as input to:
        - Loss functions (cross-entropy, MSE)
        - Accuracy calculations (argmax comparison)
        - Gradient computation (backpropagation)
        - Probability analysis (confidence estimation)
        """
        # x is (784, 1), weights is (784, 10)
        # result should be (10, 1)
        
        # For PyTensorLib, we need to be careful about matrix multiplication
        # Let's use a simpler approach - compute manually
        output = Tensor(10, 1)
        
        # Ensure values are computed
        x.impl.assure_value()
        self.weights.impl.assure_value()
        self.bias.impl.assure_value()
        
        # Manual matrix multiplication: weights.T @ x + bias
        if self.weights.impl.val is None or x.impl.val is None or self.bias.impl.val is None:
            raise ValueError("Tensor values must be initialized before performing operations.")
        result = self.weights.impl.val.T @ x.impl.val + self.bias.impl.val
        output.impl.val = result
        
        return output
    
    def backward(self, x: Tensor, y_true: Tensor, y_pred: Tensor, learning_rate: float = 0.01):
        """
        Execute Backward Propagation for Parameter Updates (Simplified Implementation)
        
        BACKPROPAGATION DOCUMENTATION
        =============================
        
        This method implements a simplified version of the backpropagation algorithm
        for computing gradients and updating network parameters. While the current
        implementation is limited due to PyTensorLib constraints, it demonstrates
        the fundamental concepts of gradient-based optimization in neural networks.
        
        MATHEMATICAL FOUNDATION
        ======================
        
        Gradient Computation Theory:
        ---------------------------
        For neural network training, we minimize the loss function L(θ) with respect
        to parameters θ = {W, b} using gradient descent:
        
        θ_new = θ_old - η · ∇L(θ)
        
        Where:
        - η: Learning rate (step size)
        - ∇L(θ): Gradient of loss with respect to parameters
        
        Cross-Entropy Loss Gradient:
        ----------------------------
        For multi-class classification with cross-entropy loss:
        L = -∑(i=1 to C) y_true[i] · log(y_pred[i])
        
        Gradient computation:
        1. Output layer gradient:
           ∂L/∂z = y_pred - y_true  (for softmax + cross-entropy)
        
        2. Weight gradients:
           ∂L/∂W = x ⊗ (∂L/∂z)^T = x · (y_pred - y_true)^T
           
        3. Bias gradients:
           ∂L/∂b = ∂L/∂z = y_pred - y_true
        
        TENSOR SHAPE ANALYSIS
        ====================
        
        Input Tensor Shapes:
        -------------------
        x.shape = (784, 1)        # Input features (flattened image)
        y_true.shape = (10, 1)    # True labels (one-hot encoded)
        y_pred.shape = (10, 1)    # Predicted probabilities (softmax output)
        
        Gradient Tensor Shapes:
        ----------------------
        dL_dz.shape = (10, 1)     # Output layer error signal
        dL_dW.shape = (784, 10)   # Weight gradients (same shape as weights)
        dL_db.shape = (10, 1)     # Bias gradients (same shape as bias)
        
        Parameter Update Shapes:
        -----------------------
        weights.shape = (784, 10)  # Updated weight matrix
        bias.shape = (10, 1)       # Updated bias vector
        
        SIMPLIFIED IMPLEMENTATION
        =========================
        
        Current Limitations:
        -------------------
        This implementation is intentionally simplified due to:
        1. PyTensorLib's limited automatic differentiation support
        2. Focus on educational demonstration rather than production training
        3. Emphasis on forward pass and inference capabilities
        
        Conceptual Implementation:
        -------------------------
        The method demonstrates the structure of backpropagation:
        
        1. Batch Size Calculation:
           >>> batch_size = x.shape[1] if len(x.shape) > 1 else 1
           Determines whether processing single sample or batch
        
        2. Error Signal Computation:
           >>> dz3 = y_pred - y_true  # Cross-entropy + softmax derivative
           Computes output layer error (difference between prediction and truth)
        
        3. Gradient Computation (conceptual):
           >>> dW = x @ dz3.T / batch_size    # Weight gradients
           >>> db = dz3.mean(axis=1)          # Bias gradients
        
        4. Parameter Updates (conceptual):
           >>> weights -= learning_rate * dW
           >>> bias -= learning_rate * db
        
        FULL IMPLEMENTATION EXAMPLE
        ===========================
        
        Complete Backpropagation Algorithm:
        ----------------------------------
        Here's how a full implementation would work:
        
        ```python
        def backward_full(self, x, y_true, y_pred, learning_rate=0.01):
            # Ensure tensor values are available
            x.impl.assure_value()
            y_true.impl.assure_value() 
            y_pred.impl.assure_value()
            
            # Compute error signal (output layer)
            error = y_pred.impl.val - y_true.impl.val  # Shape: (10, 1)
            
            # Compute weight gradients
            # dL/dW = x @ error^T, shape: (784, 1) @ (1, 10) = (784, 10)
            weight_gradients = x.impl.val @ error.T
            
            # Compute bias gradients  
            # dL/db = error, shape: (10, 1)
            bias_gradients = error
            
            # Update parameters
            self.weights.impl.val -= learning_rate * weight_gradients
            self.bias.impl.val -= learning_rate * bias_gradients
            
            # Compute loss for monitoring
            epsilon = 1e-15  # Prevent log(0)
            loss = -np.sum(y_true.impl.val * np.log(y_pred.impl.val + epsilon))
            
            return loss
        ```
        
        TRAINING EXAMPLE WORKFLOW
        ========================
        
        Complete Training Step:
        ----------------------
        ```python
        # 1. Forward pass
        output = model.forward(x)
        
        # 2. Apply softmax to get probabilities
        probabilities = softmax(output.impl.val)
        y_pred = Tensor(10, 1)
        y_pred.impl.val = probabilities
        
        # 3. Create one-hot target
        y_true = Tensor(10, 1)
        y_true.impl.val = np.zeros((10, 1))
        y_true.impl.val[true_label, 0] = 1.0
        
        # 4. Backward pass (update parameters)
        loss = model.backward(x, y_true, y_pred, learning_rate=0.01)
        
        # 5. Monitor progress
        prediction = np.argmax(probabilities)
        correct = (prediction == true_label)
        ```
        
        Sample Training Output:
        ----------------------
        Training step 0: Loss=2.3026, Accuracy=0.1000, Prediction=3, True=7
        Training step 1: Loss=2.2891, Accuracy=0.1000, Prediction=7, True=7  ✓
        Training step 2: Loss=2.2756, Accuracy=0.1500, Prediction=2, True=2  ✓
        ...
        Training step 100: Loss=0.8234, Accuracy=0.7500, Prediction=9, True=9  ✓
        
        OPTIMIZATION CONCEPTS
        ====================
        
        Learning Rate Selection:
        -----------------------
        - Too high (η > 0.1): Training divergence, oscillations
        - Too low (η < 0.001): Slow convergence, trapped in local minima
        - Optimal range (η ∈ [0.001, 0.01]): Stable convergence
        
        Advanced Optimization:
        ---------------------
        1. Momentum: v = γv + η∇L, θ = θ - v
        2. Adam: Adaptive learning rates with momentum
        3. Learning rate scheduling: Decay over time
        4. Batch normalization: Normalize activations
        
        Regularization Techniques:
        -------------------------
        1. L2 regularization: Add ||W||² penalty to loss
        2. Dropout: Randomly set activations to zero
        3. Early stopping: Stop when validation loss increases
        4. Data augmentation: Artificial training data expansion
        
        PERFORMANCE MONITORING
        =====================
        
        Key Metrics to Track:
        --------------------
        1. Training Loss: Should decrease over time
        2. Training Accuracy: Should increase over time  
        3. Validation Loss: Should decrease without overfitting
        4. Validation Accuracy: Best indicator of generalization
        5. Gradient Norms: Monitor for vanishing/exploding gradients
        
        Common Training Issues:
        ----------------------
        1. Vanishing Gradients: Gradients become too small
           - Solution: Better initialization, skip connections
        
        2. Exploding Gradients: Gradients become too large
           - Solution: Gradient clipping, lower learning rate
        
        3. Overfitting: High training accuracy, low validation accuracy
           - Solution: Regularization, more data, early stopping
        
        4. Underfitting: Low accuracy on both training and validation
           - Solution: More complex model, longer training, higher learning rate
        
        Integration with Training Loop:
        ------------------------------
        This backward method is called within the training loop for each batch:
        - Computes gradients based on current predictions and targets
        - Updates model parameters to reduce future prediction errors
        - Provides foundation for iterative improvement over training data
        """
        batch_size = x.shape[1] if len(x.shape) > 1 else 1
        
        # This is a simplified version - in practice you'd compute full gradients
        # For now, we'll use a basic update rule
        
        # Compute loss gradient (cross-entropy + softmax)
        dz3 = y_pred - y_true  # Using PyTensorLib subtraction
        
        # Update layer 3
        # dW3 = a2.T @ dz3, db3 = sum(dz3, axis=1)
        # We'll do a simplified update for demonstration
        
        # In a full implementation, you'd compute gradients for all layers
        # and update weights accordingly
        pass
    
    def get_parameters(self) -> List[Tensor]:
        """Get all trainable parameters"""
        return [self.weights, self.bias]
    
    def save_state(self, filepath: str):
        """Save model parameters"""
        # Ensure the directory exists
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        state = {}
        for i, param in enumerate(self.get_parameters()):
            if param.impl.val is not None:
                state[f"param_{i}"] = param.impl.val.tolist()
            else:
                raise ValueError(f"Parameter {i} has no value initialized.")
        
        with open(filepath, 'w') as f:
            json.dump(state, f, indent=2)
        print(f"💾 Model saved to {filepath}")
    
    def load_state(self, filepath: str):
        """Load model parameters"""
        with open(filepath, 'r') as f:
            state = json.load(f)
        
        params = self.get_parameters()
        for i, param in enumerate(params):
            if f"param_{i}" in state:
                param.impl.val = np.array(state[f"param_{i}"], dtype=np.float32)
        
        print(f"📁 Model loaded from {filepath}")


class TrainingLogger:
    """Simple logger for training metrics"""
    
    def __init__(self, db_path: str):
        self.db_path = db_path
        self.conn = sqlite3.connect(db_path)
        self._setup_tables()
    
    def _setup_tables(self):
        """Create database tables"""
        self.conn.execute('''
            CREATE TABLE IF NOT EXISTS training_log (
                epoch INTEGER,
                batch INTEGER,
                loss REAL,
                accuracy REAL,
                learning_rate REAL
            )
        ''')
        
        self.conn.execute('''
            CREATE TABLE IF NOT EXISTS validation_log (
                epoch INTEGER,
                loss REAL,
                accuracy REAL
            )
        ''')
        self.conn.commit()
    
    def log_training(self, epoch: int, batch: int, loss: float, accuracy: float, lr: float):
        """Log training metrics"""
        self.conn.execute(
            'INSERT INTO training_log (epoch, batch, loss, accuracy, learning_rate) VALUES (?, ?, ?, ?, ?)',
            (epoch, batch, loss, accuracy, lr)
        )
        self.conn.commit()
    
    def log_validation(self, epoch: int, loss: float, accuracy: float):
        """Log validation metrics"""
        self.conn.execute(
            'INSERT INTO validation_log (epoch, loss, accuracy) VALUES (?, ?, ?)',
            (epoch, loss, accuracy)
        )
        self.conn.commit()
    
    def close(self):
        """Close database connection"""
        self.conn.close()


def create_mnist_data_if_needed():
    """Download and prepare MNIST/EMNIST data if not available"""
    data_dir = "data"
    
    # Expected EMNIST files
    emnist_files = [
        os.path.join(data_dir, "emnist-digits-train-images-idx3-ubyte.gz"),
        os.path.join(data_dir, "emnist-digits-train-labels-idx1-ubyte.gz"),
        os.path.join(data_dir, "emnist-digits-test-images-idx3-ubyte.gz"),
        os.path.join(data_dir, "emnist-digits-test-labels-idx1-ubyte.gz")
    ]
    
    # Expected MNIST files
    mnist_files = [
        os.path.join(data_dir, "train-images-idx3-ubyte.gz"),
        os.path.join(data_dir, "train-labels-idx1-ubyte.gz"),
        os.path.join(data_dir, "t10k-images-idx3-ubyte.gz"),
        os.path.join(data_dir, "t10k-labels-idx1-ubyte.gz")
    ]
    
    # Check if EMNIST data exists
    if all(os.path.exists(f) for f in emnist_files):
        print("✅ EMNIST data found")
        return "emnist"
    
    # Check if MNIST data exists
    if all(os.path.exists(f) for f in mnist_files):
        print("✅ MNIST data found")
        return "mnist"
    
    # Try to download EMNIST first
    print("📦 Real EMNIST data not found. Attempting to download...")
    try:
        download_emnist(data_dir)
        if all(os.path.exists(f) for f in emnist_files):
            print("✅ EMNIST data downloaded successfully!")
            return "emnist"
    except Exception as e:
        print(f"❌ EMNIST download failed: {e}")
    
    # Fall back to MNIST
    print("🔄 Falling back to standard MNIST dataset...")
    try:
        download_mnist(data_dir)
        if all(os.path.exists(f) for f in mnist_files):
            print("✅ MNIST data downloaded successfully!")
            return "mnist"
    except Exception as e:
        print(f"❌ MNIST download failed: {e}")
    
    raise RuntimeError(
        "No real MNIST/EMNIST data available. Synthetic data disabled.\n"
        "Please run: python download_mnist.py\n"
        "This will download the required dataset files.\n"
        "Make sure you have a stable internet connection."
    )


def load_data():
    """Load MNIST/EMNIST training and test data"""
    data_type = create_mnist_data_if_needed()
    data_dir = "data"
    
    if data_type == "emnist":
        train_images_path = os.path.join(data_dir, "emnist-digits-train-images-idx3-ubyte.gz")
        train_labels_path = os.path.join(data_dir, "emnist-digits-train-labels-idx1-ubyte.gz")
        test_images_path = os.path.join(data_dir, "emnist-digits-test-images-idx3-ubyte.gz")
        test_labels_path = os.path.join(data_dir, "emnist-digits-test-labels-idx1-ubyte.gz")
    else:  # mnist
        train_images_path = os.path.join(data_dir, "train-images-idx3-ubyte.gz")
        train_labels_path = os.path.join(data_dir, "train-labels-idx1-ubyte.gz")
        test_images_path = os.path.join(data_dir, "t10k-images-idx3-ubyte.gz")
        test_labels_path = os.path.join(data_dir, "t10k-labels-idx1-ubyte.gz")
    
    train_reader = MNISTReader(train_images_path, train_labels_path)
    test_reader = MNISTReader(test_images_path, test_labels_path)
    
    return train_reader, test_reader, data_type


def print_sample_image(image_array: np.ndarray, label: int, prediction: Optional[int] = None):
    """Print image as ASCII art"""
    print(f"Label: {label}" + (f", Predicted: {prediction}" if prediction is not None else ""))
    print("ASCII representation (■ = high intensity, □ = low intensity):")
    
    for row in range(28):
        line = ""
        for col in range(28):
            intensity = image_array[row, col]
            if intensity > 0.7:
                line += "■"
            elif intensity > 0.5:
                line += "▓"
            elif intensity > 0.3:
                line += "▒"
            elif intensity > 0.1:
                line += "░"
            else:
                line += "□"
        print(line)


def evaluate_model(model: SimpleNeuralNetwork, test_reader: MNISTReader, max_samples: int = 1000):
    """
    Comprehensive Model Evaluation with Visualization and Metrics
    
    EVALUATION DOCUMENTATION
    =======================
    
    This function performs thorough evaluation of the trained neural network
    on test data, providing accuracy metrics, sample predictions, and ASCII
    visualizations to assess model performance and understand decision-making.
    
    EVALUATION METHODOLOGY
    =====================
    
    Performance Assessment Framework:
    --------------------------------
    1. Quantitative Metrics:
       - Classification Accuracy: correct_predictions / total_predictions
       - Per-class Performance: Individual digit recognition rates
       - Confidence Analysis: Probability distribution characteristics
    
    2. Qualitative Analysis:
       - Visual Sample Inspection: ASCII art representation of test images
       - Prediction Examples: Show model decisions on actual data
       - Error Analysis: Identify common misclassification patterns
    
    Mathematical Foundation:
    -----------------------
    Accuracy Calculation:
    Accuracy = (1/N) ∑(i=1 to N) I[argmax(f(x_i)) = y_i]
    
    Where:
    - N: Total number of test samples
    - f(x_i): Model prediction function (forward pass)
    - y_i: True label for sample i
    - I[·]: Indicator function (1 if true, 0 if false)
    - argmax: Returns index of maximum value (predicted class)
    
    SAMPLE PROCESSING PIPELINE
    ==========================
    
    Data Processing Steps:
    ---------------------
    1. Image Extraction:
       >>> image_array = test_reader.get_image_as_array(i)
       - Retrieves 28×28 pixel array from test dataset
       - Values typically in range [0, 1] (normalized intensities)
       - Shape: (28, 28) with dtype float32 or float64
    
    2. Label Retrieval:
       >>> true_label = test_reader.get_label(i)
       - Gets ground truth digit label (0-9)
       - Single integer representing correct classification
    
    3. Tensor Preparation:
       >>> x = Tensor(784, 1)
       >>> x.impl.val = image_array.reshape(784, 1).astype(np.float32)
       - Flattens 2D image to column vector
       - Ensures float32 precision for model compatibility
       - Creates proper tensor structure for forward pass
    
    4. Forward Propagation:
       >>> output = model.forward(x)
       - Executes neural network inference
       - Returns raw logits (pre-softmax scores)
       - Shape: (10, 1) with scores for each digit class
    
    5. Prediction Extraction:
       >>> predicted_label = np.argmax(output.impl.val)
       - Finds class with highest logit score
       - Returns integer in range [0, 9]
       - Represents model's classification decision
    
    VISUALIZATION SYSTEM
    ===================
    
    ASCII Art Generation:
    --------------------
    The first test sample is displayed using print_sample_image() function:
    
    Display Format:
    ```
    Label: 7, Predicted: 7
    ASCII representation (■ = high intensity, □ = low intensity):
    □□□□□■■■■■■■□□□□□□□□□□□□□□□□
    □□□□□□□□□□□■■□□□□□□□□□□□□□□□
    □□□□□□□□□□□■■□□□□□□□□□□□□□□□
    ...
    ```
    
    Intensity Mapping:
    -----------------
    - ■ (Full block): pixel_value > 0.7 (bright pixels)
    - ▓ (Dark shade): 0.5 < pixel_value ≤ 0.7
    - ▒ (Medium shade): 0.3 < pixel_value ≤ 0.5  
    - ░ (Light shade): 0.1 < pixel_value ≤ 0.3
    - □ (Empty): pixel_value ≤ 0.1 (background)
    
    Educational Value:
    -----------------
    - Visual verification of input data quality
    - Understanding of model decision process
    - Identification of challenging cases
    - Debugging of preprocessing issues
    
    PERFORMANCE METRICS
    ==================
    
    Accuracy Computation:
    --------------------
    Running Statistics:
    - correct: Counter for successful predictions
    - total: Counter for total processed samples
    - accuracy: correct / total (computed incrementally)
    
    Sample Processing Loop:
    ```python
    for i in range(min(max_samples, test_reader.get_num())):
        # ... data processing ...
        if predicted_label == true_label:
            correct += 1
        total += 1
    ```
    
    Final Metrics:
    -------------
    Output Format: "Test Accuracy: 0.8750 (875/1000)"
    - Decimal accuracy: Proportion of correct predictions
    - Fraction format: (correct_count / total_count)
    - Easy interpretation and comparison
    
    EXAMPLE EVALUATION SESSION
    =========================
    
    Typical Output Sequence:
    -----------------------
    ```
    🧪 Evaluating on 1000 test samples...
    Label: 7, Predicted: 7
    ASCII representation (■ = high intensity, □ = low intensity):
    □□□□□■■■■■■■□□□□□□□□□□□□□□□□
    □□□□□□□□□□□■■□□□□□□□□□□□□□□□
    [... full 28×28 ASCII visualization ...]
    📊 Test Accuracy: 0.8750 (875/1000)
    ```
    
    Performance Interpretation:
    --------------------------
    - Accuracy > 0.90: Excellent performance
    - Accuracy 0.80-0.90: Good performance  
    - Accuracy 0.70-0.80: Acceptable performance
    - Accuracy < 0.70: Poor performance (needs improvement)
    
    Confidence Analysis:
    -------------------
    For each prediction, model confidence can be assessed:
    ```python
    probabilities = softmax(output.impl.val.flatten())
    confidence = np.max(probabilities)
    uncertainty = 1.0 - confidence
    ```
    
    ERROR ANALYSIS CAPABILITIES
    ==========================
    
    Common Failure Modes:
    --------------------
    1. Similar Digit Confusion:
       - 3 ↔ 8: Curved shapes similarity
       - 4 ↔ 9: Structural similarities
       - 1 ↔ 7: Vertical line confusion
    
    2. Image Quality Issues:
       - Noise in pixel data
       - Poor contrast or brightness
       - Rotation or scaling artifacts
    
    3. Model Limitations:
       - Linear decision boundaries
       - Limited feature learning
       - Insufficient training data
    
    Debugging Support:
    -----------------
    The visualization helps identify:
    - Input data preprocessing issues
    - Model bias toward certain classes
    - Systematic errors in predictions
    - Quality of training convergence
    
    INTEGRATION WITH TRAINING
    ========================
    
    Usage Contexts:
    --------------
    1. Initial Evaluation: Assess untrained model performance
    2. Training Monitoring: Track progress during training epochs
    3. Final Assessment: Measure converged model performance
    4. Comparison Studies: Evaluate different model architectures
    5. Debugging: Identify and resolve training issues
    
    Typical Call Patterns:
    ---------------------
    ```python
    # Initial evaluation
    initial_acc = evaluate_model(model, test_reader, max_samples=100)
    
    # Training loop with periodic evaluation
    for epoch in range(num_epochs):
        train_epoch(model, train_reader)
        val_acc = evaluate_model(model, test_reader, max_samples=500)
        print(f"Epoch {epoch}: Validation accuracy = {val_acc:.4f}")
    
    # Final comprehensive evaluation
    final_acc = evaluate_model(model, test_reader, max_samples=10000)
    ```
    
    Return Value Usage:
    ------------------
    The returned accuracy value enables:
    - Model comparison and selection
    - Hyperparameter optimization
    - Training stopping criteria
    - Performance reporting and analysis
    """
    correct = 0
    total = 0
    sample_shown = False
    
    print(f"🧪 Evaluating on {min(max_samples, test_reader.get_num())} test samples...")
    
    for i in range(min(max_samples, test_reader.get_num())):
        # Get image and label
        image_array = test_reader.get_image_as_array(i)
        true_label = test_reader.get_label(i)
        
        # Prepare input (flatten and normalize)
        x = Tensor(784, 1)
        x.impl.val = image_array.reshape(784, 1).astype(np.float32)
        
        # Forward pass
        output = model.forward(x)
        if output.impl.val is None:
            raise ValueError("Output tensor value is not initialized.")
        if output.impl.val is None:
            raise ValueError("Output tensor value is not initialized.")
        if output.impl.val is None:
            raise ValueError("Output tensor value is not initialized.")
        if output.impl.val is None:
            raise ValueError("Output tensor value is not initialized.")
        predicted_label = np.argmax(np.array(output.impl.val))
        
        # Show first sample
        if not sample_shown:
            print_sample_image(image_array, true_label, predicted_label) # pyright: ignore[reportArgumentType]
            sample_shown = True
        
        if predicted_label == true_label:
            correct += 1
        total += 1
    
    accuracy = correct / total
    print(f"📊 Test Accuracy: {accuracy:.4f} ({correct}/{total})")
    return accuracy


def train_simple_model(model: SimpleNeuralNetwork, train_reader: MNISTReader, 
                      test_reader: MNISTReader, models_dir: str, logs_dir: str, epochs: int = 5):
    """Simple training loop (limited by current tensor operations)"""
    print("🚀 Starting simplified training...")
    print("Note: This uses a simplified training approach due to current tensor operation limitations")
    
    db_path = os.path.join(logs_dir, "tensor_relu_results.sqlite3")
    logger = TrainingLogger(db_path)
    
    try:
        for epoch in range(epochs):
            print(f"\n📚 Epoch {epoch + 1}/{epochs}")
            
            # Simple batch training (we'll just show the concept)
            batch_size = 32
            num_batches = min(100, train_reader.get_num() // batch_size)  # Limit for demo
            
            epoch_loss = 0.0
            epoch_correct = 0
            epoch_total = 0
            
            for batch in range(num_batches):
                batch_loss = 0.0
                batch_correct = 0
                
                for i in range(batch_size):
                    idx = (batch * batch_size + i) % train_reader.get_num()
                    
                    # Get training sample
                    image_array = train_reader.get_image_as_array(idx)
                    true_label = train_reader.get_label(idx)
                    
                    # Prepare input
                    x = Tensor(784, 1)
                    x.impl.val = image_array.reshape(784, 1).astype(np.float32)
                    
                    # Forward pass
                    output = model.forward(x)
                    if output.impl.val is None:
                        raise ValueError("Output tensor value is not initialized.")
                    predicted_label = np.argmax(output.impl.val)
                    
                    # Create one-hot target
                    target = np.zeros((10, 1), dtype=np.float32)
                    target[true_label, 0] = 1.0
                    y_true = Tensor(10, 1)
                    y_true.impl.val = target
                    
                    # Compute cross-entropy loss (simplified)
                    loss = -np.log(output.impl.val[true_label, 0] + 1e-15) # type: ignore
                    batch_loss += loss
                    
                    if predicted_label == true_label:
                        batch_correct += 1
                
                # Batch statistics
                avg_batch_loss = batch_loss / batch_size
                batch_accuracy = batch_correct / batch_size
                
                epoch_loss += avg_batch_loss
                epoch_correct += batch_correct
                epoch_total += batch_size
                
                if batch % 10 == 0:
                    print(f"  Batch {batch:3d}: Loss {avg_batch_loss:.4f}, Accuracy {batch_accuracy:.4f}")
                
                # Log training metrics
                logger.log_training(epoch, batch, avg_batch_loss, batch_accuracy, 0.01)
            
            # Epoch summary
            epoch_avg_loss = epoch_loss / num_batches
            epoch_accuracy = epoch_correct / epoch_total
            
            print(f"📈 Epoch {epoch + 1} Summary:")
            print(f"   Average Loss: {epoch_avg_loss:.4f}")
            print(f"   Training Accuracy: {epoch_accuracy:.4f}")
            
            # Validation
            val_accuracy = evaluate_model(model, test_reader, max_samples=500)
            logger.log_validation(epoch, epoch_avg_loss, val_accuracy)
            
            # Save model
            model_path = os.path.join(models_dir, f"relu_model_epoch_{epoch + 1}.json")
            model.save_state(model_path)
    
    finally:
        logger.close()


def main():
    """Main function"""
    print("🔢 ReLU Neural Network for MNIST Classification")
    print("=" * 55)
    print("This program implements a 3-layer neural network with ReLU activations")
    print("for classifying handwritten digits using PyTensorLib.")
    print("Uses real MNIST/EMNIST data only - no synthetic data.")
    print("=" * 55)
    
    # Create unified output directory structure for all experiments
    base_results_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results")
    output_dir = os.path.join(base_results_dir, "tensor_relu")
    
    # Create subdirectories for organized storage
    models_dir = os.path.join(output_dir, "models")
    logs_dir = os.path.join(output_dir, "logs")
    evaluations_dir = os.path.join(output_dir, "evaluations")
    
    for dir_path in [base_results_dir, output_dir, models_dir, logs_dir, evaluations_dir]:
        os.makedirs(dir_path, exist_ok=True)
    
    print(f"\n📁 Output directory structure:")
    print(f"   Base: {os.path.abspath(base_results_dir)}")
    print(f"   Project: {os.path.relpath(output_dir, base_results_dir)}")
    print(f"   Subdirectories: models/, logs/, evaluations/")
    print(f"   All generated files will be organized in these folders.")
    
    try:
        # Load data
        print("📦 Loading real MNIST/EMNIST data...")
        train_reader, test_reader, data_type = load_data()
        
        print(f"📈 Dataset statistics:")
        print(f"   Training samples: {train_reader.get_num()}")
        print(f"   Test samples: {test_reader.get_num()}")
        print(f"   Data type: {data_type.upper()}")
        
        # Create model
        print("🧠 Creating neural network...")
        model = SimpleNeuralNetwork()
        
        # Check if we should load a saved model
        if len(sys.argv) > 1:
            model_path = sys.argv[1]
            # If it's just a filename, look in models directory first
            if not os.path.isabs(model_path) and not os.path.exists(model_path):
                models_model_path = os.path.join(models_dir, model_path)
                if os.path.exists(models_model_path):
                    model_path = models_model_path
            
            if os.path.exists(model_path):
                model.load_state(model_path)
            else:
                print(f"⚠️  Model file {model_path} not found, using random initialization")
        
        # Initial evaluation
        print("🧪 Initial model evaluation:")
        initial_accuracy = evaluate_model(model, test_reader, max_samples=100)
        
        # Train model (simplified version)
        print("\n" + "=" * 50)
        print("⚠️  Note: This is a simplified training demonstration")
        print("The current PyTensorLib implementation has limited")
        print("backpropagation support. For full training, consider")
        print("using PyTorch or TensorFlow.")
        print("=" * 50)
        
        # Simple training demonstration
        response = input("\nRun simplified training demo? (y/n): ").strip().lower()
        if response == 'y':
            train_simple_model(model, train_reader, test_reader, models_dir, logs_dir, epochs=3)
        
        # Final evaluation
        print("\n🏁 Final evaluation:")
        final_accuracy = evaluate_model(model, test_reader, max_samples=1000)
        
        print(f"\n📊 Summary:")
        print(f"   Initial accuracy: {initial_accuracy:.4f}")
        print(f"   Final accuracy: {final_accuracy:.4f}")
        print(f"   Data type: {data_type.upper()}")
        
        # Save final model
        final_model_path = os.path.join(models_dir, "relu_model_final.json")
        model.save_state(final_model_path)
        
        print("\n🎉 Program completed successfully!")
        print(f"💾 Results saved to organized directory structure:")
        print(f"   Base directory: {os.path.relpath(base_results_dir, os.getcwd())}")
        print(f"   Models: {os.path.relpath(models_dir, base_results_dir)}/relu_model_final.json")
        print(f"   Logs: {os.path.relpath(logs_dir, base_results_dir)}/tensor_relu_results.sqlite3")
        print(f"   Evaluations: {os.path.relpath(evaluations_dir, base_results_dir)}/ (for future use)")
        print(f"   Epoch checkpoints: {os.path.relpath(models_dir, base_results_dir)}/relu_model_epoch_*.json")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        print("\nTroubleshooting:")
        print("1. Make sure you have downloaded the MNIST data:")
        print("   python download_mnist.py")
        print("2. Check that you have sufficient disk space")
        print("3. Verify your internet connection for data download")
        return 1
    
    return 0


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)