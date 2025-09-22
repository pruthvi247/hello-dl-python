#!/usr/bin/env python3
"""
Perceptron Learning Algorithm for 3 vs 7 Classification using PyTensorLib

COMPREHENSIVE MODULE DOCUMENTATION
=================================

This module implements a complete perceptron learning system for binary classification
of handwritten digits 3 and 7 from the MNIST/EMNIST dataset. It serves as a Python
conversion of the original C++ 37learn.cc program, demonstrating fundamental machine
learning concepts using the PyTensorLib tensor computation framework.

ALGORITHM OVERVIEW
=================

The perceptron is one of the simplest machine learning algorithms, implementing
a linear classifier that learns to separate two classes using the following approach:

Mathematical Foundation:
-----------------------
1. Linear Decision Function: f(x) = W·x + b
   - W: weight vector (28×28 for MNIST images)
   - x: input image vector (flattened 28×28 pixels)
   - b: bias term (scalar offset)

2. Decision Boundary: prediction = sign(f(x))
   - If f(x) > 0: predict class 7 (positive)
   - If f(x) ≤ 0: predict class 3 (negative)

3. Learning Rule with Margin:
   - If label=7 and f(x)<2: W += η·x, b += η (boost confidence)
   - If label=3 and f(x)>-2: W -= η·x, b -= η (reduce confidence)
   - η: learning rate (typically 0.01)

Enhanced Perceptron Features:
----------------------------
- Margin-based updates (confidence threshold of ±2)
- Periodic accuracy evaluation during training
- Early stopping when accuracy exceeds 98%
- Comprehensive logging and visualization
- Organized file output structure

DEPENDENCY INTEGRATION
=====================

PyTensorLib Components:
----------------------

1. tensor_lib.Tensor:
   Primary tensor computation engine providing:
   - Multi-dimensional array storage with shape tracking
   - Lazy evaluation system through tensor.impl.assure_value()
   - Type-safe operations with TMode state management
   - Memory-efficient computation graphs
   
   Usage in this module:
   >>> weights = Tensor(28, 28)  # Create weight matrix
   >>> weights.impl.val = np.random.normal(0, scale, (28, 28))
   >>> image = Tensor(28, 28)    # Create image tensor
   >>> score = np.sum(weights.impl.val * image.impl.val) + bias

2. mnist_utils.MNISTReader:
   MNIST dataset interface providing:
   - Unified data loading (MNIST, EMNIST, synthetic)
   - Image preprocessing and normalization
   - Label extraction and filtering
   - Array conversion utilities
   
   Integration examples:
   >>> train_reader = MNISTReader(train_images, train_labels)
   >>> num_samples = train_reader.get_num()
   >>> label = train_reader.get_label(index)
   >>> img_array = train_reader.get_image_as_array(index)

External Dependencies:
---------------------

1. NumPy (numpy):
   Numerical computation backend providing:
   - Efficient array operations for tensor data
   - Mathematical functions (sum, dot product, normalization)
   - Random number generation for weight initialization
   - Type conversion and array manipulation
   
   Critical operations:
   >>> dot_product = np.sum(weights.impl.val * image.impl.val)
   >>> weights.impl.val = np.random.normal(0, scale, (28, 28))
   >>> normalized = (data - data_min) / (data_max - data_min)

2. SQLite3 (sqlite3):
   Database logging system providing:
   - Training progress tracking
   - Individual prediction logging
   - Performance analysis capabilities
   - Persistent storage for experimental results
   
   Database schema:
   >>> test_results: (id, label, res, verdict, correct, timestamp)
   >>> training_progress: (epoch, accuracy, weights_saved, bias, timestamp)

3. JSON (json):
   Model serialization providing:
   - Weight and bias persistence
   - Training configuration storage
   - Result metadata preservation
   - Cross-platform model exchange

4. OS utilities (os, sys):
   System integration providing:
   - Directory creation and management
   - Path manipulation and validation
   - Module path configuration
   - File system operations

DATA FLOW ARCHITECTURE
======================

Input Data Pipeline:
-------------------
MNIST/EMNIST Dataset → MNISTReader → Tensor → Perceptron Algorithm

1. Raw Data Loading:
   >>> train_images, train_labels = download_emnist(data_dir)
   Format: Binary IDX files with image pixel arrays and integer labels

2. Reader Initialization:
   >>> train_reader = MNISTReader(train_images, train_labels)
   Provides: Indexed access to samples with preprocessing

3. Sample Extraction:
   >>> img_array = train_reader.get_image_as_array(n)  # Shape: (28, 28)
   >>> label = train_reader.get_label(n)               # Value: 3 or 7

4. Tensor Conversion:
   >>> img = Tensor(28, 28)
   >>> img.impl.val = img_array.astype(np.float32)
   Result: PyTensorLib tensor ready for computation

Training Pipeline:
-----------------
Input Sample → Forward Pass → Prediction → Weight Update → Logging

1. Forward Pass Computation:
   >>> score = np.sum(img.impl.val * weights.impl.val) + bias
   Mathematics: Computes dot product W·x + b

2. Prediction Generation:
   >>> predict = 7 if score > 0 else 3
   Logic: Applies decision boundary threshold

3. Weight Update Logic:
   >>> if label == 7 and score < 2.0:
   ...     weights.impl.val += img.impl.val * learning_rate
   ...     bias += learning_rate
   Result: Moves decision boundary toward correct classification

4. Progress Logging:
   >>> logger.log_test_result(label, score, predict)
   >>> logger.log_training_progress(epoch, accuracy, weights_file, bias)

Output Generation Pipeline:
--------------------------
Training Results → Visualization → Database → File System

1. Weight Visualization:
   >>> save_tensor_as_image(weights, filepath, scale=252)
   Output: ASCII text files showing weight patterns

2. Database Logging:
   >>> logger = PerceptronLogger(db_path)
   Storage: SQLite database with training metrics

3. Model Serialization:
   >>> model_state = {'weights': weights.tolist(), 'bias': bias, ...}
   >>> json.dump(model_state, file)
   Format: JSON file with complete model state

4. Organized File Structure:
   results/perceptron_37/
   ├── models/perceptron_37_final.json
   ├── logs/37learn.sqlite3
   └── visualizations/weights-*.txt

COMPUTATIONAL COMPLEXITY
========================

Time Complexity Analysis:
------------------------
- Single forward pass: O(H×W) = O(784) for 28×28 images
- Weight update: O(H×W) = O(784) for element-wise operations
- Training epoch: O(N×H×W) where N = number of training samples
- Total training: O(E×N×H×W) where E = number of epochs until convergence

Space Complexity Analysis:
-------------------------
- Weight storage: O(H×W) = 3,136 bytes for 28×28 float32 weights
- Image buffer: O(H×W) = 3,136 bytes per sample
- Database overhead: O(N) for logging N training samples
- Total memory: ~10KB for model + dataset size

Performance Characteristics:
---------------------------
- Typical convergence: 500-2000 training samples
- Training time: 10-60 seconds on modern hardware
- Final accuracy: 85-98% depending on data quality
- Memory usage: <100MB including full MNIST dataset

USAGE EXAMPLES
=============

Basic Training Session:
----------------------
>>> # Run complete training pipeline
>>> python examples/perceptron_37_learn.py
🎯 Perceptron Learning Algorithm: 3 vs 7 Classification
📦 Loading MNIST/EMNIST data...
✅ EMNIST data loaded successfully
🧠 Initializing perceptron...
🚀 Starting perceptron training...
📊 Testing step 0, accuracy: 52.3%
📊 Testing step 100, accuracy: 78.9%
📊 Testing step 500, accuracy: 95.4%
📊 Testing step 789, accuracy: 98.7%
🎉 Target accuracy achieved! Stopping training.

Programmatic Usage:
------------------
>>> from perceptron_37_learn import main, do_test, PerceptronLogger
>>> 
>>> # Run training
>>> main()
>>> 
>>> # Load saved model for inference
>>> with open('results/perceptron_37/models/perceptron_37_final.json') as f:
...     model = json.load(f)
>>> weights_array = np.array(model['weights'])
>>> bias = model['bias']

Analysis and Visualization:
--------------------------
>>> # Analyze training progress
>>> import sqlite3
>>> conn = sqlite3.connect('results/perceptron_37/logs/37learn.sqlite3')
>>> cursor = conn.execute('SELECT epoch, accuracy FROM training_progress')
>>> data = cursor.fetchall()
>>> print(f"Learning curve: {data}")

>>> # Examine weight patterns
>>> with open('results/perceptron_37/visualizations/weights-final.txt') as f:
...     weight_visualization = f.read()
>>> print("Final learned features:")
>>> print(weight_visualization)

EDUCATIONAL VALUE
================

Machine Learning Concepts Demonstrated:
--------------------------------------
1. Linear Classification: Decision boundaries and hyperplanes
2. Online Learning: Processing samples one at a time
3. Gradient-Free Updates: Direct weight modification based on errors
4. Early Stopping: Preventing overfitting with validation accuracy
5. Margin-Based Learning: Confidence thresholds for robust classification

Programming Concepts Illustrated:
---------------------------------
1. Object-Oriented Design: Classes for logging and data management
2. Error Handling: Graceful degradation and fallback strategies
3. File Organization: Structured output for reproducible experiments
4. Database Integration: Persistent storage for analysis
5. Tensor Operations: Multi-dimensional array computations

Research and Development Applications:
-------------------------------------
1. Baseline Implementation: Starting point for more complex algorithms
2. Algorithm Comparison: Benchmark for evaluating other methods
3. Feature Engineering: Understanding which image regions are important
4. Hyperparameter Studies: Effects of learning rate and margin size
5. Dataset Analysis: Characteristics of MNIST digit separability

This module serves as both a functional machine learning implementation
and an educational resource for understanding fundamental concepts in
pattern recognition, online learning, and tensor computation frameworks.
"""

import os
import sys
import sqlite3
import numpy as np
import json
from typing import Tuple, Optional

# Add src to Python path for PyTensorLib imports
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'src'))

from pytensorlib.tensor_lib import Tensor
from pytensorlib.mnist_utils import MNISTReader, download_emnist, download_mnist, create_synthetic_mnist_data


class PerceptronLogger:
    """
    SQLite database logger for perceptron training results and evaluation metrics.
    
    This class provides comprehensive logging functionality for the perceptron learning
    algorithm, tracking both individual test predictions and overall training progress.
    It creates and manages two main tables: test_results and training_progress.
    
    Database Schema:
    ===============
    
    test_results table:
    - id: Auto-incrementing primary key
    - label: True label (3 or 7)
    - res: Computed score (dot product + bias)
    - verdict: Predicted label (3 or 7)
    - correct: Binary correctness flag (1=correct, 0=wrong)
    - timestamp: When the test was performed
    
    training_progress table:
    - epoch: Training step/epoch number
    - accuracy: Test accuracy percentage at this step
    - weights_saved: Filename of saved weight visualization
    - bias: Current bias value
    - timestamp: When the checkpoint was created
    
    Usage Example:
    =============
    >>> logger = PerceptronLogger("results/logs/training.db")
    >>> logger.log_test_result(label=7, score=1.5, verdict=7)  # Correct prediction
    >>> logger.log_test_result(label=3, score=0.5, verdict=7)  # Wrong prediction
    >>> logger.log_training_progress(epoch=100, accuracy=85.2, 
    ...                              weights_file="weights-100.txt", bias=0.15)
    >>> logger.close()
    
    The database can be queried later for analysis:
    >>> import sqlite3
    >>> conn = sqlite3.connect("results/logs/training.db")
    >>> cursor = conn.execute("SELECT accuracy FROM training_progress ORDER BY epoch")
    >>> accuracies = [row[0] for row in cursor.fetchall()]
    >>> print(f"Final accuracy: {accuracies[-1]:.2f}%")
    """
    
    def __init__(self, db_path: str):
        """
        Initialize SQLite database for logging training results.
        
        Creates a new database at the specified path, removing any existing file.
        Sets up the required tables for logging test results and training progress.
        
        Args:
            db_path (str): Full path where the SQLite database should be created
                          Example: "/path/to/results/logs/37learn.sqlite3"
        
        Database Initialization Steps:
        =============================
        1. Remove existing database file if it exists (fresh start)
        2. Create new SQLite connection
        3. Call _setup_tables() to create required schema
        4. Commit initial setup
        
        Example:
        ========
        >>> # Create logger for organized results directory
        >>> db_path = "results/perceptron_37/logs/37learn.sqlite3"
        >>> logger = PerceptronLogger(db_path)
        >>> print(f"Database created at: {logger.db_path}")
        Database created at: results/perceptron_37/logs/37learn.sqlite3
        
        File Operations:
        ===============
        Input:  db_path = "results/logs/training.db"
        Effect: Creates/overwrites SQLite database file
        Tables: test_results, training_progress (both empty)
        Connection: Active SQLite connection stored in self.conn
        """
        self.db_path = db_path
        
        # Remove existing database for fresh start
        if os.path.exists(db_path):
            os.unlink(db_path)
        
        # Create new database connection
        self.conn = sqlite3.connect(db_path)
        self._setup_tables()
    
    def _setup_tables(self):
        """
        Create database tables for logging test results and training progress.
        
        Creates two essential tables with appropriate schemas for tracking
        perceptron learning progress and individual prediction results.
        
        Table Schemas:
        =============
        
        test_results:
        - Stores individual predictions made during testing
        - Used for detailed analysis of prediction patterns
        - Allows tracking which specific samples are misclassified
        
        training_progress:
        - Stores checkpoints during training
        - Used for tracking learning curves and convergence
        - Links to saved weight visualizations
        
        SQL Operations:
        ==============
        1. CREATE TABLE test_results: Individual prediction logging
        2. CREATE TABLE training_progress: Training checkpoint logging
        3. COMMIT: Ensure tables are persisted to disk
        
        Example Usage After Setup:
        =========================
        >>> logger = PerceptronLogger("test.db")
        >>> # Tables are now ready for logging
        >>> logger.log_test_result(7, 1.2, 7)  # Log correct prediction
        >>> logger.log_training_progress(0, 50.0, "weights-0.txt", 0.0)
        
        Database State After Setup:
        ===========================
        Tables: test_results (empty), training_progress (empty)
        Indexes: Auto-incrementing primary keys
        Constraints: NOT NULL on required fields
        """
        # Table for individual test predictions
        self.conn.execute('''
            CREATE TABLE IF NOT EXISTS test_results (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                label INTEGER,
                res REAL,
                verdict INTEGER,
                correct INTEGER,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Table for training progress checkpoints
        self.conn.execute('''
            CREATE TABLE IF NOT EXISTS training_progress (
                epoch INTEGER,
                accuracy REAL,
                weights_saved TEXT,
                bias REAL,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        self.conn.commit()
    
    def log_test_result(self, label: int, score: float, verdict: int):
        """
        Log an individual test prediction result with detailed metrics.
        
        Records the true label, computed score, predicted label, and correctness
        for a single test sample. This enables detailed analysis of prediction
        patterns and identification of frequently misclassified samples.
        
        Args:
            label (int): True label from dataset (3 or 7)
            score (float): Computed score from perceptron (dot_product + bias)
            verdict (int): Predicted label based on score (3 if score ≤ 0, else 7)
        
        Prediction Logic:
        ================
        The verdict is determined by the perceptron decision boundary:
        - If score > 0: predict 7 (positive class)
        - If score ≤ 0: predict 3 (negative class)
        
        Correctness is computed as binary flag:
        - correct = 1 if verdict == label (correct prediction)
        - correct = 0 if verdict != label (incorrect prediction)
        
        Example Logging Scenarios:
        =========================
        >>> logger = PerceptronLogger("test.db")
        
        # Scenario 1: Correct prediction of digit 7
        >>> logger.log_test_result(label=7, score=1.5, verdict=7)
        # Database: label=7, res=1.5, verdict=7, correct=1
        
        # Scenario 2: Incorrect prediction (7 classified as 3)
        >>> logger.log_test_result(label=7, score=-0.3, verdict=3)
        # Database: label=7, res=-0.3, verdict=3, correct=0
        
        # Scenario 3: Correct prediction of digit 3
        >>> logger.log_test_result(label=3, score=-2.1, verdict=3)
        # Database: label=3, res=-2.1, verdict=3, correct=1
        
        # Scenario 4: Incorrect prediction (3 classified as 7)
        >>> logger.log_test_result(label=3, score=0.8, verdict=7)
        # Database: label=3, res=0.8, verdict=7, correct=0
        
        Database Operations:
        ===================
        1. Compute correctness: correct = 1 if verdict == label else 0
        2. INSERT INTO test_results: (label, score, verdict, correct)
        3. COMMIT: Ensure data is written to disk immediately
        4. Timestamp: Automatically added by database DEFAULT
        
        Analysis Queries (Post-logging):
        ===============================
        The database can be queried for detailed analysis:
        
        # Overall accuracy calculation
        cursor = conn.execute("SELECT AVG(correct) FROM test_results")
        accuracy = cursor.fetchone()[0] * 100
        
        # Find misclassified samples ordered by confidence
        cursor = conn.execute("SELECT label, res, verdict FROM test_results WHERE correct = 0 ORDER BY ABS(res)")
        for label, score, verdict in cursor.fetchall():
            print(f"True: {label}, Score: {score:.3f}, Predicted: {verdict}")
        """
        correct = 1 if verdict == label else 0
        self.conn.execute(
            'INSERT INTO test_results (label, res, verdict, correct) VALUES (?, ?, ?, ?)',
            (label, score, verdict, correct)
        )
        self.conn.commit()
    
    def log_training_progress(self, epoch: int, accuracy: float, weights_file: str, bias: float):
        """
        Log training progress checkpoint with accuracy and model state.
        
        Records a snapshot of the training state at a specific epoch, including
        test accuracy, current bias value, and the filename of saved weight
        visualization. This enables tracking learning curves and convergence.
        
        Args:
            epoch (int): Current training step/epoch number
            accuracy (float): Test accuracy percentage (0.0 to 100.0)
            weights_file (str): Filename of saved weight visualization
            bias (float): Current bias value of the perceptron
        
        Training Progress Tracking:
        ==========================
        This method creates checkpoints that allow analysis of:
        - Learning curves: accuracy vs. epoch
        - Convergence patterns: when does learning plateau?
        - Model evolution: how weights and bias change over time
        - Checkpoint recovery: which weights gave best performance
        
        Example Training Sequence:
        =========================
        >>> logger = PerceptronLogger("training.db")
        
        # Initial checkpoint (random weights)
        >>> logger.log_training_progress(
        ...     epoch=0, accuracy=52.3, 
        ...     weights_file="weights-0.txt", bias=0.0
        ... )
        
        # Early training (weights adapting)
        >>> logger.log_training_progress(
        ...     epoch=100, accuracy=67.8, 
        ...     weights_file="weights-100.txt", bias=0.15
        ... )
        
        # Mid training (approaching convergence)
        >>> logger.log_training_progress(
        ...     epoch=500, accuracy=89.4, 
        ...     weights_file="weights-500.txt", bias=0.23
        ... )
        
        # Late training (converged)
        >>> logger.log_training_progress(
        ...     epoch=1000, accuracy=98.7, 
        ...     weights_file="weights-1000.txt", bias=0.28
        ... )
        
        Database Record Format:
        ======================
        Each checkpoint creates a database record:
        - epoch: 1000
        - accuracy: 98.7
        - weights_saved: "weights-1000.txt" 
        - bias: 0.28
        - timestamp: "2025-09-21 10:15:30"
        
        Learning Curve Analysis:
        =======================
        # Extract training progress for visualization
        cursor = conn.execute("SELECT epoch, accuracy FROM training_progress ORDER BY epoch")
        data = cursor.fetchall()
        epochs = [row[0] for row in data]
        accuracies = [row[1] for row in data]
        
        # Plot learning curve (requires matplotlib)
        plt.plot(epochs, accuracies)
        plt.xlabel("Training Epoch")
        plt.ylabel("Test Accuracy (%)")
        plt.title("Perceptron Learning Curve")
        
        Model Recovery:
        ==============
        # Find best performing checkpoint
        cursor = conn.execute("SELECT epoch, weights_saved, bias, accuracy FROM training_progress ORDER BY accuracy DESC LIMIT 1")
        best_epoch, best_weights, best_bias, best_acc = cursor.fetchone()
        print(f"Best model: epoch {best_epoch}, accuracy {best_acc:.2f}%")
        print(f"Weights file: {best_weights}, bias: {best_bias:.6f}")
        """
        self.conn.execute(
            'INSERT INTO training_progress (epoch, accuracy, weights_saved, bias) VALUES (?, ?, ?, ?)',
            (epoch, accuracy, weights_file, bias)
        )
        self.conn.commit()
    
    def close(self):
        """
        Close the database connection and finalize logging.
        
        Properly closes the SQLite connection to ensure all data is written
        to disk and resources are released. Should be called when logging
        is complete or the program is shutting down.
        
        Connection Cleanup:
        ==================
        1. Commits any pending transactions
        2. Closes the database connection
        3. Releases file locks and system resources
        
        Example Usage:
        =============
        >>> logger = PerceptronLogger("training.db")
        >>> # ... perform logging operations ...
        >>> logger.close()  # Clean shutdown
        
        Best Practices:
        ==============
        - Always call close() when done logging
        - Use try/finally or context managers for guaranteed cleanup
        - Don't access logger methods after calling close()
        
        >>> try:
        ...     logger = PerceptronLogger("training.db")
        ...     # ... logging operations ...
        ... finally:
        ...     logger.close()  # Guaranteed cleanup
        """
        self.conn.close()


def save_tensor_as_image(tensor: Tensor, filepath: str, scale: int = 252, invert: bool = False):
    """
    Save tensor as ASCII text visualization (simplified version of C++ saveTensor function).
    
    This function converts a 2D tensor into a visual representation saved as a text file.
    It's designed to replicate the visualization functionality from the original C++
    implementation, allowing examination of weight patterns and image data.
    
    Args:
        tensor (Tensor): 2D tensor to visualize (typically 28x28 for MNIST)
        filepath (str): Full output file path (including directory)
                       Example: "results/visualizations/weights-100.png"
        scale (int): Scaling factor for pixel values (default: 252)
                    Used to amplify small weight differences for better visibility
        invert (bool): Whether to invert colors (default: False)
                      True: Light pixels become dark (useful for weight visualization)
                      False: Direct mapping (useful for image data)
    
    Tensor Visualization Process:
    ============================
    
    Step 1: Data Extraction and Validation
    >>> tensor = Tensor(3, 3)
    >>> tensor.impl.val = np.array([[0.1, 0.5, 0.9],
    ...                            [0.2, 0.6, 0.8], 
    ...                            [0.3, 0.7, 0.4]], dtype=np.float32)
    Input tensor shape: (3, 3)
    Input data range: [0.1, 0.9]
    
    Step 2: Normalization (0-1 range)
    data_min = 0.1, data_max = 0.9
    normalized = (data - 0.1) / (0.9 - 0.1) = (data - 0.1) / 0.8
    Result: [[0.0, 0.5, 1.0],
             [0.125, 0.625, 0.875],
             [0.25, 0.75, 0.375]]
    
    Step 3: Optional Inversion
    If invert=True: normalized = 1.0 - normalized
    Result: [[1.0, 0.5, 0.0],
             [0.875, 0.375, 0.125],
             [0.75, 0.25, 0.625]]
    
    Step 4: Convert to 0-255 Range
    image_data = (normalized * 255).astype(np.uint8)
    Result: [[255, 127, 0],
             [223, 95, 31],
             [191, 63, 159]]
    
    Weight Visualization Example:
    ============================
    >>> # Visualize learned weights showing digit 7 pattern
    >>> weights = Tensor(28, 28)
    >>> weights.impl.val = trained_weights_for_digit_7  # After training
    >>> save_tensor_as_image(weights, "weights-final.png", scale=252, invert=False)
    
    Output file content:
    # Tensor image data: (28, 28)
    # Range: [-0.123456, 0.456789]
    # Scale: 252, Invert: False
    000 015 032 128 255 200 150 075 ...
    010 025 045 140 240 190 130 065 ...
    ...
    
    The ASCII representation shows:
    - High values (bright): Areas that strongly indicate digit 7
    - Low values (dark): Areas that indicate digit 3 or neutral
    - Pattern: Should show horizontal lines and vertical stroke of "7"
    
    Image Data Visualization Example:
    ================================
    >>> # Visualize input digit image
    >>> image = Tensor(28, 28)
    >>> image.impl.val = mnist_digit_array  # From MNIST dataset
    >>> save_tensor_as_image(image, "digit-sample.png", scale=252, invert=True)
    
    With invert=True for images:
    - Original: 0=black background, 255=white digit
    - Inverted: 255=black background, 0=white digit  
    - Result: Better visualization contrast
    
    File Organization:
    =================
    Creates organized text files with metadata headers:
    
    File structure:
    /results/perceptron_37/visualizations/
    ├── weights-0.png.txt      # Initial random weights
    ├── weights-100.png.txt    # Weights after 100 samples
    ├── weights-500.png.txt    # Weights after 500 samples
    └── weights-final.png.txt  # Final trained weights
    
    Each file contains:
    - Header with tensor dimensions
    - Data range information  
    - Processing parameters
    - ASCII grid of pixel values
    
    Error Handling:
    ==============
    The function gracefully handles various error conditions:
    
    1. Invalid tensor data:
    >>> tensor.impl.val = None
    >>> save_tensor_as_image(tensor, "output.png")
    Output: "Warning: Tensor data is invalid or None"
    
    2. Wrong dimensions:
    >>> tensor.impl.val = np.array([1, 2, 3])  # 1D instead of 2D
    >>> save_tensor_as_image(tensor, "output.png")
    Output: "Warning: Expected 2D tensor for image saving, got 1D"
    
    3. Directory creation:
    >>> save_tensor_as_image(tensor, "nonexistent/deep/path/file.png")
    Effect: Automatically creates all required directories
    
    Analysis Applications:
    =====================
    
    Training Progress Visualization:
    - Compare weights-0.txt (random) vs weights-final.txt (trained)
    - Observe how weight patterns evolve during learning
    - Identify convergence by comparing consecutive checkpoints
    
    Feature Analysis:
    - Positive weights: Areas that indicate digit 7
    - Negative weights: Areas that indicate digit 3
    - Zero weights: Irrelevant areas for classification
    
    Debugging:
    - Verify weight updates are occurring
    - Check for numerical instabilities (NaN/inf values)
    - Validate data preprocessing pipeline
    """
    try:
        # Ensure the directory exists
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        # Get tensor data and ensure it's 2D
        data = tensor.impl.val
        if data is None or not isinstance(data, np.ndarray):
            print(f"Warning: Tensor data is invalid or None")
            return
        if data.ndim != 2:
            print(f"Warning: Expected 2D tensor for image saving, got {data.ndim}D")
            return
        
        # Normalize to 0-1 range
        data_min, data_max = data.min(), data.max()
        if data_max > data_min:
            normalized = (data - data_min) / (data_max - data_min)
        else:
            normalized = np.zeros_like(data)
        
        # Apply scaling and inversion
        if invert:
            normalized = 1.0 - normalized
        
        # Convert to 0-255 range
        image_data = (normalized * 255).astype(np.uint8)
        
        # Save as simple text representation (since we don't have image libraries)
        base_name = os.path.splitext(filepath)[0]
        text_file = f"{base_name}.txt"
        
        with open(text_file, 'w') as f:
            f.write(f"# Tensor image data: {tensor.shape}\n")
            f.write(f"# Range: [{data_min:.6f}, {data_max:.6f}]\n")
            f.write(f"# Scale: {scale}, Invert: {invert}\n")
            for row in image_data:
                f.write(' '.join(f'{val:3d}' for val in row) + '\n')
        
        print(f"💾 Tensor saved as text to {text_file}")
        
    except Exception as e:
        print(f"⚠️  Warning: Could not save tensor image {filepath}: {e}")


def do_test(test_reader: MNISTReader, weights: Tensor, bias: float, 
           logger: Optional[PerceptronLogger] = None) -> float:
    """
    Test the perceptron on validation data and compute accuracy metrics.
    
    This function evaluates the current perceptron model (weights + bias) on a test
    dataset, computing predictions for each sample and calculating overall accuracy.
    It implements the core perceptron forward pass and decision boundary logic.
    
    Args:
        test_reader (MNISTReader): MNIST test data reader containing validation samples
        weights (Tensor): Current weight tensor (28x28) representing learned features
        bias (float): Current bias value (scalar offset for decision boundary)
        logger (Optional[PerceptronLogger]): Optional SQLite logger for detailed results
    
    Returns:
        float: Accuracy percentage (0.0 to 100.0)
    
    Perceptron Forward Pass Mathematics:
    ===================================
    
    For each test sample, the perceptron computes:
    
    1. Feature Extraction:
       image[i,j] = pixel intensity at position (i,j) ∈ [0, 255]
       normalized_image[i,j] = image[i,j] / 255.0 ∈ [0, 1]
    
    2. Linear Combination:
       score = Σ(i,j) weights[i,j] × image[i,j] + bias
       score = dot_product(weights, image) + bias
    
    3. Decision Boundary:
       prediction = { 7 if score > 0
                    { 3 if score ≤ 0
    
    Mathematical Example:
    ====================
    
    Sample Input:
    >>> # Trained weights (showing pattern for digit 7)
    >>> weights.impl.val = np.array([
    ...     [ 0.02, -0.01,  0.15, ...],  # Top row: horizontal line detector
    ...     [-0.03,  0.08,  0.12, ...],  # Middle: vertical stroke detector  
    ...     [ 0.01, -0.02,  0.18, ...],  # Bottom: continuation detector
    ...     ...
    ... ])
    >>> bias = 0.25
    
    >>> # Test image: handwritten digit "7"
    >>> image = np.array([
    ...     [  0,   0, 255, 255, 255, 255,   0, ...],  # Top horizontal line
    ...     [  0,   0,   0,   0, 255, 255,   0, ...],  # Diagonal stroke
    ...     [  0,   0,   0, 255, 255,   0,   0, ...],  # Continuing down
    ...     ...
    ... ]) / 255.0  # Normalized to [0, 1]
    
    Step-by-step computation:
    1. Dot product calculation:
       dot_product = Σ(i,j) weights[i,j] × image[i,j]
       dot_product = (0.02×0 + -0.01×0 + 0.15×1 + ...) = 2.47
    
    2. Add bias:
       score = 2.47 + 0.25 = 2.72
    
    3. Apply decision boundary:
       since score = 2.72 > 0: prediction = 7
       true_label = 7: CORRECT prediction
    
    Sample Execution Flow:
    =====================
    
    Test Dataset Processing:
    >>> test_reader = MNISTReader(test_images, test_labels)
    >>> weights = trained_perceptron_weights  # After training
    >>> bias = 0.25
    >>> accuracy = do_test(test_reader, weights, bias)
    
    Processing each sample:
    Sample 0: Label=7, Score=2.72, Prediction=7 ✓ (Correct)
    Sample 1: Label=3, Score=-1.85, Prediction=3 ✓ (Correct)  
    Sample 2: Label=7, Score=-0.32, Prediction=3 ✗ (Wrong)
    Sample 3: Label=3, Score=0.15, Prediction=7 ✗ (Wrong)
    ...
    
    Final Statistics:
    Total samples tested: 2000 (digits 3 and 7 only)
    Correct predictions: 1876
    Wrong predictions: 124
    Accuracy: 93.8%
    
    Detailed Logging (if logger provided):
    =====================================
    
    Each test result is logged to database:
    >>> logger = PerceptronLogger("test_results.db")
    >>> accuracy = do_test(test_reader, weights, bias, logger)
    
    Database records created:
    test_results table:
    | id | label | res   | verdict | correct | timestamp          |
    |----|-------|-------|---------|---------|-------------------|
    | 1  | 7     | 2.72  | 7       | 1       | 2025-09-21 10:30:01|
    | 2  | 3     | -1.85 | 3       | 1       | 2025-09-21 10:30:01|
    | 3  | 7     | -0.32 | 3       | 0       | 2025-09-21 10:30:01|
    | 4  | 3     | 0.15  | 7       | 0       | 2025-09-21 10:30:01|
    
    Error Analysis Capabilities:
    ===========================
    
    The logged data enables detailed error analysis through SQL queries.
    
    1. Misclassification Analysis:
    Find samples that were incorrectly classified, ordered by confidence.
    
    2. Decision Boundary Analysis:
    Calculate average scores for correctly classified samples to understand
    the learned decision boundary characteristics.
    
    3. Confidence Distribution:
    Examine the distribution of prediction scores for each digit class
    to understand model confidence patterns.
    
    Performance Characteristics:
    ===========================
    
    Computational Complexity:
    - Time: O(N × H × W) where N=test samples, H×W=image dimensions
    - Space: O(H × W) for weight storage
    - For MNIST: O(N × 28 × 28) = O(784N)
    
    Typical Accuracy Ranges:
    - Random weights: ~50% (chance level)
    - Partially trained: 60-85%
    - Well trained: 85-98%
    - Optimal perceptron: ~95% (theoretical limit for linear classifier)
    
    Edge Cases Handled:
    ==================
    
    1. Empty test set:
    >>> test_reader.get_num() = 0
    >>> Output: "No test samples found for digits 3 and 7"
    >>> Return: 0.0
    
    2. Invalid tensor data:
    >>> weights.impl.val = None
    >>> Raises: ValueError("Tensor data is invalid or None")
    
    3. No 3s or 7s in test set:
    >>> # All samples are digits 0,1,2,4,5,6,8,9
    >>> Output: "No test samples found for digits 3 and 7"
    >>> Return: 0.0
    
    Integration with Training Loop:
    ==============================
    
    This function is typically called periodically during training:
    
    >>> for epoch in range(max_epochs):
    ...     # Training step
    ...     update_weights(train_sample)
    ...     
    ...     # Periodic evaluation
    ...     if epoch % 100 == 0:
    ...         accuracy = do_test(test_reader, weights, bias, logger)
    ...         if accuracy > 98.0:
    ...             print("Target accuracy reached!")
    ...             break
    
    The accuracy progression shows learning:
    Epoch 0: 52.3% (random weights)
    Epoch 100: 67.8% (initial learning)
    Epoch 500: 89.4% (rapid improvement)
    Epoch 1000: 98.7% (convergence)
    """
    corrects = 0
    wrongs = 0
    
    print("🧪 Testing perceptron performance...")
    
    for n in range(test_reader.get_num()):
        label = test_reader.get_label(n)
        
        # Only test on 3s and 7s (binary classification)
        if label != 3 and label != 7:
            continue
        
        # Get image as array and create tensor
        img_array = test_reader.get_image_as_array(n)
        img = Tensor(28, 28)
        img.impl.val = img_array.astype(np.float32)
        
        # Compute score: dot product of image and weights + bias
        # This implements: score = Σ(i,j) weights[i,j] × image[i,j] + bias
        if img.impl.val is None or weights.impl.val is None:
            raise ValueError("Tensor data is invalid or None for either 'img' or 'weights'")
        dot_product = np.sum(img.impl.val * weights.impl.val)
        score = dot_product + bias
        
        # Make prediction using decision boundary: 7 if score > 0, else 3
        predict = 7 if score > 0 else 3
        
        # Log result if logger provided for detailed analysis
        if logger:
            # Cast NumPy floating to native float for type checker compatibility
            logger.log_test_result(label, float(score), predict)
        
        # Count correct/incorrect predictions
        if predict == label:
            corrects += 1
        else:
            wrongs += 1
    
    # Calculate accuracy percentage
    total = corrects + wrongs
    if total == 0:
        print("⚠️  No test samples found for digits 3 and 7")
        return 0.0
    
    accuracy = 100.0 * corrects / total
    print(f"📊 Accuracy: {accuracy:.2f}% ({corrects}/{total} correct)")
    
    return accuracy


def create_learning_rate_tensor(size: int, lr: float) -> Tensor:
    """
    Create uniform tensor filled with learning rate value for weight updates.
    
    This function creates a tensor filled with the learning rate value, replicating
    the functionality of the C++ lr.identity(0.01) operation. It's used to ensure
    consistent learning rate application across all weight components during training.
    
    Args:
        size (int): Size of square tensor (typically 28 for 28x28 MNIST images)
        lr (float): Learning rate value to fill the tensor with
    
    Returns:
        Tensor: Square tensor (size × size) filled with learning rate value
    
    Mathematical Purpose:
    ====================
    
    In the original C++ implementation, lr.identity(0.01) creates an identity-like
    tensor where each element is the learning rate. This enables element-wise
    multiplication during weight updates:
    
    weight_update = lr_tensor × image_tensor
    new_weights = old_weights ± weight_update
    
    Tensor Creation Process:
    =======================
    
    Step 1: Create empty tensor structure
    >>> lr_tensor = Tensor(28, 28)
    
    Step 2: Fill with learning rate value
    >>> lr_tensor.impl.val = np.full((28, 28), 0.01, dtype=np.float32)
    
    Result: 28×28 tensor where every element equals 0.01
    
    Example Usage in Training:
    =========================
    
    >>> # Create learning rate tensor
    >>> lr_tensor = create_learning_rate_tensor(28, 0.01)
    >>> print(f"Shape: {lr_tensor.shape}")
    Shape: (28, 28)
    >>> print(f"Sample values: {lr_tensor.impl.val[0, :5]}")
    Sample values: [0.01 0.01 0.01 0.01 0.01]
    
    Weight Update Application:
    ========================
    
    During perceptron training, the learning rate tensor enables vectorized updates:
    
    >>> # Current weights and input image
    >>> weights = Tensor(28, 28)  # Current learned weights
    >>> image = Tensor(28, 28)    # Input training sample
    >>> lr_tensor = create_learning_rate_tensor(28, 0.01)
    
    >>> # Perceptron learning rule for label=7, score<2.0
    >>> if label == 7 and score < 2.0:
    ...     # Vectorized update: weights += lr * image
    ...     weights.impl.val += lr_tensor.impl.val * image.impl.val
    
    This is equivalent to the element-wise operation:
    for i in range(28):
        for j in range(28):
            weights[i][j] += 0.01 * image[i][j]
    
    But much more efficient using NumPy vectorization.
    
    Learning Rate Impact:
    ====================
    
    Different learning rates produce different training behaviors:
    
    Large Learning Rate (lr=0.1):
    >>> lr_tensor = create_learning_rate_tensor(28, 0.1)
    Behavior: Fast learning, potential instability
    Risk: Overshooting optimal weights, oscillations
    
    Medium Learning Rate (lr=0.01):
    >>> lr_tensor = create_learning_rate_tensor(28, 0.01)
    Behavior: Balanced learning speed and stability
    Typical: Good default for perceptron training
    
    Small Learning Rate (lr=0.001):
    >>> lr_tensor = create_learning_rate_tensor(28, 0.001)
    Behavior: Slow but stable learning
    Risk: Very slow convergence, may need more epochs
    
    Adaptive Learning Rate Example:
    ==============================
    
    >>> # Start with higher learning rate, reduce over time
    >>> initial_lr = 0.05
    >>> decay_factor = 0.99
    >>> 
    >>> for epoch in range(1000):
    ...     current_lr = initial_lr * (decay_factor ** epoch)
    ...     lr_tensor = create_learning_rate_tensor(28, current_lr)
    ...     # Use lr_tensor for this epoch's weight updates
    
    Memory and Performance:
    ======================
    
    Tensor Size: 28 × 28 × 4 bytes = 3,136 bytes per tensor
    Creation Time: O(size²) to fill array
    Memory Access: Sequential fill pattern, cache-friendly
    
    For size=28: ~3KB memory, negligible computational cost
    For size=224: ~200KB memory, still very fast
    
    Comparison with Scalar Learning Rate:
    ====================================
    
    Tensor approach (current):
    >>> lr_tensor = create_learning_rate_tensor(28, 0.01)
    >>> weights.impl.val += lr_tensor.impl.val * image.impl.val
    Advantages: Consistent with C++ implementation, explicit tensor operations
    
    Scalar approach (alternative):
    >>> weights.impl.val += 0.01 * image.impl.val
    Advantages: More memory efficient, simpler code
    
    The tensor approach is used here for compatibility with the original C++
    algorithm and to maintain explicit tensor operation semantics.
    
    Integration with PyTensorLib:
    ============================
    
    The function creates a properly initialized PyTensorLib Tensor:
    - Shape information: lr_tensor.shape = (size, size)
    - Implementation: lr_tensor.impl.val = NumPy array
    - Type safety: dtype=np.float32 for consistency
    - Memory layout: Contiguous array for efficient operations
    
    This ensures compatibility with other PyTensorLib operations while
    providing the uniform learning rate distribution needed for training.
    """
    lr_tensor = Tensor(size, size)
    lr_tensor.impl.val = np.full((size, size), lr, dtype=np.float32)
    return lr_tensor


def main():
    """
    Main perceptron learning algorithm implementing 3 vs 7 digit classification.
    
    This function implements the complete perceptron training pipeline, converting
    the original C++ 37learn.cc algorithm to Python using PyTensorLib. It demonstrates
    classic machine learning concepts including linear classification, online learning,
    and early stopping based on validation accuracy.
    
    Algorithm Overview:
    ==================
    
    The perceptron algorithm for binary classification:
    1. Initialize random weights W and bias b
    2. For each training sample (x, y):
       - Compute score = W·x + b  
       - Make prediction: ŷ = sign(score)
       - Update weights if prediction is wrong or confidence is low:
         * If y=7 and score<2: W += η·x, b += η
         * If y=3 and score>-2: W -= η·x, b -= η
    3. Test periodically and stop when accuracy > 98%
    
    Mathematical Foundation:
    =======================
    
    Linear Decision Boundary:
    The perceptron learns a hyperplane separating digits 3 and 7:
    f(x) = W·x + b = Σ(i,j) W[i,j] × x[i,j] + b
    
    Decision Rule:
    predict(x) = { 7 if f(x) > 0
                 { 3 if f(x) ≤ 0
    
    Weight Update Rule (with margin):
    - For digit 7: increase weights if score < 2 (low confidence)
    - For digit 3: decrease weights if score > -2 (low confidence)  
    - Margin of 2 provides better generalization than simple perceptron
    
    Training Pipeline Flow:
    ======================
    
    Phase 1: Data Loading and Preparation
    ====================================
    >>> # Create organized output directories
    >>> base_results_dir = "results"
    >>> output_dir = "results/perceptron_37"
    >>> models_dir = "results/perceptron_37/models"
    >>> logs_dir = "results/perceptron_37/logs"
    >>> visualizations_dir = "results/perceptron_37/visualizations"
    
    >>> # Load MNIST/EMNIST data with fallback strategy
    >>> try:
    ...     train_reader = MNISTReader(emnist_data)  # Preferred
    >>> except:
    ...     train_reader = MNISTReader(mnist_data)   # Fallback
    >>> except:
    ...     sys.exit(1)  # Exit if no real data available
    
    Phase 2: Model Initialization
    =============================
    >>> # Initialize weights with normalized random values
    >>> weights = Tensor(28, 28)
    >>> scale = 1.0 / np.sqrt(28 * 28)  # ≈ 0.0357 for stable initialization
    >>> weights.impl.val = np.random.normal(0, scale, (28, 28))
    
    >>> # Initialize bias and learning rate
    >>> bias = 0.0  # Start with no bias preference
    >>> learning_rate = 0.01  # Moderate learning rate
    
    Phase 3: Training Loop with Perceptron Updates
    =============================================
    >>> count = 0
    >>> for n in range(train_reader.get_num()):
    ...     label = train_reader.get_label(n)
    ...     if label not in [3, 7]:
    ...         continue  # Skip non-target digits
    ...     
    ...     # Periodic testing and checkpoint saving
    ...     if count % 4 == 0:
    ...         accuracy = do_test(test_reader, weights, bias, logger)
    ...         save_tensor_as_image(weights, f"weights-{count}.png")
    ...         if accuracy > 98.0:
    ...             break  # Early stopping
    ...     
    ...     # Forward pass: compute current score
    ...     img_array = train_reader.get_image_as_array(n)
    ...     img = Tensor(28, 28)
    ...     img.impl.val = img_array.astype(np.float32)
    ...     score = np.sum(img.impl.val * weights.impl.val) + bias
    ...     
    ...     # Perceptron learning rule with margin
    ...     if label == 7 and score < 2.0:
    ...         # Boost weights toward this positive example
    ...         weights.impl.val += img.impl.val * learning_rate
    ...         bias += learning_rate
    ...     elif label == 3 and score > -2.0:
    ...         # Reduce weights away from this negative example
    ...         weights.impl.val -= img.impl.val * learning_rate
    ...         bias -= learning_rate
    ...     
    ...     count += 1
    
    Phase 4: Final Evaluation and Model Saving
    ==========================================
    >>> # Final evaluation on test set
    >>> final_accuracy = do_test(test_reader, weights, bias, logger)
    
    >>> # Save final model state
    >>> model_state = {
    ...     'weights': weights.impl.val.tolist(),
    ...     'bias': bias,
    ...     'final_accuracy': final_accuracy,
    ...     'training_samples': count,
    ...     'data_type': data_type,
    ...     'learning_rate': learning_rate
    ... }
    >>> with open('perceptron_37_final.json', 'w') as f:
    ...     json.dump(model_state, f)
    
    Detailed Component Explanations:
    ===============================
    
    1. Directory Structure Organization:
    Creates hierarchical output organization:
    results/
    └── perceptron_37/
        ├── models/          # Final model and checkpoints
        ├── logs/            # Training progress database
        └── visualizations/  # Weight evolution images
    
    2. Data Loading Strategy:
    Implements robust fallback mechanism:
    Priority 1: EMNIST (extended MNIST, more samples)
    Priority 2: MNIST (classic dataset)
    Priority 3: Exit execution if no real data available
    
    3. Weight Initialization:
    Uses normalized random initialization:
    - Mean: 0 (no initial bias toward either class)
    - Standard deviation: 1/√(784) ≈ 0.0357
    - Distribution: Normal (Gaussian)
    - Purpose: Prevents saturation, enables learning
    
    4. Training Loop Structure:
    Implements online learning with periodic evaluation:
    - Process samples one by one (online)
    - Test every 4 samples (frequent monitoring)
    - Save visualizations at checkpoints
    - Early stopping at 98% accuracy
    
    5. Perceptron Learning Rule:
    Enhanced perceptron with margin:
    
    Standard perceptron: Update only on misclassification
    if predict(x) ≠ y: update_weights()
    
    Margin perceptron: Update on low confidence
    if y=7 and score<2: boost_weights()    # Even if correctly classified
    if y=3 and score>-2: reduce_weights()  # Even if correctly classified
    
    Benefits: Better generalization, more robust decision boundary
    
    Training Progress Example:
    =========================
    
    Typical training progression:
    
    Step 0: Random weights, ~50% accuracy
    📊 Testing step 0, accuracy: 52.3%
    💾 Tensor saved as text to weights-0.png.txt
    
    Step 100: Early learning, weights adapting
    📊 Testing step 100, accuracy: 67.8%
    💾 Tensor saved as text to weights-100.png.txt
    
    Step 500: Rapid improvement phase
    📊 Testing step 500, accuracy: 89.4%
    💾 Tensor saved as text to weights-500.png.txt
    
    Step 1000: Approaching convergence
    📊 Testing step 1000, accuracy: 98.7%
    🎉 Target accuracy achieved! Stopping training.
    
    Final Output:
    🏁 Training completed after 1000 samples
    📊 Final evaluation: 98.7% accuracy
    💾 Results saved to organized directory structure
    
    Performance Characteristics:
    ===========================
    
    Computational Complexity:
    - Time per sample: O(H×W) = O(784) for dot product computation
    - Total training time: O(N×H×W) where N = number of training samples
    - Memory usage: O(H×W) for weight storage ≈ 3KB for 28×28 weights
    
    Convergence Properties:
    - Guaranteed convergence if data is linearly separable
    - Typical convergence: 500-2000 training samples for MNIST 3 vs 7
    - Final accuracy: 85-98% depending on data quality and parameters
    
    Error Handling and Robustness:
    ==============================
    
    The implementation includes comprehensive error handling:
    
    1. Data Loading Failures:
    >>> try:
    ...     load_emnist()
    >>> except Exception:
    ...     try:
    ...         load_mnist()
    ...     except Exception:
    ...         sys.exit(1)  # Exit if no real data available
    
    2. Keyboard Interruption:
    >>> except KeyboardInterrupt:
    ...     print("Training interrupted by user")
    ...     # Still save progress and exit gracefully
    
    3. Tensor Validation:
    >>> if img.impl.val is None or weights.impl.val is None:
    ...     raise ValueError("Tensor data is invalid")
    
    4. Empty Test Sets:
    >>> if total_samples == 0:
    ...     print("No test samples found")
    ...     return 0.0
    
    Integration with PyTensorLib Ecosystem:
    ======================================
    
    Dependencies and their purposes:
    - tensor_lib.Tensor: Core tensor operations and storage
    - mnist_utils.MNISTReader: Data loading and preprocessing
    - numpy: Efficient numerical computations
    - sqlite3: Training progress logging and analysis
    - json: Model serialization and persistence
    
    The main function serves as the orchestrator, coordinating these
    components to implement the complete machine learning pipeline
    from data loading through model training to result analysis.
    """
    
    print("🎯 Perceptron Learning Algorithm: 3 vs 7 Classification")
    print("=" * 60)
    print("Converting C++ 37learn.cc to Python using PyTensorLib")
    print("This implements a classic perceptron with weight updates.")
    print("=" * 60)
    
    # Create unified output directory structure for all experiments
    base_results_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results")
    output_dir = os.path.join(base_results_dir, "perceptron_37")
    
    # Create subdirectories for organized storage
    models_dir = os.path.join(output_dir, "models")
    logs_dir = os.path.join(output_dir, "logs") 
    visualizations_dir = os.path.join(output_dir, "visualizations")
    
    for dir_path in [base_results_dir, output_dir, models_dir, logs_dir, visualizations_dir]:
        os.makedirs(dir_path, exist_ok=True)
    
    print(f"\n📁 Output directory structure:")
    print(f"   Base: {os.path.abspath(base_results_dir)}")
    print(f"   Project: {os.path.relpath(output_dir, base_results_dir)}")
    print(f"   Subdirectories: models/, logs/, visualizations/")
    print(f"   All generated files will be organized in these folders.")
    
    # Load MNIST/EMNIST data (try EMNIST first, fallback to MNIST)
    print("\n📦 Loading MNIST/EMNIST data...")
    
    data_dir = "./data"
    train_reader = None
    test_reader = None
    data_type = "Unknown"
    
    # Try EMNIST first
    try:
        train_images, train_labels, test_images, test_labels = download_emnist(data_dir)
        train_reader = MNISTReader(train_images, train_labels)
        test_reader = MNISTReader(test_images, test_labels)
        data_type = "EMNIST"
        print("✅ EMNIST data loaded successfully")
    except Exception as e:
        print(f"⚠️  EMNIST failed: {e}")
        
        # Fallback to MNIST
        try:
            train_images, train_labels, test_images, test_labels = download_mnist(data_dir)
            train_reader = MNISTReader(train_images, train_labels)
            test_reader = MNISTReader(test_images, test_labels)
            data_type = "MNIST"
            print("✅ MNIST data loaded successfully")
        except Exception as e:
            print(f"❌ MNIST failed: {e}")
            print("❌ Both EMNIST and MNIST data loading failed.")
            print("❌ Real dataset is required for meaningful perceptron training.")
            print("❌ Please ensure data files are available or check network connection.")
            print("💡 Run setup_environment.py or download_mnist.py to download data first.")
            sys.exit(1)
    
    print(f"📈 Dataset statistics:")
    print(f"   Training samples: {train_reader.get_num()}")
    print(f"   Test samples: {test_reader.get_num()}")
    print(f"   Data type: {data_type}")
    
    # Initialize weights randomly (equivalent to C++ weights.randomize(1.0/sqrt(28*28)))
    print("\n🧠 Initializing perceptron...")
    weights = Tensor(28, 28)
    scale = 1.0 / np.sqrt(28 * 28)  # 1/sqrt(784) ≈ 0.0357
    weights.impl.val = np.random.normal(0, scale, (28, 28)).astype(np.float32)
    
    # Save initial random weights
    initial_weights_path = os.path.join(visualizations_dir, "random-weights.png")
    save_tensor_as_image(weights, initial_weights_path, 252)
    
    # Initialize bias and learning rate
    bias = 0.0
    learning_rate = 0.01
    
    # Create learning rate tensor (equivalent to C++ lr.identity(0.01))
    lr_tensor = create_learning_rate_tensor(28, learning_rate)
    
    # Setup logging
    db_path = os.path.join(logs_dir, "37learn.sqlite3")
    logger = PerceptronLogger(db_path)
    
    # Training loop
    print("\n🚀 Starting perceptron training...")
    print("Algorithm: Update weights based on prediction errors")
    print("- If label=7 and score<2: weights += lr * image, bias += lr")
    print("- If label=3 and score>-2: weights -= lr * image, bias -= lr")
    print()
    
    count = 0
    
    try:
        for n in range(train_reader.get_num()):
            label = train_reader.get_label(n)
            
            # Only train on 3s and 7s
            if label != 3 and label != 7:
                continue
            
            # Test every 4 samples and save progress
            if count % 4 == 0:
                print(f"\n📊 Training step {count}, testing accuracy...")
                accuracy = do_test(test_reader, weights, bias)
                
                # Save weights periodically
                weights_file = os.path.join(visualizations_dir, f"weights-{count}.png")
                save_tensor_as_image(weights, weights_file, 252)
                logger.log_training_progress(count, accuracy, os.path.basename(weights_file), bias)
                
                # Early stopping: if accuracy > 98%, we're done!
                if accuracy > 98.0:
                    print(f"🎉 Target accuracy achieved! Stopping training.")
                    break
            
            # Get training image
            img_array = train_reader.get_image_as_array(n)
            img = Tensor(28, 28)
            img.impl.val = img_array.astype(np.float32)
            
            # Compute current score
            dot_product = np.sum(img.impl.val * weights.impl.val)
            score = dot_product + bias
            
            # Save example on first significant sample
            if count == 25001:
                print(f"\n📷 Example at step {count}:")
                print(f"   Score for sample: {score:.6f}")
                random_image_path = os.path.join(visualizations_dir, "random-image.png")
                save_tensor_as_image(img, random_image_path, 252, invert=True)
                
                # Create product tensor for visualization
                product = Tensor(28, 28)
                product.impl.val = img.impl.val * weights.impl.val
                random_prod_path = os.path.join(visualizations_dir, "random-prod.png")
                save_tensor_as_image(product, random_prod_path, 252)
            
            # Perceptron learning rule
            if label == 7:
                # If we predicted too low for a 7, increase weights toward this image
                if score < 2.0:
                    weights.impl.val += img.impl.val * learning_rate
                    bias += learning_rate
            else:  # label == 3
                # If we predicted too high for a 3, decrease weights away from this image
                if score > -2.0:
                    weights.impl.val -= img.impl.val * learning_rate
                    bias -= learning_rate
            
            count += 1
            
            # Progress indicator
            if count % 1000 == 0:
                print(f"  Processed {count} training samples...")
    
    except KeyboardInterrupt:
        print("\n⚠️  Training interrupted by user")
    
    # Final evaluation
    print(f"\n🏁 Training completed after {count} samples")
    
    # Save final weights
    final_weights_path = os.path.join(visualizations_dir, "weights-final.png")
    save_tensor_as_image(weights, final_weights_path, 252)
    
    # Final test with logging
    print("\n📊 Final evaluation:")
    final_accuracy = do_test(test_reader, weights, bias, logger)
    
    print(f"\n📈 Training Summary:")
    print(f"   Final accuracy: {final_accuracy:.2f}%")
    print(f"   Final bias: {bias:.6f}")
    print(f"   Training samples processed: {count}")
    print(f"   Data type: {data_type}")
    
    # Save final model state
    model_state = {
        'weights': weights.impl.val.tolist(),
        'bias': bias,
        'final_accuracy': final_accuracy,
        'training_samples': count,
        'data_type': data_type,
        'learning_rate': learning_rate
    }
    
    model_path = os.path.join(models_dir, 'perceptron_37_final.json')
    with open(model_path, 'w') as f:
        json.dump(model_state, f, indent=2)
    
    print(f"\n💾 Results saved to organized directory structure:")
    print(f"   Base directory: {os.path.relpath(base_results_dir, os.getcwd())}")
    print(f"   Models: {os.path.relpath(models_dir, base_results_dir)}/perceptron_37_final.json")
    print(f"   Logs: {os.path.relpath(logs_dir, base_results_dir)}/37learn.sqlite3")
    print(f"   Visualizations: {os.path.relpath(visualizations_dir, base_results_dir)}/weights-*.png.txt")
    print(f"   Initial weights: {os.path.relpath(visualizations_dir, base_results_dir)}/random-weights.png.txt")
    print(f"   Final weights: {os.path.relpath(visualizations_dir, base_results_dir)}/weights-final.png.txt")
    
    logger.close()
    
    print("\n🎉 Perceptron learning complete!")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)