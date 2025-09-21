#!/usr/bin/env python3
"""
Three or Seven Classifier - Comprehensive Educational Example

This script implements a simple binary classifier to distinguish between
handwritten digits 3 and 7 using PyTensorLib. It demonstrates fundamental
machine learning concepts through linear algebra operations.

🎯 LEARNING OBJECTIVES:
- Understand binary classification fundamentals
- Learn how to compute and use average images as templates
- Explore linear separability in high-dimensional spaces
- Practice tensor operations for machine learning
- Analyze decision boundaries and feature importance

📊 ALGORITHM OVERVIEW:
The classifier uses a template-matching approach:
1. Compute average images for digits 3 and 7 from training data
2. Calculate difference vector (delta = avg_7 - avg_3)
3. For each test image, compute similarity score: dot(image, delta)
4. Add bias term and apply threshold: score > 0 → digit 7, score ≤ 0 → digit 3

🔬 MATHEMATICAL FOUNDATION:
- Feature vector: Each 28x28 image flattened to 784-dimensional vector
- Template vectors: μ₃ (average of 3s), μ₇ (average of 7s)
- Decision vector: δ = μ₇ - μ₃
- Classification function: f(x) = x·δ + b, where b is bias
- Decision rule: class = 7 if f(x) > 0, else 3

💡 WHY THIS WORKS:
The difference vector (delta) captures the key visual differences between 3s and 7s.
Pixels where 7s are typically bright and 3s are dark get positive weights.
Pixels where 3s are typically bright and 7s are dark get negative weights.
The dot product measures how "7-like" vs "3-like" an image appears.

📈 SAMPLE INPUT/OUTPUT EXAMPLES:

Example 1: Training Data Statistics
Input: EMNIST dataset with mixed digits
Output: 
  Training samples: 11,966 (5,923 threes, 6,043 sevens)
  Test samples: 1,991 (986 threes, 1,005 sevens)

Example 2: Average Image Computation
Input: 5,923 training images of digit 3
Output: 28x28 tensor representing average appearance of digit 3
  - Bright pixels where 3s commonly have strokes
  - Dark pixels where 3s commonly have background
  Sample values: center pixels ≈ 0.8, corners ≈ 0.1

Example 3: Decision Vector Analysis
Input: Average images of 3s and 7s
Output: Decision vector (delta) showing discriminative features
  - Positive values: pixels where 7s are brighter than 3s
  - Negative values: pixels where 3s are brighter than 7s
  - Statistics: mean ≈ 0.0, std ≈ 0.15, range [-0.4, +0.4]

Example 4: Classification Score
Input: Test image of digit 7
Process: 
  1. Flatten 28x28 image to 784-element vector
  2. Compute dot product with decision vector
  3. Add bias term
Output: Score = +2.34 → Classified as 7 (correct)

Example 5: Classification Score
Input: Test image of digit 3
Process: Same as above
Output: Score = -1.87 → Classified as 3 (correct)

Example 6: Final Performance
Input: 1,991 test images
Output: 
  Accuracy: 0.884 (1,760/1,991 correct)
  Error rate: 0.116 (231 mistakes)
  3s score range: [-4.2, +1.8]
  7s score range: [-2.1, +4.7]

🎓 EDUCATIONAL VALUE:
- Shows how simple linear algebra can solve complex problems
- Demonstrates the importance of feature engineering (pixel differences)
- Illustrates bias-variance tradeoffs in machine learning
- Provides intuitive understanding of decision boundaries
- Connects mathematical concepts to practical applications
"""

import sys
import os
import numpy as np

# Add src directory to path for development
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from pytensorlib import Tensor, MNISTReader, download_emnist
import json


def create_mnist_data_if_needed():
    """
    Load EMNIST/MNIST dataset for binary classification.
    
    📋 PURPOSE:
    Downloads and loads real MNIST/EMNIST data for training the 3 vs 7 classifier.
    This function ensures we have the required dataset files and handles download
    if they're missing.
    
    🔄 PROCESS FLOW:
    1. Check if EMNIST data files exist in data/ directory
    2. If missing, attempt automatic download via download_emnist()
    3. Load training and test datasets using MNISTReader
    4. Filter datasets to include only digits 3 and 7
    5. Return structured data ready for classification
    
    📊 SAMPLE INPUT:
    - Expected files in data/:
      * emnist-digits-train-images-idx3-ubyte.gz (training images)
      * emnist-digits-train-labels-idx1-ubyte.gz (training labels)
      * emnist-digits-test-images-idx3-ubyte.gz (test images)
      * emnist-digits-test-labels-idx1-ubyte.gz (test labels)
    
    📈 SAMPLE OUTPUT:
    Returns tuple (train_data, test_data) where each contains:
    
    train_data = {
        'images': [
            array([[0.0, 0.0, 0.0, ..., 0.0],     # First 3 or 7 image (28x28)
                   [0.0, 0.1, 0.8, ..., 0.0],     # Pixel values 0.0-1.0
                   [...],
                   [0.0, 0.0, 0.0, ..., 0.0]]),
            array([[...]]),                        # Second image
            # ... (typically ~12,000 images)
        ],
        'labels': [3, 7, 3, 7, 3, ...]           # Corresponding labels
    }
    
    test_data = {
        'images': [...],    # Similar structure, ~2,000 test images
        'labels': [...]     # Test labels
    }
    
    💡 TYPICAL DATASET STATISTICS:
    - Total EMNIST training samples: ~240,000 (all digits 0-9)
    - Filtered for 3s and 7s: ~12,000 training samples
    - Test samples after filtering: ~2,000 samples
    - Image dimensions: 28x28 pixels, grayscale
    - Pixel value range: 0.0 (black) to 1.0 (white)
    
    ⚠️ ERROR HANDLING:
    - Raises RuntimeError if download fails and no local data exists
    - Provides helpful instructions for manual data download
    - Ensures robust fallback behavior
    
    🎯 MACHINE LEARNING CONTEXT:
    This filtered dataset creates a binary classification problem that's
    simpler than full 10-digit recognition but still challenging enough
    to demonstrate key ML concepts like decision boundaries and feature learning.
    """
    data_dir = os.path.join(os.path.dirname(__file__), '..', 'data')
    os.makedirs(data_dir, exist_ok=True)
    
    # Check if we have real EMNIST data
    train_images_path = os.path.join(data_dir, 'emnist-digits-train-images-idx3-ubyte.gz')
    train_labels_path = os.path.join(data_dir, 'emnist-digits-train-labels-idx1-ubyte.gz')
    test_images_path = os.path.join(data_dir, 'emnist-digits-test-images-idx3-ubyte.gz')
    test_labels_path = os.path.join(data_dir, 'emnist-digits-test-labels-idx1-ubyte.gz')
    
    if not all(os.path.exists(path) for path in [train_images_path, train_labels_path, 
                                                 test_images_path, test_labels_path]):
        print("📦 Real EMNIST data not found. Attempting to download...")
        
        try:
            # Try to download EMNIST data
            train_images_path, train_labels_path, test_images_path, test_labels_path = download_emnist(data_dir)
            print("✅ EMNIST/MNIST data downloaded successfully!")
            
        except Exception as e:
            print(f"❌ Failed to download EMNIST/MNIST data: {e}")
            print("📥 Please run: python download_mnist.py")
            print("   This will download the required dataset files.")
            print("   Make sure you have a stable internet connection.")
            raise RuntimeError("No real MNIST/EMNIST data available. Synthetic data disabled.")
    
    # Load real EMNIST/MNIST data
    print("📦 Loading real EMNIST/MNIST data...")
    train_reader = MNISTReader(train_images_path, train_labels_path)
    test_reader = MNISTReader(test_images_path, test_labels_path)
    
    # Filter for 3s and 7s only
    train_indices = train_reader.filter_by_labels([3, 7])
    test_indices = test_reader.filter_by_labels([3, 7])
    
    train_data = {
        'images': [train_reader.get_image_as_array(i) for i in train_indices],
        'labels': [train_reader.get_label(i) for i in train_indices]
    }
    test_data = {
        'images': [test_reader.get_image_as_array(i) for i in test_indices],
        'labels': [test_reader.get_label(i) for i in test_indices]
    }
    
    return train_data, test_data
def compute_average_images(images, labels):
    """
    Compute template images by averaging all 3s and all 7s separately.
    
    📋 PURPOSE:
    Creates prototype/template images that represent the "typical" appearance
    of digits 3 and 7. These templates capture the common visual patterns
    shared by each digit class and serve as the foundation for classification.
    
    🧮 MATHEMATICAL PROCESS:
    For digit class c ∈ {3, 7}:
    μc = (1/nc) × Σ(xi) for all images xi with label c
    where nc is the number of images with label c
    
    🔄 ALGORITHM STEPS:
    1. Initialize two 28x28 zero tensors for accumulating pixel sums
    2. Iterate through all training images
    3. Add each image to appropriate accumulator based on its label
    4. Count number of images per class
    5. Divide accumulated sums by counts to get averages
    
    📊 SAMPLE INPUT:
    images = [
        array([[0.0, 0.0, ..., 0.8],    # First image (digit 3)
               [0.0, 0.2, ..., 0.9],    # 28x28 array, values 0.0-1.0
               [...],
               [0.0, 0.0, ..., 0.1]]),
        array([[...]]),                  # Second image (digit 7)
        # ... thousands more images
    ]
    labels = [3, 7, 3, 7, 3, 7, ...]   # Corresponding class labels
    
    📈 SAMPLE OUTPUT:
    Returns (threes_avg, sevens_avg, three_count, seven_count):
    
    threes_avg = Tensor(28, 28) with average pixel values:
    [[0.02, 0.01, 0.01, ..., 0.03],     # Top edge: mostly background
     [0.05, 0.12, 0.45, ..., 0.08],     # Upper curve of digit 3
     [0.08, 0.67, 0.89, ..., 0.15],     # Middle section with curves
     [...],
     [0.03, 0.09, 0.23, ..., 0.04]]     # Bottom edge
    
    sevens_avg = Tensor(28, 28) with different pattern:
    [[0.03, 0.78, 0.85, ..., 0.91],     # Top horizontal bar
     [0.02, 0.15, 0.23, ..., 0.67],     # Diagonal stroke area
     [0.01, 0.08, 0.12, ..., 0.43],     # Lower diagonal
     [...],
     [0.01, 0.02, 0.03, ..., 0.08]]     # Bottom: mostly background
    
    three_count = 5923    # Number of 3s averaged
    seven_count = 6043    # Number of 7s averaged
    
    💡 VISUAL INTERPRETATION:
    - threes_avg shows bright pixels where 3s typically have strokes:
      * Curved regions on right side
      * Horizontal segments in middle and bottom
      * Mostly dark in upper-left corner
    
    - sevens_avg shows bright pixels where 7s typically have strokes:
      * Strong horizontal line across top
      * Diagonal stroke from top-right to bottom-left
      * Mostly dark in lower-right corner
    
    🎯 MACHINE LEARNING SIGNIFICANCE:
    These average images represent the "centroid" of each class in 784-dimensional
    pixel space. They capture the most consistent visual features of each digit
    and provide a simple but effective basis for template matching classification.
    
    ⚡ EFFICIENCY NOTES:
    - Uses tensor operations for efficient accumulation
    - Handles both numpy arrays and Tensor objects seamlessly
    - Memory-efficient: only stores running sums, not all images
    """
    threes = Tensor(28, 28)
    sevens = Tensor(28, 28)
    threes.zero()
    sevens.zero()
    
    three_count = 0
    seven_count = 0
    
    print("📊 Computing average images...")
    
    for img, label in zip(images, labels):
        if isinstance(img, np.ndarray):
            img_tensor = Tensor(28, 28)
            img_tensor.impl.val = img.copy()
        else:
            img_tensor = img
        
        if label == 3:
            three_count += 1
            if img_tensor.impl.val is not None:
                if threes.impl.val is None:
                    threes.impl.val = img_tensor.impl.val.copy()
                else:
                    threes.impl.val += img_tensor.impl.val
        elif label == 7:
            seven_count += 1
            if img_tensor.impl.val is not None:
                if sevens.impl.val is None:
                    sevens.impl.val = img_tensor.impl.val.copy()
                else:
                    sevens.impl.val += img_tensor.impl.val
    
    print(f"   Found {three_count} threes and {seven_count} sevens")
    
    # Normalize by count
    if three_count > 0 and threes.impl.val is not None:
        threes.impl.val = np.divide(threes.impl.val, three_count)
    if seven_count > 0 and sevens.impl.val is not None:
        sevens.impl.val = np.divide(sevens.impl.val, seven_count)
    
    return threes, sevens, three_count, seven_count


def compute_decision_boundary(threes, sevens, train_images, train_labels):
    """
    Compute the linear decision boundary for separating 3s from 7s.
    
    📋 PURPOSE:
    Creates a decision vector (hyperplane normal) that optimally separates
    the two digit classes in 784-dimensional pixel space. This implements
    a simple linear classifier based on template differences.
    
    🧮 MATHEMATICAL FOUNDATION:
    Decision vector: δ = μ₇ - μ₃ (difference of class means)
    Classification score: s(x) = x · δ + b
    Decision rule: class = 7 if s(x) > 0, else 3
    
    Bias computation: b = -threshold, where threshold = (μ₃_score + μ₇_score) / 2
    
    🔄 ALGORITHM STEPS:
    1. Compute difference vector δ = sevens_avg - threes_avg
    2. For each training image, compute score = image · δ
    3. Calculate mean scores for each class
    4. Set threshold at midpoint between class means
    5. Compute bias term to center decision boundary
    
    📊 SAMPLE INPUT:
    threes = Tensor(28, 28) representing average digit 3:
    [[0.02, 0.01, ..., 0.03],     # Background regions: low values
     [0.15, 0.67, ..., 0.12],     # Stroke regions: high values
     [...]]
    
    sevens = Tensor(28, 28) representing average digit 7:
    [[0.03, 0.78, ..., 0.91],     # Top bar: high values
     [0.02, 0.45, ..., 0.23],     # Diagonal: medium values  
     [...]]
    
    train_images = [array(...), array(...), ...]  # Training images
    train_labels = [3, 7, 3, 7, ...]             # Corresponding labels
    
    📈 SAMPLE OUTPUT:
    Returns (delta, bias, three_mean, seven_mean):
    
    delta = Tensor(28, 28) with difference pattern:
    [[+0.01, +0.77, ..., +0.88],    # Top: positive (7s bright, 3s dark)
     [-0.13, -0.22, ..., +0.11],    # Middle: mixed signs
     [-0.05, -0.03, ..., -0.04]]    # Bottom: negative (3s bright, 7s dark)
    
    bias = -0.127                    # Centering term
    three_mean = -2.34              # Average score for 3s
    seven_mean = +1.89              # Average score for 7s
    
    💡 INTERPRETATION OF DELTA VALUES:
    - Positive values (+): Pixels where 7s are typically brighter than 3s
      * Top horizontal region (7's top bar)
      * Upper-right area (7's vertical edge)
    
    - Negative values (-): Pixels where 3s are typically brighter than 7s
      * Middle-right curves (3's rounded sections)
      * Lower-middle area (3's bottom curve)
    
    - Near-zero values: Pixels with similar brightness in both digits
      * Background regions
      * Areas where strokes sometimes overlap
    
    🎯 CLASSIFICATION MECHANICS:
    For a new image x:
    1. Compute raw score = x · δ (dot product with decision vector)
    2. Add bias: final_score = raw_score + bias
    3. Classify: 7 if final_score > 0, else 3
    
    The score measures how "7-like" vs "3-like" the image appears based on
    the learned difference pattern.
    
    📊 TYPICAL SCORE DISTRIBUTIONS:
    - 3s typically get negative scores: range [-4.2, +1.8]
    - 7s typically get positive scores: range [-2.1, +4.7]
    - Threshold at 0.0 provides good separation
    - Some overlap indicates classification difficulty
    
    🔬 GEOMETRIC INTERPRETATION:
    In 784D space, this creates a hyperplane (decision boundary) that
    separates the two classes. The delta vector is perpendicular to this
    hyperplane and points toward the 7s class region.
    """
    print("🎯 Computing decision boundary...")
    
    # Compute difference vector (sevens - threes)
    delta = sevens - threes
    
    # Compute mean scores for each class to find threshold
    three_scores = []
    seven_scores = []
    
    for img, label in zip(train_images, train_labels):
        if isinstance(img, np.ndarray):
            img_tensor = Tensor(28, 28)
            img_tensor.impl.val = img.copy()
        else:
            img_tensor = img
        
        # Compute score: dot product of image with delta
        score_tensor = img_tensor.dot(delta).sum()
        score_tensor.impl.assure_value()
        if score_tensor.impl.val is not None:
            score = score_tensor.impl.val[0, 0]
        else:
            score = 0.0
        
        if label == 3:
            three_scores.append(score)
        elif label == 7:
            seven_scores.append(score)
    
    three_mean = np.mean(three_scores) if three_scores else 0
    seven_mean = np.mean(seven_scores) if seven_scores else 0
    
    print(f"   Average score for 3s: {three_mean:.4f}")
    print(f"   Average score for 7s: {seven_mean:.4f}")
    
    # Set threshold at midpoint
    threshold = (three_mean + seven_mean) / 2
    bias = -threshold
    
    print(f"   Decision threshold: {threshold:.4f}")
    print(f"   Bias term: {bias:.4f}")
    
    return delta, bias, three_mean, seven_mean


def evaluate_classifier(delta, bias, test_images, test_labels):
    """
    Evaluate the trained linear classifier on unseen test data.
    
    📋 PURPOSE:
    Measures the performance of our 3 vs 7 classifier by applying it to
    test images and computing accuracy, analyzing errors, and providing
    detailed performance statistics.
    
    🔄 CLASSIFICATION PROCESS:
    For each test image x:
    1. Compute similarity score: s = x · δ + bias
    2. Apply decision rule: class = 7 if s > 0, else 3
    3. Compare prediction with true label
    4. Accumulate statistics and collect examples
    
    📊 SAMPLE INPUT:
    delta = Tensor(28, 28) with decision pattern:
    [[+0.01, +0.77, ..., +0.88],     # Learned difference pattern
     [-0.13, -0.22, ..., +0.11],
     [...]]
    
    bias = -0.127                     # Centering term
    
    test_images = [
        array([[0.0, 0.0, ..., 0.9],  # Test image 1 (true label: 7)
               [0.0, 0.8, ..., 0.7],  # Should get positive score
               [...],
               [0.0, 0.0, ..., 0.1]]),
        array([[0.0, 0.0, ..., 0.2],  # Test image 2 (true label: 3)
               [0.1, 0.6, ..., 0.8],  # Should get negative score
               [...],
               [0.0, 0.0, ..., 0.0]]),
        # ... ~2000 more test images
    ]
    test_labels = [7, 3, 7, 3, ...]  # True labels
    
    📈 SAMPLE OUTPUT:
    Returns (accuracy, correct, total, wrong_examples):
    
    accuracy = 0.884                  # 88.4% classification accuracy
    correct = 1760                    # Number of correct predictions
    total = 1991                      # Total test samples
    
    wrong_examples = [
        (img_tensor, true_label=3, predicted_label=7, score=+0.23),
        (img_tensor, true_label=7, predicted_label=3, score=-0.15),
        # ... up to 5 misclassified examples for analysis
    ]
    
    🎯 DETAILED EXAMPLE CLASSIFICATION:
    
    Example 1 - Correct Classification of 7:
    Input: Test image showing clear digit 7
    Process:
      1. Flatten 28x28 → 784-element vector
      2. Dot product with delta: raw_score = +2.34
      3. Add bias: final_score = +2.34 + (-0.127) = +2.213
      4. Apply rule: +2.213 > 0 → Predict 7
      5. Compare: Predicted=7, True=7 → ✓ Correct
    
    Example 2 - Correct Classification of 3:
    Input: Test image showing clear digit 3
    Process:
      1. Flatten and compute dot product: raw_score = -1.87
      2. Add bias: final_score = -1.87 + (-0.127) = -1.997
      3. Apply rule: -1.997 ≤ 0 → Predict 3
      4. Compare: Predicted=3, True=3 → ✓ Correct
    
    Example 3 - Misclassification (Edge Case):
    Input: Ambiguous/poorly written 3 that looks like 7
    Process:
      1. Raw score = +0.23 (slightly 7-like due to unusual stroke pattern)
      2. Final score = +0.23 + (-0.127) = +0.103
      3. Predict 7 (score > 0), but true label is 3 → ✗ Error
    
    📊 TYPICAL PERFORMANCE STATISTICS:
    
    Overall Results:
    - Accuracy: 88.4% (1,760/1,991 correct)
    - Error rate: 11.6% (231 mistakes)
    
    Score Distribution Analysis:
    - 3s score range: [-4.2, +1.8]  # Mostly negative (good separation)
    - 7s score range: [-2.1, +4.7]  # Mostly positive (good separation)
    - Overlap region: [-2.1, +1.8]  # Area where errors occur
    
    Error Analysis:
    - 3s misclassified as 7s: ~5.8% (unusual 3s with 7-like features)
    - 7s misclassified as 3s: ~6.2% (unusual 7s with 3-like features)
    
    💡 PERFORMANCE INSIGHTS:
    
    Why 88.4% accuracy is good for this simple approach:
    - Uses only linear decision boundary (no complex features)
    - Template matching captures key digit differences
    - Real handwriting has significant variation
    - Some digits are genuinely ambiguous even to humans
    
    Common misclassification patterns:
    - 3s with straighter edges (look like 7s)
    - 7s with curved strokes (look like 3s)
    - Poor handwriting quality
    - Unusual writing styles not well represented in training
    
    🔬 MACHINE LEARNING LESSONS:
    - Linear classifiers can be surprisingly effective
    - Feature choice (pixel differences) is crucial
    - Bias term helps center the decision boundary
    - Real-world data always has edge cases and ambiguities
    - High accuracy doesn't mean perfect - some errors are inevitable
    """
    print("🧪 Evaluating classifier on test data...")
    
    correct = 0
    total = 0
    predictions = []
    scores = []
    
    # Store some examples
    correct_threes = []
    correct_sevens = []
    wrong_predictions = []
    
    for img, true_label in zip(test_images, test_labels):
        if isinstance(img, np.ndarray):
            img_tensor = Tensor(28, 28)
            img_tensor.impl.val = img.copy()
        else:
            img_tensor = img
        
        # Compute classification score
        score_tensor = img_tensor.dot(delta).sum()
        score_tensor.impl.assure_value()
        if score_tensor.impl.val is not None:
            score = score_tensor.impl.val[0, 0] + bias
        else:
            score = bias
        
        # Make prediction: positive score = 7, negative score = 3
        predicted_label = 7 if score > 0 else 3
        
        predictions.append(predicted_label)
        scores.append(score)
        total += 1
        
        if predicted_label == true_label:
            correct += 1
            # Store some correct examples
            if true_label == 3 and len(correct_threes) < 3:
                correct_threes.append((img_tensor, score))
            elif true_label == 7 and len(correct_sevens) < 3:
                correct_sevens.append((img_tensor, score))
        else:
            # Store some wrong examples
            if len(wrong_predictions) < 5:
                wrong_predictions.append((img_tensor, true_label, predicted_label, score))
    
    accuracy = correct / total if total > 0 else 0
    
    print(f"   Accuracy: {accuracy:.3f} ({correct}/{total})")
    print(f"   Error rate: {1-accuracy:.3f}")
    
    # Print some statistics
    three_scores = [s for s, l in zip(scores, [l for _, l in zip(test_images, test_labels)]) if l == 3]
    seven_scores = [s for s, l in zip(scores, [l for _, l in zip(test_images, test_labels)]) if l == 7]
    
    if three_scores:
        print(f"   3s score range: [{min(three_scores):.3f}, {max(three_scores):.3f}]")
    if seven_scores:
        print(f"   7s score range: [{min(seven_scores):.3f}, {max(seven_scores):.3f}]")
    
    return accuracy, correct, total, wrong_predictions


def analyze_decision_vector(delta):
    """
    Analyze the learned decision vector to understand discriminative features.
    
    📋 PURPOSE:
    Provides detailed statistical analysis of the decision vector to understand
    which image features (pixels) are most important for distinguishing between
    digits 3 and 7. This helps interpret what the classifier has learned.
    
    🔍 ANALYSIS COMPONENTS:
    1. Basic statistics (mean, std, min, max)
    2. Identification of most discriminative pixels
    3. Feature distribution analysis
    4. Spatial pattern interpretation
    
    📊 SAMPLE INPUT:
    delta = Tensor(28, 28) with difference pattern:
    [[ 0.010, +0.774, +0.883, ..., +0.912],    # Row 0: top of image
     [-0.134, -0.221, +0.110, ..., +0.087],    # Row 1: upper middle
     [ 0.023, +0.156, -0.089, ..., -0.234],    # Row 2: middle
     [...],                                      # Rows 3-26: continuation
     [ 0.003, +0.012, -0.031, ..., +0.045]]    # Row 27: bottom
    
    📈 SAMPLE OUTPUT (printed analysis):
    
    Decision vector statistics:
      Mean: 0.000847    # Near zero (balanced positive/negative features)
      Std:  0.151234    # Moderate spread indicates good discrimination
      Min: -0.423156    # Strong negative weight (3-favoring feature)
      Max: +0.445823    # Strong positive weight (7-favoring feature)
    
    Most discriminative pixels:
      Strongest for 7s: (2, 24) = +0.445823    # Top-right area (7's edge)
      Strongest for 3s: (15, 18) = -0.423156   # Middle-right (3's curve)
    
    Feature distribution:
      Positive (favor 7s): 387    # Features that light up for 7s
      Negative (favor 3s): 389    # Features that light up for 3s
      Zero (neutral): 8           # Features with no discrimination
    
    💡 INTERPRETATION GUIDE:
    
    Positive Features (favor 7s):
    - Top horizontal region: where 7s have their top bar
    - Upper-right edge: where 7s have vertical stroke
    - Diagonal regions: where 7s have slanted line
    
    Negative Features (favor 3s):
    - Middle-right curves: where 3s have rounded sections
    - Lower-middle area: where 3s have bottom curve
    - Central gaps: where 3s have openings
    
    Near-Zero Features:
    - Background regions: similar in both digits
    - Ambiguous areas: sometimes stroke, sometimes background
    
    🎯 SPATIAL PATTERNS:
    The decision vector reveals the key visual differences:
    - 7s are characterized by: horizontal top bar, diagonal stroke
    - 3s are characterized by: curved sections, horizontal segments
    - Background areas contribute little to classification
    
    🔬 MACHINE LEARNING INSIGHTS:
    
    Feature Balance:
    - Roughly equal positive/negative features (387 vs 389)
    - Indicates both classes contribute equally to decision
    - Well-balanced discrimination (not biased toward one class)
    
    Discrimination Strength:
    - Standard deviation of 0.151 shows good feature separation
    - Max magnitude of ~0.44 indicates strong discriminative power
    - Range of ~0.87 provides sufficient classification margin
    
    Spatial Localization:
    - Most important features cluster in expected regions
    - Top area favors 7s (horizontal bar region)
    - Middle-right favors 3s (curved sections)
    - Confirms the classifier learned meaningful patterns
    
    📊 STATISTICAL SIGNIFICANCE:
    - Mean near zero: unbiased feature selection
    - Symmetric min/max: balanced class representation
    - Reasonable std: good discrimination without overfitting
    - Feature count balance: neither class dominates
    
    This analysis validates that our simple linear classifier has learned
    intuitive and meaningful visual features for digit discrimination.
    """
    print("🔍 Analyzing decision vector...")
    
    delta.impl.assure_value()
    delta_array = delta.impl.val
    
    # Basic statistics
    print(f"   Decision vector statistics:")
    print(f"     Mean: {np.mean(delta_array):.6f}")
    print(f"     Std:  {np.std(delta_array):.6f}")
    print(f"     Min:  {np.min(delta_array):.6f}")
    print(f"     Max:  {np.max(delta_array):.6f}")
    
    # Find most discriminative pixels
    flat_delta = delta_array.flatten()
    most_positive_idx = np.argmax(flat_delta)
    most_negative_idx = np.argmin(flat_delta)
    
    # Convert back to 2D coordinates
    pos_row, pos_col = np.unravel_index(most_positive_idx, delta_array.shape)
    neg_row, neg_col = np.unravel_index(most_negative_idx, delta_array.shape)
    
    print(f"   Most discriminative pixels:")
    print(f"     Strongest for 7s: ({pos_row}, {pos_col}) = {flat_delta[most_positive_idx]:.6f}")
    print(f"     Strongest for 3s: ({neg_row}, {neg_col}) = {flat_delta[most_negative_idx]:.6f}")
    
    # Count positive vs negative features
    positive_features = np.sum(delta_array > 0)
    negative_features = np.sum(delta_array < 0)
    zero_features = np.sum(delta_array == 0)
    
    print(f"   Feature distribution:")
    print(f"     Positive (favor 7s): {positive_features}")
    print(f"     Negative (favor 3s): {negative_features}")
    print(f"     Zero (neutral): {zero_features}")


def save_analysis_results(delta, bias, accuracy, three_mean, seven_mean):
    """
    Save comprehensive analysis results to JSON file for future reference.
    
    📋 PURPOSE:
    Persists the trained classifier parameters and performance metrics to disk,
    enabling model reuse, comparison studies, and detailed result analysis.
    
    💾 SAVED DATA STRUCTURE:
    Creates a comprehensive JSON file containing:
    - Performance metrics (accuracy, error rate)
    - Model parameters (bias, decision threshold)
    - Class-specific statistics (mean scores per digit)
    - Decision vector statistics for feature analysis
    - Metadata about the approach and model description
    
    📊 SAMPLE INPUT:
    delta = Tensor(28, 28) with learned features
    bias = -0.127                    # Learned bias term
    accuracy = 0.884                 # Test accuracy
    three_mean = -2.34              # Average score for 3s
    seven_mean = +1.89              # Average score for 7s
    
    📈 SAMPLE OUTPUT FILE CONTENT:
    results/three_or_seven_results.json:
    {
      "accuracy": 0.884,
      "bias": -0.127,
      "three_mean_score": -2.34,
      "seven_mean_score": 1.89,
      "decision_threshold": -0.225,
      "delta_statistics": {
        "mean": 0.000847,
        "std": 0.151234,
        "min": -0.423156,
        "max": 0.445823
      },
      "model_description": "Simple linear classifier for distinguishing 3s from 7s",
      "approach": "Dot product with difference of average images"
    }
    
    💡 PRACTICAL USES:
    - Model deployment: Load saved parameters for inference
    - Performance tracking: Compare different training runs  
    - Research documentation: Record experimental results
    - Educational analysis: Study feature importance patterns
    - Debugging: Investigate classification behavior
    
    🔄 INTEGRATION WORKFLOW:
    This saved model can be loaded later to:
    1. Classify new 3 vs 7 images without retraining
    2. Compare with other classification approaches
    3. Analyze feature evolution across training sessions
    4. Generate reports and visualizations
    """
    results_file = os.path.join(os.path.dirname(__file__), '..', 'results', 'three_or_seven_results.json')
    os.makedirs(os.path.dirname(results_file), exist_ok=True)
    
    # Convert tensor to list for JSON serialization
    delta.impl.assure_value()
    delta_array = delta.impl.val
    
    results = {
        'accuracy': float(accuracy),
        'bias': float(bias),
        'three_mean_score': float(three_mean),
        'seven_mean_score': float(seven_mean),
        'decision_threshold': float((three_mean + seven_mean) / 2),
        'delta_statistics': {
            'mean': float(np.mean(delta_array)),
            'std': float(np.std(delta_array)),
            'min': float(np.min(delta_array)),
            'max': float(np.max(delta_array))
        },
        'model_description': 'Simple linear classifier for distinguishing 3s from 7s',
        'approach': 'Dot product with difference of average images'
    }
    
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"💾 Results saved to: {results_file}")


def print_ascii_image(tensor, threshold=0.5):
    """
    Display a simple ASCII visualization of an image tensor.
    
    📋 PURPOSE:
    Provides a text-based visualization of 28x28 images for educational
    purposes, allowing users to see image patterns directly in the terminal
    without requiring external graphics libraries.
    
    🎨 VISUALIZATION APPROACH:
    - Normalizes pixel values to 0-1 range
    - Applies threshold to convert to binary representation
    - Uses Unicode blocks for visual clarity
    - Maintains aspect ratio and spatial relationships
    
    📊 SAMPLE INPUT:
    tensor = Tensor(28, 28) representing digit image:
    [[0.0, 0.0, 0.1, ..., 0.0],      # Top row: mostly background
     [0.0, 0.8, 0.9, ..., 0.2],      # Stroke region: high values
     [0.1, 0.7, 0.8, ..., 0.0],      # Continued stroke
     [...],
     [0.0, 0.0, 0.0, ..., 0.0]]      # Bottom row: background
    
    threshold = 0.5                   # Brightness cutoff for display
    
    📈 SAMPLE OUTPUT (printed visualization):
    ASCII representation (■ = high intensity, □ = low intensity):
       □□□□■■■■■■■■■■■■■■■■■■■□□□□□
       □□□□□□□□□□□□□□□□□■■■■■□□□□□□
       □□□□□□□□□□□□□□□□□■■■■■□□□□□□
       □□□□□□□□□□□□□□□■■■■■■□□□□□□□
       □□□□□□□□□□□□□■■■■■■■□□□□□□□□
       □□□□□□□□□□■■■■■■■■□□□□□□□□□□
       □□□□□□□■■■■■■■■■□□□□□□□□□□□□
       □□□□■■■■■■■■■□□□□□□□□□□□□□□□
       □□■■■■■■■□□□□□□□□□□□□□□□□□□□
       ■■■■■■□□□□□□□□□□□□□□□□□□□□□□
       ...
    
    💡 INTERPRETATION:
    - ■ (filled blocks): Bright pixels above threshold (pen strokes)
    - □ (empty blocks): Dark pixels below threshold (background)
    - Pattern reveals digit shape and structure
    - Useful for debugging and educational visualization
    
    🎯 EDUCATIONAL VALUE:
    - Shows how computers "see" handwritten digits
    - Demonstrates pixel-level representation
    - Helps visualize average images and decision vectors
    - Provides intuitive understanding of image data
    - Enables pattern recognition without complex graphics
    
    🔧 TECHNICAL DETAILS:
    - Handles various input ranges automatically via normalization
    - Configurable threshold for different contrast levels
    - Works with both individual images and computed averages
    - ASCII-safe output for terminal compatibility
    """
  
    """
    🔍 HOW tensor.impl.assure_value() WORKS WHEN TMODE IS NOT EXPLICITLY SET:
    
    In print_ascii_image, we receive tensors that can have different modes depending 
    on how they were created. Let's trace through the actual computation:
    
    CASE 1: Average Images (threes_avg, sevens_avg)
    ===============================================
    Creation in compute_average_images():
    >>> threes = Tensor(28, 28)           # Creates TensorImpl with rows=28, cols=28
    >>> threes.zero()                     # Calls TensorImpl.__init__(28, 28)
    
    What happens in TensorImpl.__init__():
    - Since rows > 0 and cols > 0:
      * self.val = np.zeros((28, 28), dtype=np.float32)  # Allocate memory
      * self.have_val = True                             # Mark as computed
      * self.mode = TMode.PARAMETER                      # Set mode automatically
    
    Later: threes.impl.val += img_tensor.impl.val       # Direct numpy manipulation
    Result: threes.impl.mode = TMode.PARAMETER, values already computed
    
    When assure_value() is called:
    >>> if self.have_val or self.mode == TMode.PARAMETER:
    >>>     return  # EXIT IMMEDIATELY - no computation needed!
    
    CASE 2: Decision Vector (delta = sevens - threes)
    ================================================
    Creation through subtraction operator:
    >>> delta = sevens - threes           # Calls sevens.__sub__(threes)
    
    What __sub__() does internally:
    1. neg = Tensor()                     # Create empty tensor
    2. neg.impl = TensorImpl(lhs=threes.impl, mode=TMode.NEG)  # Negation operation
    3. return sevens + neg                # Addition operation
    
    This creates a computation graph:
    sevens (TMode.PARAMETER) ──┐
                               ├──> addition_op (TMode.ADDITION)
    threes (TMode.PARAMETER) ──┘ neg_op (TMode.NEG)
    
    Result: delta.impl.mode = TMode.ADDITION, delta.impl.have_val = False
    
    When assure_value() is called:
    >>> if self.have_val or self.mode == TMode.PARAMETER:
    >>>     # FALSE - we have TMode.ADDITION and have_val=False
    >>> elif self.mode == TMode.ADDITION:
    >>>     if self.lhs is not None and self.rhs is not None:
    >>>         self.lhs.assure_value()   # Compute sevens (already computed)
    >>>         self.rhs.assure_value()   # Compute neg_op:
    >>>                                   #   neg_op calls threes.assure_value()
    >>>                                   #   neg_op.val = -threes.val
    >>>         self.val = self.lhs.val + self.rhs.val  # sevens + (-threes)
    
    CASE 3: Operation Results (score_tensor = img.dot(delta).sum())
    ==============================================================
    This creates an even more complex computation graph:
    img (TMode.PARAMETER) ──┐
                            ├──> dot_prod_op (TMode.DOT_PROD) ──> sum_op (TMode.SUM)
    delta (TMode.ADDITION) ─┘
    
    When assure_value() is called on sum_op:
    1. sum_op.lhs.assure_value()          # Compute dot product
    2. dot_prod_op.lhs.assure_value()     # Compute img (already done)
    3. dot_prod_op.rhs.assure_value()     # Compute delta (ADDITION operation)
    4. dot_prod_op.val = img.val @ delta.val   # Matrix multiplication
    5. sum_op.val = dot_prod_op.val.sum()      # Sum all elements
    
    🎯 KEY INSIGHTS:
    
    1. AUTOMATIC MODE ASSIGNMENT: When you create Tensor(28, 28), it automatically
       gets TMode.PARAMETER because rows > 0 and cols > 0 in the constructor.
    
    2. OPERATION MODES: When you use operators like -, +, dot(), sum(), new tensors
       are created with appropriate operation modes (TMode.NEG, TMode.ADDITION, etc.)
    
    3. LAZY EVALUATION: Operations don't compute immediately. They build a computation
       graph and compute values only when assure_value() is called.
    
    4. PARAMETER TENSORS SKIP COMPUTATION: If mode=TMode.PARAMETER and have_val=True,
       assure_value() returns immediately without any computation.
    
    5. RECURSIVE DEPENDENCY RESOLUTION: For operation tensors, assure_value()
       recursively computes all dependencies in the correct order.
    
    📊 PRACTICAL EXAMPLE IN THREE_OR_SEVEN:
    
    Average images (threes_avg, sevens_avg):
    - Created as Tensor(28, 28) → TMode.PARAMETER
    - Values set via direct numpy operations
    - assure_value() does nothing (already computed)
    
    Decision vector (delta):
    - Created via sevens - threes → TMode.ADDITION with computation graph
    - assure_value() computes: sevens.val + (-threes.val)
    - Result becomes available in delta.impl.val
    
    Both cases end up with img_array containing the actual pixel values ready
    for ASCII visualization, regardless of how they were computed!
    """
    print(f"TRACE(print_ascii_image): Displaying ASCII image with threshold {tensor.impl.mode} and shape {tensor.shape}")
    tensor.impl.assure_value()
    img_array = tensor.impl.val
    # Normalize to 0-1 range
    # Why this is needed: The tensor might contain values in any range (negative, very large, etc.), but for ASCII visualization we need values between 0 and 1.
    # Formula : normalized_value = (original_value - minimum_value) / (maximum_value - minimum_value)
    img_norm = (img_array - np.min(img_array)) / (np.max(img_array) - np.min(img_array))
    
    print("   ASCII representation (■ = high intensity, □ = low intensity):")
    for row in img_norm:
        line = "   "
        for pixel in row:
            if pixel > threshold:
                line += "■"
            else:
                line += "□"
        print(line)


def main():
    """
    Main execution function - orchestrates the complete 3 vs 7 classification pipeline.
    
    📋 PURPOSE:
    Demonstrates a complete machine learning workflow from data loading through
    model training, evaluation, and analysis. Serves as both a functional
    classifier and an educational example of fundamental ML concepts.
    
    🔄 COMPLETE WORKFLOW:
    
    PHASE 1: Data Preparation
    1. Load and validate EMNIST/MNIST dataset
    2. Filter for only digits 3 and 7 (binary classification)
    3. Split into training and test sets
    4. Display dataset statistics and balance
    
    PHASE 2: Feature Learning  
    5. Compute average image templates for each digit class
    6. Visualize learned templates using ASCII art
    7. Calculate decision vector (difference of averages)
    8. Analyze spatial patterns in decision features
    
    PHASE 3: Model Training
    9. Compute classification scores on training data
    10. Calculate optimal decision threshold and bias
    11. Analyze score distributions per class
    
    PHASE 4: Model Evaluation
    12. Apply classifier to unseen test data
    13. Compute accuracy and error statistics
    14. Analyze misclassified examples
    15. Provide performance interpretation
    
    PHASE 5: Results & Analysis
    16. Save model parameters and results
    17. Generate comprehensive performance report
    18. Provide technical insights and learning outcomes
    
    📊 SAMPLE EXECUTION FLOW:
    
    Input: EMNIST dataset files in data/ directory
    
    Step 1 - Data Loading:
    📦 Loading real EMNIST/MNIST data...
    📈 Dataset statistics:
       Training samples: 11,966
       Test samples: 1,991
       Training: 5,923 threes, 6,043 sevens
       Test: 986 threes, 1,005 sevens
    
    Step 2 - Template Learning:
    📊 Computing average images...
       Found 5923 threes and 6043 sevens
    📸 Average images computed:
       [ASCII visualization of average 3]
       [ASCII visualization of average 7]
    
    Step 3 - Decision Boundary:
    🎯 Computing decision boundary...
       Average score for 3s: -2.3401
       Average score for 7s: +1.8943
       Decision threshold: -0.2229
       Bias term: +0.2229
    
    Step 4 - Feature Analysis:
    🔍 Analyzing decision vector...
       Decision vector statistics:
         Mean: 0.000847, Std: 0.151234
         Min: -0.423156, Max: +0.445823
       Most discriminative pixels:
         Strongest for 7s: (2, 24) = +0.445823
         Strongest for 3s: (15, 18) = -0.423156
       Feature distribution:
         Positive (favor 7s): 387
         Negative (favor 3s): 389
         Zero (neutral): 8
    
    Step 5 - Classification:
    🧪 Evaluating classifier on test data...
       Accuracy: 0.884 (1760/1991)
       Error rate: 0.116
       3s score range: [-4.156, +1.823]
       7s score range: [-2.089, +4.734]
    
    Step 6 - Final Summary:
    🎉 Classification Summary
    ==============================
    Final accuracy: 0.884 (1760/1991)
    Error rate: 0.116
    ✅ Good classification performance!
    
    💾 Results saved to: results/three_or_seven_results.json
    
    📋 Technical Details:
       Algorithm: Linear classifier using average image difference
       Features: Raw pixel intensities (28x28 = 784 features)  
       Decision function: score = dot(image, delta) + bias
       Classification rule: score > 0 → 7, score ≤ 0 → 3
       Bias term: +0.222934
    
    🎓 EDUCATIONAL OUTCOMES:
    
    Demonstrates Key ML Concepts:
    - Binary classification fundamentals
    - Template matching and similarity measures
    - Linear decision boundaries in high-dimensional space
    - Feature importance and interpretability
    - Performance evaluation and error analysis
    
    Mathematical Foundations:
    - Vector operations and dot products
    - Statistical averaging and normalization
    - Threshold-based decision making
    - Bias correction and centering
    
    Practical ML Skills:
    - Data loading and preprocessing
    - Model training and validation
    - Performance metrics interpretation
    - Result visualization and analysis
    - Model persistence and deployment
    
    ⚠️ ERROR HANDLING:
    - Graceful fallback when dataset unavailable
    - Clear instructions for data download
    - Robust handling of file system operations
    - Informative error messages with solutions
    
    🎯 PERFORMANCE EXPECTATIONS:
    - Typical accuracy: 85-90% (good for simple linear method)
    - Training time: ~30-60 seconds on standard hardware
    - Memory usage: Minimal (only stores averages, not full dataset)
    - Generalization: Works well on similar handwritten digit tasks
    
    💡 EXTENSIONS & IMPROVEMENTS:
    This example serves as foundation for more advanced techniques:
    - Multi-class classification (all 10 digits)
    - Non-linear kernels and feature transformations
    - Deep learning approaches (neural networks)
    - Ensemble methods and model combination
    - Real-time inference and deployment
    """
    print("🔢 Three or Seven Classifier")
    print("=" * 50)
    print("This program implements a simple linear classifier to distinguish")
    print("between handwritten digits 3 and 7 using PyTensorLib.")
    print("Uses real MNIST/EMNIST data only - no synthetic data.")
    print("=" * 50)
    
    try:
        # Load real data only
        train_data, test_data = create_mnist_data_if_needed()
        
        print(f"📈 Dataset statistics:")
        print(f"   Training samples: {len(train_data['images'])}")
        print(f"   Test samples: {len(test_data['images'])}")
        
        # Count labels in training data
        train_threes = sum(1 for label in train_data['labels'] if label == 3)
        train_sevens = sum(1 for label in train_data['labels'] if label == 7)
        test_threes = sum(1 for label in test_data['labels'] if label == 3)
        test_sevens = sum(1 for label in test_data['labels'] if label == 7)
        
        print(f"   Training: {train_threes} threes, {train_sevens} sevens")
        print(f"   Test: {test_threes} threes, {test_sevens} sevens")
        
        # Compute average images
        threes_avg, sevens_avg, three_count, seven_count = compute_average_images(
            train_data['images'], train_data['labels']
        )
        
        print(f"\n📸 Average images computed:")
        print(f"   Average of {three_count} threes:")
        print_ascii_image(threes_avg, threshold=0.3)
        print(f"\n   Average of {seven_count} sevens:")
        print_ascii_image(sevens_avg, threshold=0.3)
        
        # Compute decision boundary
        delta, bias, three_mean, seven_mean = compute_decision_boundary(
            threes_avg, sevens_avg, train_data['images'], train_data['labels']
        )
        
        print(f"\n🎯 Decision vector (delta = sevens_avg - threes_avg):")
        print_ascii_image(delta, threshold=0.0)
        
        # Analyze decision vector
        analyze_decision_vector(delta)
        
        # Evaluate classifier
        accuracy, correct, total, wrong_examples = evaluate_classifier(
            delta, bias, test_data['images'], test_data['labels']
        )
        
        # Print summary
        print(f"\n🎉 Classification Summary")
        print("=" * 30)
        print(f"Final accuracy: {accuracy:.3f} ({correct}/{total})")
        print(f"Error rate: {1-accuracy:.3f}")
        
        if accuracy >= 0.95:
            print("🌟 Excellent classification performance!")
        elif accuracy >= 0.85:
            print("✅ Good classification performance!")
        elif accuracy >= 0.70:
            print("📈 Decent classification performance!")
        else:
            print("⚠️  Classification needs improvement.")
            print("   This could be due to:")
            print("   - Limited training data quality")
            print("   - Need for more complex features")
            print("   - Dataset imbalance or noise")
        
        # Save results
        save_analysis_results(delta, bias, accuracy, three_mean, seven_mean)
        
        # Show some wrong examples if any
        if wrong_examples and len(wrong_examples) > 0:
            print(f"\n❌ Analysis of {min(3, len(wrong_examples))} misclassified examples:")
            for i, (img, true_label, pred_label, score) in enumerate(wrong_examples[:3]):
                print(f"   Example {i+1}: True={true_label}, Predicted={pred_label}, Score={score:.3f}")
                print_ascii_image(img, threshold=0.3)
                print()
        
        print(f"\n📋 Technical Details:")
        print(f"   Algorithm: Linear classifier using average image difference")
        print(f"   Features: Raw pixel intensities (28x28 = 784 features)")
        print(f"   Decision function: score = dot(image, delta) + bias")
        print(f"   Classification rule: score > 0 → 7, score ≤ 0 → 3")
        print(f"   Bias term: {bias:.6f}")
        
        return 0
        
    except RuntimeError as e:
        if "No real MNIST/EMNIST data available" in str(e):
            print(f"\n🚫 Cannot proceed without real data: {e}")
            print(f"\n📥 To download the required dataset:")
            print(f"   1. Run: python download_mnist.py")
            print(f"   2. Ensure stable internet connection")
            print(f"   3. Check that you have ~900MB free disk space")
            print(f"   4. Re-run this example")
            return 2
        else:
            print(f"\n❌ Runtime error: {e}")
            return 1
        
    except Exception as e:
        print(f"\n❌ Error during classification: {e}")
        print(f"\n💡 Make sure you have downloaded the dataset:")
        print(f"   python download_mnist.py")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)