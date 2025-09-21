#!/usr/bin/env python3
"""
Perceptron Learning Algorithm for 3 vs 7 Classification

This is a Python conversion of the C++ 37learn.cc program using PyTensorLib.
It implements a simple perceptron with weight updates to learn to distinguish
between handwritten digits 3 and 7 from the MNIST/EMNIST dataset.

Algorithm:
1. Initialize random weights and bias
2. For each training image (3 or 7):
   - Compute score = dot(image, weights) + bias
   - Predict: 7 if score > 0, else 3
   - Update weights based on error:
     * If label=7 and score<2: weights += learning_rate * image, bias += lr
     * If label=3 and score>-2: weights -= learning_rate * image, bias -= lr
3. Test accuracy periodically and stop when >98% correct

This demonstrates classic machine learning concepts:
- Linear classifiers and decision boundaries
- Online learning with weight updates
- Early stopping based on validation accuracy
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
    """SQLite database logger for perceptron training results"""
    
    def __init__(self, db_path: str):
        """Initialize SQLite database for logging training results"""
        self.db_path = db_path
        
        # Remove existing database
        if os.path.exists(db_path):
            os.unlink(db_path)
        
        self.conn = sqlite3.connect(db_path)
        self._setup_tables()
    
    def _setup_tables(self):
        """Create database tables for logging test results"""
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
        """Log individual test result"""
        correct = 1 if verdict == label else 0
        self.conn.execute(
            'INSERT INTO test_results (label, res, verdict, correct) VALUES (?, ?, ?, ?)',
            (label, score, verdict, correct)
        )
        self.conn.commit()
    
    def log_training_progress(self, epoch: int, accuracy: float, weights_file: str, bias: float):
        """Log training progress"""
        self.conn.execute(
            'INSERT INTO training_progress (epoch, accuracy, weights_saved, bias) VALUES (?, ?, ?, ?)',
            (epoch, accuracy, weights_file, bias)
        )
        self.conn.commit()
    
    def close(self):
        """Close database connection"""
        self.conn.close()


def save_tensor_as_image(tensor: Tensor, filepath: str, scale: int = 252, invert: bool = False):
    """
    Save tensor as image (simplified version of C++ saveTensor function)
    
    Args:
        tensor: 2D tensor to save
        filepath: Full output file path (including directory)
        scale: Scaling factor for pixel values
        invert: Whether to invert colors
    """
    try:
        # Ensure the directory exists
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        # Get tensor data and ensure it's 2D
        data = tensor.impl.val
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
    Test the perceptron on validation data
    
    Args:
        test_reader: MNIST test data reader
        weights: Current weight tensor (28x28)
        bias: Current bias value
        logger: Optional SQLite logger for results
    
    Returns:
        Accuracy percentage
    """
    corrects = 0
    wrongs = 0
    
    print("🧪 Testing perceptron performance...")
    
    for n in range(test_reader.get_num()):
        label = test_reader.get_label(n)
        
        # Only test on 3s and 7s
        if label != 3 and label != 7:
            continue
        
        # Get image as array and create tensor
        img_array = test_reader.get_image_as_array(n)
        img = Tensor(28, 28)
        img.impl.val = img_array.astype(np.float32)
        
        # Compute score: dot product of image and weights + bias
        # This is equivalent to: score = (img.dot(weights).sum())(0,0) + bias
        dot_product = np.sum(img.impl.val * weights.impl.val)
        score = dot_product + bias
        
        # Make prediction: 7 if score > 0, else 3
        predict = 7 if score > 0 else 3
        
        # Log result if logger provided
        if logger:
            logger.log_test_result(label, score, predict)
        
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
    Create identity-like tensor for learning rate (mimics C++ lr.identity(0.01))
    
    Args:
        size: Size of square tensor (28 for 28x28 images)
        lr: Learning rate value
    
    Returns:
        Tensor filled with learning rate value
    """
    lr_tensor = Tensor(size, size)
    lr_tensor.impl.val = np.full((size, size), lr, dtype=np.float32)
    return lr_tensor


def main():
    """
    Main perceptron learning algorithm
    
    Implements the same logic as the C++ 37learn.cc:
    1. Load MNIST/EMNIST training and test data
    2. Initialize random weights and zero bias
    3. Train using perceptron update rule
    4. Test periodically and save progress
    5. Stop when accuracy > 98% or training complete
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
            print(f"⚠️  MNIST failed: {e}")
            print("🔄 Using synthetic data for demonstration")
            
            # Use synthetic data
            images, labels = create_synthetic_mnist_data(1000)
            # Split into train/test
            train_images = images[:800]
            train_labels = labels[:800]
            test_images = images[800:]
            test_labels = labels[800:]
            
            train_reader = MNISTReader(train_images, train_labels)
            test_reader = MNISTReader(test_images, test_labels)
            data_type = "Synthetic"
    
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