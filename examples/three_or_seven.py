#!/usr/bin/env python3
"""
Three or Seven Classifier

This script implements a simple binary classifier to distinguish between
handwritten digits 3 and 7 using PyTensorLib. It's a Python port of the
original C++ threeorseven.cc program.

The approach:
1. Load MNIST dataset and filter for 3s and 7s
2. Compute average images for each digit
3. Calculate difference vector (delta)
4. Use dot product with delta as the classification score
5. Apply threshold to classify digits

This demonstrates basic image classification using linear algebra operations.
"""

import sys
import os
import numpy as np

# Add src directory to path for development
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from pytensorlib import Tensor, MNISTReader, download_emnist
import json


def create_mnist_data_if_needed():
    """Load EMNIST/MNIST data - no synthetic fallback"""
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
            print("� Please run: python download_mnist.py")
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
    """Compute average images for 3s and 7s"""
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
            threes.impl.val += img_tensor.impl.val
        elif label == 7:
            seven_count += 1
            sevens.impl.val += img_tensor.impl.val
    
    print(f"   Found {three_count} threes and {seven_count} sevens")
    
    # Normalize by count
    if three_count > 0:
        threes.impl.val /= three_count
    if seven_count > 0:
        sevens.impl.val /= seven_count
    
    return threes, sevens, three_count, seven_count


def compute_decision_boundary(threes, sevens, train_images, train_labels):
    """Compute the decision boundary (delta vector) and bias"""
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
        score = score_tensor.impl.val[0, 0]
        
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
    """Evaluate the classifier on test data"""
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
        score = score_tensor.impl.val[0, 0] + bias
        
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
    """Analyze the decision vector to understand what features it captures"""
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
    """Save analysis results to JSON file"""
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
    """Print a simple ASCII representation of an image tensor"""
    tensor.impl.assure_value()
    img_array = tensor.impl.val
    
    # Normalize to 0-1 range
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
    """Main execution function"""
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