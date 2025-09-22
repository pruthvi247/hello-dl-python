#!/usr/bin/env python3
"""
Neural Network Matrix Transformations: Comprehensive Guide with Examples

This guide provides detailed explanations of how matrices change shape through
neural network layers, with concrete numerical examples to illustrate the
transformation process from input image to final classification.
"""

import numpy as np

def print_matrix_info(matrix, name, description=""):
    """Print detailed information about a matrix"""
    print(f"\n{name}:")
    if description:
        print(f"Description: {description}")
    print(f"Shape: {matrix.shape}")
    print(f"Data type: {matrix.dtype}")
    if matrix.size <= 50:  # Only show values for small matrices
        print(f"Values:\n{matrix}")
    else:
        print(f"Sample values (first 5 elements): {matrix.flatten()[:5]}")
        print(f"Min: {np.min(matrix):.4f}, Max: {np.max(matrix):.4f}, Mean: {np.mean(matrix):.4f}")

def demonstrate_3_layer_network():
    """
    Demonstrate a 3-layer neural network with actual matrix operations
    Architecture: 784 → 128 → 64 → 10
    """
    print("=" * 80)
    print("3-LAYER NEURAL NETWORK MATRIX TRANSFORMATION DEMONSTRATION")
    print("=" * 80)
    print("Architecture: Input(784) → Hidden1(128) → Hidden2(64) → Output(10)")
    print("=" * 80)

    # ========================================================================
    # INPUT LAYER: Raw Image Data
    # ========================================================================
    print("\n" + "="*60)
    print("STEP 1: INPUT LAYER - IMAGE TO VECTOR CONVERSION")
    print("="*60)
    
    # Create a sample 28x28 image (simulating a handwritten digit)
    np.random.seed(42)  # For reproducible results
    image_2d = np.random.rand(28, 28) * 0.8  # Pixel values 0-0.8
    # Add some structure to make it look more like a digit
    image_2d[10:18, 8:20] = 0.9  # Bright horizontal bar
    image_2d[8:20, 10:12] = 0.9   # Bright vertical bar
    
    print_matrix_info(image_2d, "Original Image", "28×28 pixel intensities")
    
    # Flatten to column vector for neural network input
    x = image_2d.reshape(784, 1)
    print_matrix_info(x, "Flattened Input Vector (x)", "Column vector for neural network")
    
    print(f"\nTransformation: (28, 28) → reshape → (784, 1)")
    print("Each pixel becomes one input feature to the neural network")

    # ========================================================================
    # LAYER 1: INPUT TO FIRST HIDDEN LAYER
    # ========================================================================
    print("\n" + "="*60)
    print("STEP 2: LAYER 1 TRANSFORMATION - INPUT TO HIDDEN LAYER 1")
    print("="*60)
    
    # Layer 1 weights: W1 maps 784 inputs to 128 hidden neurons
    W1 = np.random.normal(0, 0.1, (128, 784))  # Xavier initialization
    b1 = np.zeros((128, 1))  # Bias vector
    
    print_matrix_info(W1, "Weight Matrix W1", "Maps 784 input features to 128 hidden neurons")
    print_matrix_info(b1, "Bias Vector b1", "Offset for each hidden neuron")
    
    print(f"\nMatrix Multiplication Rule:")
    print(f"(W1 @ x) + b1 = z1")
    print(f"({W1.shape[0]}×{W1.shape[1]}) @ ({x.shape[0]}×{x.shape[1]}) + ({b1.shape[0]}×{b1.shape[1]}) = ({W1.shape[0]}×{x.shape[1]})")
    
    # Compute linear combination
    z1 = W1 @ x + b1
    print_matrix_info(z1, "Linear Output z1", "Raw activations before ReLU")
    
    # Apply ReLU activation
    a1 = np.maximum(0, z1)  # ReLU: max(0, x)
    print_matrix_info(a1, "Activated Output a1", "After ReLU activation (negative values → 0)")
    
    print(f"\nLayer 1 Summary:")
    print(f"• Input: {x.shape} → Linear: {z1.shape} → ReLU: {a1.shape}")
    print(f"• Parameters: {W1.size + b1.size} ({W1.size} weights + {b1.size} biases)")
    print(f"• Non-zero activations: {np.count_nonzero(a1)}/{a1.size} ({np.count_nonzero(a1)/a1.size*100:.1f}%)")

    # ========================================================================
    # LAYER 2: FIRST HIDDEN TO SECOND HIDDEN LAYER
    # ========================================================================
    print("\n" + "="*60)
    print("STEP 3: LAYER 2 TRANSFORMATION - HIDDEN LAYER 1 TO HIDDEN LAYER 2")
    print("="*60)
    
    # Layer 2 weights: W2 maps 128 hidden1 neurons to 64 hidden2 neurons
    W2 = np.random.normal(0, 0.1, (64, 128))
    b2 = np.zeros((64, 1))
    
    print_matrix_info(W2, "Weight Matrix W2", "Maps 128 hidden1 neurons to 64 hidden2 neurons")
    print_matrix_info(b2, "Bias Vector b2", "Offset for each hidden2 neuron")
    
    print(f"\nMatrix Multiplication Rule:")
    print(f"(W2 @ a1) + b2 = z2")
    print(f"({W2.shape[0]}×{W2.shape[1]}) @ ({a1.shape[0]}×{a1.shape[1]}) + ({b2.shape[0]}×{b2.shape[1]}) = ({W2.shape[0]}×{a1.shape[1]})")
    
    # Compute linear combination
    z2 = W2 @ a1 + b2
    print_matrix_info(z2, "Linear Output z2", "Raw activations before ReLU")
    
    # Apply ReLU activation
    a2 = np.maximum(0, z2)
    print_matrix_info(a2, "Activated Output a2", "After ReLU activation")
    
    print(f"\nLayer 2 Summary:")
    print(f"• Input: {a1.shape} → Linear: {z2.shape} → ReLU: {a2.shape}")
    print(f"• Parameters: {W2.size + b2.size} ({W2.size} weights + {b2.size} biases)")
    print(f"• Non-zero activations: {np.count_nonzero(a2)}/{a2.size} ({np.count_nonzero(a2)/a2.size*100:.1f}%)")

    # ========================================================================
    # LAYER 3: SECOND HIDDEN TO OUTPUT LAYER
    # ========================================================================
    print("\n" + "="*60)
    print("STEP 4: LAYER 3 TRANSFORMATION - HIDDEN LAYER 2 TO OUTPUT")
    print("="*60)
    
    # Layer 3 weights: W3 maps 64 hidden2 neurons to 10 output classes
    W3 = np.random.normal(0, 0.1, (10, 64))
    b3 = np.zeros((10, 1))
    
    print_matrix_info(W3, "Weight Matrix W3", "Maps 64 hidden2 neurons to 10 output classes")
    print_matrix_info(b3, "Bias Vector b3", "Offset for each output class")
    
    print(f"\nMatrix Multiplication Rule:")
    print(f"(W3 @ a2) + b3 = z3")
    print(f"({W3.shape[0]}×{W3.shape[1]}) @ ({a2.shape[0]}×{a2.shape[1]}) + ({b3.shape[0]}×{b3.shape[1]}) = ({W3.shape[0]}×{a2.shape[1]})")
    
    # Compute linear combination (final layer - no ReLU)
    z3 = W3 @ a2 + b3
    print_matrix_info(z3, "Final Logits z3", "Raw scores for each digit class (0-9)")
    
    # Apply softmax for probability interpretation
    exp_z3 = np.exp(z3 - np.max(z3))  # Subtract max for numerical stability
    softmax_output = exp_z3 / np.sum(exp_z3)
    print_matrix_info(softmax_output, "Softmax Probabilities", "Probability distribution over classes")
    
    predicted_class = np.argmax(z3)
    confidence = softmax_output[predicted_class, 0]
    
    print(f"\nLayer 3 Summary:")
    print(f"• Input: {a2.shape} → Linear: {z3.shape} → Softmax: {softmax_output.shape}")
    print(f"• Parameters: {W3.size + b3.size} ({W3.size} weights + {b3.size} biases)")
    print(f"• Predicted class: {predicted_class} with confidence: {confidence:.4f}")

    # ========================================================================
    # COMPLETE NETWORK SUMMARY
    # ========================================================================
    print("\n" + "="*60)
    print("COMPLETE NETWORK TRANSFORMATION SUMMARY")
    print("="*60)
    
    total_params = W1.size + b1.size + W2.size + b2.size + W3.size + b3.size
    
    print(f"Data Flow:")
    print(f"  Image (28×28) → Flatten → Input (784×1)")
    print(f"  ↓ Layer 1: W1(128×784) × x(784×1) + b1(128×1) → z1(128×1) → ReLU → a1(128×1)")
    print(f"  ↓ Layer 2: W2(64×128) × a1(128×1) + b2(64×1) → z2(64×1) → ReLU → a2(64×1)")
    print(f"  ↓ Layer 3: W3(10×64) × a2(64×1) + b3(10×1) → z3(10×1) → Softmax → P(class)")
    
    print(f"\nParameter Count:")
    print(f"  Layer 1: {W1.size + b1.size} ({W1.size} + {b1.size})")
    print(f"  Layer 2: {W2.size + b2.size} ({W2.size} + {b2.size})")
    print(f"  Layer 3: {W3.size + b3.size} ({W3.size} + {b3.size})")
    print(f"  Total: {total_params} parameters")
    
    print(f"\nMemory Requirements (float32):")
    print(f"  Total parameters: {total_params * 4} bytes = {total_params * 4 / 1024:.1f} KB")
    
    return {
        'weights': [W1, W2, W3],
        'biases': [b1, b2, b3],
        'activations': [x, a1, a2, z3],
        'predicted_class': predicted_class,
        'confidence': confidence
    }

def demonstrate_simple_example():
    """
    Demonstrate with very small matrices for easy understanding
    """
    print("\n" + "="*80)
    print("SIMPLE EXAMPLE: TINY NETWORK FOR EASY UNDERSTANDING")
    print("="*80)
    print("Architecture: Input(4) → Hidden(3) → Output(2)")
    print("="*80)
    
    # Tiny input vector (4 features)
    x = np.array([[0.5], [0.8], [0.2], [0.9]])
    print_matrix_info(x, "Input Vector x", "4 input features")
    
    # Layer 1: 4 inputs → 3 hidden neurons
    W1 = np.array([
        [0.1, 0.2, 0.3, 0.4],  # Weights for hidden neuron 1
        [0.5, 0.6, 0.7, 0.8],  # Weights for hidden neuron 2
        [0.9, 1.0, 1.1, 1.2]   # Weights for hidden neuron 3
    ])
    b1 = np.array([[0.1], [0.2], [0.3]])
    
    print_matrix_info(W1, "Weight Matrix W1", "Maps 4 inputs to 3 hidden neurons")
    print_matrix_info(b1, "Bias Vector b1")
    
    print(f"\nStep-by-step matrix multiplication:")
    print(f"W1 @ x:")
    print(f"[0.1 0.2 0.3 0.4]   [0.5]   [0.1×0.5 + 0.2×0.8 + 0.3×0.2 + 0.4×0.9]   [{np.dot(W1[0], x.flatten()):.2f}]")
    print(f"[0.5 0.6 0.7 0.8] × [0.8] = [0.5×0.5 + 0.6×0.8 + 0.7×0.2 + 0.8×0.9] = [{np.dot(W1[1], x.flatten()):.2f}]")
    print(f"[0.9 1.0 1.1 1.2]   [0.2]   [0.9×0.5 + 1.0×0.8 + 1.1×0.2 + 1.2×0.9]   [{np.dot(W1[2], x.flatten()):.2f}]")
    print(f"                    [0.9]")
    
    z1 = W1 @ x + b1
    print_matrix_info(z1, "Linear Output z1 = W1@x + b1")
    
    a1 = np.maximum(0, z1)  # ReLU
    print_matrix_info(a1, "After ReLU a1 = max(0, z1)")
    
    # Layer 2: 3 hidden → 2 outputs
    W2 = np.array([
        [0.2, 0.4, 0.6],  # Weights for output 1
        [0.8, 1.0, 1.2]   # Weights for output 2
    ])
    b2 = np.array([[0.1], [0.2]])
    
    print_matrix_info(W2, "Weight Matrix W2", "Maps 3 hidden to 2 outputs")
    
    z2 = W2 @ a1 + b2
    print_matrix_info(z2, "Final Output z2 = W2@a1 + b2")
    
    # Softmax for interpretation
    exp_z2 = np.exp(z2)
    softmax = exp_z2 / np.sum(exp_z2)
    print_matrix_info(softmax, "Softmax Probabilities")
    
    print(f"\nFinal prediction: Class {np.argmax(z2)} with probability {np.max(softmax):.4f}")

def demonstrate_matrix_multiplication_rules():
    """
    Explain the fundamental rules of matrix multiplication for neural networks
    """
    print("\n" + "="*80)
    print("MATRIX MULTIPLICATION RULES FOR NEURAL NETWORKS")
    print("="*80)
    
    print("FUNDAMENTAL RULE:")
    print("For matrix multiplication A @ B = C to be valid:")
    print("• A must have shape (m, n)")
    print("• B must have shape (n, p)  ← Note: n must match!")
    print("• Result C will have shape (m, p)")
    print()
    
    print("NEURAL NETWORK APPLICATION:")
    print("Weight @ Input + Bias = Output")
    print("(output_size × input_size) @ (input_size × 1) + (output_size × 1) = (output_size × 1)")
    print()
    
    # Examples with different sizes
    examples = [
        {"layer": "Layer 1", "W_shape": (128, 784), "x_shape": (784, 1), "result": (128, 1)},
        {"layer": "Layer 2", "W_shape": (64, 128), "x_shape": (128, 1), "result": (64, 1)},
        {"layer": "Layer 3", "W_shape": (10, 64), "x_shape": (64, 1), "result": (10, 1)},
    ]
    
    for example in examples:
        W_shape = example["W_shape"]
        x_shape = example["x_shape"]
        result_shape = example["result"]
        
        print(f"{example['layer']}:")
        print(f"  Weight matrix: {W_shape}")
        print(f"  Input vector:  {x_shape}")
        print(f"  Output:        {result_shape}")
        print(f"  Rule check:    ({W_shape[0]}×{W_shape[1]}) @ ({x_shape[0]}×{x_shape[1]}) = ({result_shape[0]}×{result_shape[1]}) ✓")
        print()
    
    print("WHY THESE SHAPES WORK:")
    print("• Weight matrix rows = number of neurons in current layer")
    print("• Weight matrix columns = number of neurons in previous layer")
    print("• Input vector rows = number of neurons in previous layer")
    print("• Output vector rows = number of neurons in current layer")
    print("• The 'matching dimension' ensures proper connections between layers")

if __name__ == "__main__":
    # Run all demonstrations
    demonstrate_matrix_multiplication_rules()
    demonstrate_simple_example()
    results = demonstrate_3_layer_network()
    
    print("\n" + "="*80)
    print("MATRIX TRANSFORMATION GUIDE COMPLETE")
    print("="*80)
    print("This guide showed how a 28×28 image gets transformed through")
    print("multiple layers to produce a final classification decision.")
    print("Each layer reduces the dimensionality while extracting")
    print("increasingly abstract features from the input data.")