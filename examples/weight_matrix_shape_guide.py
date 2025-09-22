#!/usr/bin/env python3
"""
Weight Matrix Shape Calculation: Complete Guide

This guide provides a comprehensive explanation of how weight matrix shapes
are calculated and why they have specific dimensions in neural networks,
based on matrix multiplication rules and the autograd blog insights.
"""

import numpy as np

def explain_weight_matrix_shape_calculation():
    """
    Complete explanation of how weight matrix shapes are determined
    """
    print("=" * 80)
    print("WEIGHT MATRIX SHAPE CALCULATION: THE FUNDAMENTAL RULES")
    print("=" * 80)
    
    print("🎯 THE CORE PRINCIPLE")
    print("=" * 50)
    print("Weight matrix shapes are determined by ONE fundamental rule:")
    print("  Matrix multiplication: (A × B) requires A.cols = B.rows")
    print("  For neural networks: (Weight × Input) = Output")
    print("  Therefore: Weight.shape = (output_size, input_size)")
    print()
    
    print("📐 MATHEMATICAL FOUNDATION")
    print("=" * 50)
    print("General matrix multiplication rule:")
    print("  Matrix A: shape (m, n)")
    print("  Matrix B: shape (n, p)  ← n MUST match A's columns")
    print("  Result C: shape (m, p)")
    print()
    print("Applied to neural networks:")
    print("  Weight W: shape (neurons_out, neurons_in)")
    print("  Input X:  shape (neurons_in, 1)")
    print("  Output Y: shape (neurons_out, 1)")
    print("  Operation: Y = W @ X")

def demonstrate_shape_calculation_step_by_step():
    """
    Step-by-step demonstration of how shapes are calculated
    """
    print("\n" + "=" * 80)
    print("STEP-BY-STEP SHAPE CALCULATION")
    print("=" * 80)
    
    print("🏗️ BUILDING A 3-LAYER NETWORK")
    print("=" * 50)
    print("Target architecture: 784 → 128 → 64 → 10")
    print()
    
    # Layer 1 calculation
    print("LAYER 1: Input(784) → Hidden1(128)")
    print("-" * 40)
    print("We want to transform:")
    print("  Input:  784 features → Output: 128 neurons")
    print("  Input shape: (784, 1)")
    print("  Desired output shape: (128, 1)")
    print()
    print("Matrix multiplication requirement:")
    print("  W1 @ input = output")
    print("  W1 @ (784, 1) = (128, 1)")
    print("  (?, ?) @ (784, 1) = (128, 1)")
    print()
    print("Solving for W1 shape:")
    print("  For (a, b) @ (784, 1) = (128, 1)")
    print("  We need: b = 784 (to match input rows)")
    print("  We need: a = 128 (to produce output rows)")
    print("  Therefore: W1.shape = (128, 784) ✓")
    print()
    
    # Verify with actual computation
    input_vec = np.random.rand(784, 1)
    W1 = np.random.rand(128, 784)
    output1 = W1 @ input_vec
    print(f"Verification:")
    print(f"  Input: {input_vec.shape}")
    print(f"  W1: {W1.shape}")
    print(f"  Output: {output1.shape}")
    print(f"  Expected: (128, 1) ✓")
    print()
    
    # Layer 2 calculation
    print("LAYER 2: Hidden1(128) → Hidden2(64)")
    print("-" * 40)
    print("Now the input is the output from Layer 1:")
    print("  Input from Layer 1: (128, 1) [after ReLU]")
    print("  Desired output: (64, 1)")
    print()
    print("Matrix multiplication requirement:")
    print("  W2 @ hidden1_output = hidden2_output")
    print("  W2 @ (128, 1) = (64, 1)")
    print("  (?, ?) @ (128, 1) = (64, 1)")
    print()
    print("Solving for W2 shape:")
    print("  For (a, b) @ (128, 1) = (64, 1)")
    print("  We need: b = 128 (to match Layer 1 output rows)")
    print("  We need: a = 64 (to produce desired output rows)")
    print("  Therefore: W2.shape = (64, 128) ✓")
    print()
    
    # Verify Layer 2
    hidden1_output = np.maximum(0, output1)  # ReLU activation
    W2 = np.random.rand(64, 128)
    output2 = W2 @ hidden1_output
    print(f"Verification:")
    print(f"  Hidden1 output (after ReLU): {hidden1_output.shape}")
    print(f"  W2: {W2.shape}")
    print(f"  Output: {output2.shape}")
    print(f"  Expected: (64, 1) ✓")
    print()
    
    # Layer 3 calculation
    print("LAYER 3: Hidden2(64) → Output(10)")
    print("-" * 40)
    print("Final layer for classification:")
    print("  Input from Layer 2: (64, 1) [after ReLU]")
    print("  Desired output: (10, 1) [10 digit classes]")
    print()
    print("Matrix multiplication requirement:")
    print("  W3 @ hidden2_output = final_output")
    print("  W3 @ (64, 1) = (10, 1)")
    print("  (?, ?) @ (64, 1) = (10, 1)")
    print()
    print("Solving for W3 shape:")
    print("  For (a, b) @ (64, 1) = (10, 1)")
    print("  We need: b = 64 (to match Layer 2 output rows)")
    print("  We need: a = 10 (for 10 digit classes)")
    print("  Therefore: W3.shape = (10, 64) ✓")
    print()
    
    # Verify Layer 3
    hidden2_output = np.maximum(0, output2)  # ReLU activation
    W3 = np.random.rand(10, 64)
    final_output = W3 @ hidden2_output
    print(f"Verification:")
    print(f"  Hidden2 output (after ReLU): {hidden2_output.shape}")
    print(f"  W3: {W3.shape}")
    print(f"  Final output: {final_output.shape}")
    print(f"  Expected: (10, 1) ✓")

def explain_blog_post_insights():
    """
    Explain insights from the bert hubert autograd blog post
    """
    print("\n" + "=" * 80)
    print("INSIGHTS FROM AUTOGRAD BLOG POST")
    print("=" * 80)
    
    print("🔗 BLOG POST ARCHITECTURE ANALYSIS")
    print("=" * 50)
    print("The blog describes this exact network architecture:")
    print("1. Flatten 28x28 image to a 784x1 matrix")
    print("2. Multiply this matrix by a 128x784 matrix")
    print("3. Replace all negative elements by 0 (ReLU)")
    print("4. Multiply the resulting matrix by a 64x128 matrix")
    print("5. Replace all negative elements by 0 (ReLU)")
    print("6. Multiply the resulting matrix by a 10x64 matrix")
    print("7. Pick the highest row (argmax for classification)")
    print()
    
    print("📊 PARAMETER COUNT VERIFICATION")
    print("=" * 50)
    print("Blog states: 128*784 + 64*128 + 10*64 = 109,184 weights")
    print("Plus biases: 128 + 64 + 10 = 202 bias parameters")
    print()
    
    # Calculate and verify
    layer1_weights = 128 * 784
    layer2_weights = 64 * 128
    layer3_weights = 10 * 64
    total_weights = layer1_weights + layer2_weights + layer3_weights
    
    layer1_biases = 128
    layer2_biases = 64
    layer3_biases = 10
    total_biases = layer1_biases + layer2_biases + layer3_biases
    
    print(f"Verification:")
    print(f"  Layer 1 weights: 128 × 784 = {layer1_weights:,}")
    print(f"  Layer 2 weights: 64 × 128 = {layer2_weights:,}")
    print(f"  Layer 3 weights: 10 × 64 = {layer3_weights:,}")
    print(f"  Total weights: {total_weights:,} ✓")
    print()
    print(f"  Layer 1 biases: {layer1_biases}")
    print(f"  Layer 2 biases: {layer2_biases}")
    print(f"  Layer 3 biases: {layer3_biases}")
    print(f"  Total biases: {total_biases} ✓")
    print(f"  Grand total parameters: {total_weights + total_biases:,}")
    print()
    
    print("🧠 KEY INSIGHT: MATRIX DIMENSION MATCHING")
    print("=" * 50)
    print("The blog emphasizes that matrix multiplication requires dimension matching:")
    print("  'The connection between input image intensity and weight was clear'")
    print("  In multi-layer networks, this connection propagates through layers")
    print()
    print("Each layer's output becomes the next layer's input:")
    print("  Layer 1 output shape → Layer 2 input shape")
    print("  Layer 2 output shape → Layer 3 input shape")
    print("  This creates a CHAIN of shape requirements")

def demonstrate_common_mistakes():
    """
    Show common mistakes in weight matrix shape definition
    """
    print("\n" + "=" * 80)
    print("COMMON MISTAKES IN WEIGHT MATRIX SHAPES")
    print("=" * 80)
    
    print("❌ MISTAKE 1: Swapping dimensions")
    print("=" * 50)
    print("Wrong thinking: 'I have 784 inputs and want 128 outputs'")
    print("Incorrect shape: (784, 128)  # This is backwards!")
    print()
    print("Why this fails:")
    try:
        wrong_W = np.random.rand(784, 128)  # Wrong shape
        input_vec = np.random.rand(784, 1)
        # This will fail:
        # result = wrong_W @ input_vec  # Shape mismatch!
        print(f"  Attempted: {wrong_W.shape} @ {input_vec.shape}")
        print(f"  Matrix multiplication rule: (784, 128) @ (784, 1)")
        print(f"  Problem: 128 ≠ 784, dimensions don't match!")
    except Exception as e:
        print(f"  Error: {e}")
    print()
    
    print("✅ CORRECT approach:")
    print("Correct shape: (128, 784)")
    correct_W = np.random.rand(128, 784)
    input_vec = np.random.rand(784, 1)
    result = correct_W @ input_vec
    print(f"  Success: {correct_W.shape} @ {input_vec.shape} = {result.shape}")
    print()
    
    print("❌ MISTAKE 2: Forgetting the data flows")
    print("=" * 50)
    print("Wrong thinking: 'Layer 2 has 64 neurons, so W2 should be (64, 64)'")
    print("Problem: This ignores that Layer 2 receives input from Layer 1")
    print()
    print("Correct thinking:")
    print("  Layer 1 produces: 128 neurons → Layer 2 receives: 128 inputs")
    print("  Layer 2 produces: 64 neurons")
    print("  Therefore: W2 must be (64, 128) to map 128→64")
    print()
    
    print("❌ MISTAKE 3: Confusing rows and columns")
    print("=" * 50)
    print("Memory trick: Weight matrix shape = (WHERE we're going, WHERE we came from)")
    print("  (output_neurons, input_neurons)")
    print("  (destination, source)")
    print("  Think: 'How many outputs × How many inputs'")

def show_weight_matrix_interpretation():
    """
    Show how to interpret the values in weight matrices
    """
    print("\n" + "=" * 80)
    print("INTERPRETING WEIGHT MATRIX VALUES")
    print("=" * 80)
    
    print("🔍 WHAT EACH WEIGHT REPRESENTS")
    print("=" * 50)
    
    # Create a small example for illustration
    W1 = np.random.rand(3, 4)  # 4 inputs → 3 outputs
    
    print("Example: W1 with shape (3, 4) mapping 4 inputs to 3 outputs")
    print(f"Weight matrix W1:")
    print(f"{W1}")
    print()
    
    print("Interpretation:")
    print("  Row 0: [w00, w01, w02, w03] = weights from all 4 inputs TO output neuron 0")
    print("  Row 1: [w10, w11, w12, w13] = weights from all 4 inputs TO output neuron 1")
    print("  Row 2: [w20, w21, w22, w23] = weights from all 4 inputs TO output neuron 2")
    print()
    print("  Column 0: [w00, w10, w20] = how input 0 affects ALL output neurons")
    print("  Column 1: [w01, w11, w21] = how input 1 affects ALL output neurons")
    print("  Column 2: [w02, w12, w22] = how input 2 affects ALL output neurons")
    print("  Column 3: [w03, w13, w23] = how input 3 affects ALL output neurons")
    print()
    
    print("🧮 MATRIX MULTIPLICATION MECHANICS")
    print("=" * 50)
    print("When we compute Y = W @ X:")
    
    X = np.array([[1.0], [2.0], [3.0], [4.0]])  # 4×1 input vector
    Y = W1 @ X
    
    print(f"Input X: {X.flatten()}")
    print()
    print("Each output is computed as:")
    for i in range(3):
        weights_for_output_i = W1[i, :]
        computation = " + ".join([f"{W1[i,j]:.3f}×{X[j,0]:.1f}" for j in range(4)])
        result = np.dot(weights_for_output_i, X.flatten())
        print(f"  Output {i}: {computation} = {result:.3f}")
    print()
    print(f"Final output Y: {Y.flatten()}")

def practical_implementation_guide():
    """
    Practical guide for implementing weight matrices
    """
    print("\n" + "=" * 80)
    print("PRACTICAL IMPLEMENTATION GUIDE")
    print("=" * 80)
    
    print("🛠️ STEP-BY-STEP IMPLEMENTATION")
    print("=" * 50)
    
    print("1. DEFINE YOUR ARCHITECTURE")
    print("   Input size: 784 (28×28 flattened)")
    print("   Hidden1 size: 128")
    print("   Hidden2 size: 64")
    print("   Output size: 10 (digit classes)")
    print()
    
    print("2. CALCULATE WEIGHT SHAPES")
    architecture = [784, 128, 64, 10]
    weight_shapes = []
    
    for i in range(len(architecture) - 1):
        input_size = architecture[i]
        output_size = architecture[i + 1]
        shape = (output_size, input_size)
        weight_shapes.append(shape)
        print(f"   Layer {i+1}: {input_size} → {output_size}, Weight shape: {shape}")
    print()
    
    print("3. IMPLEMENT IN CODE")
    print("```python")
    print("import numpy as np")
    print()
    print("class NeuralNetwork:")
    print("    def __init__(self):")
    print("        # Layer 1: 784 → 128")
    print("        self.W1 = np.random.normal(0, 0.1, (128, 784))")
    print("        self.b1 = np.zeros((128, 1))")
    print("        ")
    print("        # Layer 2: 128 → 64")
    print("        self.W2 = np.random.normal(0, 0.1, (64, 128))")
    print("        self.b2 = np.zeros((64, 1))")
    print("        ")
    print("        # Layer 3: 64 → 10")
    print("        self.W3 = np.random.normal(0, 0.1, (10, 64))")
    print("        self.b3 = np.zeros((10, 1))")
    print("    ")
    print("    def forward(self, x):")
    print("        # x shape: (784, 1)")
    print("        z1 = self.W1 @ x + self.b1      # (128, 1)")
    print("        a1 = np.maximum(0, z1)          # ReLU")
    print("        ")
    print("        z2 = self.W2 @ a1 + self.b2     # (64, 1)")
    print("        a2 = np.maximum(0, z2)          # ReLU")
    print("        ")
    print("        z3 = self.W3 @ a2 + self.b3     # (10, 1)")
    print("        return z3")
    print("```")
    print()
    
    print("4. VERIFY YOUR SHAPES")
    print("   Always check that matrix multiplications work:")
    print("   - Print shapes during forward pass")
    print("   - Use assertions to verify expected shapes")
    print("   - Test with small dummy data first")

if __name__ == "__main__":
    explain_weight_matrix_shape_calculation()
    demonstrate_shape_calculation_step_by_step()
    explain_blog_post_insights()
    demonstrate_common_mistakes()
    show_weight_matrix_interpretation()
    practical_implementation_guide()
    
    print("\n" + "=" * 80)
    print("SUMMARY: THE GOLDEN RULE")
    print("=" * 80)
    print("Weight matrix shape = (neurons_out, neurons_in)")
    print("This ensures: Weight @ Input = Output")
    print("Where @ follows the mathematical rule: (m,n) @ (n,1) = (m,1)")
    print()
    print("Remember: The 'matching dimension' (n) ensures proper connections")
    print("between layers, enabling data to flow through the network correctly.")