#!/usr/bin/env python3
"""
LogSoftMax and Cross-Entropy Loss: Complete Guide for Digit Classification

This comprehensive guide explains the mathematical foundations, practical implementation,
and intuitive understanding of LogSoftMax and Cross-Entropy Loss in the context of
our MNIST digit classifier.

Mathematical Pipeline:
1. Neural Network → Raw Logits (scores)
2. LogSoftMax → Log-Probabilities  
3. Cross-Entropy Loss → Single Loss Value
4. Autograd → Gradients
5. Optimizer → Parameter Updates

Author: AI Assistant
Date: September 2025
"""

import numpy as np
from pathlib import Path
import json

def explain_softmax_fundamentals():
    """
    Deep dive into SoftMax fundamentals with digit classification examples
    """
    print("=" * 80)
    print("SOFTMAX AND LOGSOFTMAX: MATHEMATICAL FOUNDATIONS")
    print("=" * 80)
    
    print("🎯 THE PROBLEM: CONVERTING SCORES TO PROBABILITIES")
    print("=" * 60)
    print("Our neural network outputs 10 raw scores (logits) for digits 0-9.")
    print("We need to convert these scores into probabilities that:")
    print("  1. Are all positive (between 0 and 1)")
    print("  2. Sum to exactly 1.0")
    print("  3. Preserve the relative ordering of scores")
    print()
    
    # Example with real-looking digit classifier outputs
    print("Example: Network output for a handwritten '7'")
    print("-" * 50)
    
    # Realistic logits where '7' has highest score
    logits = np.array([
        -2.1,  # digit 0
        -0.8,  # digit 1  
        -1.5,  # digit 2
        -0.3,  # digit 3
         0.2,  # digit 4
        -1.8,  # digit 5
        -0.9,  # digit 6
         3.7,  # digit 7 (highest - correct!)
        -1.2,  # digit 8
         0.8   # digit 9
    ])
    
    print("Raw logits (network outputs):")
    for i, logit in enumerate(logits):
        marker = " ← highest!" if i == 7 else ""
        print(f"  Digit {i}: {logit:6.1f}{marker}")
    print()
    
    print("🧮 SOFTMAX TRANSFORMATION")
    print("=" * 60)
    print("SoftMax formula: P(class_i) = exp(logit_i) / Σ_j exp(logit_j)")
    print()
    
    # Calculate softmax step by step
    print("Step 1: Calculate exponentials")
    exp_logits = np.exp(logits)
    for i, (logit, exp_val) in enumerate(zip(logits, exp_logits)):
        print(f"  exp({logit:5.1f}) = {exp_val:8.2f}")
    print()
    
    print("Step 2: Sum all exponentials")
    sum_exp = np.sum(exp_logits)
    print(f"  Sum = {sum_exp:.2f}")
    print()
    
    print("Step 3: Normalize to get probabilities")
    probabilities = exp_logits / sum_exp
    for i, (exp_val, prob) in enumerate(zip(exp_logits, probabilities)):
        percentage = prob * 100
        marker = " ← most likely!" if i == 7 else ""
        print(f"  P(digit {i}) = {exp_val:8.2f} / {sum_exp:.2f} = {prob:.6f} ({percentage:5.2f}%){marker}")
    print()
    
    print("Verification:")
    print(f"  Sum of probabilities: {np.sum(probabilities):.10f} ✓")
    print(f"  All positive: {np.all(probabilities >= 0)} ✓")
    print(f"  Highest probability: digit {np.argmax(probabilities)} ✓")
    
    return logits, probabilities

def explain_logsoftmax_advantages():
    """
    Explain why we use LogSoftMax instead of SoftMax
    """
    print("\n" + "=" * 80)
    print("WHY LOGSOFTMAX? NUMERICAL STABILITY AND EFFICIENCY")
    print("=" * 80)
    
    print("🚫 PROBLEMS WITH REGULAR SOFTMAX")
    print("=" * 60)
    
    # Demonstrate numerical instability
    print("Example: Large logits cause numerical overflow")
    large_logits = np.array([50.0, 51.0, 52.0])
    print(f"Large logits: {large_logits}")
    
    try:
        large_exp = np.exp(large_logits)
        print(f"Exponentials: {large_exp}")
        print("Problem: Numbers too large to represent!")
    except:
        print("Error: Numerical overflow!")
    print()
    
    print("✅ LOGSOFTMAX SOLUTION")
    print("=" * 60)
    print("LogSoftMax formula: log_softmax(x_i) = log(softmax(x_i))")
    print("                                    = x_i - log(Σ_j exp(x_j))")
    print()
    print("Advantages:")
    print("  1. Numerically stable (no exponential overflow)")
    print("  2. More efficient computation")
    print("  3. Better gradient properties")
    print("  4. Natural for Cross-Entropy Loss")
    print()
    
    # Use previous example
    logits = np.array([-2.1, -0.8, -1.5, -0.3, 0.2, -1.8, -0.9, 3.7, -1.2, 0.8])
    
    print("Computing LogSoftMax step by step:")
    print("-" * 40)
    
    # Numerically stable computation
    print("Step 1: Subtract max for numerical stability")
    logits_shifted = logits - np.max(logits)
    print(f"  Original max: {np.max(logits):.1f}")
    print(f"  Shifted logits: {logits_shifted}")
    print()
    
    print("Step 2: Compute log-sum-exp")
    log_sum_exp = np.log(np.sum(np.exp(logits_shifted)))
    print(f"  log(Σ exp(shifted)) = {log_sum_exp:.6f}")
    print()
    
    print("Step 3: Subtract from each shifted logit")
    log_probabilities = logits_shifted - log_sum_exp
    for i, (orig, log_prob) in enumerate(zip(logits, log_probabilities)):
        prob = np.exp(log_prob)
        marker = " ← highest!" if i == 7 else ""
        print(f"  log_softmax(digit {i}) = {log_prob:8.4f} (prob = {prob:.6f}){marker}")
    print()
    
    # Verify it matches regular softmax
    regular_softmax = np.exp(logits) / np.sum(np.exp(logits))
    log_of_softmax = np.log(regular_softmax)
    
    print("Verification: LogSoftMax = log(SoftMax)")
    print(f"  Max difference: {np.max(np.abs(log_probabilities - log_of_softmax)):.10f} ✓")
    
    return log_probabilities

def explain_cross_entropy_loss():
    """
    Deep dive into Cross-Entropy Loss with digit classification
    """
    print("\n" + "=" * 80)
    print("CROSS-ENTROPY LOSS: FROM PROBABILITIES TO LOSS")
    print("=" * 80)
    
    print("🎯 THE GOAL: MEASURING PREDICTION QUALITY")
    print("=" * 60)
    print("Cross-Entropy Loss measures how far our predictions are from the truth.")
    print("Formula: L = -Σ_i y_i * log(p_i)")
    print("  where y_i = 1 for correct class, 0 otherwise")
    print("        p_i = predicted probability for class i")
    print()
    
    print("For classification (one-hot encoding):")
    print("  L = -log(p_correct_class)")
    print("  We only care about the probability of the correct class!")
    print()
    
    # Use previous example where true digit is 7
    true_digit = 7
    log_probs = np.array([-5.8, -4.5, -5.2, -4.0, -3.5, -5.5, -4.6, 
                          0.0, -4.9, -2.9])  # Simplified from previous calculation
    
    print("Example: Handwritten '7' classification")
    print("-" * 50)
    print(f"True digit: {true_digit}")
    print("One-hot encoding: [0, 0, 0, 0, 0, 0, 0, 1, 0, 0]")
    print()
    
    print("Log-probabilities from LogSoftMax:")
    for i, log_prob in enumerate(log_probs):
        prob = np.exp(log_prob)
        marker = " ← TRUE CLASS" if i == true_digit else ""
        print(f"  Digit {i}: log_prob = {log_prob:6.3f}, prob = {prob:.6f}{marker}")
    print()
    
    print("Cross-Entropy Loss calculation:")
    print("-" * 40)
    print("L = -log(p_correct_class)")
    print(f"L = -log_softmax[{true_digit}]")
    print(f"L = -({log_probs[true_digit]:.6f})")
    loss = -log_probs[true_digit]
    print(f"L = {loss:.6f}")
    print()
    
    print("🧠 INTUITION: WHAT DOES THIS LOSS MEAN?")
    print("=" * 60)
    print("Cross-Entropy Loss interpretation:")
    correct_prob = np.exp(log_probs[true_digit])
    print(f"  Model assigns {correct_prob:.2%} probability to correct digit")
    print(f"  Loss = {loss:.3f}")
    print()
    
    # Show how loss changes with confidence
    print("How loss changes with model confidence:")
    confidence_levels = [0.1, 0.3, 0.5, 0.7, 0.9, 0.99, 0.999]
    for conf in confidence_levels:
        ce_loss = -np.log(conf)
        print(f"  Confidence {conf:5.1%} → Loss = {ce_loss:.3f}")
    print()
    
    print("Key insights:")
    print("  • Perfect confidence (prob=1.0) → Loss = 0")
    print("  • Low confidence → High loss")
    print("  • Loss grows quickly as confidence approaches 0")
    print("  • Loss is always positive (for prob < 1)")

def demonstrate_gradient_flow():
    """
    Show how gradients flow through LogSoftMax and Cross-Entropy
    """
    print("\n" + "=" * 80)
    print("GRADIENT FLOW: CONNECTING TO AUTOMATIC DIFFERENTIATION")
    print("=" * 80)
    
    print("🔄 THE COMPLETE PIPELINE")
    print("=" * 60)
    print("1. Neural Network → Logits (raw scores)")
    print("2. LogSoftMax → Log-probabilities")
    print("3. Cross-Entropy → Loss value")
    print("4. Autograd → Gradients ∂L/∂weights")
    print("5. Optimizer → Update weights")
    print()
    
    print("📊 BEAUTIFUL MATHEMATICAL PROPERTY")
    print("=" * 60)
    print("The gradient of Cross-Entropy + LogSoftMax has a simple form:")
    print()
    print("∂L/∂logit_i = p_i - y_i")
    print()
    print("Where:")
    print("  p_i = softmax probability for class i")
    print("  y_i = 1 if i is correct class, 0 otherwise")
    print()
    
    # Demonstrate with example
    logits = np.array([-2.1, -0.8, -1.5, -0.3, 0.2, -1.8, -0.9, 3.7, -1.2, 0.8])
    true_class = 7
    
    # Calculate softmax probabilities
    exp_logits = np.exp(logits - np.max(logits))  # Numerical stability
    probs = exp_logits / np.sum(exp_logits)
    
    # One-hot encoding
    one_hot = np.zeros(10)
    one_hot[true_class] = 1
    
    # Gradient
    gradients = probs - one_hot
    
    print("Example gradient calculation:")
    print("-" * 40)
    print("True class: 7")
    print()
    print("Gradients ∂L/∂logit_i = p_i - y_i:")
    for i, (prob, y, grad) in enumerate(zip(probs, one_hot, gradients)):
        marker = " ← TRUE CLASS" if i == true_class else ""
        print(f"  Digit {i}: {prob:.6f} - {y:.0f} = {grad:8.6f}{marker}")
    print()
    
    print("Gradient insights:")
    print(f"  • Correct class gradient: {gradients[true_class]:.6f} (negative, reduces logit)")
    print(f"  • Wrong class gradients: positive (reduce their logits)")
    print(f"  • Gradient magnitude ∝ confidence error")
    print(f"  • Sum of gradients: {np.sum(gradients):.10f} (≈ 0)")

def implement_complete_example():
    """
    Complete working example with real digit classification
    """
    print("\n" + "=" * 80)
    print("COMPLETE DIGIT CLASSIFICATION EXAMPLE")
    print("=" * 80)
    
    print("🏗️ SIMULATING COMPLETE PIPELINE")
    print("=" * 60)
    
    # Simulate a simple 3-layer network forward pass
    print("1. Network Architecture: 784 → 128 → 64 → 10")
    
    # Simulate input (flattened 28x28 image)
    np.random.seed(42)  # Reproducible results
    image_input = np.random.rand(784, 1) * 0.5  # Normalized pixel values
    
    print("2. Forward Pass:")
    print("   Input: 28x28 image flattened to (784, 1)")
    
    # Layer 1: 784 → 128
    W1 = np.random.normal(0, 0.1, (128, 784))
    b1 = np.zeros((128, 1))
    z1 = W1 @ image_input + b1
    a1 = np.maximum(0, z1)  # ReLU
    print(f"   Layer 1: (784,1) → (128,1), {np.sum(a1 > 0)}/128 neurons active")
    
    # Layer 2: 128 → 64  
    W2 = np.random.normal(0, 0.1, (64, 128))
    b2 = np.zeros((64, 1))
    z2 = W2 @ a1 + b2
    a2 = np.maximum(0, z2)  # ReLU
    print(f"   Layer 2: (128,1) → (64,1), {np.sum(a2 > 0)}/64 neurons active")
    
    # Layer 3: 64 → 10
    W3 = np.random.normal(0, 0.1, (10, 64))
    b3 = np.zeros((10, 1))
    logits = W3 @ a2 + b3  # Raw scores
    print(f"   Layer 3: (64,1) → (10,1), final logits")
    print()
    
    # Display logits
    print("3. Raw Logits (network outputs):")
    logits_flat = logits.flatten()
    for i, logit in enumerate(logits_flat):
        print(f"   Digit {i}: {logit:8.4f}")
    print(f"   Predicted digit: {np.argmax(logits_flat)}")
    print()
    
    # Apply LogSoftMax
    print("4. LogSoftMax Application:")
    log_probs = logits_flat - np.log(np.sum(np.exp(logits_flat)))
    probs = np.exp(log_probs)
    
    print("   Log-probabilities and probabilities:")
    for i, (log_prob, prob) in enumerate(zip(log_probs, probs)):
        confidence = prob * 100
        print(f"   Digit {i}: log_prob={log_prob:8.4f}, prob={prob:.6f} ({confidence:5.2f}%)")
    print()
    
    # Simulate true label and compute loss
    true_digit = 3  # Simulate this image is actually a '3'
    print(f"5. Ground Truth: This image is digit {true_digit}")
    print()
    
    # Cross-Entropy Loss
    print("6. Cross-Entropy Loss:")
    ce_loss = -log_probs[true_digit]
    print(f"   L = -log_softmax[{true_digit}] = -({log_probs[true_digit]:.6f}) = {ce_loss:.6f}")
    print(f"   Model confidence in correct answer: {probs[true_digit]:.2%}")
    print()
    
    # Gradients
    print("7. Gradient Computation:")
    one_hot = np.zeros(10)
    one_hot[true_digit] = 1
    gradients = probs - one_hot
    
    print("   ∂L/∂logit_i = softmax_i - one_hot_i:")
    for i, grad in enumerate(gradients):
        marker = " ← TRUE" if i == true_digit else ""
        direction = "↓" if grad < 0 else "↑"
        print(f"   ∂L/∂logit[{i}] = {grad:8.6f} {direction}{marker}")
    print()
    
    print("8. Training Step (conceptual):")
    learning_rate = 0.01
    print(f"   With learning_rate = {learning_rate}:")
    print(f"   logit[{true_digit}] += {-learning_rate * gradients[true_digit]:.6f} (increase correct)")
    for i, grad in enumerate(gradients):
        if i != true_digit and abs(grad) > 0.01:
            print(f"   logit[{i}] += {-learning_rate * grad:.6f} (decrease wrong)")
    
    return logits_flat, log_probs, ce_loss, gradients

def explain_information_theory_connection():
    """
    Connect Cross-Entropy to information theory concepts
    """
    print("\n" + "=" * 80)
    print("INFORMATION THEORY: WHY CROSS-ENTROPY MAKES SENSE")
    print("=" * 80)
    
    print("🧠 INFORMATION CONTENT")
    print("=" * 60)
    print("Information theory tells us:")
    print("  • Rare events carry more information")
    print("  • Information content I(x) = -log(P(x))")
    print("  • More surprising → More information")
    print()
    
    print("Examples from digit classification:")
    probabilities = [0.9, 0.5, 0.1, 0.01, 0.001]
    for prob in probabilities:
        info_content = -np.log(prob)
        print(f"  P(correct) = {prob:5.1%} → Information = {info_content:.2f} bits")
    print()
    
    print("📏 CROSS-ENTROPY AS EXPECTED INFORMATION")
    print("=" * 60)
    print("Cross-Entropy Loss = Expected information content")
    print("  • If model is confident and correct → Low loss")
    print("  • If model is confident but wrong → High loss")
    print("  • If model is uncertain → Medium loss")
    print()
    
    print("This connects to:")
    print("  • Maximum Likelihood Estimation")
    print("  • Kullback-Leibler Divergence")
    print("  • Minimizing 'surprise' in predictions")

def show_common_mistakes_and_tips():
    """
    Common implementation mistakes and best practices
    """
    print("\n" + "=" * 80)
    print("COMMON MISTAKES AND BEST PRACTICES")
    print("=" * 80)
    
    print("❌ MISTAKE 1: Numerical instability")
    print("=" * 60)
    print("Wrong:")
    print("  softmax = exp(logits) / sum(exp(logits))  # Can overflow!")
    print()
    print("Right:")
    print("  max_logit = max(logits)")
    print("  softmax = exp(logits - max_logit) / sum(exp(logits - max_logit))")
    print()
    
    print("❌ MISTAKE 2: Computing softmax then taking log")
    print("=" * 60)
    print("Inefficient:")
    print("  softmax = compute_softmax(logits)")
    print("  log_softmax = log(softmax)  # Numerical errors!")
    print()
    print("Better:")
    print("  log_softmax = logits - log_sum_exp(logits)")
    print()
    
    print("❌ MISTAKE 3: Wrong gradient shapes")
    print("=" * 60)
    print("Remember:")
    print("  • Logits shape: (batch_size, num_classes)")
    print("  • Labels shape: (batch_size,) or (batch_size, num_classes)")
    print("  • Loss shape: scalar or (batch_size,)")
    print()
    
    print("✅ BEST PRACTICES")
    print("=" * 60)
    print("1. Use library functions (torch.nn.LogSoftmax, torch.nn.CrossEntropyLoss)")
    print("2. Always check tensor shapes")
    print("3. Use numerical stability tricks")
    print("4. Verify gradients with finite differences")
    print("5. Monitor loss curves during training")

if __name__ == "__main__":
    print("LOGSOFTMAX AND CROSS-ENTROPY LOSS: COMPLETE GUIDE")
    print("Understanding the heart of neural network classification")
    print()
    
    # Run all explanations
    logits, probs = explain_softmax_fundamentals()
    log_probs = explain_logsoftmax_advantages()
    explain_cross_entropy_loss()
    demonstrate_gradient_flow()
    final_logits, final_log_probs, loss, grads = implement_complete_example()
    explain_information_theory_connection()
    show_common_mistakes_and_tips()
    
    print("\n" + "=" * 80)
    print("SUMMARY: THE COMPLETE PICTURE")
    print("=" * 80)
    print("Neural Network Pipeline for Digit Classification:")
    print("1. 🖼️  Image (28×28) → Flatten → (784,1)")
    print("2. 🧠 Neural Network → Raw logits (10 scores)")
    print("3. 📊 LogSoftMax → Log-probabilities (sum to 1)")
    print("4. 🎯 Cross-Entropy → Loss value (lower = better)")
    print("5. ∇  Autograd → Gradients ∂L/∂weights")
    print("6. ⚡ Optimizer → Update weights")
    print("7. 🔄 Repeat until convergence")
    print()
    print("Mathematical beauty: ∂L/∂logit_i = softmax_i - one_hot_i")
    print("This simple gradient drives all learning in classification!")