#!/usr/bin/env python3
"""
Neural Network Example

This script demonstrates building and training a simple neural network
using PyTensorLib for the XOR problem.
"""

import sys
import os

# Add src directory to path for development
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from pytensorlib import Tensor, relu, sigmoid, mse_loss
import numpy as np


class SimpleNeuralNetwork:
    """Simple 2-layer neural network"""
    
    def __init__(self, input_size=2, hidden_size=4, output_size=1):
        """Initialize network with random weights"""
        # Initialize weights with small random values
        self.w1 = Tensor(np.random.randn(input_size, hidden_size) * 0.5, requires_grad=True)
        self.b1 = Tensor(np.zeros((1, hidden_size)), requires_grad=True)
        
        self.w2 = Tensor(np.random.randn(hidden_size, output_size) * 0.5, requires_grad=True)
        self.b2 = Tensor(np.zeros((1, output_size)), requires_grad=True)
    
    def forward(self, x):
        """Forward pass through the network"""
        # First layer: linear + ReLU
        h = relu(x @ self.w1 + self.b1)
        
        # Second layer: linear + sigmoid
        output = sigmoid(h @ self.w2 + self.b2)
        
        return output
    
    def parameters(self):
        """Return all trainable parameters"""
        return [self.w1, self.b1, self.w2, self.b2]
    
    def zero_grad(self):
        """Clear all gradients"""
        for param in self.parameters():
            param.grad = None


def create_xor_dataset():
    """Create XOR dataset"""
    # XOR truth table
    X = np.array([
        [0, 0],
        [0, 1], 
        [1, 0],
        [1, 1]
    ], dtype=np.float32)
    
    y = np.array([
        [0],  # 0 XOR 0 = 0
        [1],  # 0 XOR 1 = 1
        [1],  # 1 XOR 0 = 1
        [0]   # 1 XOR 1 = 0
    ], dtype=np.float32)
    
    return Tensor(X), Tensor(y)


def train_network():
    """Train neural network on XOR problem"""
    print("🧠 Neural Network Training on XOR Problem")
    print("=" * 50)
    
    # Create dataset
    X, y = create_xor_dataset()
    print("Dataset:")
    print("  Input (X)  | Target (y)")
    print("  -----------|-----------")
    for i in range(len(X.data)):
        print(f"  {X.data[i]}   |     {y.data[i][0]}")
    
    # Create network
    network = SimpleNeuralNetwork(input_size=2, hidden_size=8, output_size=1)
    
    # Training parameters
    learning_rate = 0.5
    epochs = 2000
    
    print(f"\nTraining parameters:")
    print(f"  Learning rate: {learning_rate}")
    print(f"  Epochs: {epochs}")
    print(f"  Hidden units: 8")
    
    print(f"\nTraining progress:")
    print("Epoch | Loss    | Accuracy")
    print("------|---------|----------")
    
    # Training loop
    losses = []
    for epoch in range(epochs):
        # Forward pass
        predictions = network.forward(X)
        
        # Compute loss
        loss = mse_loss(predictions, y)
        losses.append(loss.data)
        
        # Backward pass
        loss.backward()
        
        # Update parameters
        for param in network.parameters():
            param.data -= learning_rate * param.grad
        
        # Clear gradients
        network.zero_grad()
        
        # Print progress
        if epoch % 200 == 0 or epoch == epochs - 1:
            # Compute accuracy
            pred_binary = (predictions.data > 0.5).astype(int)
            accuracy = (pred_binary == y.data).mean()
            
            print(f"{epoch:5d} | {loss.data:.5f} | {accuracy:.3f}")
    
    return network, X, y, losses


def test_network(network, X, y):
    """Test the trained network"""
    print(f"\n🎯 Testing Trained Network")
    print("=" * 30)
    
    # Get predictions
    predictions = network.forward(X)
    
    print("Input | Target | Prediction | Binary")
    print("------|--------|------------|-------")
    
    correct = 0
    for i in range(len(X.data)):
        pred_val = predictions.data[i][0]
        pred_binary = int(pred_val > 0.5)
        target = int(y.data[i][0])
        
        if pred_binary == target:
            correct += 1
            status = "✓"
        else:
            status = "✗"
        
        print(f"{X.data[i]} |   {target}    |   {pred_val:.4f}   |   {pred_binary} {status}")
    
    accuracy = correct / len(X.data)
    print(f"\nFinal Accuracy: {accuracy:.3f} ({correct}/{len(X.data)})")
    
    return accuracy


def gradient_analysis(network, X, y):
    """Analyze gradients during training"""
    print(f"\n📊 Gradient Analysis")
    print("=" * 25)
    
    # Single forward-backward pass
    predictions = network.forward(X)
    loss = mse_loss(predictions, y)
    loss.backward()
    
    print("Parameter gradients:")
    print(f"  W1 gradient norm: {np.linalg.norm(network.w1.grad):.6f}")
    print(f"  b1 gradient norm: {np.linalg.norm(network.b1.grad):.6f}")
    print(f"  W2 gradient norm: {np.linalg.norm(network.w2.grad):.6f}")
    print(f"  b2 gradient norm: {np.linalg.norm(network.b2.grad):.6f}")
    
    # Check for vanishing/exploding gradients
    max_grad = max(
        np.max(np.abs(network.w1.grad)),
        np.max(np.abs(network.b1.grad)),
        np.max(np.abs(network.w2.grad)),
        np.max(np.abs(network.b2.grad))
    )
    
    print(f"  Maximum gradient magnitude: {max_grad:.6f}")
    
    if max_grad < 1e-6:
        print("  ⚠️  Warning: Very small gradients (vanishing gradient problem)")
    elif max_grad > 10:
        print("  ⚠️  Warning: Very large gradients (exploding gradient problem)")
    else:
        print("  ✓ Gradients are in a reasonable range")


def visualize_decision_boundary(network):
    """Visualize the decision boundary learned by the network"""
    print(f"\n🎨 Decision Boundary Visualization")
    print("=" * 40)
    
    print("Testing decision boundary on a grid:")
    print("  0.0  0.2  0.4  0.6  0.8  1.0")
    
    for y_val in [1.0, 0.8, 0.6, 0.4, 0.2, 0.0]:
        row = f"{y_val:.1f} "
        for x_val in [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]:
            test_input = Tensor([[x_val, y_val]])
            pred = network.forward(test_input)
            
            if pred.data[0][0] > 0.5:
                row += " ■ "
            else:
                row += " □ "
        
        print(row)
    
    print("\nLegend: ■ = Class 1 (output > 0.5), □ = Class 0 (output ≤ 0.5)")


def main():
    """Main execution function"""
    print("🎯 PyTensorLib Neural Network Example")
    print("=" * 60)
    
    try:
        # Set random seed for reproducibility
        np.random.seed(42)
        
        # Train the network
        network, X, y, losses = train_network()
        
        # Test the network
        accuracy = test_network(network, X, y)
        
        # Analyze gradients
        gradient_analysis(network, X, y)
        
        # Visualize decision boundary
        visualize_decision_boundary(network)
        
        # Summary
        print(f"\n🎉 Training Summary")
        print("=" * 20)
        print(f"Final loss: {losses[-1]:.6f}")
        print(f"Final accuracy: {accuracy:.3f}")
        print(f"Training converged: {'Yes' if accuracy >= 0.75 else 'No'}")
        
        if accuracy >= 1.0:
            print("🌟 Perfect classification achieved!")
        elif accuracy >= 0.75:
            print("✅ Good classification performance!")
        else:
            print("⚠️  Classification needs improvement. Try:")
            print("   - Increase learning rate")
            print("   - Train for more epochs")
            print("   - Add more hidden units")
        
        return 0
        
    except Exception as e:
        print(f"\n❌ Error during training: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)