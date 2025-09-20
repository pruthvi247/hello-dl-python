# PyTensorLib 🧠

A Deep Learning Framework in Python with Automatic Differentiation

[![Python Version](https://img.shields.io/badge/python-3.7%2B-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

PyTensorLib is a lightweight deep learning framework implemented in pure Python with NumPy. It provides automatic differentiation, tensor operations, and common neural network components.

## ✨ Features

- **Automatic Differentiation**: Full gradient computation for complex computational graphs
- **Tensor Operations**: Broadcasting, reshaping, mathematical operations
- **Activation Functions**: ReLU, Sigmoid, Tanh, GELU, and more
- **Loss Functions**: Mean Squared Error, Cross Entropy
- **MNIST Utilities**: Dataset loading and processing tools
- **Pure Python**: No external dependencies except NumPy
- **Educational**: Clean, readable code for learning deep learning concepts

## 🚀 Quick Start

### Installation

1. **Clone and set up the environment:**
   ✅ Complete setup - Handles everything automatically ✅ Virtual environment - Keeps dependencies isolated
   ✅ Testing - Verifies everything works ✅ Instructions - Tells you exactly what to do next ✅ Beginner-friendly - Guided process with clear output

Use setup.py only if:

You're already in a virtual environment
You want to install without the guided setup
You're packaging for distribution

```bash
git clone <repository-url>
cd hello-dl-python
python setup_environment.py
```

2. **Activate the virtual environment:**

```bash
# On macOS/Linux
source pytensor-env/bin/activate

# On Windows
pytensor-env\Scripts\activate.bat
```

3. **Download EMNIST dataset (recommended):**

```bash
python download_mnist.py
```

This attempts to download the EMNIST (Extended MNIST) dataset from NIST, which provides high-quality handwritten digit data. If EMNIST is not accessible, it automatically falls back to the standard MNIST dataset. The download is about 900MB and may take several minutes.

**Note:** EMNIST download may require institutional access. The script will automatically use MNIST or synthetic data as fallbacks.

4. **Verify installation:**

```bash
python -c "from pytensorlib import quick_test; quick_test()"
```

### Basic Usage

```python
from pytensorlib import Tensor, relu, sigmoid

# Create tensors
x = Tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
y = Tensor([[0.5, 1.5]], requires_grad=True)

# Operations with automatic differentiation
z = relu(x @ y.T)  # Matrix multiplication + ReLU
loss = z.sum()

# Compute gradients
loss.backward()

print(f"Loss: {loss.data}")
print(f"Gradient of x: {x.grad}")
```

### Neural Network Example

```python
from pytensorlib import Tensor, relu, mse_loss
import numpy as np

# Simple neural network for XOR problem
class SimpleNet:
    def __init__(self):
        self.w1 = Tensor(np.random.randn(2, 4) * 0.5, requires_grad=True)
        self.b1 = Tensor(np.zeros((1, 4)), requires_grad=True)
        self.w2 = Tensor(np.random.randn(4, 1) * 0.5, requires_grad=True)
        self.b2 = Tensor(np.zeros((1, 1)), requires_grad=True)

    def forward(self, x):
        h = relu(x @ self.w1 + self.b1)
        return h @ self.w2 + self.b2

    def parameters(self):
        return [self.w1, self.b1, self.w2, self.b2]

# Training data (XOR)
X = Tensor([[0, 0], [0, 1], [1, 0], [1, 1]], requires_grad=False)
y = Tensor([[0], [1], [1], [0]], requires_grad=False)

net = SimpleNet()
learning_rate = 0.1

# Training loop
for epoch in range(1000):
    # Forward pass
    pred = net.forward(X)
    loss = mse_loss(pred, y)

    # Backward pass
    loss.backward()

    # Update parameters
    for param in net.parameters():
        param.data -= learning_rate * param.grad
        param.grad = None  # Clear gradients

    if epoch % 100 == 0:
        print(f"Epoch {epoch}, Loss: {loss.data:.4f}")
```

### MNIST Data Processing

```python
from pytensorlib.mnist_utils import create_synthetic_mnist_data

# Generate synthetic MNIST-like data for testing
images, labels = create_synthetic_mnist_data(100)

print(f"Generated {len(images)} images")
print(f"Image shape: {images[0].shape}")
print(f"Unique labels: {set(labels)}")

# Use with PyTensorLib tensors
from pytensorlib import Tensor

# Convert to tensors for training
X = Tensor(images.reshape(100, -1))  # Flatten to 784 features
y = Tensor(labels)

print(f"Tensor X shape: {X.shape}")
print(f"Tensor y shape: {y.shape}")
```

## 📚 API Reference

### Core Classes

#### `Tensor`

Main tensor class with automatic differentiation support.

```python
Tensor(data, requires_grad=False, device=None)
```

**Methods:**

- `backward()` - Compute gradients
- `sum()`, `mean()` - Reduction operations
- `reshape(shape)` - Change tensor shape
- `transpose()` - Matrix transpose
- `@` - Matrix multiplication
- `+`, `-`, `*`, `/` - Element-wise operations

#### `TensorImpl`

Internal implementation class (not for direct use).

### Activation Functions

```python
from pytensorlib import relu, sigmoid, tanh, gelu

x = Tensor([[1.0, -2.0, 3.0]])
relu_out = relu(x)
sigmoid_out = sigmoid(x)
```

Available functions:

- `relu(x)` - Rectified Linear Unit
- `sigmoid(x)` - Sigmoid activation
- `tanh(x)` - Hyperbolic tangent
- `gelu(x)` - Gaussian Error Linear Unit

### Loss Functions

```python
from pytensorlib import mse_loss, cross_entropy_loss

predictions = Tensor([[0.8, 0.2], [0.3, 0.7]])
targets = Tensor([[1.0, 0.0], [0.0, 1.0]])

mse = mse_loss(predictions, targets)
ce = cross_entropy_loss(predictions, targets)
```

### EMNIST/MNIST Data Utilities

PyTensorLib supports both EMNIST (Extended MNIST) and standard MNIST datasets:

```python
from pytensorlib.mnist_utils import download_emnist, MNISTReader, create_synthetic_mnist_data

# Download EMNIST dataset (recommended - higher quality)
train_images, train_labels, test_images, test_labels = download_emnist("./data")

# Load the data
reader = MNISTReader(train_images, train_labels)
image = reader.get_image_as_array(0)
label = reader.get_label(0)

# Or create synthetic data for testing
images, labels = create_synthetic_mnist_data(1000)
```

**EMNIST vs MNIST:**

- **EMNIST** (Extended MNIST): Higher quality, more diverse handwritten digits from NIST
- **MNIST**: Classic dataset, smaller download but older data
- **Synthetic**: Generated data for testing when no internet connection

The examples automatically use EMNIST if available, fall back to MNIST, or generate synthetic data.

## 🧪 Testing

Run the complete test suite:

```bash
# Activate environment first
source pytensor-env/bin/activate

# Run all tests
cd tests
python test_tensor_lib.py
python test_mnist_utils.py

# Or run specific test categories
python -c "
import sys
sys.path.append('../src')
from pytensorlib import quick_test
quick_test()
"
```

## 📁 Project Structure

```
python-tensor-lib/
├── src/pytensorlib/          # Main package
│   ├── __init__.py          # Package exports and API
│   ├── tensor_lib.py        # Core tensor implementation
│   └── mnist_utils.py       # MNIST dataset utilities
├── tests/                   # Test suite
│   ├── test_tensor_lib.py   # Tensor and gradient tests
│   └── test_mnist_utils.py  # MNIST utilities tests
├── examples/                # Example scripts
│   ├── basic_operations.py  # Basic tensor operations demo
│   ├── neural_network.py    # XOR neural network example
│   └── three_or_seven.py    # Binary digit classifier (3 vs 7)
├── docs/                    # Documentation
├── setup.py                 # Package installation
├── setup_environment.py     # Environment setup script
├── requirements.txt         # Dependencies
└── README.md               # This file
```

## 🎯 Examples

### Example 1: Gradient Descent

```python
from pytensorlib import Tensor
import numpy as np

# Optimize f(x) = (x - 3)^2 to find minimum at x = 3
x = Tensor([0.0], requires_grad=True)
learning_rate = 0.1

for i in range(50):
    loss = (x - 3) ** 2
    loss.backward()

    x.data -= learning_rate * x.grad
    x.grad = None

    if i % 10 == 0:
        print(f"Step {i}: x = {x.data[0]:.3f}, loss = {loss.data:.3f}")
```

### Example 2: Multivariate Function

```python
from pytensorlib import Tensor

# Minimize f(x,y) = x^2 + y^2 + 2xy
x = Tensor([1.0], requires_grad=True)
y = Tensor([1.0], requires_grad=True)

for i in range(100):
    loss = x**2 + y**2 + 2*x*y
    loss.backward()

    x.data -= 0.1 * x.grad
    y.data -= 0.1 * y.grad
    x.grad = None
    y.grad = None

    if i % 20 == 0:
        print(f"Step {i}: x={x.data[0]:.3f}, y={y.data[0]:.3f}, loss={loss.data:.3f}")
```

### Example 3: Classification Network

```python
from pytensorlib import Tensor, relu, sigmoid, cross_entropy_loss
import numpy as np

# Binary classification dataset
np.random.seed(42)
X = Tensor(np.random.randn(100, 2))
y = Tensor((X.data[:, 0] + X.data[:, 1] > 0).astype(float).reshape(-1, 1))

# Simple network
W1 = Tensor(np.random.randn(2, 3) * 0.5, requires_grad=True)
b1 = Tensor(np.zeros((1, 3)), requires_grad=True)
W2 = Tensor(np.random.randn(3, 1) * 0.5, requires_grad=True)
b2 = Tensor(np.zeros((1, 1)), requires_grad=True)

# Training
for epoch in range(200):
    # Forward pass
    h = relu(X @ W1 + b1)
    logits = h @ W2 + b2
    probs = sigmoid(logits)

    # Loss
    loss = cross_entropy_loss(probs, y)

    # Backward pass
    loss.backward()

    # Update
    lr = 0.1
    for param in [W1, b1, W2, b2]:
        param.data -= lr * param.grad
        param.grad = None

    if epoch % 50 == 0:
        accuracy = ((probs.data > 0.5) == y.data).mean()
        print(f"Epoch {epoch}: Loss = {loss.data:.4f}, Accuracy = {accuracy:.3f}")
```

### Example 4: Digit Classification (3 vs 7)

```bash
# Run the three_or_seven classifier example
python examples/three_or_seven.py
```

This example demonstrates:

- Loading and processing MNIST-like data (with automatic EMNIST→MNIST fallback)
- Computing average images for different digit classes
- Building a linear classifier using difference vectors
- Evaluating classification performance with real data
- Analyzing decision boundaries and misclassified examples

The classifier achieves ~96.6% accuracy on real MNIST data by:

1. Computing average images for 3s and 7s
2. Creating a decision vector (difference of averages)
3. Using dot products for classification scores
4. Applying a learned threshold for final predictions

**Results with real MNIST data:**

- Training samples: 12,396 (6,131 threes, 6,265 sevens)
- Test samples: 2,038 (1,010 threes, 1,028 sevens)
- Accuracy: 96.6% (1,969/2,038 correct predictions)
- Shows actual misclassified examples with ASCII visualization

## 🔧 Development

### Setting Up Development Environment

```bash
# Clone repository
git clone <repository-url>
cd python-tensor-lib

# Run setup (creates virtual environment and installs dependencies)
python setup_environment.py

# Activate environment
source pytensor-env/bin/activate

# Install in development mode
pip install -e .
```

### Running Tests

```bash
# Run tensor tests
python tests/test_tensor_lib.py

# Run MNIST tests
python tests/test_mnist_utils.py

# Quick functionality test
python -c "from pytensorlib import quick_test; quick_test()"
```

### Adding New Features

1. Implement your feature in the appropriate module (`src/pytensorlib/`)
2. Add comprehensive tests in `tests/`
3. Update the `__init__.py` to export new functionality
4. Add examples to this README

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Make your changes and add tests
4. Ensure all tests pass (`python tests/test_*.py`)
5. Commit your changes (`git commit -m 'Add amazing feature'`)
6. Push to the branch (`git push origin feature/amazing-feature`)
7. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Inspired by PyTorch and other modern deep learning frameworks
- Built for educational purposes and learning automatic differentiation
- Thanks to the NumPy team for the excellent numerical computing library

## 📞 Support

If you encounter any issues or have questions:

1. Check the examples in this README
2. Run the test suite to verify installation
3. Create an issue on GitHub with details about your problem

---

**Happy deep learning with PyTensorLib! 🚀🧠**
