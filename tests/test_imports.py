#!/usr/bin/env python3
"""
Test script to verify that Go to Definition works properly
"""

# Import the problematic class
from pytensorlib import MNISTReader, Tensor, relu

# Create instances to test  
# Note: MNISTReader requires file paths, so we'll just test the import
# reader = MNISTReader("dummy_images", "dummy_labels")  # <- Try Cmd+Click on MNISTReader here
tensor = Tensor(3, 3)   # <- Try Cmd+Click on Tensor here 
result = relu(tensor)   # <- Try Cmd+Click on relu here

print("Imports successful!")
print(f"MNISTReader module: {MNISTReader.__module__}")
print(f"Tensor type: {type(tensor)}")

# Check the file location
import pytensorlib.mnist_utils
print(f"mnist_utils location: {pytensorlib.mnist_utils.__file__}")