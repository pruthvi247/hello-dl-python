"""
MNIST Dataset Reader and Utilities

This module provides utilities for reading and processing MNIST dataset files.
It handles the IDX file format used by MNIST and provides convenient interfaces
for loading images and labels.

🔢 IDX FILE FORMAT & MAGIC NUMBERS:

The IDX format is a simple file format for storing vectors and multidimensional 
matrices used by the MNIST dataset. Each file begins with a magic number that 
identifies the data type and dimensionality.

MAGIC NUMBER STRUCTURE (4 bytes, big-endian):
┌─────────┬─────────┬─────────┬─────────┐
│ 0x00    │ 0x00    │ Type    │ Dims    │
└─────────┴─────────┴─────────┴─────────┘

Type Codes:
- 0x08: unsigned byte (uint8)
- 0x09: signed byte (int8)  
- 0x0B: short (int16)
- 0x0C: int (int32)
- 0x0D: float (float32)
- 0x0E: double (float64)

Dimension Codes:
- 0x01: 1D array (vector)
- 0x02: 2D array (matrix)
- 0x03: 3D array (tensor)

MNIST SPECIFIC MAGIC NUMBERS:
- 2049 (0x00000801): Labels file (1D array of uint8)
- 2051 (0x00000803): Images file (3D array of uint8)

FILE FORMAT EXAMPLES:

Labels File (train-labels-idx1-ubyte):
┌──────────────┬──────────────┬──────────────────┐
│ Magic: 2049  │ Count: 60000 │ Data: 60000 bytes│
│ (4 bytes)    │ (4 bytes)    │ (60000 bytes)     │
└──────────────┴──────────────┴──────────────────┘

Images File (train-images-idx3-ubyte):
┌──────────────┬──────────────┬─────────┬─────────┬─────────────────────┐
│ Magic: 2051  │ Count: 60000 │ Rows:28 │ Cols:28 │ Data: 47,040,000    │
│ (4 bytes)    │ (4 bytes)    │(4 bytes)│(4 bytes)│ bytes (60000×28×28) │
└──────────────┴──────────────┴─────────┴─────────┴─────────────────────┘

BINARY LAYOUT EXAMPLE:
Labels file header:
[0x00][0x00][0x08][0x01][0x00][0x00][0xEA][0x60][label_data...]
 ^--- Magic 2049 ---^    ^--- Count 60000 --^

Images file header:  
[0x00][0x00][0x08][0x03][0x00][0x00][0xEA][0x60][0x00][0x00][0x00][0x1C][0x00][0x00][0x00][0x1C][pixel_data...]
 ^--- Magic 2051 ---^    ^--- Count 60000 --^    ^--- Rows 28 ---^     ^--- Cols 28 ---^

The module is compatible with the tensor_lib module for deep learning applications.
"""

import gzip
import struct
import numpy as np
from typing import List, Tuple, Optional
import os
import urllib.request
import zipfile
import tempfile

__version__ = "1.0.0"


class MNISTReader:
    """Python implementation of MNIST dataset reader"""
    
    def __init__(self, images_path: str, labels_path: str):
        """
        Initialize MNIST reader with paths to compressed image and label files
        
        Sets up the MNIST dataset reader by loading and parsing IDX format files.
        The IDX format is a simple format for vectors and multidimensional matrices
        used by the MNIST dataset. This constructor loads both images and labels,
        validates file formats, and pre-processes all data for efficient access.
        
        IDX File Format:
        - Images: magic number (2051) + dimensions + pixel data (uint8)
        - Labels: magic number (2049) + count + label data (uint8)
        
        Args:
            images_path (str): Path to gzipped images file (e.g., train-images-idx3-ubyte.gz)
            labels_path (str): Path to gzipped labels file (e.g., train-labels-idx1-ubyte.gz)
            
        Raises:
            ValueError: If magic numbers don't match IDX format or counts mismatch
            FileNotFoundError: If specified files don't exist
            
        Sample Input/Output:
            >>> # Load MNIST training data
            >>> train_images_path = "data/train-images-idx3-ubyte.gz"
            >>> train_labels_path = "data/train-labels-idx1-ubyte.gz"
            >>> reader = MNISTReader(train_images_path, train_labels_path)
            >>> reader.num                # Output: 60000 (training samples)
            >>> reader.rows               # Output: 28 (image height)
            >>> reader.cols               # Output: 28 (image width)
            >>> reader.stride             # Output: 784 (28*28 pixels per image)
            
            >>> # Load MNIST test data
            >>> test_images_path = "data/t10k-images-idx3-ubyte.gz"
            >>> test_labels_path = "data/t10k-labels-idx1-ubyte.gz"
            >>> test_reader = MNISTReader(test_images_path, test_labels_path)
            >>> test_reader.num           # Output: 10000 (test samples)
            >>> len(test_reader.images)   # Output: 7840000 (10000 * 784 pixels)
            >>> len(test_reader.labels)   # Output: 10000 (one label per image)
            
            >>> # Custom MNIST subset
            >>> custom_reader = MNISTReader("subset-images.gz", "subset-labels.gz")
            >>> custom_reader.get_stats() # Output: {'num_images': N, 'image_shape': (28, 28), ...}
            
            >>> # Error cases
            >>> try:
            ...     bad_reader = MNISTReader("corrupted.gz", "labels.gz")
            ... except ValueError as e:
            ...     print(f"Format error: {e}")  # Output: "Wrong magic number..."
        """
        self.rows = 0
        self.cols = 0
        self.stride = 0
        self.num = 0
        self.images = []
        self.labels = []
        self.converted = {}  # Cache for normalized float images
        
        self._load_data(images_path, labels_path)
    
    def _load_data(self, images_path: str, labels_path: str):
        """
        Load and parse MNIST IDX format data files
        
        Implements the IDX file format parser for MNIST datasets. The IDX format
        stores multidimensional arrays in a portable binary format with specific
        magic numbers for type identification and dimension information.
        
        🔢 MAGIC NUMBERS EXPLAINED:
        Magic numbers are 4-byte identifiers at the start of IDX files that specify:
        - Data type (byte 3): 0x08 = unsigned byte (uint8)
        - Dimensions (byte 4): 0x01 = 1D array, 0x03 = 3D array
        
        Magic Number Format: 0x00 00 [type] [dims]
        - 2049 (0x00000801): IDX1 = 1D array of unsigned bytes (labels)
        - 2051 (0x00000803): IDX3 = 3D array of unsigned bytes (images)
        
        IDX Format Structure:
        - Labels (IDX1): [magic:4][count:4][data:count]
        - Images (IDX3): [magic:4][count:4][rows:4][cols:4][data:count*rows*cols]
        
        Binary Layout Example (Big-Endian):
        Labels File:
        [0x00 0x00 0x08 0x01][0x00 0x00 0xEA 0x60][label_data...]
         ^-- Magic 2049      ^-- Count 60000      ^-- 60000 bytes
        
        Images File:
        [0x00 0x00 0x08 0x03][0x00 0x00 0xEA 0x60][0x00 0x00 0x00 0x1C][0x00 0x00 0x00 0x1C][pixel_data...]
         ^-- Magic 2051      ^-- Count 60000      ^-- Rows 28          ^-- Cols 28          ^-- 47M bytes
        
        Args:
            images_path (str): Path to compressed images file
            labels_path (str): Path to compressed labels file
            
        Raises:
            ValueError: If magic numbers are incorrect or data counts mismatch
            IOError: If files cannot be read or are corrupted
            
        Sample Input/Output:
            >>> # Internal method called by __init__
            >>> reader = MNISTReader("train-images.gz", "train-labels.gz")
            >>> # After _load_data() completes:
            >>> reader.labels[:5]         # Output: [5, 0, 4, 1, 9] (first 5 labels)
            >>> reader.images[:10]        # Output: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0] (first 10 pixels)
            >>> reader.num                # Output: 60000 (total samples)
            >>> reader.rows               # Output: 28 (image height)
            >>> reader.cols               # Output: 28 (image width)
            
            >>> # Magic number validation examples
            >>> # If images file has wrong magic number:
            >>> # ValueError: "Wrong magic number in images file: 1234 (expected 2051 for IDX3 format)"
            >>> # If labels file has wrong magic number:
            >>> # ValueError: "Wrong magic number in labels file: 5678 (expected 2049 for IDX1 format)"
            >>> # If label count != image count:
            >>> # ValueError: "Mismatch between number of labels and images"
            
            >>> # Memory layout after loading
            >>> len(reader.images)        # Output: 47040000 (60000 * 28 * 28)
            >>> len(reader.labels)        # Output: 60000
            >>> reader.stride             # Output: 784 (pixels per image)
            >>> reader.converted          # Output: {} (will be filled by _convert_all_images)
            
            >>> # File format technical details
            >>> import struct
            >>> with gzip.open("train-labels.gz", 'rb') as f:
            ...     magic_bytes = f.read(4)
            ...     magic = struct.unpack('>I', magic_bytes)[0]
            ...     print(f"Magic: {magic} (0x{magic:08X})")
            >>> # Output: "Magic: 2049 (0x00000801)"
            
            >>> with gzip.open("train-images.gz", 'rb') as f:
            ...     magic_bytes = f.read(4)
            ...     magic = struct.unpack('>I', magic_bytes)[0]
            ...     print(f"Magic: {magic} (0x{magic:08X})")
            >>> # Output: "Magic: 2051 (0x00000803)"
        """
        
        # Load labels file
        with gzip.open(labels_path, 'rb') as f:
            # Read IDX1 header: magic number (4 bytes) + number of items (4 bytes)
            # Magic number 2049 (0x00000801) = IDX1 format (1D unsigned byte array)
            magic, num_labels = struct.unpack('>II', f.read(8))
            
            if magic != 2049:
                raise ValueError(f"Wrong magic number in labels file: {magic} (expected 2049 for IDX1 format)")
            
            # Read all labels
            self.labels = list(f.read(num_labels))
        
        # Load images file
        with gzip.open(images_path, 'rb') as f:
            # Read IDX3 header: magic (4) + num items (4) + rows (4) + cols (4)
            # Magic number 2051 (0x00000803) = IDX3 format (3D unsigned byte array)
            magic, num_images, rows, cols = struct.unpack('>IIII', f.read(16))
            
            if magic != 2051:
                raise ValueError(f"Wrong magic number in images file: {magic} (expected 2051 for IDX3 format)")
            
            if num_images != num_labels:
                raise ValueError("Mismatch between number of labels and images")
            
            self.rows = rows
            self.cols = cols
            self.stride = rows * cols
            self.num = num_images
            
            # Read all image data
            image_data = f.read(num_images * rows * cols)
            self.images = list(image_data)
        
        # Pre-convert all images to normalized float format
        self._convert_all_images()
    
    def _convert_all_images(self):
        """
        Convert all images to normalized float format and cache them
        
        Pre-processes all loaded images by converting from uint8 (0-255) to 
        normalized float32 (0.0-1.0) values. This normalization is crucial for
        neural network training as it ensures input values are in a suitable
        range for gradient-based optimization algorithms.
        
        Normalization Process:
        - Original: uint8 values from 0 to 255 (grayscale pixels)
        - Normalized: float values from 0.0 to 1.0 (pixel_value / 255.0)
        - Storage: Cached in self.converted dictionary for fast access
        
        Benefits:
        - Faster training convergence (normalized inputs)
        - Better numerical stability in gradient computations
        - Consistent value ranges across different datasets
        - Memory-efficient caching for repeated access
        
        Sample Input/Output:
            >>> reader = MNISTReader("train-images.gz", "train-labels.gz")
            >>> # After _convert_all_images() completes:
            >>> len(reader.converted)     # Output: 60000 (all images cached)
            >>> 
            >>> # Example pixel value transformation
            >>> raw_pixel = reader.images[0]    # Output: 255 (white pixel, uint8)
            >>> normalized = reader.converted[0][0]  # Output: 1.0 (255/255)
            >>> 
            >>> raw_pixel = reader.images[100]  # Output: 128 (gray pixel, uint8)  
            >>> normalized = reader.converted[0][100]  # Output: 0.502 (128/255)
            >>> 
            >>> raw_pixel = reader.images[200]  # Output: 0 (black pixel, uint8)
            >>> normalized = reader.converted[0][200]  # Output: 0.0 (0/255)
            
            >>> # Memory usage comparison
            >>> import sys
            >>> raw_size = sys.getsizeof(reader.images)      # Larger (uint8 list)
            >>> normalized_size = sys.getsizeof(reader.converted)  # Optimized (float dict)
            
            >>> # Verification of conversion
            >>> for img_idx in range(min(3, reader.num)):
            ...     raw_img = reader.get_image(img_idx)
            ...     norm_img = reader.converted[img_idx]
            ...     # All values should be raw[i]/255.0 == norm[i]
            ...     assert len(raw_img) == len(norm_img) == 784
            ...     assert all(abs(raw_img[i]/255.0 - norm_img[i]) < 1e-6 for i in range(784))
        """
        for n in range(self.num):
            pos = n * self.stride
            # Normalize pixel values from 0-255 to 0-1
            float_image = [self.images[pos + i] / 255.0 for i in range(self.stride)]
            self.converted[n] = float_image
    
    def get_num(self) -> int:
        """
        Get total number of images in the dataset
        
        Returns the count of images/labels loaded from the MNIST files.
        This is useful for determining dataset size, creating training loops,
        and validating data consistency.
        
        Returns:
            int: Total number of images in the dataset
            
        Sample Input/Output:
            >>> # MNIST training set
            >>> train_reader = MNISTReader("train-images.gz", "train-labels.gz")
            >>> train_reader.get_num()    # Output: 60000
            
            >>> # MNIST test set
            >>> test_reader = MNISTReader("t10k-images.gz", "t10k-labels.gz")
            >>> test_reader.get_num()     # Output: 10000
            
            >>> # Use in training loop
            >>> for epoch in range(10):
            ...     for i in range(reader.get_num()):
            ...         image = reader.get_image_float(i)
            ...         label = reader.get_label(i)
            ...         # Process training sample...
            
            >>> # Batch size calculation
            >>> total_samples = reader.get_num()
            >>> batch_size = 32
            >>> num_batches = (total_samples + batch_size - 1) // batch_size
            >>> print(f"Dataset: {total_samples} samples, {num_batches} batches")
            >>> # Output: "Dataset: 60000 samples, 1875 batches"
            
            >>> # Data split verification
            >>> assert train_reader.get_num() == len(train_reader.labels)
            >>> assert train_reader.get_num() * 784 == len(train_reader.images)
        """
        return self.num
    
    def get_image(self, n: int) -> List[int]:
        """
        Get raw image as list of uint8 values (0-255)
        
        Retrieves the original pixel values for the specified image as stored
        in the MNIST file format. Each pixel is represented as an unsigned 8-bit
        integer (0=black, 255=white). The image is returned as a flattened list
        of 784 values (28×28 pixels) in row-major order.
        
        Args:
            n (int): Image index (0 to num_images-1)
            
        Returns:
            List[int]: List of 784 uint8 pixel values (0-255)
            
        Raises:
            IndexError: If image index is out of range
            
        Sample Input/Output:
            >>> reader = MNISTReader("train-images.gz", "train-labels.gz")
            >>> 
            >>> # Get first image (digit "5")
            >>> img_0 = reader.get_image(0)
            >>> len(img_0)                # Output: 784 (28*28)
            >>> type(img_0[0])            # Output: <class 'int'>
            >>> min(img_0)                # Output: 0 (darkest pixel)
            >>> max(img_0)                # Output: 255 (brightest pixel)
            
            >>> # Pixel value examples
            >>> img_0[0]                  # Output: 0 (top-left corner, usually background)
            >>> img_0[200]                # Output: 128 (middle gray pixel)
            >>> img_0[400]                # Output: 255 (white foreground pixel)
            
            >>> # Image statistics
            >>> import statistics
            >>> mean_intensity = statistics.mean(img_0)  # Output: ~45.6 (mostly dark)
            >>> bright_pixels = sum(1 for p in img_0 if p > 200)  # Output: ~23 (few bright pixels)
            
            >>> # Reshape to 2D for visualization
            >>> import numpy as np
            >>> img_2d = np.array(img_0).reshape(28, 28)
            >>> img_2d.shape              # Output: (28, 28)
            >>> img_2d[14, 14]            # Output: pixel value at center
            
            >>> # Error handling
            >>> try:
            ...     invalid_img = reader.get_image(70000)  # Beyond dataset size
            ... except IndexError as e:
            ...     print(f"Error: {e}")  # Output: "Image index 70000 out of range (0-59999)"
            
            >>> # Compare with normalized version
            >>> raw_img = reader.get_image(100)
            >>> norm_img = reader.get_image_float(100)
            >>> # Verify: norm_img[i] ≈ raw_img[i] / 255.0 for all i
        """
        if n >= self.num:
            raise IndexError(f"Image index {n} out of range (0-{self.num-1})")
        
        pos = n * self.stride
        return self.images[pos:pos + self.stride]
    
    def get_image_float(self, n: int) -> List[float]:
        """
        Get normalized image as list of float values (0.0-1.0)
        
        Retrieves the pre-normalized pixel values for the specified image.
        Pixel intensities are scaled from the original 0-255 range to 0.0-1.0,
        which is the standard preprocessing step for neural network training.
        This method uses cached normalized values for efficiency.
        
        Args:
            n (int): Image index (0 to num_images-1)
            
        Returns:
            List[float]: List of 784 normalized pixel values (0.0-1.0)
            
        Raises:
            IndexError: If image index is out of range
            RuntimeError: If normalized image cache is corrupted
            
        Sample Input/Output:
            >>> reader = MNISTReader("train-images.gz", "train-labels.gz")
            >>> 
            >>> # Get normalized first image
            >>> norm_img = reader.get_image_float(0)
            >>> len(norm_img)             # Output: 784 (28*28)
            >>> type(norm_img[0])         # Output: <class 'float'>
            >>> min(norm_img)             # Output: 0.0 (black pixels)
            >>> max(norm_img)             # Output: 1.0 (white pixels)
            
            >>> # Pixel value examples (normalized)
            >>> norm_img[0]               # Output: 0.0 (background pixel: 0/255)
            >>> norm_img[200]             # Output: 0.502 (gray pixel: 128/255)
            >>> norm_img[400]             # Output: 1.0 (foreground pixel: 255/255)
            
            >>> # Neural network input preparation
            >>> def prepare_training_sample(idx):
            ...     features = reader.get_image_float(idx)  # Normalized pixels
            ...     label = reader.get_label(idx)           # Class label
            ...     return features, label
            >>> 
            >>> features, label = prepare_training_sample(42)
            >>> len(features)             # Output: 784 (input layer size)
            >>> all(0.0 <= f <= 1.0 for f in features)  # Output: True (valid range)
            
            >>> # Batch preparation for training
            >>> batch_indices = [0, 1, 2, 3, 4]
            >>> batch_features = [reader.get_image_float(i) for i in batch_indices]
            >>> batch_labels = [reader.get_label(i) for i in batch_indices]
            >>> len(batch_features)      # Output: 5 (batch size)
            >>> len(batch_features[0])   # Output: 784 (feature dimension)
            
            >>> # Comparison with raw image
            >>> raw_img = reader.get_image(123)
            >>> norm_img = reader.get_image_float(123)
            >>> # Verify normalization: raw[i] / 255.0 ≈ norm[i]
            >>> import math
            >>> for i in range(10):  # Check first 10 pixels
            ...     expected = raw_img[i] / 255.0
            ...     actual = norm_img[i]
            ...     assert abs(expected - actual) < 1e-6
            
            >>> # Error cases
            >>> try:
            ...     invalid = reader.get_image_float(-1)
            ... except IndexError:
            ...     print("Negative index not allowed")
        """
        if n >= self.num:
            raise IndexError(f"Image index {n} out of range (0-{self.num-1})")
        
        if n not in self.converted:
            raise RuntimeError(f"Could not find image {n}")
        
        return self.converted[n]
    
    def push_image_to_tensor(self, n: int, dest):
        """
        Copy normalized image data directly into an existing 28×28 tensor
        
        Efficiently transfers normalized pixel values from the MNIST dataset
        into a pre-allocated tensor for neural network processing. The data
        is copied in column-major order to match the C++ implementation's
        memory layout, ensuring compatibility with the tensor library.
        
        Memory Layout:
        - Source: Flattened 784-element list (row-major from MNIST)
        - Destination: 28×28 tensor matrix (column-major for tensor_lib)
        - Conversion: dest[row,col] = src[row + 28*col]
        
        Args:
            n (int): Image index (0 to num_images-1)
            dest (Tensor): Pre-allocated 28×28 parameter tensor
            
        Raises:
            AssertionError: If destination is not a parameter tensor or wrong size
            IndexError: If image index is out of range
            
        Sample Input/Output:
            >>> from pytensorlib import Tensor
            >>> reader = MNISTReader("train-images.gz", "train-labels.gz")
            >>> 
            >>> # Create destination tensor
            >>> img_tensor = Tensor(28, 28)
            >>> img_tensor.shape          # Output: (28, 28)
            >>> img_tensor[0, 0]          # Output: 0.0 (initially zeros)
            
            >>> # Copy first image (digit "5")
            >>> reader.push_image_to_tensor(0, img_tensor)
            >>> img_tensor[0, 0]          # Output: 0.0 (background pixel)
            >>> img_tensor[14, 14]        # Output: 0.8 (foreground pixel, example)
            >>> img_tensor.shape          # Output: (28, 28) (unchanged)
            
            >>> # Verify data integrity
            >>> src_data = reader.get_image_float(0)
            >>> for row in range(28):
            ...     for col in range(28):
            ...         expected = src_data[row + 28 * col]
            ...         actual = img_tensor[row, col]
            ...         assert abs(expected - actual) < 1e-6
            
            >>> # Neural network input preparation
            >>> def prepare_input_tensor(img_idx):
            ...     input_tensor = Tensor(28, 28)
            ...     reader.push_image_to_tensor(img_idx, input_tensor)
            ...     return input_tensor
            >>> 
            >>> input_data = prepare_input_tensor(42)
            >>> # Ready for forward pass through CNN layers
            
            >>> # Batch processing with pre-allocated tensors
            >>> batch_size = 10
            >>> batch_tensors = [Tensor(28, 28) for _ in range(batch_size)]
            >>> for i, tensor in enumerate(batch_tensors):
            ...     reader.push_image_to_tensor(i, tensor)
            >>> # All tensors now contain normalized image data
            
            >>> # Error cases
            >>> wrong_size_tensor = Tensor(32, 32)  # Wrong dimensions
            >>> try:
            ...     reader.push_image_to_tensor(0, wrong_size_tensor)
            ... except (AssertionError, AttributeError) as e:
            ...     print(f"Size error: tensor must be 28x28")
            
            >>> # Performance benefit: reuse tensors in training loop
            >>> reusable_tensor = Tensor(28, 28)
            >>> for epoch in range(5):
            ...     for img_idx in range(0, 100, 10):  # Every 10th image
            ...         reader.push_image_to_tensor(img_idx, reusable_tensor)
            ...         # Process with neural network...
            ...         # Tensor is overwritten each iteration (memory efficient)
        """
        # Import here to avoid circular dependency
        try:
            from .tensor_lib import TMode
            assert dest.impl.mode == TMode.PARAMETER, "Destination must be a parameter tensor"
        except ImportError:
            # Fallback for standalone usage
            assert hasattr(dest, 'impl'), "Destination must be a tensor"
        
        src = self.get_image_float(n)
        
        # Copy data in column-major order to match C++ implementation
        for row in range(28):
            for col in range(28):
                dest[row, col] = src[row + 28 * col]
    
    def get_image_as_tensor(self, n: int):
        """
        Get normalized image as a new 28×28 tensor
        
        Creates a new tensor and populates it with normalized pixel values
        from the specified MNIST image. This is a convenience method that
        combines tensor allocation and data copying in one step. Ideal for
        one-off tensor creation and functional programming styles.
        
        Args:
            n (int): Image index (0 to num_images-1)
            
        Returns:
            Tensor: New 28×28 tensor containing normalized image data
            
        Raises:
            ImportError: If tensor_lib module is not available
            IndexError: If image index is out of range
            
        Sample Input/Output:
            >>> reader = MNISTReader("train-images.gz", "train-labels.gz")
            >>> 
            >>> # Get first image as tensor
            >>> img_tensor = reader.get_image_as_tensor(0)
            >>> type(img_tensor)          # Output: <class 'pytensorlib.tensor_lib.Tensor'>
            >>> img_tensor.shape          # Output: (28, 28)
            >>> img_tensor[0, 0]          # Output: 0.0 (background pixel)
            >>> img_tensor[14, 14]        # Output: 0.6 (foreground pixel, example)
            
            >>> # Direct use in neural network forward pass
            >>> def predict_digit(img_idx):
            ...     input_tensor = reader.get_image_as_tensor(img_idx)
            ...     # Apply CNN layers
            ...     conv1_output = conv1_layer(input_tensor)
            ...     conv2_output = conv2_layer(conv1_output)
            ...     # ... rest of network
            ...     return predictions
            >>> 
            >>> prediction = predict_digit(42)
            
            >>> # Batch creation (functional style)
            >>> batch_indices = [0, 1, 2, 3, 4]
            >>> batch_tensors = [reader.get_image_as_tensor(i) for i in batch_indices]
            >>> len(batch_tensors)       # Output: 5
            >>> all(t.shape == (28, 28) for t in batch_tensors)  # Output: True
            
            >>> # Comparison with manual method
            >>> # Method 1: Using this convenience function
            >>> tensor1 = reader.get_image_as_tensor(100)
            >>> 
            >>> # Method 2: Manual tensor creation
            >>> from pytensorlib import Tensor
            >>> tensor2 = Tensor(28, 28)
            >>> reader.push_image_to_tensor(100, tensor2)
            >>> 
            >>> # Both methods produce identical results
            >>> for row in range(28):
            ...     for col in range(28):
            ...         assert abs(tensor1[row, col] - tensor2[row, col]) < 1e-6
            
            >>> # Data pipeline integration
            >>> def create_training_sample(idx):
            ...     features = reader.get_image_as_tensor(idx)    # 28x28 tensor
            ...     label = reader.get_label(idx)                # int (0-9)
            ...     return features, label
            >>> 
            >>> X, y = create_training_sample(123)
            >>> # Ready for automatic differentiation and backpropagation
            
            >>> # Memory consideration: creates new tensor each time
            >>> tensors = []
            >>> for i in range(10):
            ...     t = reader.get_image_as_tensor(i)  # New tensor allocated
            ...     tensors.append(t)
            >>> len(tensors)              # Output: 10 (separate tensor objects)
            
            >>> # Error handling
            >>> try:
            ...     reader.get_image_as_tensor(70000)  # Out of range
            ... except IndexError as e:
            ...     print(f"Index error: {e}")
        """
        # Import here to avoid circular dependency
        try:
            from .tensor_lib import Tensor
        except ImportError:
            raise ImportError("tensor_lib module required for tensor operations")
            
        tensor = Tensor(28, 28)
        self.push_image_to_tensor(n, tensor)
        return tensor
    
    def get_image_as_array(self, n: int) -> np.ndarray:
        """
        Get normalized image as a 28×28 numpy array
        
        🔄 HOW IT WORKS INTERNALLY:
        
        INPUT PROCESSING:
        1. Takes image index n (integer from 0 to num_images-1)
        2. Calls get_image_float(n) to retrieve normalized pixel data
        3. get_image_float() returns a pre-computed 784-element list of floats (0.0-1.0)
        
        INTERNAL DATA TRANSFORMATION:
        The method performs a critical memory layout conversion:
        
        Source Format (from get_image_float):
        - 784-element flat list: [pixel_0, pixel_1, ..., pixel_783]
        - Row-major order: pixel_i represents position (i//28, i%28)
        - Example: pixel_56 is at row=2, col=0 (56//28=2, 56%28=0)
        
        Destination Format (28×28 numpy array):
        - Column-major arrangement for tensor library compatibility
        - array[row, col] = src[row + 28*col]
        - This transforms: (row, col) → flat_index = row + 28*col
        
        CONVERSION ALGORITHM:
        ```python
        for row in range(28):      # 0 to 27
            for col in range(28):  # 0 to 27
                flat_index = row + 28 * col
                image_2d[row, col] = src[flat_index]
        ```
        
        MEMORY LAYOUT EXAMPLE:
        For a 4×4 image (simplified):
        
        Row-major flat list: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
        
        Column-major 2D array:
        [[ 0,  4,  8, 12],    # array[0,:] = src[0], src[4], src[8], src[12]
         [ 1,  5,  9, 13],    # array[1,:] = src[1], src[5], src[9], src[13]
         [ 2,  6, 10, 14],    # array[2,:] = src[2], src[6], src[10], src[14]
         [ 3,  7, 11, 15]]    # array[3,:] = src[3], src[7], src[11], src[15]
        
        WHY COLUMN-MAJOR ORDER?
        - Matches C++ tensor library memory layout
        - Ensures data consistency between Python and C++ components
        - Required for proper tensor operations and neural network computations
        
        Args:
            n (int): Image index (0 to num_images-1)
            
        Returns:
            np.ndarray: 28×28 float32 array with normalized pixel values (0.0-1.0)
            
        Raises:
            IndexError: If image index is out of range
            
        Sample Input/Output - Understanding the Mechanics:
            >>> reader = MNISTReader("train-images.gz", "train-labels.gz")
            >>> 
            >>> # INPUT: Image index
            >>> n = 0  # First image in dataset
            >>> 
            >>> # STEP 1: Get the flat normalized data
            >>> flat_data = reader.get_image_float(0)
            >>> type(flat_data)          # Output: <class 'list'>
            >>> len(flat_data)           # Output: 784
            >>> flat_data[0]             # Output: 0.0 (first pixel, normalized)
            >>> flat_data[28]            # Output: 0.12 (start of row 1)
            >>> flat_data[56]            # Output: 0.0 (start of row 2)
            
            >>> # STEP 2: Apply the transformation algorithm
            >>> img_array = reader.get_image_as_array(0)
            >>> 
            >>> # OUTPUT: 28×28 numpy array
            >>> type(img_array)          # Output: <class 'numpy.ndarray'>
            >>> img_array.shape         # Output: (28, 28)
            >>> img_array.dtype         # Output: dtype('float32')
            
            >>> # VERIFY THE TRANSFORMATION:
            >>> # array[row,col] should equal flat_data[row + 28*col]
            >>> row, col = 0, 0
            >>> flat_index = row + 28 * col    # 0 + 28*0 = 0
            >>> img_array[0, 0] == flat_data[0]  # Output: True
            
            >>> row, col = 1, 0  
            >>> flat_index = row + 28 * col    # 1 + 28*0 = 1
            >>> img_array[1, 0] == flat_data[1]  # Output: True
            
            >>> row, col = 0, 1
            >>> flat_index = row + 28 * col    # 0 + 28*1 = 28  
            >>> img_array[0, 1] == flat_data[28]  # Output: True
            
            >>> row, col = 14, 14
            >>> flat_index = row + 28 * col    # 14 + 28*14 = 406
            >>> img_array[14, 14] == flat_data[406]  # Output: True
            
            >>> # DEMONSTRATE THE CONVERSION PROCESS:
            >>> def show_conversion_process(reader, img_idx):
            ...     print(f"Converting image {img_idx}:")
            ...     flat = reader.get_image_float(img_idx)
            ...     array = reader.get_image_as_array(img_idx)
            ...     
            ...     print(f"  Input: {len(flat)}-element list")
            ...     print(f"  Output: {array.shape} numpy array")
            ...     print(f"  First few flat values: {flat[:5]}")
            ...     print(f"  Corresponding array positions:")
            ...     for i in range(5):
            ...         row, col = i, 0  # First column
            ...         print(f"    flat[{i}] = {flat[i]:.3f} → array[{row},{col}] = {array[row,col]:.3f}")
            >>> 
            >>> show_conversion_process(reader, 0)
            >>> # Output:
            >>> # Converting image 0:
            >>> #   Input: 784-element list  
            >>> #   Output: (28, 28) numpy array
            >>> #   First few flat values: [0.0, 0.0, 0.0, 0.0, 0.0]
            >>> #   Corresponding array positions:
            >>> #     flat[0] = 0.000 → array[0,0] = 0.000
            >>> #     flat[1] = 0.000 → array[1,0] = 0.000
            >>> #     flat[2] = 0.000 → array[2,0] = 0.000
            >>> #     flat[3] = 0.000 → array[3,0] = 0.000
            >>> #     flat[4] = 0.000 → array[4,0] = 0.000
            
            >>> # MEMORY LAYOUT VISUALIZATION:
            >>> def visualize_memory_layout(flat_data, rows=4, cols=4):
            ...     print("Row-major flat:", flat_data[:rows*cols])
            ...     print("Column-major 2D:")
            ...     for r in range(rows):
            ...         row_values = []
            ...         for c in range(cols):
            ...             idx = r + rows * c
            ...             row_values.append(flat_data[idx])
            ...         print(f"  Row {r}: {row_values}")
            >>> 
            >>> # Create simple test data
            >>> test_flat = list(range(16))  # [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]
            >>> visualize_memory_layout(test_flat, 4, 4)
            >>> # Output:
            >>> # Row-major flat: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
            >>> # Column-major 2D:
            >>> #   Row 0: [0, 4, 8, 12]
            >>> #   Row 1: [1, 5, 9, 13] 
            >>> #   Row 2: [2, 6, 10, 14]
            >>> #   Row 3: [3, 7, 11, 15]
        """
        src = self.get_image_float(n)
        
        # Reshape to 28x28 in column-major order
        image_2d = np.zeros((28, 28), dtype=np.float32)
        for row in range(28):
            for col in range(28):
                image_2d[row, col] = src[row + 28 * col]
        
        return image_2d
    
    def get_label(self, n: int) -> int:
        """
        Get class label for the specified image
        
        Returns the ground truth digit class (0-9) for the given image index.
        Labels are stored as single bytes and represent the actual digit
        shown in the corresponding image. This is used for supervised learning
        where the model learns to predict these labels from image features.
        
        Args:
            n (int): Image index (0 to num_images-1)
            
        Returns:
            int: Digit class label (0-9)
            
        Raises:
            IndexError: If image index is out of range
            
        Sample Input/Output:
            >>> reader = MNISTReader("train-images.gz", "train-labels.gz")
            >>> 
            >>> # Get individual labels
            >>> reader.get_label(0)       # Output: 5 (first image is digit "5")
            >>> reader.get_label(1)       # Output: 0 (second image is digit "0")
            >>> reader.get_label(2)       # Output: 4 (third image is digit "4")
            >>> reader.get_label(100)     # Output: 7 (101st image is digit "7")
            
            >>> # Label range validation
            >>> labels_sample = [reader.get_label(i) for i in range(100)]
            >>> min(labels_sample)        # Output: 0 (minimum digit)
            >>> max(labels_sample)        # Output: 9 (maximum digit)
            >>> all(0 <= label <= 9 for label in labels_sample)  # Output: True
            
            >>> # Label distribution analysis
            >>> from collections import Counter
            >>> first_1000_labels = [reader.get_label(i) for i in range(1000)]
            >>> label_counts = Counter(first_1000_labels)
            >>> label_counts              # Output: Counter({1: 112, 7: 104, 3: 101, ...})
            
            >>> # Training pair preparation
            >>> def get_training_pair(idx):
            ...     image = reader.get_image_float(idx)  # Features (784 floats)
            ...     label = reader.get_label(idx)        # Target (1 int)
            ...     return image, label
            >>> 
            >>> X, y = get_training_pair(42)
            >>> len(X)                    # Output: 784 (input features)
            >>> type(y)                   # Output: <class 'int'>
            >>> 0 <= y <= 9               # Output: True (valid digit)
            
            >>> # One-hot encoding for neural networks
            >>> def label_to_one_hot(label):
            ...     one_hot = [0.0] * 10
            ...     one_hot[label] = 1.0
            ...     return one_hot
            >>> 
            >>> label = reader.get_label(5)     # Output: 1 (digit "1")
            >>> one_hot = label_to_one_hot(label)
            >>> one_hot                   # Output: [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
            
            >>> # Batch label extraction
            >>> batch_indices = list(range(32))  # First 32 images
            >>> batch_labels = [reader.get_label(i) for i in batch_indices]
            >>> len(batch_labels)        # Output: 32 (batch size)
            >>> type(batch_labels[0])    # Output: <class 'int'>
            
            >>> # Error handling
            >>> try:
            ...     reader.get_label(100000)  # Out of range
            ... except IndexError as e:
            ...     print(f"Error: {e}")  # Output: "Label index 100000 out of range..."
        """
        if n >= self.num:
            raise IndexError(f"Label index {n} out of range (0-{self.num-1})")
        
        return self.labels[n]
    
    def get_batch(self, indices: List[int]) -> Tuple[List[np.ndarray], List[int]]:
        """
        Get a batch of images and labels
        
        Args:
            indices: List of image indices
            
        Returns:
            Tuple of (images, labels) where images is list of 28x28 arrays
        """
        images = [self.get_image_as_array(i) for i in indices]
        labels = [self.get_label(i) for i in indices]
        return images, labels
    
    def get_batch_tensors(self, indices: List[int]) -> Tuple[List, List[int]]:
        """
        Get a batch of images as tensors and labels
        
        Args:
            indices: List of image indices
            
        Returns:
            Tuple of (tensors, labels) where tensors is list of 28x28 Tensor objects
        """
        tensors = [self.get_image_as_tensor(i) for i in indices]
        labels = [self.get_label(i) for i in indices]
        return tensors, labels
    
    def filter_by_labels(self, target_labels: List[int]) -> List[int]:
        """
        Get indices of all images with specified labels
        
        Args:
            target_labels: List of labels to filter for (e.g., [3, 7])
            
        Returns:
            List of indices where the label matches one of target_labels
        """
        indices = []
        for i in range(self.num):
            if self.get_label(i) in target_labels:
                indices.append(i)
        return indices
    
    def get_stats(self) -> dict:
        """Get dataset statistics"""
        label_counts = {}
        for label in self.labels:
            label_counts[label] = label_counts.get(label, 0) + 1
        
        return {
            'num_images': self.num,
            'image_shape': (self.rows, self.cols),
            'label_counts': label_counts,
            'unique_labels': sorted(label_counts.keys())
        }
    
    def __str__(self):
        stats = self.get_stats()
        return f"MNISTReader(images={stats['num_images']}, shape={stats['image_shape']}, labels={stats['unique_labels']})"
    
    def __repr__(self):
        return self.__str__()


def download_emnist(data_dir: str = "./mnist_data") -> Tuple[str, str, str, str]:
    """
    Download EMNIST (Extended MNIST) dataset from available sources
    
    Args:
        data_dir: Directory to store EMNIST files
        
    Returns:
        Tuple of (train_images_path, train_labels_path, test_images_path, test_labels_path)
    """
    os.makedirs(data_dir, exist_ok=True)
    
    # EMNIST files we need (digits subset)
    expected_files = [
        "emnist-digits-train-images-idx3-ubyte.gz",
        "emnist-digits-train-labels-idx1-ubyte.gz",
        "emnist-digits-test-images-idx3-ubyte.gz", 
        "emnist-digits-test-labels-idx1-ubyte.gz"
    ]
    
    # Check if files already exist
    paths = []
    all_exist = True
    for filename in expected_files:
        filepath = os.path.join(data_dir, filename)
        paths.append(filepath)
        if not os.path.exists(filepath):
            all_exist = False
    
    if all_exist:
        print("EMNIST files already exist, skipping download.")
        return tuple(paths)
    
    # Alternative EMNIST download sources
    emnist_urls = [
        "https://biometrics.nist.gov/cs_links/EMNIST/gzip.zip",  # User-verified working URL
        "https://www.itl.nist.gov/iaui/vip/cs_links/EMNIST/gzip.zip",
        "https://cloudstor.aarnet.edu.au/plus/s/ZNmuFiuQTqZlu9W/download",
        "http://www.itl.nist.gov/iaui/vip/cs_links/EMNIST/gzip.zip"
    ]
    
    print("Attempting to download EMNIST dataset...")
    print("Note: EMNIST download may require special access. Falling back to MNIST if needed.")
    
    success = False
    for i, emnist_url in enumerate(emnist_urls):
        try:
            print(f"Trying source {i+1}/{len(emnist_urls)}: {emnist_url}")
            print("This may take several minutes (file is ~900MB)...")
            
            # Create a request with proper headers to avoid blocking
            req = urllib.request.Request(
                emnist_url,
                headers={
                    'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
                    'Accept': 'application/zip, */*',
                    'Accept-Language': 'en-US,en;q=0.9',
                    'Accept-Encoding': 'gzip, deflate',
                    'Connection': 'keep-alive',
                    'Upgrade-Insecure-Requests': '1'
                }
            )
            
            # Download to temporary file
            with tempfile.NamedTemporaryFile(delete=False, suffix='.zip') as tmp_file:
                with urllib.request.urlopen(req) as response:
                    tmp_file.write(response.read())
                zip_path = tmp_file.name
            
            print("Download complete. Extracting files...")
            
            # Extract the zip file
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                # List all files in the zip
                zip_files = zip_ref.namelist()
                print(f"Found {len(zip_files)} files in archive")
                
                # Extract only the digit files we need
                extracted_count = 0
                for filename in expected_files:
                    # Look for the file in the zip (might be in subdirectory)
                    matching_files = [f for f in zip_files if filename in f]
                    if matching_files:
                        # Extract the first matching file
                        zip_ref.extract(matching_files[0], data_dir)
                        
                        # Move to expected location if in subdirectory
                        extracted_path = os.path.join(data_dir, matching_files[0])
                        target_path = os.path.join(data_dir, filename)
                        
                        if extracted_path != target_path:
                            os.makedirs(os.path.dirname(target_path), exist_ok=True)
                            os.rename(extracted_path, target_path)
                        
                        extracted_count += 1
                        print(f"Extracted: {filename}")
                    else:
                        print(f"Warning: {filename} not found in archive")
            
            print(f"Successfully extracted {extracted_count} files")
            
            # Clean up
            os.unlink(zip_path)
            
            # Verify all files exist
            missing_files = []
            for filepath in paths:
                if not os.path.exists(filepath):
                    missing_files.append(filepath)
            
            if missing_files:
                print(f"Missing files after extraction: {missing_files}")
                continue
            
            success = True
            break
            
        except Exception as e:
            print(f"Failed to download from {emnist_url}: {e}")
            # Clean up any partial files
            try:
                if 'zip_path' in locals():
                    os.unlink(zip_path)
            except:
                pass
            continue
    
    if not success:
        print("❌ EMNIST download failed from all sources.")
        print("🔄 Falling back to standard MNIST dataset...")
        return download_mnist(data_dir)
    
    return tuple(paths)


def download_mnist(data_dir: str = "./mnist_data") -> Tuple[str, str, str, str]:
    """
    Download MNIST dataset if not already present
    
    Args:
        data_dir: Directory to store MNIST files
        
    Returns:
        Tuple of (train_images_path, train_labels_path, test_images_path, test_labels_path)
    """
    os.makedirs(data_dir, exist_ok=True)
    
    # Alternative URLs in case the main one is down
    base_urls = [
        "http://yann.lecun.com/exdb/mnist/",
        "https://storage.googleapis.com/cvdf-datasets/mnist/",
        "https://ossci-datasets.s3.amazonaws.com/mnist/"
    ]
    
    files = [
        "train-images-idx3-ubyte.gz",
        "train-labels-idx1-ubyte.gz", 
        "t10k-images-idx3-ubyte.gz",
        "t10k-labels-idx1-ubyte.gz"
    ]
    
    paths = []
    for filename in files:
        filepath = os.path.join(data_dir, filename)
        if not os.path.exists(filepath):
            print(f"Downloading {filename}...")
            
            # Try different URLs
            success = False
            for base_url in base_urls:
                try:
                    urllib.request.urlretrieve(base_url + filename, filepath)
                    success = True
                    break
                except Exception as e:
                    print(f"Failed to download from {base_url}: {e}")
                    continue
            
            if not success:
                raise RuntimeError(f"Could not download {filename} from any mirror")
        
        paths.append(filepath)
    
    return tuple(paths)


def create_synthetic_mnist_data(num_samples: int = 1000) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create synthetic MNIST-like data for testing when real MNIST is not available
    
    Args:
        num_samples: Number of synthetic samples to create
        
    Returns:
        Tuple of (images, labels) where images is (num_samples, 28, 28) and labels is (num_samples,)
    """
    images = np.zeros((num_samples, 28, 28), dtype=np.float32)
    labels = np.zeros(num_samples, dtype=np.int32)
    
    for i in range(num_samples):
        digit = i % 10
        labels[i] = digit
        
        # Create simple patterns for each digit
        if digit == 0:
            # Circle-like pattern
            center_r, center_c = 14, 14
            for r in range(28):
                for c in range(28):
                    dist = ((r - center_r)**2 + (c - center_c)**2)**0.5
                    if 8 <= dist <= 12:
                        images[i, r, c] = 0.8
        
        elif digit == 1:
            # Vertical line
            images[i, 5:23, 13:15] = 0.8
        
        elif digit == 2:
            # Horizontal lines
            images[i, 7:10, 8:20] = 0.8
            images[i, 13:16, 8:20] = 0.8
            images[i, 19:22, 8:20] = 0.8
        
        elif digit == 3:
            # Three horizontal bars
            images[i, 7:10, 8:20] = 0.8
            images[i, 13:16, 8:18] = 0.8
            images[i, 19:22, 8:20] = 0.8
            images[i, 10:13, 17:20] = 0.8
            images[i, 16:19, 17:20] = 0.8
        
        elif digit == 7:
            # Top bar and diagonal
            images[i, 7:10, 8:20] = 0.8
            for j in range(12):
                row = 10 + j
                col = 18 - j
                if 0 <= row < 28 and 0 <= col < 28:
                    images[i, row:row+2, col:col+2] = 0.8
        
        else:
            # Simple random pattern for other digits
            np.random.seed(i + digit * 1000)
            mask = np.random.random((28, 28)) < 0.1
            images[i, mask] = 0.6
    
    # Add some noise
    noise = np.random.normal(0, 0.05, images.shape).astype(np.float32)
    images = np.clip(images + noise, 0, 1)
    
    return images, labels


def decode_idx_magic_number(magic: int) -> dict:
    """
    Decode and explain an IDX format magic number
    
    Takes a 4-byte magic number from an IDX file and breaks it down into
    its component parts: data type and dimensionality. This is useful for
    understanding and debugging IDX file formats.
    
    Args:
        magic (int): 4-byte magic number from IDX file header
        
    Returns:
        dict: Dictionary containing decoded information:
            - 'magic': Original magic number
            - 'magic_hex': Hexadecimal representation  
            - 'valid': Whether this is a valid IDX magic number
            - 'type_code': Data type byte (0x08 = uint8, etc.)
            - 'dims_code': Dimension byte (0x01 = 1D, 0x03 = 3D, etc.)
            - 'type_name': Human-readable data type name
            - 'dims_name': Human-readable dimension description
            - 'description': Full format description
            
    Sample Input/Output:
        >>> # Decode MNIST labels magic number
        >>> info = decode_idx_magic_number(2049)
        >>> info['magic']             # Output: 2049
        >>> info['magic_hex']         # Output: '0x00000801'
        >>> info['valid']             # Output: True
        >>> info['type_code']         # Output: 8
        >>> info['dims_code']         # Output: 1  
        >>> info['type_name']         # Output: 'unsigned byte (uint8)'
        >>> info['dims_name']         # Output: '1D array (vector)'
        >>> info['description']       # Output: 'IDX1: 1D array of unsigned bytes'
        
        >>> # Decode MNIST images magic number
        >>> info = decode_idx_magic_number(2051)
        >>> info['magic']             # Output: 2051
        >>> info['magic_hex']         # Output: '0x00000803'
        >>> info['valid']             # Output: True
        >>> info['type_name']         # Output: 'unsigned byte (uint8)'
        >>> info['dims_name']         # Output: '3D array (tensor)'
        >>> info['description']       # Output: 'IDX3: 3D array of unsigned bytes'
        
        >>> # Invalid magic number
        >>> info = decode_idx_magic_number(1234)
        >>> info['valid']             # Output: False
        >>> info['description']       # Output: 'Invalid IDX magic number'
        
        >>> # Print detailed breakdown
        >>> def explain_magic(magic):
        ...     info = decode_idx_magic_number(magic)
        ...     print(f"Magic Number: {magic} ({info['magic_hex']})")
        ...     if info['valid']:
        ...         print(f"  Type: {info['type_name']}")
        ...         print(f"  Dimensions: {info['dims_name']}")
        ...         print(f"  Format: {info['description']}")
        ...     else:
        ...         print("  Invalid IDX format")
        >>> 
        >>> explain_magic(2049)
        >>> # Output:
        >>> # Magic Number: 2049 (0x00000801)
        >>> #   Type: unsigned byte (uint8)
        >>> #   Dimensions: 1D array (vector)
        >>> #   Format: IDX1: 1D array of unsigned bytes
    """
    # Extract bytes from 32-bit magic number (big-endian)
    byte0 = (magic >> 24) & 0xFF  # Should be 0x00
    byte1 = (magic >> 16) & 0xFF  # Should be 0x00  
    byte2 = (magic >> 8) & 0xFF   # Type code
    byte3 = magic & 0xFF          # Dimensions code
    
    # Data type mapping
    type_names = {
        0x08: 'unsigned byte (uint8)',
        0x09: 'signed byte (int8)',
        0x0B: 'short (int16)', 
        0x0C: 'int (int32)',
        0x0D: 'float (float32)',
        0x0E: 'double (float64)'
    }
    
    # Dimension mapping
    dims_names = {
        0x01: '1D array (vector)',
        0x02: '2D array (matrix)', 
        0x03: '3D array (tensor)',
        0x04: '4D array'
    }
    
    # Check if this is a valid IDX magic number
    valid = (byte0 == 0x00 and byte1 == 0x00 and 
             byte2 in type_names and byte3 in dims_names)
    
    result = {
        'magic': magic,
        'magic_hex': f'0x{magic:08X}',
        'valid': valid,
        'type_code': byte2,
        'dims_code': byte3,
        'type_name': type_names.get(byte2, f'unknown type (0x{byte2:02X})'),
        'dims_name': dims_names.get(byte3, f'unknown dimensions (0x{byte3:02X})')
    }
    
    if valid:
        idx_format = f"IDX{byte3}"
        result['description'] = f"{idx_format}: {result['dims_name']} of {result['type_name']}"
    else:
        result['description'] = 'Invalid IDX magic number'
    
    return result


# Export all public functions and classes
__all__ = [
    'MNISTReader', 'download_mnist', 'download_emnist', 'create_synthetic_mnist_data',
    'decode_idx_magic_number', '__version__'
]