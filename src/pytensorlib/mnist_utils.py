"""
MNIST Dataset Reader and Utilities

This module provides utilities for reading and processing MNIST dataset files.
It handles the IDX file format used by MNIST and provides convenient interfaces
for loading images and labels.

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
        
        Args:
            images_path: Path to gzipped images file (e.g., train-images-idx3-ubyte.gz)
            labels_path: Path to gzipped labels file (e.g., train-labels-idx1-ubyte.gz)
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
        """Load and parse MNIST data files"""
        
        # Load labels file
        with gzip.open(labels_path, 'rb') as f:
            # Read IDX1 header: magic number (4 bytes) + number of items (4 bytes)
            magic, num_labels = struct.unpack('>II', f.read(8))
            
            if magic != 2049:
                raise ValueError(f"Wrong magic number in labels file: {magic}")
            
            # Read all labels
            self.labels = list(f.read(num_labels))
        
        # Load images file
        with gzip.open(images_path, 'rb') as f:
            # Read IDX3 header: magic (4) + num items (4) + rows (4) + cols (4)
            magic, num_images, rows, cols = struct.unpack('>IIII', f.read(16))
            
            if magic != 2051:
                raise ValueError(f"Wrong magic number in images file: {magic}")
            
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
        """Convert all images to normalized float format and cache them"""
        for n in range(self.num):
            pos = n * self.stride
            # Normalize pixel values from 0-255 to 0-1
            float_image = [self.images[pos + i] / 255.0 for i in range(self.stride)]
            self.converted[n] = float_image
    
    def get_num(self) -> int:
        """Get number of images in dataset"""
        return self.num
    
    def get_image(self, n: int) -> List[int]:
        """Get raw image as list of uint8 values"""
        if n >= self.num:
            raise IndexError(f"Image index {n} out of range (0-{self.num-1})")
        
        pos = n * self.stride
        return self.images[pos:pos + self.stride]
    
    def get_image_float(self, n: int) -> List[float]:
        """Get normalized image as list of float values (0-1)"""
        if n >= self.num:
            raise IndexError(f"Image index {n} out of range (0-{self.num-1})")
        
        if n not in self.converted:
            raise RuntimeError(f"Could not find image {n}")
        
        return self.converted[n]
    
    def push_image_to_tensor(self, n: int, dest):
        """
        Copy image data into a 28x28 tensor
        
        Args:
            n: Image index
            dest: Destination tensor (must be 28x28 parameter tensor)
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
        Get image as a 28x28 tensor
        
        Args:
            n: Image index
            
        Returns:
            28x28 tensor containing the image
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
        Get image as a 28x28 numpy array
        
        Args:
            n: Image index
            
        Returns:
            28x28 numpy array containing the image
        """
        src = self.get_image_float(n)
        
        # Reshape to 28x28 in column-major order
        image_2d = np.zeros((28, 28), dtype=np.float32)
        for row in range(28):
            for col in range(28):
                image_2d[row, col] = src[row + 28 * col]
        
        return image_2d
    
    def get_label(self, n: int) -> int:
        """Get label for image n"""
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


# Export all public functions and classes
__all__ = [
    'MNISTReader', 'download_mnist', 'download_emnist', 'create_synthetic_mnist_data',
    '__version__'
]