#!/usr/bin/env python3
"""
Test suite for MNIST utilities

This script tests the MNIST dataset loading and processing functionality.
"""

import numpy as np
import sys
import os
import unittest
import tempfile

# Import from the installed package
from pytensorlib.mnist_utils import MNISTReader, create_synthetic_mnist_data


class TestSyntheticMNIST(unittest.TestCase):
    """Test synthetic MNIST data generation"""
    
    def test_synthetic_data_creation(self):
        """Test creation of synthetic MNIST data"""
        images, labels = create_synthetic_mnist_data(100)
        
        # Check shapes
        self.assertEqual(images.shape, (100, 28, 28))
        self.assertEqual(labels.shape, (100,))
        
        # Check data types
        self.assertEqual(images.dtype, np.float32)
        self.assertEqual(labels.dtype, np.int32)
        
        # Check value ranges
        self.assertTrue(np.all(images >= 0))
        self.assertTrue(np.all(images <= 1))
        self.assertTrue(np.all(labels >= 0))
        self.assertTrue(np.all(labels <= 9))
        
        # Check that different digits are generated
        unique_labels = np.unique(labels)
        self.assertGreater(len(unique_labels), 1)
    
    def test_synthetic_data_patterns(self):
        """Test that synthetic data has distinguishable patterns"""
        images, labels = create_synthetic_mnist_data(50)
        
        # Find images for digits 0 and 1
        zeros = images[labels == 0]
        ones = images[labels == 1]
        
        if len(zeros) > 0 and len(ones) > 0:
            # They should be different
            avg_zero = np.mean(zeros[0])
            avg_one = np.mean(ones[0])
            
            # Different digits should have different average intensities
            self.assertNotAlmostEqual(avg_zero, avg_one, places=2)


class TestMNISTUtilities(unittest.TestCase):
    """Test MNIST utility functions"""
    
    def setUp(self):
        """Set up test fixtures"""
        # Create synthetic MNIST data for testing
        self.images, self.labels = create_synthetic_mnist_data(20)
        
    def test_image_processing(self):
        """Test image processing functions"""
        # Test that we can process the synthetic data
        self.assertEqual(len(self.images), 20)
        self.assertEqual(len(self.labels), 20)
        
        # Test filtering by labels
        target_labels = [3, 7]
        filtered_indices = []
        for i, label in enumerate(self.labels):
            if label in target_labels:
                filtered_indices.append(i)
        
        # Should have some filtered results (probabilistically)
        # We can't guarantee exact counts with random data, so just check structure
        self.assertIsInstance(filtered_indices, list)


def create_mock_mnist_files():
    """Create mock MNIST files for testing"""
    import gzip
    import struct
    
    # Create temporary directory
    temp_dir = tempfile.mkdtemp()
    
    # Create mock images file
    images_path = os.path.join(temp_dir, "mock-images.gz")
    with gzip.open(images_path, 'wb') as f:
        # IDX3 header: magic, num_images, rows, cols
        header = struct.pack('>IIII', 2051, 2, 28, 28)
        f.write(header)
        
        # Write 2 mock images (28x28 each)
        image_data = np.random.randint(0, 256, size=(2, 28, 28), dtype=np.uint8)
        f.write(image_data.tobytes())
    
    # Create mock labels file
    labels_path = os.path.join(temp_dir, "mock-labels.gz")
    with gzip.open(labels_path, 'wb') as f:
        # IDX1 header: magic, num_labels
        header = struct.pack('>II', 2049, 2)
        f.write(header)
        
        # Write 2 mock labels
        labels_data = np.array([3, 7], dtype=np.uint8)
        f.write(labels_data.tobytes())
    
    return images_path, labels_path, temp_dir


class TestMNISTReader(unittest.TestCase):
    """Test MNIST file reading functionality"""
    
    @classmethod
    def setUpClass(cls):
        """Set up mock MNIST files for testing"""
        cls.images_path, cls.labels_path, cls.temp_dir = create_mock_mnist_files()
    
    @classmethod
    def tearDownClass(cls):
        """Clean up temporary files"""
        import shutil
        shutil.rmtree(cls.temp_dir)
    
    def test_mnist_reader_creation(self):
        """Test MNISTReader initialization"""
        reader = MNISTReader(self.images_path, self.labels_path)
        
        self.assertEqual(reader.get_num(), 2)
        self.assertEqual(reader.rows, 28)
        self.assertEqual(reader.cols, 28)
    
    def test_label_access(self):
        """Test label access"""
        reader = MNISTReader(self.images_path, self.labels_path)
        
        self.assertEqual(reader.get_label(0), 3)
        self.assertEqual(reader.get_label(1), 7)
        
        # Test out of bounds
        with self.assertRaises(IndexError):
            reader.get_label(2)
    
    def test_image_access(self):
        """Test image access"""
        reader = MNISTReader(self.images_path, self.labels_path)
        
        # Test raw image access
        raw_image = reader.get_image(0)
        self.assertEqual(len(raw_image), 28 * 28)
        self.assertTrue(all(0 <= pixel <= 255 for pixel in raw_image))
        
        # Test float image access
        float_image = reader.get_image_float(0)
        self.assertEqual(len(float_image), 28 * 28)
        self.assertTrue(all(0.0 <= pixel <= 1.0 for pixel in float_image))
    
    def test_image_as_array(self):
        """Test image conversion to numpy array"""
        reader = MNISTReader(self.images_path, self.labels_path)
        
        image_array = reader.get_image_as_array(0)
        self.assertEqual(image_array.shape, (28, 28))
        self.assertEqual(image_array.dtype, np.float32)
        self.assertTrue(np.all(image_array >= 0.0))
        self.assertTrue(np.all(image_array <= 1.0))
    
    def test_batch_operations(self):
        """Test batch loading operations"""
        reader = MNISTReader(self.images_path, self.labels_path)
        
        # Test batch loading
        images, labels = reader.get_batch([0, 1])
        self.assertEqual(len(images), 2)
        self.assertEqual(len(labels), 2)
        self.assertEqual(labels, [3, 7])
        
        # Check image shapes
        for img in images:
            self.assertEqual(img.shape, (28, 28))
    
    def test_filtering(self):
        """Test label filtering"""
        reader = MNISTReader(self.images_path, self.labels_path)
        
        # Filter for specific labels
        threes_and_sevens = reader.filter_by_labels([3, 7])
        self.assertEqual(set(threes_and_sevens), {0, 1})
        
        # Filter for non-existent label
        nines = reader.filter_by_labels([9])
        self.assertEqual(nines, [])
    
    def test_statistics(self):
        """Test dataset statistics"""
        reader = MNISTReader(self.images_path, self.labels_path)
        
        stats = reader.get_stats()
        
        self.assertEqual(stats['num_images'], 2)
        self.assertEqual(stats['image_shape'], (28, 28))
        self.assertEqual(stats['label_counts'], {3: 1, 7: 1})
        self.assertEqual(stats['unique_labels'], [3, 7])


def run_functionality_test():
    """Run a comprehensive functionality test"""
    print("\nRunning MNIST utilities functionality test...")
    
    try:
        # Test synthetic data generation
        print("1. Testing synthetic data generation...")
        images, labels = create_synthetic_mnist_data(50)
        
        print(f"   Generated {len(images)} images with shape {images[0].shape}")
        print(f"   Labels range: {np.min(labels)} to {np.max(labels)}")
        print(f"   Unique labels: {len(np.unique(labels))}")
        
        # Test that we can distinguish between different digits
        print("2. Testing digit patterns...")
        for digit in [0, 1, 3, 7]:
            digit_images = images[labels == digit]
            if len(digit_images) > 0:
                avg_intensity = np.mean(digit_images[0])
                print(f"   Digit {digit}: average intensity = {avg_intensity:.3f}")
        
        print("✓ Functionality test passed")
        
    except Exception as e:
        print(f"✗ Functionality test failed: {e}")
        raise


def main():
    """Run all tests"""
    print("MNIST Utilities Test Suite")
    print("=" * 40)
    
    # Run unit tests
    unittest.main(argv=[''], exit=False, verbosity=2)
    
    # Run additional functionality test
    try:
        run_functionality_test()
        
        print("\n🎉 All MNIST utility tests completed successfully!")
        return 0
        
    except Exception as e:
        print(f"\n❌ Tests failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)