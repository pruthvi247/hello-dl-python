"""
Test class to verify all documentation examples from mnist_utils.py

This test suite validates that all the sample inputs and outputs provided
in the comprehensive documentation for the mnist_utils module are accurate 
and executable. This specific test file focuses on mnist_utils.py functionality
including MNISTReader class methods and utility functions.
"""

import unittest
import numpy as np
import sys
import os
import tempfile
import gzip
import struct

# Add src to path to import pytensorlib
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from pytensorlib.mnist_utils import MNISTReader, create_synthetic_mnist_data
from pytensorlib import Tensor


class TestMNISTUtilsDocumentationExamples(unittest.TestCase):
    """Test class for verifying mnist_utils.py documentation examples"""
    
    @classmethod
    def setUpClass(cls):
        """Set up test fixtures for the entire test class"""
        # Check if EMNIST data files exist
        data_dir = os.path.join(os.path.dirname(__file__), '..', 'data')
        cls.train_images_path = os.path.join(data_dir, 'emnist-digits-train-images-idx3-ubyte.gz')
        cls.train_labels_path = os.path.join(data_dir, 'emnist-digits-train-labels-idx1-ubyte.gz')
        cls.test_images_path = os.path.join(data_dir, 'emnist-digits-test-images-idx3-ubyte.gz')
        cls.test_labels_path = os.path.join(data_dir, 'emnist-digits-test-labels-idx1-ubyte.gz')
        
        # Check if files exist, if not create synthetic data for testing
        if (os.path.exists(cls.train_images_path) and 
            os.path.exists(cls.train_labels_path)):
            cls.use_real_data = True
            cls.reader = MNISTReader(cls.train_images_path, cls.train_labels_path)
        else:
            cls.use_real_data = False
            # Create synthetic MNIST data for testing
            cls.synthetic_images, cls.synthetic_labels = create_synthetic_mnist_data(1000)
            cls._create_synthetic_files()
            cls.reader = MNISTReader(cls.synthetic_train_images, cls.synthetic_train_labels)
    
    @classmethod
    def _create_synthetic_files(cls):
        """Create synthetic MNIST-format files for testing"""
        # Create temporary files for synthetic data
        cls.temp_dir = tempfile.mkdtemp()
        cls.synthetic_train_images = os.path.join(cls.temp_dir, 'synthetic-train-images.gz')
        cls.synthetic_train_labels = os.path.join(cls.temp_dir, 'synthetic-train-labels.gz')
        
        # Write synthetic images file (IDX3 format)
        with gzip.open(cls.synthetic_train_images, 'wb') as f:
            # IDX3 header: magic (2051), count, rows (28), cols (28)
            f.write(struct.pack('>IIII', 2051, len(cls.synthetic_images), 28, 28))
            # Convert float images back to uint8 for file format
            for img in cls.synthetic_images:
                img_uint8 = (img * 255).astype(np.uint8)
                f.write(img_uint8.tobytes())
        
        # Write synthetic labels file (IDX1 format)  
        with gzip.open(cls.synthetic_train_labels, 'wb') as f:
            # IDX1 header: magic (2049), count
            f.write(struct.pack('>II', 2049, len(cls.synthetic_labels)))
            # Write labels as bytes
            f.write(cls.synthetic_labels.astype(np.uint8).tobytes())
    
    @classmethod
    def tearDownClass(cls):
        """Clean up test fixtures"""
        if not cls.use_real_data:
            # Clean up temporary files
            import shutil
            if hasattr(cls, 'temp_dir'):
                shutil.rmtree(cls.temp_dir, ignore_errors=True)
    
    def setUp(self):
        """Set up test fixtures before each test method"""
        pass
    
    def test_mnist_reader_initialization(self):
        """Test MNISTReader.__init__() documentation examples"""
        print("Testing MNISTReader initialization examples...")
        
        # Test basic properties after initialization
        self.assertIsInstance(self.reader.num, int)
        self.assertGreater(self.reader.num, 0)
        self.assertEqual(self.reader.rows, 28)
        self.assertEqual(self.reader.cols, 28)
        self.assertEqual(self.reader.stride, 784)  # 28*28
        
        # Test data structures are populated
        self.assertIsInstance(self.reader.images, list)
        self.assertIsInstance(self.reader.labels, list)
        self.assertIsInstance(self.reader.converted, dict)
        
        # Test dimensions match expectations
        self.assertEqual(len(self.reader.labels), self.reader.num)
        self.assertEqual(len(self.reader.images), self.reader.num * 784)
        self.assertEqual(len(self.reader.converted), self.reader.num)
        
        print("✓ MNISTReader initialization examples passed")
    
    def test_data_access_methods(self):
        """Test get_num(), get_image(), get_image_float(), get_label() documentation examples"""
        print("Testing data access method examples...")
        
        # Test get_num()
        num_images = self.reader.get_num()
        self.assertIsInstance(num_images, int)
        self.assertGreater(num_images, 0)
        self.assertEqual(num_images, self.reader.num)
        
        # Test get_image() - raw uint8 values
        if num_images > 0:
            raw_img = self.reader.get_image(0)
            # print(raw_img)
            self.assertIsInstance(raw_img, list)
            self.assertEqual(len(raw_img), 784)
            self.assertTrue(all(isinstance(pixel, int) for pixel in raw_img[:10]))
            self.assertTrue(all(0 <= pixel <= 255 for pixel in raw_img))
        
        # Test get_image_float() - normalized values
        if num_images > 0:
            norm_img = self.reader.get_image_float(0)
            self.assertIsInstance(norm_img, list)
            self.assertEqual(len(norm_img), 784)
            self.assertTrue(all(isinstance(pixel, float) for pixel in norm_img[:10]))
            self.assertTrue(all(0.0 <= pixel <= 1.0 for pixel in norm_img))
        
        # Test get_label() 
        if num_images > 0:
            label = self.reader.get_label(0)
            self.assertIsInstance(label, int)
            self.assertTrue(0 <= label <= 9)
        
        # Test index error handling
        with self.assertRaises(IndexError):
            self.reader.get_image(num_images)  # Out of range
        
        with self.assertRaises(IndexError):
            self.reader.get_label(num_images)  # Out of range
        
        print("✓ Data access method examples passed")
    
    def test_normalization_consistency(self):
        """Test that raw and normalized images are consistent"""
        print("Testing normalization consistency...")
        
        if self.reader.get_num() > 0:
            # Compare raw and normalized versions
            raw_img = self.reader.get_image(0)
            norm_img = self.reader.get_image_float(0)
            
            # Check normalization: raw[i] / 255.0 ≈ norm[i]
            for i in range(min(100, len(raw_img))):  # Test first 100 pixels
                expected = raw_img[i] / 255.0
                actual = norm_img[i]
                self.assertAlmostEqual(expected, actual, places=5)
        
        print("✓ Normalization consistency examples passed")
    
    def test_tensor_integration_methods(self):
        """Test push_image_to_tensor() and get_image_as_tensor() documentation examples"""
        print("Testing tensor integration method examples...")
        
        if self.reader.get_num() > 0:
            # Test get_image_as_tensor()
            img_tensor = self.reader.get_image_as_tensor(0)
            self.assertIsInstance(img_tensor, Tensor)
            self.assertEqual(img_tensor.shape, (28, 28))
            
            # Test push_image_to_tensor()
            dest_tensor = Tensor(28, 28)
            original_value = dest_tensor[0, 0]  # Should be 0.0 initially
            self.assertEqual(original_value, 0.0)
            
            self.reader.push_image_to_tensor(0, dest_tensor)
            # Tensor should now contain image data
            
            # Compare both methods produce same result
            for row in range(28):
                for col in range(28):
                    self.assertAlmostEqual(
                        float(img_tensor[row, col]), 
                        float(dest_tensor[row, col]), 
                        places=5
                    )
        
        print("✓ Tensor integration method examples passed")
    
    def test_image_as_array_method(self):
        """Test get_image_as_array() documentation examples"""
        print("Testing get_image_as_array() method examples...")
        
        if self.reader.get_num() > 0:
            img_array = self.reader.get_image_as_array(0)
            self.assertIsInstance(img_array, np.ndarray)
            self.assertEqual(img_array.shape, (28, 28))
            self.assertEqual(img_array.dtype, np.float32)
            self.assertTrue(np.all(img_array >= 0.0))
            self.assertTrue(np.all(img_array <= 1.0))
        
        print("✓ get_image_as_array() method examples passed")
    
    def test_get_image_as_array_comprehensive(self):
        """Test comprehensive get_image_as_array() functionality and documentation examples"""
        print("Testing comprehensive get_image_as_array() functionality...")
        
        if self.reader.get_num() > 0:
            # Test basic functionality
            img_array = self.reader.get_image_as_array(0)
            
            # Basic type and shape validation
            self.assertIsInstance(img_array, np.ndarray)
            self.assertEqual(img_array.shape, (28, 28))
            self.assertEqual(img_array.dtype, np.float32)
            
            # Value range validation (normalized 0.0-1.0)
            self.assertTrue(np.all(img_array >= 0.0))
            self.assertTrue(np.all(img_array <= 1.0))
            
            # Test element access from documentation examples
            # Array should be accessible with standard numpy indexing
            top_left = img_array[0, 0]
            center = img_array[14, 14] 
            bottom_right = img_array[27, 27]
            self.assertIsInstance(top_left, (float, np.float32))
            self.assertIsInstance(center, (float, np.float32))
            self.assertIsInstance(bottom_right, (float, np.float32))
            
            # Test consistency with get_image_float (column-major transformation)
            flat_img = self.reader.get_image_float(0)
            self.assertEqual(len(flat_img), 784)  # 28*28
            
            # Verify transformation: array[row,col] = flat[row + 28*col]
            transformation_consistent = True
            for row in range(28):
                for col in range(28):
                    flat_idx = row + 28 * col
                    if abs(flat_img[flat_idx] - img_array[row, col]) > 1e-6:
                        transformation_consistent = False
                        break
                if not transformation_consistent:
                    break
            
            self.assertTrue(transformation_consistent, 
                          "Array transformation should match: array[row,col] == flat[row + 28*col]")
            
            # Test specific examples from documentation
            # Check first few positions for consistency
            for row, col in [(0,0), (5,10), (14,14), (27,27)]:
                flat_idx = row + 28 * col
                if flat_idx < len(flat_img):
                    self.assertAlmostEqual(
                        float(img_array[row, col]), 
                        float(flat_img[flat_idx]), 
                        places=6,
                        msg=f"Mismatch at position ({row},{col}) -> flat[{flat_idx}]"
                    )
            
            # Test array statistics from documentation
            mean_intensity = np.mean(img_array)
            std_intensity = np.std(img_array)
            non_zero_pixels = np.count_nonzero(img_array)
            
            # Validate statistical properties
            self.assertIsInstance(mean_intensity, (float, np.float32, np.float64))
            self.assertIsInstance(std_intensity, (float, np.float32, np.float64))
            self.assertIsInstance(non_zero_pixels, (int, np.integer))
            self.assertGreaterEqual(mean_intensity, 0.0)
            self.assertLessEqual(mean_intensity, 1.0)
            self.assertGreaterEqual(std_intensity, 0.0)
            self.assertGreaterEqual(non_zero_pixels, 0)
            self.assertLessEqual(non_zero_pixels, 784)
            
            # Test batch processing from documentation examples
            if self.reader.get_num() >= 5:
                batch_size = 5
                batch_arrays = [self.reader.get_image_as_array(i) for i in range(batch_size)]
                
                # Validate batch creation
                self.assertEqual(len(batch_arrays), batch_size)
                self.assertTrue(all(isinstance(arr, np.ndarray) for arr in batch_arrays))
                self.assertTrue(all(arr.shape == (28, 28) for arr in batch_arrays))
                self.assertTrue(all(arr.dtype == np.float32 for arr in batch_arrays))
                
                # Test stacking for CNN-style processing
                batch_stack = np.stack(batch_arrays)
                self.assertEqual(batch_stack.shape, (batch_size, 28, 28))
                self.assertEqual(batch_stack.dtype, np.float32)
            
            # Test channel dimension expansion for CNN (documentation example)
            img_with_channel = np.expand_dims(img_array, axis=-1)
            self.assertEqual(img_with_channel.shape, (28, 28, 1))
            
            # Test array transformations from documentation
            rotated = np.rot90(img_array)
            self.assertEqual(rotated.shape, (28, 28))
            
            flipped = np.flipud(img_array)
            self.assertEqual(flipped.shape, (28, 28))
            
            transposed = img_array.T
            self.assertEqual(transposed.shape, (28, 28))
            
            # Test multiple images to ensure consistency
            if self.reader.get_num() >= 3:
                for img_idx in range(3):
                    test_array = self.reader.get_image_as_array(img_idx)
                    test_flat = self.reader.get_image_float(img_idx)
                    
                    # Verify transformation consistency for each image
                    for row in range(0, 28, 7):  # Sample every 7th row for efficiency
                        for col in range(0, 28, 7):  # Sample every 7th col
                            flat_idx = row + 28 * col
                            self.assertAlmostEqual(
                                float(test_array[row, col]), 
                                float(test_flat[flat_idx]), 
                                places=6
                            )
        
        # Test error handling
        with self.assertRaises(IndexError):
            self.reader.get_image_as_array(self.reader.get_num())  # Out of range
        
        print("✓ Comprehensive get_image_as_array() functionality passed")
    
    def test_batch_processing_methods(self):
        """Test get_batch() and get_batch_tensors() documentation examples"""
        print("Testing batch processing method examples...")
        
        if self.reader.get_num() >= 5:
            # Test get_batch()
            batch_indices = [0, 1, 2, 3, 4]
            images, labels = self.reader.get_batch(batch_indices)
            
            self.assertEqual(len(images), 5)
            self.assertEqual(len(labels), 5)
            self.assertTrue(all(isinstance(img, np.ndarray) for img in images))
            self.assertTrue(all(img.shape == (28, 28) for img in images))
            self.assertTrue(all(isinstance(label, int) for label in labels))
            self.assertTrue(all(0 <= label <= 9 for label in labels))
            
            # Test get_batch_tensors()
            tensors, batch_labels = self.reader.get_batch_tensors(batch_indices)
            
            self.assertEqual(len(tensors), 5)
            self.assertEqual(len(batch_labels), 5)
            self.assertTrue(all(isinstance(tensor, Tensor) for tensor in tensors))
            self.assertTrue(all(tensor.shape == (28, 28) for tensor in tensors))
            
            # Labels should be the same from both methods
            self.assertEqual(labels, batch_labels)
        
        print("✓ Batch processing method examples passed")
    
    def test_filter_by_labels_method(self):
        """Test filter_by_labels() documentation examples"""
        print("Testing filter_by_labels() method examples...")
        
        if self.reader.get_num() > 0:
            # Test filtering for specific digits
            target_labels = [0, 1]  # Look for digits 0 and 1
            filtered_indices = self.reader.filter_by_labels(target_labels)
            
            self.assertIsInstance(filtered_indices, list)
            # Verify all filtered indices have correct labels
            for idx in filtered_indices[:10]:  # Check first 10 results
                label = self.reader.get_label(idx)
                self.assertIn(label, target_labels)
        
        print("✓ filter_by_labels() method examples passed")
    
    def test_utility_methods(self):
        """Test get_stats(), __str__(), __repr__() documentation examples"""
        print("Testing utility method examples...")
        
        # Test get_stats()
        stats = self.reader.get_stats()
        self.assertIsInstance(stats, dict)
        self.assertIn('num_images', stats)
        self.assertIn('image_shape', stats)
        self.assertIn('label_counts', stats)
        self.assertIn('unique_labels', stats)
        
        self.assertEqual(stats['num_images'], self.reader.get_num())
        self.assertEqual(stats['image_shape'], (28, 28))
        self.assertIsInstance(stats['label_counts'], dict)
        self.assertIsInstance(stats['unique_labels'], list)
        
        # Test string representations
        str_repr = str(self.reader)
        self.assertIsInstance(str_repr, str)
        self.assertIn('MNISTReader', str_repr)
        
        repr_str = repr(self.reader)
        self.assertIsInstance(repr_str, str)
        self.assertIn('MNISTReader', repr_str)
        
        print("✓ Utility method examples passed")
    
    def test_synthetic_data_creation(self):
        """Test create_synthetic_mnist_data() documentation examples"""
        print("Testing synthetic data creation examples...")
        
        # Test default parameters
        images, labels = create_synthetic_mnist_data(100)
        self.assertEqual(images.shape, (100, 28, 28))
        self.assertEqual(labels.shape, (100,))
        self.assertEqual(images.dtype, np.float32)
        self.assertEqual(labels.dtype, np.int32)
        
        # Test value ranges
        self.assertTrue(np.all(images >= 0.0))
        self.assertTrue(np.all(images <= 1.0))
        self.assertTrue(np.all(labels >= 0))
        self.assertTrue(np.all(labels <= 9))
        
        # Test label distribution (should cycle through 0-9)
        expected_labels = [i % 10 for i in range(100)]
        np.testing.assert_array_equal(labels, expected_labels)
        
        print("✓ Synthetic data creation examples passed")
    
    def test_convert_all_images_method(self):
        """Test _convert_all_images() documentation examples and functionality"""
        print("Testing _convert_all_images() method examples...")
        
        # Test that conversion was called during initialization
        self.assertIsInstance(self.reader.converted, dict)
        self.assertEqual(len(self.reader.converted), self.reader.get_num())
        
        if self.reader.get_num() > 0:
            # Test that all images are cached in converted dictionary
            for img_idx in range(min(5, self.reader.get_num())):  # Test first 5 images
                self.assertIn(img_idx, self.reader.converted)
                converted_img = self.reader.converted[img_idx]
                
                # Test converted image properties
                self.assertIsInstance(converted_img, list)
                self.assertEqual(len(converted_img), 784)  # 28*28 pixels
                
                # Test all pixels are floats in range [0.0, 1.0]
                for pixel in converted_img:
                    self.assertIsInstance(pixel, float)
                    self.assertGreaterEqual(pixel, 0.0)
                    self.assertLessEqual(pixel, 1.0)
            
            # Test normalization accuracy: raw[i]/255.0 should equal converted[i]
            raw_img = self.reader.get_image(0)
            converted_img = self.reader.converted[0]
            
            self.assertEqual(len(raw_img), len(converted_img))
            
            # Verify normalization formula: pixel_value / 255.0
            for i in range(len(raw_img)):
                expected_normalized = raw_img[i] / 255.0
                actual_normalized = converted_img[i]
                self.assertAlmostEqual(expected_normalized, actual_normalized, places=6)
            
            # Test specific pixel value transformations from documentation
            # Find pixels with different values to test edge cases
            unique_values = set(raw_img[:100])  # Check first 100 pixels
            for raw_value in list(unique_values)[:5]:  # Test up to 5 different values
                # Find first occurrence of this value
                pixel_index = raw_img.index(raw_value)
                expected = raw_value / 255.0
                actual = converted_img[pixel_index]
                self.assertAlmostEqual(expected, actual, places=6)
                
                # Test specific cases mentioned in documentation
                if raw_value == 255:  # White pixel
                    self.assertAlmostEqual(actual, 1.0, places=6)
                elif raw_value == 0:  # Black pixel
                    self.assertAlmostEqual(actual, 0.0, places=6)
                elif raw_value == 128:  # Gray pixel (approximately)
                    self.assertAlmostEqual(actual, 128/255.0, places=6)
                    self.assertAlmostEqual(actual, 0.502, places=3)  # As in documentation
            
            # Test that get_image_float() returns the same as converted cache
            for img_idx in range(min(3, self.reader.get_num())):
                float_img_method = self.reader.get_image_float(img_idx)
                cached_img = self.reader.converted[img_idx]
                
                self.assertEqual(len(float_img_method), len(cached_img))
                for i in range(len(float_img_method)):
                    self.assertAlmostEqual(float_img_method[i], cached_img[i], places=6)
        
        # Test memory efficiency: converted dict should exist and be accessible
        import sys
        if self.reader.get_num() > 0:
            raw_size = sys.getsizeof(self.reader.images)
            converted_size = sys.getsizeof(self.reader.converted)
            print(f"Raw images size: {raw_size} bytes")
            # print(f"{type(self.reader.converted)} ")
            # print(f"{len(self.reader.converted)} converted images size: {converted_size} bytes")
            # print(f"lenght of keys in converted: {len(self.reader.converted.keys())}")
            # print(f"First 10 key in converted: {list(self.reader.converted.keys())[:10]}")
            # print(f"First converted image sample (first 10 pixels): {self.reader.converted[0][:10]}")
            # Both should be reasonable sizes (not testing exact sizes due to variability)
            self.assertGreater(raw_size, 0)
            self.assertGreater(converted_size, 0)
        
        print("✓ _convert_all_images() method examples passed")
    
    def test_edge_cases_and_error_handling(self):
        """Test edge cases and error conditions"""
        print("Testing edge cases and error handling...")
        
        # Test out-of-range positive indices (these should raise IndexError)
        with self.assertRaises(IndexError):
            self.reader.get_image(self.reader.get_num())  # Just beyond range
        
        with self.assertRaises(IndexError):
            self.reader.get_image_float(self.reader.get_num())  # Just beyond range
        
        with self.assertRaises(IndexError):
            self.reader.get_label(self.reader.get_num())  # Just beyond range
        
        # Test negative indices (Python allows these, so they might not raise IndexError)
        # Instead, let's test that they behave consistently
        if self.reader.get_num() > 0:
            try:
                # Negative indices might work in Python (wrap around)
                img_neg = self.reader.get_image(-1)
                # If it works, it should be the last image
                img_last = self.reader.get_image(self.reader.get_num() - 1)
                # They might be the same (Python list behavior)
            except IndexError:
                # If negative indices raise IndexError, that's also valid
                pass
        
        # Test empty batch
        empty_indices = []
        images, labels = self.reader.get_batch(empty_indices)
        self.assertEqual(len(images), 0)
        self.assertEqual(len(labels), 0)
        
        # Test tensor integration with wrong size tensor (if applicable)
        if self.reader.get_num() > 0:
            wrong_size_tensor = Tensor(32, 32)  # Wrong size
            # Note: The current implementation might not validate tensor size strictly
            # This is an area for potential improvement in the library
        
        print("✓ Edge cases and error handling passed")


def run_mnist_utils_documentation_tests():
    """Run all mnist_utils.py documentation example tests"""
    print("=" * 60)
    print("RUNNING MNIST_UTILS DOCUMENTATION EXAMPLE TESTS")
    print("=" * 60)
    
    # Create test suite
    suite = unittest.TestLoader().loadTestsFromTestCase(TestMNISTUtilsDocumentationExamples)
    
    # Run tests with detailed output
    runner = unittest.TextTestRunner(verbosity=2, stream=sys.stdout)
    result = runner.run(suite)
    
    print("\n" + "=" * 60)
    if result.wasSuccessful():
        print("🎉 ALL MNIST_UTILS DOCUMENTATION EXAMPLES PASSED!")
        print(f"✓ {result.testsRun} tests completed successfully")
    else:
        print("❌ SOME MNIST_UTILS DOCUMENTATION EXAMPLES FAILED")
        print(f"Failed: {len(result.failures)}, Errors: {len(result.errors)}")
        
        if result.failures:
            print("\nFailures:")
            for test, failure in result.failures:
                print(f"  - {test}: {failure}")
        
        if result.errors:
            print("\nErrors:")
            for test, error in result.errors:
                print(f"  - {test}: {error}")
    
    print("=" * 60)
    return result.wasSuccessful()


if __name__ == "__main__":
    # Configure environment
    os.environ.setdefault('PYTHONPATH', 
                         os.path.join(os.path.dirname(__file__), '..', 'src'))
    
    # Run the tests
    success = run_mnist_utils_documentation_tests()
    
    # Exit with appropriate code
    sys.exit(0 if success else 1)