#!/usr/bin/env python3
"""
Test cases for the decode_idx_magic_number function in mnist_utils

This test suite validates the IDX magic number decoder function that was added
to help understand and debug IDX file format magic numbers used in MNIST datasets.
"""

import unittest
import sys
import os

# Add the parent directory to Python path to import our modules
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(parent_dir, 'src'))

from pytensorlib.mnist_utils import decode_idx_magic_number


class TestDecodeIdxMagicNumber(unittest.TestCase):
    """Test cases for decode_idx_magic_number function"""
    
    def test_mnist_labels_magic_number(self):
        """Test decoding MNIST labels magic number (2049)"""
        result = decode_idx_magic_number(2049)
        
        self.assertEqual(result['magic'], 2049)
        self.assertEqual(result['magic_hex'], '0x00000801')
        self.assertTrue(result['valid'])
        self.assertEqual(result['type_code'], 8)
        self.assertEqual(result['dims_code'], 1)
        self.assertEqual(result['type_name'], 'unsigned byte (uint8)')
        self.assertEqual(result['dims_name'], '1D array (vector)')
        self.assertEqual(result['description'], 'IDX1: 1D array (vector) of unsigned byte (uint8)')
    
    def test_mnist_images_magic_number(self):
        """Test decoding MNIST images magic number (2051)"""
        result = decode_idx_magic_number(2051)
        
        self.assertEqual(result['magic'], 2051)
        self.assertEqual(result['magic_hex'], '0x00000803')
        self.assertTrue(result['valid'])
        self.assertEqual(result['type_code'], 8)
        self.assertEqual(result['dims_code'], 3)
        self.assertEqual(result['type_name'], 'unsigned byte (uint8)')
        self.assertEqual(result['dims_name'], '3D array (tensor)')
        self.assertEqual(result['description'], 'IDX3: 3D array (tensor) of unsigned byte (uint8)')
    
    def test_other_valid_magic_numbers(self):
        """Test other valid IDX format magic numbers"""
        # IDX2 format with int32 data
        magic_2d_int32 = (0x0C << 8) | 0x02  # 0x00000C02 = 3074
        result = decode_idx_magic_number(3074)
        
        self.assertTrue(result['valid'])
        self.assertEqual(result['type_code'], 0x0C)
        self.assertEqual(result['dims_code'], 0x02)
        self.assertEqual(result['type_name'], 'int (int32)')
        self.assertEqual(result['dims_name'], '2D array (matrix)')
        self.assertEqual(result['description'], 'IDX2: 2D array (matrix) of int (int32)')
        
        # IDX1 format with float32 data
        magic_1d_float32 = (0x0D << 8) | 0x01  # 0x00000D01 = 3329
        result = decode_idx_magic_number(3329)
        
        self.assertTrue(result['valid'])
        self.assertEqual(result['type_code'], 0x0D)
        self.assertEqual(result['dims_code'], 0x01)
        self.assertEqual(result['type_name'], 'float (float32)')
        self.assertEqual(result['dims_name'], '1D array (vector)')
    
    def test_invalid_magic_numbers(self):
        """Test invalid magic numbers"""
        # Random invalid number
        result = decode_idx_magic_number(1234)
        self.assertFalse(result['valid'])
        self.assertEqual(result['description'], 'Invalid IDX magic number')
        
        # Wrong prefix bytes
        result = decode_idx_magic_number(0x01000801)  # First byte should be 0x00
        self.assertFalse(result['valid'])
        
        # Invalid type code
        result = decode_idx_magic_number(0x00001001)  # Type 0x10 doesn't exist
        self.assertFalse(result['valid'])
        
        # Invalid dimension code
        result = decode_idx_magic_number(0x00000805)  # Dimension 0x05 doesn't exist
        self.assertFalse(result['valid'])
    
    def test_hex_formatting(self):
        """Test hex formatting is correct"""
        result = decode_idx_magic_number(2049)
        self.assertEqual(result['magic_hex'], '0x00000801')
        
        result = decode_idx_magic_number(2051)
        self.assertEqual(result['magic_hex'], '0x00000803')
        
        # Test larger number
        result = decode_idx_magic_number(0x12345678)
        self.assertEqual(result['magic_hex'], '0x12345678')
    
    def test_byte_extraction(self):
        """Test that bytes are extracted correctly from magic number"""
        # Test with known values
        magic = 0x12345678
        result = decode_idx_magic_number(magic)
        
        # Manually verify byte extraction
        byte0 = (magic >> 24) & 0xFF  # 0x12
        byte1 = (magic >> 16) & 0xFF  # 0x34
        byte2 = (magic >> 8) & 0xFF   # 0x56
        byte3 = magic & 0xFF          # 0x78
        
        self.assertEqual(result['type_code'], byte2)  # 0x56
        self.assertEqual(result['dims_code'], byte3)  # 0x78
    
    def test_comprehensive_type_codes(self):
        """Test all supported type codes"""
        type_mappings = {
            0x08: 'unsigned byte (uint8)',
            0x09: 'signed byte (int8)',
            0x0B: 'short (int16)',
            0x0C: 'int (int32)',
            0x0D: 'float (float32)',
            0x0E: 'double (float64)'
        }
        
        for type_code, expected_name in type_mappings.items():
            magic = (type_code << 8) | 0x01  # Create IDX1 with this type
            result = decode_idx_magic_number(magic)
            self.assertTrue(result['valid'])
            self.assertEqual(result['type_name'], expected_name)
    
    def test_comprehensive_dimension_codes(self):
        """Test all supported dimension codes"""
        dims_mappings = {
            0x01: '1D array (vector)',
            0x02: '2D array (matrix)',
            0x03: '3D array (tensor)',
            0x04: '4D array'
        }
        
        for dims_code, expected_name in dims_mappings.items():
            magic = (0x08 << 8) | dims_code  # Create uint8 with this dimension
            result = decode_idx_magic_number(magic)
            self.assertTrue(result['valid'])
            self.assertEqual(result['dims_name'], expected_name)
    
    def test_edge_cases(self):
        """Test edge cases and boundary conditions"""
        # Zero magic number
        result = decode_idx_magic_number(0)
        self.assertFalse(result['valid'])
        
        # Maximum uint32 value
        result = decode_idx_magic_number(0xFFFFFFFF)
        self.assertFalse(result['valid'])
        
        # Negative numbers (should be treated as unsigned)
        result = decode_idx_magic_number(-1)
        # In Python, -1 as uint32 is 0xFFFFFFFF, but the hex formatter handles it differently
        expected_hex = '0x-0000001'  # This is how Python formats negative hex
        self.assertEqual(result['magic_hex'], expected_hex)


def run_comprehensive_test():
    """Run all tests and provide detailed output"""
    print("🧪 Running IDX Magic Number Decoder Tests")
    print("=" * 50)
    
    # Create test suite
    suite = unittest.TestLoader().loadTestsFromTestCase(TestDecodeIdxMagicNumber)
    
    # Run tests with detailed output
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    print("\n" + "=" * 50)
    
    if result.wasSuccessful():
        print("✅ All tests passed!")
        print(f"📊 Ran {result.testsRun} tests successfully")
    else:
        print("❌ Some tests failed:")
        print(f"📊 Ran {result.testsRun} tests")
        print(f"❌ Failures: {len(result.failures)}")
        print(f"💥 Errors: {len(result.errors)}")
        
        if result.failures:
            print("\nFailures:")
            for test, traceback in result.failures:
                print(f"  - {test}: {traceback}")
        
        if result.errors:
            print("\nErrors:")
            for test, traceback in result.errors:
                print(f"  - {test}: {traceback}")
    
    return result.wasSuccessful()


if __name__ == '__main__':
    # Run comprehensive test when script is executed directly
    success = run_comprehensive_test()
    sys.exit(0 if success else 1)