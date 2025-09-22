#!/usr/bin/env python3
"""
Simple test to verify the data loading behavior.
"""

import sys
import os

# Add src to Python path for PyTensorLib imports
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'src'))

# Mock the download functions to simulate failure
import unittest.mock

def mock_download_emnist(data_dir):
    raise Exception("EMNIST download failed (simulated)")

def mock_download_mnist(data_dir):
    raise Exception("MNIST download failed (simulated)")

def test_data_loading_exit():
    """Test that the main function exits when data loading fails."""
    
    print("🧪 Testing data loading exit behavior...")
    
    # Patch the download functions to simulate failure
    with unittest.mock.patch('pytensorlib.mnist_utils.download_emnist', side_effect=mock_download_emnist):
        with unittest.mock.patch('pytensorlib.mnist_utils.download_mnist', side_effect=mock_download_mnist):
            try:
                # Import and run the main function
                from examples.perceptron_37_learn import main
                main()
                print("❌ ERROR: main() returned instead of exiting!")
                return False
            except SystemExit as e:
                if e.code == 1:
                    print("✅ SUCCESS: main() correctly called sys.exit(1)")
                    return True
                else:
                    print(f"❌ ERROR: main() called sys.exit({e.code}), expected sys.exit(1)")
                    return False
            except Exception as e:
                print(f"❌ ERROR: main() raised unexpected exception: {e}")
                return False

if __name__ == "__main__":
    success = test_data_loading_exit()
    if success:
        print("\n🎉 Test passed: Program correctly exits when data loading fails!")
    else:
        print("\n❌ Test failed!")