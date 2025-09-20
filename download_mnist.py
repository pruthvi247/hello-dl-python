#!/usr/bin/env python3
"""
EMNIST Data Downloader

This script downloads the EMNIST (Extended MNIST) dataset files for use with PyTensorLib.
EMNIST includes handwritten digits and letters from NIST Special Database 19.
Run this to get real EMNIST data instead of synthetic data.
"""

import sys
import os

# Add src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from pytensorlib.mnist_utils import download_emnist


def main():
    """Download EMNIST dataset"""
    print("📦 EMNIST Dataset Downloader")
    print("=" * 40)
    print("EMNIST (Extended MNIST) is a comprehensive dataset of handwritten")
    print("digits and letters derived from NIST Special Database 19.")
    print()
    
    # Create data directory
    data_dir = os.path.join(os.path.dirname(__file__), 'data')
    
    try:
        print(f"📁 Creating data directory: {data_dir}")
        os.makedirs(data_dir, exist_ok=True)
        
        print("🌐 Downloading EMNIST dataset...")
        print("   Source: NIST EMNIST (Extended MNIST)")
        print("   This may take several minutes (file is ~900MB)...")
        print("   Note: If EMNIST is not accessible, will fall back to MNIST")
        print("   Please be patient...")
        
        # Download the files
        train_images, train_labels, test_images, test_labels = download_emnist(data_dir)
        
        # Check if we got EMNIST or MNIST
        if "emnist" in os.path.basename(train_images):
            dataset_type = "EMNIST"
            print("\n✅ EMNIST download completed successfully!")
        else:
            dataset_type = "MNIST (fallback)"
            print("\n✅ MNIST download completed successfully!")
            print("   Note: EMNIST was not accessible, using standard MNIST dataset")
        print(f"📊 Downloaded {dataset_type} files:")
        print(f"   Training images: {train_images}")
        print(f"   Training labels: {train_labels}")
        print(f"   Test images: {test_images}")
        print(f"   Test labels: {test_labels}")
        
        # Verify files exist and show sizes
        total_size = 0
        for path in [train_images, train_labels, test_images, test_labels]:
            if os.path.exists(path):
                size = os.path.getsize(path)
                total_size += size
                print(f"   {os.path.basename(path)}: {size:,} bytes")
            else:
                print(f"   ❌ Missing: {path}")
        
        print(f"\n📏 Total download size: {total_size:,} bytes ({total_size/1024/1024:.1f} MB)")
        
        # Test loading
        print(f"\n🧪 Testing {dataset_type} loading...")
        from pytensorlib.mnist_utils import MNISTReader
        
        reader = MNISTReader(train_images, train_labels)
        print(f"   ✅ Successfully loaded {reader.num} training images")
        print(f"   📐 Image dimensions: {reader.rows}x{reader.cols}")
        
        test_reader = MNISTReader(test_images, test_labels)
        print(f"   ✅ Successfully loaded {test_reader.num} test images")
        
        # Show some statistics
        stats = reader.get_stats()
        print(f"\n📈 Dataset Statistics:")
        print(f"   Available digits: {stats['unique_labels']}")
        print(f"   Label distribution: {stats['label_counts']}")
        
        print(f"\n🎉 {dataset_type} dataset is ready to use!")
        print(f"   You can now run examples with real data:")
        print(f"   python examples/three_or_seven.py")
        print(f"\n📚 About {dataset_type}:")
        if dataset_type == "EMNIST":
            print(f"   - Extended version of the MNIST dataset")
            print(f"   - Contains handwritten digits (0-9) and letters (A-Z, a-z)")
            print(f"   - Higher quality and more diverse than original MNIST")
        else:
            print(f"   - Classic handwritten digit dataset")
            print(f"   - Contains digits 0-9 from 250 writers")
            print(f"   - Widely used benchmark dataset")
        print(f"   - Perfect for digit classification experiments")
        
        return 0
        
    except Exception as e:
        print(f"\n❌ Error downloading EMNIST data: {e}")
        print(f"\n🔧 Troubleshooting:")
        print(f"   1. Check your internet connection")
        print(f"   2. Make sure you have write permissions to: {data_dir}")
        print(f"   3. Ensure you have enough disk space (~900MB)")
        print(f"   4. Try running with different network settings")
        print(f"   5. Contact your network administrator if behind firewall")
        print(f"\n🌐 Alternative: The function will automatically fall back to")
        print(f"   standard MNIST if EMNIST download fails.")
        print(f"\n⚠️  Note: This example requires real data and will not use synthetic fallbacks.")
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)