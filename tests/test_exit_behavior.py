#!/usr/bin/env python3
"""
Test script to demonstrate that perceptron_37_learn.py exits when data loading fails.
"""

import sys
import os
import subprocess
import tempfile
import shutil

def test_exit_on_data_failure():
    """Test that the perceptron script exits when data loading fails."""
    
    print("🧪 Testing exit behavior when data loading fails...")
    
    # Create a temporary directory to simulate missing data
    with tempfile.TemporaryDirectory() as temp_dir:
        # Copy the perceptron script to temp directory
        script_path = os.path.join(temp_dir, "perceptron_37_learn.py")
        original_script = "examples/perceptron_37_learn.py"
        shutil.copy2(original_script, script_path)
        
        # Modify the script to use a non-existent data directory and disable downloads
        with open(script_path, 'r') as f:
            content = f.read()
        
        # Replace data directory to non-existent path and disable network downloads
        modified_content = content.replace(
            'data_dir = "./data"',
            'data_dir = "./nonexistent_data_dir"'
        ).replace(
            'def download_emnist(data_dir):',
            'def download_emnist(data_dir):\n    raise Exception("Network downloads disabled for test")\n\ndef original_download_emnist(data_dir):'
        ).replace(
            'def download_mnist(data_dir):',
            'def download_mnist(data_dir):\n    raise Exception("Network downloads disabled for test")\n\ndef original_download_mnist(data_dir):'
        )
        
        with open(script_path, 'w') as f:
            f.write(modified_content)
        
        # Set up Python environment
        python_exe = sys.executable
        
        # Run the modified script
        try:
            result = subprocess.run(
                [python_exe, script_path],
                capture_output=True,
                text=True,
                timeout=30,  # Timeout after 30 seconds
                cwd=temp_dir
            )
            
            print(f"📤 Exit code: {result.returncode}")
            print(f"📝 Last few lines of output:")
            
            # Show the last few lines of stdout
            stdout_lines = result.stdout.strip().split('\n')
            for line in stdout_lines[-10:]:  # Show last 10 lines
                if line.strip():
                    print(f"   {line}")
            
            # Show stderr if any
            if result.stderr.strip():
                print(f"📝 Error output:")
                stderr_lines = result.stderr.strip().split('\n')
                for line in stderr_lines[-5:]:  # Show last 5 lines
                    if line.strip():
                        print(f"   {line}")
            
            # Check if program exited as expected
            if result.returncode == 1:
                print("✅ SUCCESS: Program correctly exited with code 1 when data loading failed")
                return True
            else:
                print(f"❌ UNEXPECTED: Program exited with code {result.returncode}, expected 1")
                return False
                
        except subprocess.TimeoutExpired:
            print("⏰ TIMEOUT: Program did not exit within 30 seconds")
            return False
        except Exception as e:
            print(f"❌ ERROR running test: {e}")
            return False

if __name__ == "__main__":
    success = test_exit_on_data_failure()
    if success:
        print("\n🎉 Test passed: Program correctly exits when data loading fails!")
    else:
        print("\n❌ Test failed: Program did not exit as expected!")
    
    sys.exit(0 if success else 1)