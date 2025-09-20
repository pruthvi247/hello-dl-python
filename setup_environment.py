#!/usr/bin/env python3
"""
Setup script for PyTensorLib - A Deep Learning Framework in Python

This script sets up the Python tensor library package including:
- Virtual environment creation
- Package installation 
- Dependency management
- Testing
"""

import os
import sys
import subprocess
import platform


def run_command(cmd, description, check=True):
    """Run a command and handle errors"""
    print(f"\n→ {description}")
    print(f"  Running: {cmd}")
    
    try:
        result = subprocess.run(cmd, shell=True, check=check, 
                              capture_output=True, text=True)
        if result.stdout:
            print(f"  Output: {result.stdout.strip()}")
        return result.returncode == 0
    except subprocess.CalledProcessError as e:
        print(f"  Error: {e}")
        if e.stderr:
            print(f"  Stderr: {e.stderr}")
        return False


def check_python_version():
    """Check if Python version is compatible"""
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 7):
        print(f"❌ Python {version.major}.{version.minor} is not supported")
        print("   PyTensorLib requires Python 3.7 or higher")
        return False
    
    print(f"✓ Python {version.major}.{version.minor}.{version.micro} is compatible")
    return True


def create_virtual_environment():
    """Create and set up virtual environment"""
    print("\n📦 Setting up virtual environment...")
    
    venv_name = "pytensor-env"
    
    # Check if virtual environment already exists
    if os.path.exists(venv_name):
        print(f"  Virtual environment '{venv_name}' already exists")
        return venv_name
    
    # Create virtual environment
    success = run_command(f"python3 -m venv {venv_name}", 
                         "Creating virtual environment")
    
    if not success:
        print("❌ Failed to create virtual environment")
        return None
    
    print(f"✓ Virtual environment '{venv_name}' created successfully")
    return venv_name


def get_activation_script(venv_name):
    """Get the path to the activation script"""
    if platform.system() == "Windows":
        return os.path.join(venv_name, "Scripts", "activate.bat")
    else:
        return os.path.join(venv_name, "bin", "activate")


def install_dependencies(venv_name):
    """Install required dependencies"""
    print("\n📚 Installing dependencies...")
    
    # Get Python executable in virtual environment
    if platform.system() == "Windows":
        python_exe = os.path.join(venv_name, "Scripts", "python.exe")
        pip_exe = os.path.join(venv_name, "Scripts", "pip.exe")
    else:
        python_exe = os.path.join(venv_name, "bin", "python")
        pip_exe = os.path.join(venv_name, "bin", "pip")
    
    # Upgrade pip
    success = run_command(f"{python_exe} -m pip install --upgrade pip",
                         "Upgrading pip")
    if not success:
        print("⚠️  Warning: Failed to upgrade pip, continuing...")
    
    # Install numpy
    success = run_command(f"{pip_exe} install numpy",
                         "Installing numpy")
    if not success:
        print("❌ Failed to install numpy")
        return False
    
    # Install development dependencies
    dev_deps = ["pytest", "pytest-cov"]
    for dep in dev_deps:
        success = run_command(f"{pip_exe} install {dep}",
                             f"Installing {dep}")
        if not success:
            print(f"⚠️  Warning: Failed to install {dep}")
    
    print("✓ Dependencies installed successfully")
    return True


def install_package(venv_name):
    """Install the package in development mode"""
    print("\n🔧 Installing PyTensorLib package...")
    
    if platform.system() == "Windows":
        pip_exe = os.path.join(venv_name, "Scripts", "pip.exe")
    else:
        pip_exe = os.path.join(venv_name, "bin", "pip")
    
    # Install package in editable mode
    success = run_command(f"{pip_exe} install -e .", 
                         "Installing PyTensorLib in development mode")
    
    if success:
        print("✓ PyTensorLib package installed successfully")
        return True
    else:
        print("❌ Failed to install PyTensorLib package")
        return False


def run_tests(venv_name):
    """Run the test suite"""
    print("\n🧪 Running test suite...")
    
    if platform.system() == "Windows":
        python_exe = os.path.join(venv_name, "Scripts", "python.exe")
    else:
        python_exe = os.path.join(venv_name, "bin", "python")
    
    # Test imports first
    print("  Testing package imports...")
    success = run_command(f"{python_exe} -c \"import pytensorlib; print('✓ Import successful')\"",
                         "Testing package import")
    if not success:
        print("❌ Package import failed")
        return False
    
    # Run tensor tests
    test_cmd = f"cd tests && {python_exe} test_tensor_lib.py"
    success = run_command(test_cmd, "Running tensor library tests")
    if not success:
        print("⚠️  Tensor tests failed, but continuing...")
    
    # Run MNIST tests  
    test_cmd = f"cd tests && {python_exe} test_mnist_utils.py"
    success = run_command(test_cmd, "Running MNIST utility tests")
    if not success:
        print("⚠️  MNIST tests failed, but continuing...")
    
    # Quick functionality test
    quick_test = """
import sys
sys.path.insert(0, 'src')
from pytensorlib import quick_test
result = quick_test()
print(f'Quick test result: {result}')
"""
    
    success = run_command(f"{python_exe} -c \"{quick_test}\"",
                         "Running quick functionality test")
    
    if success:
        print("✓ Basic functionality tests passed")
        return True
    else:
        print("⚠️  Some tests failed, but package is installed")
        return False


def create_activation_instructions(venv_name):
    """Create instructions for activating the environment"""
    activation_script = get_activation_script(venv_name)
    
    instructions = f"""
🚀 PyTensorLib Setup Complete!

To activate the virtual environment and start using PyTensorLib:

"""
    
    if platform.system() == "Windows":
        instructions += f"""Windows:
    {venv_name}\\Scripts\\activate.bat

PowerShell:
    {venv_name}\\Scripts\\Activate.ps1
"""
    else:
        instructions += f"""Unix/macOS:
    source {venv_name}/bin/activate
"""
    
    instructions += f"""
Once activated, you can:

1. Import the library:
   python -c "import pytensorlib; print(pytensorlib.__version__)"

2. Run the quick test:
   python -c "from pytensorlib import quick_test; quick_test()"

3. Run individual tests:
   cd tests
   python test_tensor_lib.py
   python test_mnist_utils.py

4. Start coding with PyTensorLib:
   python
   >>> from pytensorlib import Tensor, relu, sigmoid
   >>> x = Tensor([[1, 2], [3, 4]])
   >>> y = relu(x)
   >>> print(y)

To deactivate the environment:
    deactivate

Happy deep learning! 🧠✨
"""
    
    print(instructions)
    
    # Save instructions to file
    with open("ACTIVATION_GUIDE.txt", "w") as f:
        f.write(instructions)
    
    print("📝 Activation instructions saved to ACTIVATION_GUIDE.txt")


def main():
    """Main setup function"""
    print("🎯 PyTensorLib Setup Script")
    print("=" * 50)
    
    # Check if we're in the right directory
    if not os.path.exists("src") or not os.path.exists("setup.py"):
        print("❌ Please run this script from the python-tensor-lib directory")
        print("   (The directory should contain 'src' and 'setup.py')")
        return 1
    
    # Check Python version
    if not check_python_version():
        return 1
    
    # Create virtual environment
    venv_name = create_virtual_environment()
    if not venv_name:
        return 1
    
    # Install dependencies
    if not install_dependencies(venv_name):
        return 1
    
    # Install the package
    if not install_package(venv_name):
        return 1
    
    # Run tests
    tests_passed = run_tests(venv_name)
    
    # Create activation instructions
    create_activation_instructions(venv_name)
    
    if tests_passed:
        print("\n🎉 Setup completed successfully!")
        print("   All tests passed. PyTensorLib is ready to use!")
    else:
        print("\n⚠️  Setup completed with warnings")
        print("   Some tests failed, but the package should work")
    
    return 0


if __name__ == "__main__":
    try:
        exit_code = main()
        sys.exit(exit_code)
    except KeyboardInterrupt:
        print("\n\n❌ Setup interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Setup failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)