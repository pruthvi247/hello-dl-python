#!/usr/bin/env python3
"""
Setup script for PyTensorLib - A Deep Learning Framework in Python

This script installs PyTensorLib as a proper Python package.
"""

from setuptools import setup, find_packages
import os


def read_readme():
    """Read the README file"""
    readme_path = os.path.join(os.path.dirname(__file__), 'README.md')
    if os.path.exists(readme_path):
        with open(readme_path, 'r', encoding='utf-8') as f:
            return f.read()
    return "PyTensorLib - A Deep Learning Framework in Python"


def read_requirements():
    """Read requirements from requirements.txt if it exists"""
    req_path = os.path.join(os.path.dirname(__file__), 'requirements.txt')
    if os.path.exists(req_path):
        with open(req_path, 'r', encoding='utf-8') as f:
            return [line.strip() for line in f if line.strip() and not line.startswith('#')]
    return ['numpy>=1.18.0']


setup(
    name="pytensorlib",
    version="1.0.0",
    author="PyTensorLib Contributors",
    author_email="pytensorlib@example.com",
    description="A Deep Learning Framework in Python with Automatic Differentiation",
    long_description=read_readme(),
    long_description_content_type="text/markdown",
    url="https://github.com/pytensorlib/pytensorlib",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    python_requires=">=3.7",
    install_requires=read_requirements(),
    extras_require={
        "dev": [
            "pytest>=6.0",
            "pytest-cov>=2.10",
        ],
        "test": [
            "pytest>=6.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "pytensorlib-test=pytensorlib:quick_test",
        ],
    },
    include_package_data=True,
    zip_safe=False,
    keywords="deep-learning machine-learning neural-networks automatic-differentiation tensor",
    project_urls={
        "Bug Reports": "https://github.com/pytensorlib/pytensorlib/issues",
        "Source": "https://github.com/pytensorlib/pytensorlib",
        "Documentation": "https://github.com/pytensorlib/pytensorlib/blob/main/README.md",
    },
)