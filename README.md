# ROCm Testing Suite

A comprehensive testing suite for validating ROCm (Radeon Open Compute) functionality with Python machine learning frameworks and libraries.

## Overview

This project provides a collection of tests and utilities to verify ROCm installation and compatibility with popular Python ML frameworks including PyTorch, TensorFlow, Keras, and pgmpy. It's designed to help developers ensure their ROCm setup is working correctly for GPU-accelerated machine learning workloads.

## Features

- **ROCm Installation Verification**: Check if ROCm is properly installed and accessible
- **GPU Information Retrieval**: Display detailed GPU information using `rocm-smi`
- **Framework Testing**: Test PyTorch, TensorFlow, and Keras with ROCm backend
- **Bayesian Network Testing**: Validate pgmpy functionality for probabilistic modeling
- **Comprehensive Test Suite**: Multiple test scenarios for different use cases

## Prerequisites

### System Requirements
- AMD GPU with ROCm support
- Ubuntu 20.04/22.04 or compatible Linux distribution
- Python 3.8+ with virtual environment support

### ROCm Installation
Ensure ROCm is installed on your system. Follow the official [ROCm Installation Guide](https://rocmdocs.amd.com/en/latest/Installation_Guide/Installation-Guide.html).

## Quick Start

### 1. Clone and Setup
```bash
git clone <repository-url>
cd ROCM
```

### 2. Create Virtual Environment
```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Run Tests
```bash
# Run all tests
python -m pytest src/tests/ -v

# Run specific tests
python src/tests/test_rocm.py
python src/tests/test_pytorch.py
python src/tests/test_tensorflow.py
```

## Project Structure

```
ROCM/
├── README.md                 # Project documentation
├── requirements.txt          # Python dependencies
├── pyproject.toml           # Project configuration
├── .github/                 # GitHub workflows and templates
├── .copilot/                # GitHub Copilot configuration
├── .vscode/                 # VS Code settings
├── docs/                    # Additional documentation
│   ├── directions.txt       # Setup instructions
│   ├── git_directions.txt   # Git workflow
│   └── github_directions.txt # GitHub integration
└── src/                     # Source code
    ├── tests/               # Test modules
    │   ├── test_rocm.py     # ROCm installation tests
    │   ├── test_pytorch.py  # PyTorch ROCm tests
    │   ├── test_tensorflow.py # TensorFlow ROCm tests
    │   ├── test_keras.py    # Keras compatibility tests
    │   ├── test_pgmpy.py    # Bayesian network tests
    │   ├── test_tf_help.py  # TensorFlow utilities
    │   └── test_torch_min.py # Minimal PyTorch test
    ├── tools/               # Utility tools
    └── utils/               # Helper utilities
```

## Test Descriptions

### ROCm Core Tests (`test_rocm.py`)
- Verifies ROCm installation
- Checks `rocm-smi` availability
- Displays GPU information and status

### PyTorch Tests (`test_pytorch.py`)
- Tests PyTorch ROCm backend availability
- Performs tensor operations on GPU
- Validates device placement and computation

### TensorFlow Tests (`test_tensorflow.py`)
- Complete MNIST neural network training
- Tests TensorFlow with ROCm backend
- Includes detailed training process explanation

### Keras Tests (`test_keras.py`)
- Verifies Keras availability within TensorFlow
- Checks version compatibility

### Probabilistic Modeling Tests (`test_pgmpy.py`)
- Bayesian network construction and inference
- Demonstrates probabilistic reasoning capabilities

## Development

### Adding New Tests
1. Create test files in `src/tests/`
2. Follow the existing naming convention: `test_<framework>.py`
3. Include proper error handling and informative output
4. Update this README with test descriptions

### Contributing
1. Fork the repository
2. Create a feature branch
3. Add tests and documentation
4. Submit a pull request

## Troubleshooting

### Common Issues

**ROCm Not Detected**
- Verify ROCm installation: `/opt/rocm/bin/rocm-smi`
- Check GPU compatibility with ROCm
- Ensure proper driver installation

**PyTorch ROCm Issues**
- Install ROCm-compatible PyTorch version
- Verify PYTORCH_ROCM_ARCH environment variable
- Check GPU memory availability

**Virtual Environment Issues**
- Install `python3-venv`: `sudo apt install python3-venv`
- Use system Python if distribution packages conflict
- Consider using conda for complex dependency management

### Getting Help
- Check the `docs/` directory for detailed setup instructions
- Review test output for specific error messages
- Consult ROCm documentation for hardware-specific issues

## License

This project is open source. See LICENSE file for details.

## Acknowledgments

- AMD ROCm team for GPU compute platform
- PyTorch and TensorFlow communities for ROCm support
- Contributors to the Python scientific computing ecosystem