# GitHub Copilot Configuration for ROCm Testing Suite

This directory contains configuration files for GitHub Copilot to provide better context-aware suggestions for the ROCm Testing Suite project.

## Project Context

This is a ROCm (Radeon Open Compute) testing framework that validates GPU computing capabilities across multiple Python machine learning frameworks including PyTorch, TensorFlow, Keras, and other scientific libraries.

### Key Technologies
- **ROCm**: AMD's GPU computing platform
- **PyTorch**: Deep learning framework with ROCm backend
- **TensorFlow**: Machine learning platform with GPU acceleration
- **Keras**: High-level neural networks API
- **pgmpy**: Probabilistic graphical models library
- **pytest**: Testing framework
- **Python 3.8+**: Primary development language

### Common Patterns

#### ROCm Detection
```python
import subprocess

def check_rocm_installation():
    try:
        result = subprocess.run(["/opt/rocm/bin/rocm-smi"], 
                              stdout=subprocess.PIPE, 
                              stderr=subprocess.PIPE)
        return result.returncode == 0
    except FileNotFoundError:
        return False
```

#### PyTorch ROCm Testing
```python
import torch

def test_pytorch_rocm():
    if torch.cuda.is_available():
        device = 'cuda'
        tensor = torch.tensor([[1, 2], [3, 4]], device=device)
        return tensor.device.type == 'cuda'
    return False
```

#### TensorFlow GPU Testing
```python
import tensorflow as tf

def test_tensorflow_gpu():
    gpus = tf.config.experimental.list_physical_devices('GPU')
    return len(gpus) > 0
```

### Project Structure Context
- `src/tests/`: Test modules for different frameworks
- `src/tools/`: Utility tools for ROCm operations
- `src/utils/`: Helper functions and common utilities
- `docs/`: Documentation and setup guides

### Coding Conventions
- Use type hints for all function parameters and return values
- Include comprehensive error handling for GPU operations
- Add informative print statements for test results
- Follow pytest conventions for test naming and structure
