# Contributing to ROCm Testing Suite

Thank you for your interest in contributing to the ROCm Testing Suite! This document provides guidelines and information for contributors.

## Code of Conduct

This project adheres to a code of conduct that promotes a welcoming and inclusive environment. By participating, you are expected to uphold this code.

## How to Contribute

### Reporting Bugs

1. **Search existing issues** to ensure the bug hasn't been reported already
2. **Use the bug report template** when creating a new issue
3. **Provide detailed information** including:
   - System configuration (OS, Python version, ROCm version)
   - Steps to reproduce the issue
   - Expected vs. actual behavior
   - Relevant log output or error messages

### Suggesting Features

1. **Check the project roadmap** to see if the feature is already planned
2. **Use the feature request template** when proposing new features
3. **Describe the use case** and why the feature would be valuable
4. **Consider implementation complexity** and potential impact

### Contributing Code

#### Development Setup

1. **Fork the repository** on GitHub
2. **Clone your fork** locally:
   ```bash
   git clone https://github.com/your-username/ROCM.git
   cd ROCM
   ```
3. **Create a virtual environment**:
   ```bash
   python3 -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```
4. **Install development dependencies**:
   ```bash
   pip install -r requirements.txt
   pip install -r requirements-dev.txt
   ```

#### Development Workflow

1. **Create a feature branch**:
   ```bash
   git checkout -b feature/your-feature-name
   ```
2. **Make your changes** following the coding standards
3. **Write or update tests** for your changes
4. **Run the test suite**:
   ```bash
   pytest src/tests/ -v
   ```
5. **Check code quality**:
   ```bash
   black src/
   flake8 src/
   mypy src/
   ```
6. **Commit your changes** with clear, descriptive messages
7. **Push to your fork** and **create a pull request**

#### Coding Standards

**Python Style**
- Follow PEP 8 style guidelines
- Use Black for code formatting (line length: 88 characters)
- Use meaningful variable and function names
- Include docstrings for all public functions and classes

**Type Hints**
- Use type hints for all function parameters and return values
- Import types from `typing` module when needed
- Use `mypy` for static type checking

**Documentation**
- Write clear, concise docstrings using Google style
- Update README.md if your changes affect user-facing functionality
- Add comments for complex logic or algorithms

**Testing**
- Write unit tests for new functionality
- Ensure all tests pass before submitting PR
- Aim for high test coverage (>90%)
- Use descriptive test names that explain what is being tested

#### Example Code Structure

```python
from typing import Optional, Dict, Any
import subprocess

def check_rocm_installation() -> bool:
    """Check if ROCm is properly installed on the system.
    
    Returns:
        bool: True if ROCm is installed and accessible, False otherwise.
        
    Raises:
        subprocess.SubprocessError: If there's an error running rocm-smi.
    """
    try:
        result = subprocess.run(
            ["/opt/rocm/bin/rocm-smi"], 
            stdout=subprocess.PIPE, 
            stderr=subprocess.PIPE,
            timeout=30
        )
        return result.returncode == 0
    except (FileNotFoundError, subprocess.TimeoutExpired) as e:
        print(f"ROCm check failed: {e}")
        return False
```

### Testing Guidelines

#### Test Categories

**Unit Tests**
- Test individual functions and classes in isolation
- Mock external dependencies (ROCm commands, GPU operations)
- Fast execution (< 1 second per test)

**Integration Tests**
- Test component interactions
- May require ROCm installation for full validation
- Medium execution time (1-10 seconds per test)

**System Tests**
- End-to-end testing with real ROCm hardware
- Test complete workflows and user scenarios
- Longer execution time (10+ seconds per test)

#### Test Naming Convention

```python
def test_check_rocm_installation_success():
    """Test ROCm detection when properly installed."""
    pass

def test_check_rocm_installation_failure():
    """Test ROCm detection when not installed."""
    pass

def test_pytorch_tensor_operations_gpu():
    """Test PyTorch tensor operations on GPU."""
    pass
```

#### Test Organization

```
src/tests/
├── unit/                 # Unit tests
│   ├── test_rocm_detection.py
│   └── test_framework_checks.py
├── integration/          # Integration tests
│   ├── test_pytorch_integration.py
│   └── test_tensorflow_integration.py
├── system/              # System/E2E tests
│   └── test_full_workflow.py
└── conftest.py          # Pytest configuration and fixtures
```

### Documentation

#### Documentation Types

**API Documentation**
- Generated from docstrings using Sphinx
- Include examples and usage patterns
- Keep up-to-date with code changes

**User Guides**
- Step-by-step tutorials for common tasks
- Installation and setup instructions
- Troubleshooting guides

**Developer Documentation**
- Architecture decisions and design rationale
- Contributing guidelines (this document)
- Development setup and workflow

#### Documentation Standards

- Use Markdown for most documentation
- Include code examples with expected output
- Provide context and explain "why" not just "how"
- Keep language clear and accessible

### Pull Request Process

#### Before Submitting

1. **Ensure all tests pass** locally
2. **Run code quality checks** (black, flake8, mypy)
3. **Update documentation** if needed
4. **Write clear commit messages** following conventional commits format
5. **Rebase your branch** against the latest main branch

#### PR Requirements

1. **Clear title and description** explaining the changes
2. **Reference related issues** using keywords (fixes #123)
3. **Include test coverage** for new functionality
4. **Pass all CI checks** (automated testing, linting)
5. **Request review** from maintainers

#### PR Template

```markdown
## Description
Brief description of changes and motivation.

## Type of Change
- [ ] Bug fix (non-breaking change which fixes an issue)
- [ ] New feature (non-breaking change which adds functionality)
- [ ] Breaking change (fix or feature that would cause existing functionality to not work as expected)
- [ ] Documentation update

## Testing
- [ ] Unit tests added/updated
- [ ] Integration tests added/updated
- [ ] Manual testing performed

## Checklist
- [ ] Code follows project style guidelines
- [ ] Self-review completed
- [ ] Comments added for complex logic
- [ ] Documentation updated
- [ ] Tests added and passing
```

### Release Process

#### Versioning

We follow [Semantic Versioning](https://semver.org/):
- **MAJOR**: Incompatible API changes
- **MINOR**: Backward-compatible functionality additions
- **PATCH**: Backward-compatible bug fixes

#### Release Workflow

1. **Create release branch** from main
2. **Update version numbers** in relevant files
3. **Update CHANGELOG.md** with release notes
4. **Run full test suite** including manual testing
5. **Create GitHub release** with detailed notes
6. **Tag the release** following v{major}.{minor}.{patch} format

### Community

#### Communication Channels

- **GitHub Issues**: Bug reports and feature requests
- **GitHub Discussions**: Questions and general discussion
- **Pull Requests**: Code review and collaboration

#### Getting Help

- **Documentation**: Check the docs/ directory first
- **Issues**: Search existing issues for similar problems
- **Discussions**: Ask questions in GitHub Discussions
- **Maintainers**: Tag @hkevin01 for urgent issues

### Recognition

Contributors will be recognized in:
- **CONTRIBUTORS.md** file listing all contributors
- **Release notes** acknowledging significant contributions
- **GitHub repository** insights and contributor graphs

Thank you for contributing to the ROCm Testing Suite! Your efforts help make GPU computing more accessible and reliable for the entire community.
