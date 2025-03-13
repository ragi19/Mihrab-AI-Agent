# Contributing to LLM Agents Framework

Thank you for your interest in contributing to the LLM Agents Framework! This document provides guidelines and instructions for contributing to this project.

## Code of Conduct

By participating in this project, you agree to abide by our Code of Conduct. Please read it before contributing.

## How Can I Contribute?

### Reporting Bugs

Before creating bug reports, please check the issue tracker to see if the problem has already been reported. If it has and the issue is still open, add a comment to the existing issue instead of opening a new one.

When you are creating a bug report, please include as many details as possible:

- Use a clear and descriptive title
- Describe the exact steps to reproduce the problem
- Provide specific examples to demonstrate the steps
- Describe the behavior you observed after following the steps
- Explain which behavior you expected to see instead and why
- Include screenshots if applicable
- Include details about your configuration and environment

### Suggesting Enhancements

Enhancement suggestions are tracked as GitHub issues. When creating an enhancement suggestion, please include:

- A clear and descriptive title
- A detailed description of the proposed functionality
- Any possible implementation details or ideas
- Why this enhancement would be useful to most users

### Pull Requests

- Fill in the required template
- Do not include issue numbers in the PR title
- Include screenshots and animated GIFs in your pull request whenever possible
- Follow the Python style guide
- Include tests for your changes
- Document new code based on the project's documentation style
- End all files with a newline

## Development Process

### Setting Up Development Environment

1. Fork the repository
2. Clone your fork locally
3. Install development dependencies:
   ```bash
   python install.py
   ```

### Code Style

This project uses:
- [Black](https://black.readthedocs.io/en/stable/) for code formatting
- [isort](https://pycqa.github.io/isort/) for import sorting
- [mypy](https://mypy.readthedocs.io/en/stable/) for type checking
- [flake8](https://flake8.pycqa.org/en/latest/) for linting

Before submitting a pull request, please run:
```bash
# Format code
black src tests
isort src tests

# Check types
mypy src tests

# Run linter
flake8 src tests
```

### Testing

All new code should include appropriate tests. Run the test suite with:
```bash
pytest
```

For coverage information:
```bash
pytest --cov=src
```

### Documentation

- Update the README.md if necessary
- Add docstrings to all public modules, functions, classes, and methods
- Keep docstrings up-to-date as code changes

## Release Process

1. Update version in pyproject.toml
2. Update CHANGELOG.md
3. Create a new GitHub release with the version number
4. Tag the release in git
5. Build and upload to PyPI (maintainers only)

## Questions?

If you have any questions, please create an issue with the label "question".

Thank you for contributing to LLM Agents Framework! 