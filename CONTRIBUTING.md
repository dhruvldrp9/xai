# Contributing to XAI Toolkit

Thank you for your interest in contributing to the XAI Toolkit! This document provides guidelines and instructions for contributing to this project.

## Code of Conduct

By participating in this project, you agree to abide by our Code of Conduct.

## How to Contribute

1. Fork the repository
2. Create a new branch for your feature or bugfix
3. Make your changes
4. Run tests to ensure your changes don't break existing functionality
5. Submit a pull request

## Development Setup

1. Clone your fork of the repository:
```bash
git clone https://github.com/your-username/xai_toolkit.git
cd xai_toolkit
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Code Style

We use the following tools to maintain code quality:

- [Black](https://github.com/psf/black) for code formatting
- [isort](https://pycqa.github.io/isort/) for import sorting
- [flake8](https://flake8.pycqa.org/) for linting
- [mypy](http://mypy-lang.org/) for type checking

Before submitting a pull request, please run:

```bash
# Format code
black .

# Sort imports
isort .

# Run linter
flake8

# Run type checker
mypy .
```

## Running Tests

We use [pytest](https://docs.pytest.org/) for testing. To run the tests:

```bash
pytest
```

## Documentation

- Update the README.md if you add new features
- Add docstrings to all functions and classes
- Update the examples if your changes affect them

## Pull Request Process

1. Update the README.md with details of changes if needed
2. Update the documentation if you're changing functionality
3. The PR will be merged once you have the sign-off of at least one maintainer

## Reporting Bugs

If you find a bug, please create an issue with the following information:

- A clear, descriptive title
- The exact steps which reproduce the problem
- The expected behavior
- The actual behavior
- Screenshots if applicable
- The version of the software you're using

## Feature Requests

We welcome feature requests! Please create an issue with:

- A clear, descriptive title
- A detailed description of the proposed feature
- Any relevant examples or use cases

## Questions?

If you have any questions, please open an issue or contact the maintainers.

Thank you for contributing to XAI Toolkit! 