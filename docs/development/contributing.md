# Contributing to SLAF

Thank you for your interest in contributing to SLAF! This guide will help you get started.

## Prerequisites

- **Python 3.10+** - SLAF requires Python 3.10 or higher
- **Git** - For version control
- **uv** - For dependency management (recommended)

## Development Setup

```python
# 1. Fork the repository on GitHub
# Go to https://github.com/slaf-project/slaf and click "Fork"

# 2. Clone your fork
git clone https://github.com/YOUR_USERNAME/slaf.git
cd slaf

# 3. Add upstream remote
git remote add upstream https://github.com/slaf-project/slaf.git

# 4. Install development dependencies
uv pip install -e ".[dev,test,docs]"

# 5. Install pre-commit hooks (runs linting/formatting automatically)
uv run pre-commit install

# 6. Run tests to verify setup
pytest tests/
```

## Development Workflow

### 1. Create a Feature Branch

```bash
git checkout -b feature/your-feature-name
```

### 2. Make Your Changes

- Follow the existing code style (enforced by pre-commit hooks)
- Add tests for new functionality
- Update documentation as needed
- Add type hints to new functions
- For API changes, follow the [docstring template](docstring_template.md)

### 3. Commit Your Changes

```bash
git add .
git commit -m "feat: add your feature description"
```

### 4. Push and Create a Pull Request

```bash
git push origin feature/your-feature-name
```

Then create a pull request on GitHub.

## Code Style

### Python

- Follow [PEP 8](https://pep8.org/) style guidelines
- Use type hints for all function parameters and return values
- Write docstrings using Google style
- Keep functions focused and small
- Use meaningful variable names

### Testing

```python
# Run all tests
uv run pytest tests/

# Run specific test file
uv run pytest tests/test_slaf.py

# Run with coverage
uv run pytest pytest --cov=slaf tests/
```

## Documentation

### Building Documentation

```shell
# Serve docs locally for development
slaf docs --serve

# Build docs for testing
slaf docs --build
```

### Working with Examples

Our examples are written in marimo, that gets installed automatically with the dev dependencies.

```shell
# Edit examples interactively
cd examples
marimo edit

# After editing, export to HTML for docs
slaf examples --export

# List available examples
slaf examples --list
```

## CI/CD

The project uses GitHub Actions for automated testing and deployment:

- **Tests**: Run on every push and pull request
- **Documentation**: Automatically deployed to GitHub Pages on main branch
- **Coverage**: Requires minimum 70% code coverage
- **Security**: Automated vulnerability scanning

All checks run automatically - you don't need to run them locally unless you want to catch issues early.

## Getting Help

- 📖 **Documentation**: Check the [API Reference](../api/core.md)
- 💬 **GitHub Issues**: Report bugs on [GitHub](https://github.com/slaf-project/slaf)

## License

By contributing to SLAF, you agree that your contributions will be licensed under the same license as the project.
