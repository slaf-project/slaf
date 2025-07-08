# Contributing to SLAF

Thank you for your interest in contributing to SLAF! This guide will help you get started.

## Getting Started

### Prerequisites

1. **Python 3.9+** - SLAF requires Python 3.9 or higher
2. **Git** - For version control
3. **uv** - For dependency management (recommended)

### Development Setup

1. **Clone the repository**:

   ```bash
   git clone https://github.com/slaf-project/slaf.git
   cd slaf
   ```

2. **Install development dependencies**:

   ```bash
   uv sync --group dev
   ```

3. **Install in development mode**:

   ```bash
   uv pip install -e .
   ```

4. **Run tests**:
   ```bash
   pytest tests/
   ```

## Development Workflow

### 1. Create a Feature Branch

```bash
git checkout -b feature/your-feature-name
```

### 2. Make Your Changes

- Follow the existing code style
- Add tests for new functionality
- Update documentation as needed
- Add type hints to new functions

### 3. Run Tests and Checks

```bash
# Run all tests
pytest tests/

# Run linting
ruff check .

# Run type checking
mypy slaf/

# Run formatting
ruff format .
```

### 4. Commit Your Changes

```bash
git add .
git commit -m "feat: add your feature description"
```

Follow [conventional commits](https://www.conventionalcommits.org/) for commit messages.

### 5. Push and Create a Pull Request

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

### Documentation

- Update docstrings for any changed functions
- Add examples in docstrings where helpful
- Update relevant documentation pages
- Test documentation builds locally

## Testing

### Writing Tests

- Place tests in the `tests/` directory
- Use descriptive test names
- Test both success and failure cases
- Use fixtures for common test data

### Running Tests

```bash
# Run all tests
pytest

# Run specific test file
pytest tests/test_slaf.py

# Run with coverage
pytest --cov=slaf tests/
```

## Documentation

### Building Documentation

```bash
# Build docs
slaf docs --build

# Serve docs locally
slaf docs --serve
```

### Documentation Guidelines

- Write clear, concise explanations
- Include code examples
- Use proper markdown formatting
- Test all code examples

## Release Process

### For Maintainers

1. **Update version** in `pyproject.toml`
2. **Update changelog** with new features/fixes
3. **Create release tag**:
   ```bash
   git tag v1.0.0
   git push origin v1.0.0
   ```
4. **Build and publish**:
   ```bash
   uv build
   uv publish
   ```

## Getting Help

- **GitHub Issues**: Report bugs or request features
- **Discussions**: Ask questions in GitHub Discussions
- **Email**: Contact pavan.ramkumar@gmail.com

## Code of Conduct

We are committed to providing a welcoming and inclusive environment for all contributors. Please be respectful and constructive in all interactions.

## License

By contributing to SLAF, you agree that your contributions will be licensed under the same license as the project.
