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

Our examples are written in [Marimo](https://marimo.io/) notebooks. Marimo provides an excellent interactive environment for data science and machine learning workflows.

#### Interactive Development

```shell
# Edit examples interactively
cd examples
marimo edit

# Run a specific example
marimo edit examples/01-getting-started.py
```

#### Exporting Examples for Documentation

After editing examples, export them to HTML for the documentation:

```shell
# Export all examples to HTML
slaf examples --export

# Export a specific example
marimo export html examples/01-getting-started.py -o examples/01-getting-started.html

# List available examples
slaf examples --list
```

#### Programmatic Export

You can also export notebooks programmatically:

```python
import marimo

# Export notebook to HTML
marimo.export_html("examples/01-getting-started.py", "examples/01-getting-started.html")
```

#### Example Structure

Our examples follow a consistent structure:

- **01-getting-started.py**: Comprehensive introduction to SLAF
- **02-lazy-processing.py**: Demonstrates lazy evaluation and processing
- **03-ml-training-pipeline.py**: Shows ML training workflows

#### Best Practices for Examples

**For Interactive Use:**

- Use descriptive cell names
- Include markdown cells for explanations
- Add progress indicators for long-running operations
- Use the variables panel to explore data

**For Documentation:**

- Keep examples focused and concise
- Include clear explanations in markdown cells
- Use consistent formatting
- Test examples with different datasets

**For Export:**

- Ensure all dependencies are available
- Test the exported HTML in different browsers
- Optimize for readability in static format
- Include navigation if exporting multiple notebooks

#### Embedding Examples in Documentation

To include examples in documentation:

1. Export the notebooks to HTML using Marimo's built-in export
2. Place the HTML files in your documentation directory
3. Include them using iframes:

```html
<iframe src="01-getting-started.html" width="100%" height="800px"></iframe>
```

#### Troubleshooting Examples

**Common Issues:**

1. **Import errors**: Ensure all dependencies are installed
2. **Data not found**: Check file paths and dataset availability
3. **Memory issues**: Use lazy evaluation for large datasets
4. **Export problems**: Verify Marimo version and export options

**Getting Help:**

- Check the [Marimo documentation](https://docs.marimo.io/)
- Review the [SLAF API reference](../api/core.md)
- Open an issue on GitHub for specific problems

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
