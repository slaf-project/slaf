# Installation

This guide covers how to install SLAF and its dependencies.

## System Requirements

- **Python**: 3.12 or higher
- **Operating System**: Linux, macOS, or Windows
- **Memory**: 4GB RAM minimum, 16GB+ recommended for large datasets
- **Storage**: SSD recommended for better performance

## Quick Installation

### Using uv (recommended)

```bash
# Install uv if you haven't already
curl -LsSf https://astral.sh/uv/install.sh | sh

# Install SLAF
uv add slaf
```

### Using pip

```bash
pip install slaf
```

### Using conda

```bash
conda install -c conda-forge slaf
```

### From source

```bash
git clone https://github.com/slaf-project/slaf.git
cd slaf
uv pip install -e .
```

## Installation Options

### Basic Installation

For basic usage with core functionality:

```bash
pip install slaf
```

This includes:

- Core SLAF functionality
- Lance and DuckDB integration
- Basic data loading and querying

### Development Installation

For development and contributing:

```bash
uv pip install -e ".[dev]"
```

This includes:

- All basic dependencies
- Development tools (pytest, black, flake8)
- Marimo for interactive demos
- Benchmarking tools

### Documentation Installation

For building documentation locally:

```bash
uv pip install -e ".[docs]"
```

This includes:

- MkDocs and Material theme
- Documentation generation tools
- Example conversion utilities

### Full Installation

For all features:

```bash
uv pip install -e ".[dev,docs]"
```

## Dependencies

### Core Dependencies

- **pylance**: Lance format support
- **duckdb**: SQL query engine
- **pyarrow**: Arrow data format
- **numpy**: Numerical computing
- **pandas**: Data manipulation
- **scipy**: Scientific computing

### Single-Cell Ecosystem

- **scanpy**: Single-cell analysis
- **anndata**: Annotated data format

### Utilities

- **orjson**: Fast JSON serialization
- **tqdm**: Progress bars
- **rich**: Rich text formatting
- **torch**: Machine learning support

### Development Dependencies

- **pytest**: Testing framework
- **marimo**: Interactive notebooks
- **matplotlib**: Plotting
- **seaborn**: Statistical visualization
- **memray**: Memory profiling
- **psutil**: System monitoring

## Platform-Specific Notes

### Linux

SLAF works well on Linux with standard Python installations. No special configuration required.

### macOS

On macOS, you may need to install additional dependencies:

```bash
# Install Xcode command line tools (if not already installed)
xcode-select --install

# Install via conda (recommended for macOS)
conda install -c conda-forge slaf
```

### Windows

On Windows, some dependencies may require Visual Studio Build Tools:

```bash
# Install Visual Studio Build Tools
# Download from: https://visualstudio.microsoft.com/downloads/

# Then install SLAF
pip install slaf
```

## CLI Installation

SLAF includes a command-line interface for common tasks:

```bash
# Show SLAF version
slaf version

# Show help
slaf --help

# List available commands
slaf --help
```

### CLI Commands

- `slaf docs` - Manage documentation (build, serve, deploy)
- `slaf examples` - Export Marimo examples to HTML
- `slaf convert` - Convert datasets to SLAF format
- `slaf info` - Show information about a SLAF dataset
- `slaf query` - Execute SQL queries on SLAF datasets

## Verification

After installation, verify that SLAF is working correctly:

```python
import slaf
print(f"SLAF version: {slaf.__version__}")

# Test basic functionality
from slaf import SLAFArray
print("SLAF imported successfully!")
```

### CLI Verification

```bash
# Test CLI
slaf version
```

## Troubleshooting

### Common Issues

#### Import Errors

If you encounter import errors:

```bash
# Check Python version
python --version

# Reinstall with specific Python version
uv pip install slaf
```

#### Memory Issues

For large datasets, ensure you have sufficient memory:

```bash
# Check available memory
free -h  # Linux
vm_stat   # macOS
```

#### Performance Issues

For better performance:

1. Use SSD storage
2. Ensure sufficient RAM
3. Use conda instead of pip if possible
4. Consider using a virtual environment

### Getting Help

If you encounter issues:

1. Check the [troubleshooting guide](../user-guide/troubleshooting.md)
2. Search [GitHub issues](https://github.com/slaf-project/slaf/issues)
3. Open a new issue with:
   - Python version
   - Operating system
   - Error message
   - Steps to reproduce

## Next Steps

After installation:

1. Read the [Quick Start](quickstart.md) guide
2. Explore the [Examples](../examples/getting-started.md)
3. Check the [API Reference](../api/core.md)
4. Join the [community](https://github.com/slaf-project/slaf/discussions)
