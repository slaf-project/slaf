# Command Line Interface

SLAF includes a powerful command-line interface for common tasks like building documentation, exporting examples, and converting datasets.

## Installation

The CLI is automatically installed with SLAF:

```bash
# Using uv (recommended)
uv add slaf

# Or using pip
pip install slaf
```

## Getting Help

```bash
# Show general help
slaf --help

# Show help for specific command
slaf docs --help
slaf examples --help
slaf convert --help
```

## Commands Overview

### `slaf version`

Show the current SLAF version.

```bash
slaf version
```

### `slaf docs`

Manage SLAF documentation.

```bash
# Build documentation
slaf docs --build

# Serve documentation locally
slaf docs --serve

# Deploy to GitHub Pages
slaf docs --deploy

# Default: serve documentation locally
slaf docs
```

**Options:**

- `--build, -b`: Build the documentation to `site/` directory
- `--serve, -s`: Start a local development server
- `--deploy, -d`: Deploy to GitHub Pages

### `slaf examples`

Manage SLAF examples.

```bash
# List available examples
slaf examples --list

# Export all examples to HTML
slaf examples --export

# Export specific example
slaf examples --export getting-started
```

**Options:**

- `--export, -e`: Export examples to HTML
- `--list, -l`: List available examples
- `example`: Specific example name to export

### `slaf convert`

Convert datasets to SLAF format.

```bash
# Convert AnnData file
slaf convert data.h5ad output.slaf

# Convert with specific format
slaf convert data.h5ad output.slaf --format anndata

# Convert with verbose output
slaf convert data.h5ad output.slaf --verbose
```

**Arguments:**

- `input_path`: Input file path (AnnData, HDF5, etc.)
- `output_path`: Output SLAF directory path

**Options:**

- `--format, -f`: Input format (auto-detected if not specified)
- `--verbose, -v`: Verbose output

### `slaf info`

Show information about a SLAF dataset.

```bash
slaf info path/to/dataset.slaf
```

**Arguments:**

- `dataset_path`: Path to SLAF dataset

### `slaf query`

Execute SQL queries on SLAF datasets.

```bash
# Basic query
slaf query dataset.slaf "SELECT COUNT(*) FROM cells"

# Query with output file
slaf query dataset.slaf "SELECT * FROM cells LIMIT 10" --output results.csv

# Query with custom limit
slaf query dataset.slaf "SELECT * FROM cells" --limit 5
```

**Arguments:**

- `dataset_path`: Path to SLAF dataset
- `sql`: SQL query to execute

**Options:**

- `--output, -o`: Output file path (CSV)
- `--limit, -l`: Limit number of results (default: 10)

## Examples

### Building Documentation

```bash
# Quick development server
slaf docs

# Build for production
slaf docs --build

# Deploy to GitHub Pages
slaf docs --deploy
```

### Working with Examples

```bash
# See what examples are available
slaf examples --list

# Export all examples for documentation
slaf examples --export

# Export just the getting started example
slaf examples --export getting-started
```

### Converting Datasets

```bash
# Convert an AnnData file
slaf convert pbmc3k.h5ad pbmc3k.slaf

# Convert with verbose output to see details
slaf convert pbmc3k.h5ad pbmc3k.slaf --verbose
```

### Exploring Datasets

```bash
# Get basic info about a dataset
slaf info pbmc3k.slaf

# Run a simple query
slaf query pbmc3k.slaf "SELECT cell_type, COUNT(*) FROM cells GROUP BY cell_type"

# Export query results to CSV
slaf query pbmc3k.slaf "SELECT * FROM cells WHERE cell_type = 'T cells'" --output t_cells.csv
```

## Integration with Development Workflow

### Local Development

```bash
# Start documentation server for development
slaf docs --serve

# Export examples after making changes
slaf examples --export

# Build documentation for testing
slaf docs --build
```

### CI/CD Pipeline

```bash
# Build documentation
slaf docs --build

# Deploy to GitHub Pages
slaf docs --deploy
```

### Data Processing Pipeline

```bash
# Convert input data
slaf convert input.h5ad output.slaf

# Verify conversion
slaf info output.slaf

# Run analysis queries
slaf query output.slaf "SELECT cell_type, AVG(total_counts) FROM cells GROUP BY cell_type"
```

## Troubleshooting

### Common Issues

1. **Command not found**: Ensure SLAF is installed and in your PATH
2. **Missing dependencies**: Install required packages with `uv add <package>`
3. **Permission errors**: Check file permissions for input/output paths
4. **Memory issues**: Use smaller datasets or increase system memory

### Getting Help

```bash
# Show all available commands
slaf --help

# Show help for specific command
slaf convert --help

# Check version
slaf version
```

## Next Steps

- Learn about [How SLAF Works](how-slaf-works.md)
- Explore [Migrating to SLAF](migrating-to-slaf.md)
- Check the [API Reference](../api/core.md)
- Try the [Examples](../examples/getting-started.md)
