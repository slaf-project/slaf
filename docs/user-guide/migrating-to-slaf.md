# Migrating to SLAF

## Converting from Multiple Formats

SLAF supports conversion from multiple single-cell data formats with automatic format detection.

### Using the CLI

The easiest way to convert your data:

```bash
# Convert any supported format (auto-detection)
slaf convert data.h5ad output.slaf
slaf convert filtered_feature_bc_matrix/ output.slaf
slaf convert data.h5 output.slaf

# Convert with verbose output
slaf convert data.h5ad output.slaf --verbose

# Explicit format specification (if auto-detection fails)
slaf convert data.h5 output.slaf --format 10x_h5
```

### Using the Python API

```python
import slaf
from slaf.data import SLAFConverter

# Basic conversion with auto-detection (recommended)
converter = SLAFConverter()
converter.convert("data.h5ad", "output.slaf")
converter.convert("filtered_feature_bc_matrix/", "output.slaf")
converter.convert("data.h5", "output.slaf")

# Explicit format specification
converter.convert("data.h5", "output.slaf", input_format="10x_h5")
converter.convert("mtx_dir/", "output.slaf", input_format="10x_mtx")

# Convert existing AnnData object
import scanpy as sc
adata = sc.read_h5ad("data.h5ad")
converter.convert_anndata(adata, "output.slaf")
```

### Supported Input Formats

SLAF automatically detects the following formats:

- **AnnData** (.h5ad files) - The standard single-cell format
- **10x MTX** (filtered_feature_bc_matrix directories) - Cell Ranger output
- **10x H5** (.h5 files) - Cell Ranger H5 output format

### Format Auto-Detection

The converter automatically detects your input format based on:

- **File extension**: `.h5ad` → AnnData, `.h5` → 10x H5
- **Directory structure**: Directories containing `matrix.mtx` and `genes.tsv`/`features.tsv` → 10x MTX

### Large Dataset Conversion

For datasets larger than available memory (100k cells or more, depending on RAM):

```python
from slaf.data import SLAFConverter

# Use chunked processing for memory efficiency
converter = SLAFConverter(chunked=True, chunk_size=10000)
converter.convert("large_data.h5ad", "output.slaf")
```

**When to use chunked conversion:**

- Datasets with 100k+ cells (depending on available RAM)
- When you encounter memory errors during conversion
- For optimal memory usage on large datasets

## Current Support

**Currently Supported:**

- **AnnData** (.h5ad files) - Full support
- **10x Genomics MTX** (Cell Ranger output directories) - Full support
- **10x Genomics H5** (.h5 files) - Full support

**Format Details:**

- **10x MTX**: Supports both compressed (.gz) and uncompressed files
- **10x H5**: Handles both true 10x H5 files and regular h5ad files with .h5 extension
- **Auto-detection**: Works with both old (genes.tsv) and new (features.tsv) 10x formats

**Coming Soon:**

- **Parquet** files (Tahoe-100M format)
- **Additional single-cell formats**

We're actively working on expanding format support. If you need to convert from other formats, please [open an issue](https://github.com/your-repo/slaf/issues) to let us know what formats are most important to you.

## Conversion Examples

### From 10x Cell Ranger Output

```bash
# Convert 10x MTX directory
slaf convert filtered_feature_bc_matrix/ output.slaf

# Convert 10x H5 file
slaf convert filtered_feature_bc_matrix.h5 output.slaf
```

### From AnnData

```bash
# Convert h5ad file
slaf convert data.h5ad output.slaf
```

### From Python

```python
from slaf.data import SLAFConverter

# Auto-detect and convert
converter = SLAFConverter()
converter.convert("your_data.h5ad", "output.slaf")
converter.convert("your_10x_directory/", "output.slaf")
converter.convert("your_10x_file.h5", "output.slaf")
```

## Next Steps

After converting your data:

1. **Explore** your data: `slaf info output.slaf`
2. **Query** your data: `slaf query output.slaf "SELECT * FROM expression LIMIT 10"`
3. **Use in Python**: `import slaf; data = slaf.load("output.slaf")`

See the [Getting Started](../getting-started/quickstart.md) guide for more examples.
