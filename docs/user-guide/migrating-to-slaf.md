# Migrating to SLAF

## Quick Start

Convert your single-cell data to SLAF format with just one command:

```bash
# Convert any supported format (auto-detection)
slaf convert data.h5ad output.slaf
slaf convert filtered_feature_bc_matrix/ output.slaf
slaf convert data.h5 output.slaf
slaf convert data.tiledb output.slaf
```

That's it! SLAF automatically detects your file format and converts it with optimized settings.

## Supported Formats

SLAF supports conversion from these common single-cell formats:

- **AnnData** (.h5ad files) - The standard single-cell format
- **10x MTX** (filtered_feature_bc_matrix directories) - Cell Ranger output
- **10x H5** (.h5 files) - Cell Ranger H5 output format
- **TileDB SOMA** (.tiledb directories) - High-performance single-cell format

## Python API

```python
from slaf.data import SLAFConverter

# Basic conversion (auto-detection)
converter = SLAFConverter()
converter.convert("data.h5ad", "output.slaf")
converter.convert("filtered_feature_bc_matrix/", "output.slaf")
converter.convert("data.h5", "output.slaf")
converter.convert("data.tiledb", "output.slaf")

# Convert existing AnnData object
import scanpy as sc
adata = sc.read_h5ad("data.h5ad")
converter.convert_anndata(adata, "output.slaf")
```

## Large Datasets

For large datasets (>100k cells), you can optimize performance:

```bash
# Use larger chunks for speed (if you have enough RAM)
slaf convert large_data.h5ad output.slaf --chunk-size 100000

# Create indices for faster queries
slaf convert large_data.h5ad output.slaf --create-indices
```

```python
# Python API for large datasets
converter = SLAFConverter(chunk_size=100000, create_indices=True)
converter.convert("large_data.h5ad", "output.slaf")
```

## Advanced Options

Most users won't need these, but they're available if needed:

### CLI Options

```bash
# Specify format explicitly (if auto-detection fails)
slaf convert data.h5 output.slaf --format 10x_h5
slaf convert data.tiledb output.slaf --format tiledb

# Use non-chunked processing (not recommended for large datasets)
slaf convert small_data.h5ad output.slaf --no-chunked

# Disable storage optimization (larger files but includes string IDs)
slaf convert data.h5ad output.slaf --no-optimize-storage

# Verbose output
slaf convert data.h5ad output.slaf --verbose

# TileDB-specific options
slaf convert data.tiledb output.slaf --tiledb-collection RNA
```

### Python API Options

```python
# Custom settings
converter = SLAFConverter(
    chunk_size=50000,           # Cells per chunk
    create_indices=True,        # Faster queries
    optimize_storage=True,      # Smaller files (default)
    use_optimized_dtypes=True,  # Better compression (default)
    tiledb_collection_name="RNA",  # TileDB collection name (default: "RNA")
)

converter.convert("data.h5ad", "output.slaf")
converter.convert("data.tiledb", "output.slaf")
```

## TileDB SOMA Conversion

SLAF provides excellent support for TileDB SOMA format, which is increasingly popular for large-scale single-cell datasets:

### Basic TileDB Conversion

```bash
# Auto-detect TileDB format
slaf convert experiment.tiledb output.slaf

# Specify collection name (default: "RNA")
slaf convert experiment.tiledb output.slaf --tiledb-collection RNA
```

### Python API for TileDB

```python
from slaf.data import SLAFConverter

# Basic TileDB conversion
converter = SLAFConverter()
converter.convert("experiment.tiledb", "output.slaf")

# With custom collection name
converter = SLAFConverter(tiledb_collection_name="RNA")
converter.convert("experiment.tiledb", "output.slaf")
```

### TileDB Benefits

- **Memory Efficient**: TileDB's chunked storage works seamlessly with SLAF's chunked processing
- **Large Datasets**: Optimized for datasets with millions of cells
- **Data Preservation**: Maintains exact floating-point precision from TileDB
- **Fast Conversion**: Leverages TileDB's efficient data access patterns

## What SLAF Does

SLAF converts your data to an optimized format that:

- **Enables fast SQL queries** on your data
- **Works with any size dataset** (memory-efficient processing)
- **Preserves all metadata** (cell types, gene info, etc.)

## Next Steps

After converting your data:

1. **Explore**: `slaf info output.slaf`
2. **Query**: `slaf query output.slaf "SELECT * FROM expression LIMIT 10"`
3. **Use in Python**: `import slaf; data = slaf.SLAFArray("output.slaf")`

See the [Getting Started](../getting-started/quickstart.md) guide for more examples.
