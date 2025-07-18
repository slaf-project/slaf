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

# Convert with indices for query performance (recommended for large datasets)
slaf convert data.h5ad output.slaf --create-indices

# Optimize for speed with larger chunks
slaf convert large_data.h5ad output.slaf --chunked --chunk-size 100000

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

SLAF supports **chunked conversion** for all input formats, enabling memory-efficient processing of large datasets:

```bash
# CLI chunked conversion (all formats supported)
slaf convert large_data.h5ad output.slaf --chunked --chunk-size 10000
slaf convert filtered_feature_bc_matrix/ output.slaf --chunked --chunk-size 5000
slaf convert large_data.h5 output.slaf --chunked --chunk-size 15000

# Convert with indices for query performance
slaf convert large_data.h5ad output.slaf --create-indices
slaf convert filtered_feature_bc_matrix/ output.slaf --create-indices

# Optimize for speed with larger chunks
slaf convert large_data.h5ad output.slaf --chunked --chunk-size 100000
slaf convert filtered_feature_bc_matrix/ output.slaf --chunked --chunk-size 100000

# For very large datasets (>10GB), use smaller chunks to prevent memory issues
slaf convert huge_data.h5ad output.slaf --chunked --chunk-size 25000

# Optimize storage by only storing integer IDs (default: True)
slaf convert large_data.h5ad output.slaf --optimize-storage
slaf convert large_data.h5ad output.slaf --no-optimize-storage  # Include string IDs

# Use optimized data types for better compression (default: True)
slaf convert large_data.h5ad output.slaf --optimized-dtypes
slaf convert large_data.h5ad output.slaf --no-optimized-dtypes  # Use standard dtypes

# Enable v2 manifest paths for better query performance (default: True)
slaf convert large_data.h5ad output.slaf --v2-manifest
slaf convert large_data.h5ad output.slaf --no-v2-manifest  # Use v1 manifest

# Compact dataset after writing for optimal storage (default: True)
slaf convert large_data.h5ad output.slaf --compact
slaf convert large_data.h5ad output.slaf --no-compact  # Skip compaction
```

```python
# Python API chunked conversion
from slaf.data import SLAFConverter

# Use chunked processing for memory efficiency (all formats)
converter = SLAFConverter(chunked=True, chunk_size=10000)
converter.convert("large_data.h5ad", "output.slaf")
converter.convert("filtered_feature_bc_matrix/", "output.slaf")
converter.convert("large_data.h5", "output.slaf")

# Convert with indices for query performance
converter = SLAFConverter(create_indices=True)
converter.convert("large_data.h5ad", "output.slaf")
converter.convert("filtered_feature_bc_matrix/", "output.slaf")

# Optimize for speed with larger chunks
converter = SLAFConverter(chunked=True, chunk_size=100000)
converter.convert("large_data.h5ad", "output.slaf")
converter.convert("filtered_feature_bc_matrix/", "output.slaf")

# Optimize storage by only storing integer IDs (default: True)
converter = SLAFConverter(optimize_storage=True)
converter.convert("large_data.h5ad", "output.slaf")

# Include string IDs for compatibility (larger storage)
converter = SLAFConverter(optimize_storage=False)
converter.convert("large_data.h5ad", "output.slaf")

# Use optimized data types for better compression (default: True)
converter = SLAFConverter(use_optimized_dtypes=True)
converter.convert("large_data.h5ad", "output.slaf")

# Use standard data types for compatibility
converter = SLAFConverter(use_optimized_dtypes=False)
converter.convert("large_data.h5ad", "output.slaf")

# Enable v2 manifest paths for better query performance (default: True)
converter = SLAFConverter(enable_v2_manifest=True)
converter.convert("large_data.h5ad", "output.slaf")

# Compact dataset after writing for optimal storage (default: True)
converter = SLAFConverter(compact_after_write=True)
converter.convert("large_data.h5ad", "output.slaf")

# Maximum compression settings for very large datasets
converter = SLAFConverter(
    use_optimized_dtypes=True,
    enable_v2_manifest=True,
    compact_after_write=True
)
converter.convert("huge_data.h5ad", "output.slaf")
```

**When to use chunked conversion:**

- Datasets with 100k+ cells (depending on available RAM)
- When you encounter memory errors during conversion
- For optimal memory usage on large datasets
- **All formats supported**: h5ad, 10x MTX, 10x H5

**Storage Optimization:**

SLAF provides **storage optimization** that dramatically reduces file size by only storing integer IDs in the expression table:

- **Default behavior**: Only stores integer IDs (`cell_integer_id`, `gene_integer_id`, `value`)
- **String mapping**: String IDs are available in metadata tables for lookup
- **Storage reduction**: Can reduce file size by 50-80% compared to storing both string and integer IDs
- **Query performance**: Integer-only storage enables faster range queries and filtering

**When to use storage optimization:**

- **Default (recommended)**: Use `--optimize-storage` for all conversions
- **Compatibility**: Use `--no-optimize-storage` only if you need direct string ID access in queries
- **Large datasets**: Storage optimization is especially beneficial for datasets >1GB
- **Query patterns**: Integer-only storage is optimal for range queries and filtering

**Advanced Storage Optimizations:**

SLAF includes several advanced optimizations for maximum storage efficiency:

**Optimized Data Types (`--optimized-dtypes`):**

- **uint16/uint32**: Uses uint16 for gene IDs and uint32 for cell IDs for better compression
- **Storage reduction**: Can reduce file size by an additional 20-30% compared to standard int32/float32
- **Limitations**: Requires gene count ≤ 65,535 and cell count ≤ 4,294,967,295
- **Auto-validation**: Automatically falls back to standard types if data exceeds limits

**V2 Manifest Paths (`--v2-manifest`):**

- **Better performance**: Enables faster query performance on large datasets
- **Recommended**: Use for datasets with 100k+ cells
- **Compatibility**: Works with all Lance-compatible query engines

**Post-Write Compaction (`--compact`):**

- **Storage optimization**: Compacts dataset after writing to optimize storage layout
- **File size reduction**: Can reduce file size by 10-20% through better data organization
- **Trade-off**: Increases conversion time but improves storage efficiency and query performance

**Performance optimization tips:**

- **Chunk sizes**: Use 25k-50k cells per chunk for memory efficiency, 100k+ for speed (if memory allows)
- **Memory monitoring**: Monitor memory usage and adjust chunk size accordingly
- **SSD storage**: Use SSD storage for faster I/O operations
- **Large datasets**: For datasets >10GB, use smaller chunk sizes (10k-25k) to prevent memory issues

**Chunked conversion benefits:**

- **Memory efficient**: Processes data in chunks without loading entire dataset
- **Format agnostic**: Works with all supported input formats
- **Native readers**: Uses optimized chunked readers for each format
- **Scalable**: Handle datasets larger than available RAM

**Technical details:**

- **Native chunked readers**: Each format (h5ad, 10x MTX, 10x H5) has optimized chunked readers
- **Memory streaming**: Data is processed in configurable chunks to minimize memory usage
- **Format detection**: Automatically selects the appropriate chunked reader based on input format
- **Consistent API**: Same chunked interface works across all supported formats

### Query Performance Optimization

SLAF supports **index creation** for improved query performance on large datasets:

**When to use indices:**

- Datasets with 100k+ cells where query performance is important
- When you plan to run frequent SQL queries on the data
- For interactive analysis and exploration
- **Trade-off**: Indices increase storage size but improve query speed

**Index benefits:**

- **Faster queries**: Indexed columns enable efficient range and equality queries
- **Better performance**: Especially for filtering by cell_id, gene_id, or expression values
- **Interactive analysis**: Enables responsive exploration of large datasets

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

# Chunked conversion for large datasets
slaf convert filtered_feature_bc_matrix/ output.slaf --chunked --chunk-size 5000
slaf convert filtered_feature_bc_matrix.h5 output.slaf --chunked --chunk-size 10000

# Convert with indices for query performance
slaf convert filtered_feature_bc_matrix/ output.slaf --create-indices
slaf convert filtered_feature_bc_matrix.h5 output.slaf --create-indices
```

### From AnnData

```bash
# Convert h5ad file
slaf convert data.h5ad output.slaf

# Chunked conversion for large h5ad files
slaf convert large_data.h5ad output.slaf --chunked --chunk-size 15000

# Convert with indices for query performance
slaf convert large_data.h5ad output.slaf --create-indices
```

### From Python

```python
from slaf.data import SLAFConverter

# Auto-detect and convert
converter = SLAFConverter()
converter.convert("your_data.h5ad", "output.slaf")
converter.convert("your_10x_directory/", "output.slaf")
converter.convert("your_10x_file.h5", "output.slaf")

# Chunked conversion for all formats
converter = SLAFConverter(chunked=True, chunk_size=10000)
converter.convert("large_data.h5ad", "output.slaf")
converter.convert("large_10x_directory/", "output.slaf")
converter.convert("large_10x_file.h5", "output.slaf")

# Convert with indices for query performance
converter = SLAFConverter(create_indices=True)
converter.convert("large_data.h5ad", "output.slaf")
converter.convert("large_10x_directory/", "output.slaf")
converter.convert("large_10x_file.h5", "output.slaf")
```

### Memory-Efficient Processing Examples

```python
# Handle datasets larger than available RAM
converter = SLAFConverter(chunked=True, chunk_size=5000)
converter.convert("million_cells.h5ad", "output.slaf")

# Process with custom chunk sizes
converter = SLAFConverter(chunked=True, chunk_size=2000)
converter.convert("dense_expression.h5ad", "output.slaf")

# Combine with format detection
converter = SLAFConverter(chunked=True, chunk_size=10000)
converter.convert("unknown_format_data", "output.slaf")  # Auto-detects format
```

## Next Steps

After converting your data:

1. **Explore** your data: `slaf info output.slaf`
2. **Query** your data: `slaf query output.slaf "SELECT * FROM expression LIMIT 10"`
3. **Use in Python**: `import slaf; data = slaf.load("output.slaf")`

See the [Getting Started](../getting-started/quickstart.md) guide for more examples.
