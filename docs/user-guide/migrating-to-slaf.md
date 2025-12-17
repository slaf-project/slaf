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

## Multi-File Conversion

Convert multiple files to a single SLAF dataset:

```bash
# Convert multiple files from a directory
slaf convert data_folder/ output.slaf

# Convert specific files
slaf convert file1.h5ad file2.h5ad file3.h5ad output.slaf

# Auto-detection works for all formats
slaf convert 10x_data_folder/ output.slaf
```

SLAF automatically:

- ✅ Validates all files are compatible
- ✅ Assigns unique cell IDs across all files
- ✅ Tracks which file each cell came from
- ✅ Combines metadata intelligently

## Appending to Existing Datasets

Add new data to an existing SLAF dataset:

```bash
# Append a single file
slaf append new_data.h5ad existing.slaf

# Append multiple files from a directory
slaf append new_data_folder/ existing.slaf

# Skip validation if already validated (faster)
slaf append new_data.h5ad existing.slaf --skip-validation
```

Perfect for:

- **Incremental data collection** - Add new batches as they arrive
- **Data updates** - Append new samples to existing datasets
- **Combining datasets** - Merge related datasets over time

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

# Multi-file conversion
converter.convert("data_folder/", "output.slaf")  # Directory of files
converter.convert(["file1.h5ad", "file2.h5ad"], "output.slaf")  # List of files

# Append to existing dataset
converter.append("new_data.h5ad", "existing.slaf")
converter.append("new_data_folder/", "existing.slaf")

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

## Validation and Quality Control

The `slaf validate-input-files` command helps you catch compatibility issues before conversion:

### Basic Validation

```bash
# Validate a single file
slaf validate-input-files data.h5ad

# Validate multiple files from a directory
slaf validate-input-files data_folder/

# Validate specific files
slaf validate-input-files file1.h5ad file2.h5ad file3.h5ad
```

### Verbose Output

```bash
# Get detailed information about files being validated
slaf validate-input-files data_folder/ --verbose

# Output shows:
# 📁 Found 3 h5ad files
#   1. batch_001.h5ad
#   2. batch_002.h5ad
#   3. batch_003.h5ad
# ✅ All files are compatible for conversion
```

### Format-Specific Validation

```bash
# Validate 10x MTX directories
slaf validate-input-files filtered_feature_bc_matrix/ --format 10x_mtx

# Validate 10x H5 files
slaf validate-input-files data.h5 --format 10x_h5

# Validate TileDB SOMA files
slaf validate-input-files experiment.tiledb --format tiledb
```

### What Validation Checks

The validation command performs comprehensive compatibility checks:

- ✅ **File Integrity**: Files exist, are readable, and not empty
- ✅ **Format Consistency**: All files use the same format (h5ad, 10x_mtx, etc.)
- ✅ **Gene Compatibility**: All files have identical gene sets
- ✅ **Metadata Schema**: Cell metadata columns are compatible across files
- ✅ **Value Types**: Expression data types are consistent (uint16, float32, etc.)
- ✅ **File Sizes**: Ensures files contain actual data (not empty)

### Common Validation Scenarios

```bash
# Validate before multi-file conversion
slaf validate-input-files batch1/ batch2/ batch3/
slaf convert batch1/ output.slaf  # Safe to proceed

# Validate before appending
slaf validate-input-files new_batch/
slaf append existing.slaf new_batch/  # Safe to proceed

# Check specific format compatibility
slaf validate-input-files 10x_data/ --format 10x_mtx
```

### Error Examples

When validation fails, you get clear error messages:

```bash
# Gene mismatch error
❌ Validation failed: File batch_002.h5ad is incompatible:
  Missing genes: GENE_001, GENE_002, GENE_003
  Extra genes: GENE_999, GENE_1000

# Schema mismatch error
❌ Validation failed: File batch_003.h5ad has incompatible cell metadata schema:
  Missing columns: ['cell_type', 'batch']
  Extra columns: ['cluster_id']

# Format mismatch error
❌ Validation failed: Multiple formats detected in directory
  Found: h5ad, 10x_mtx
  All files must use the same format
```

### Integration with Conversion

Validation runs automatically during conversion, but you can skip it for performance:

```bash
# Automatic validation (default)
slaf convert data_folder/ output.slaf

# Skip validation (faster, but less safe)
slaf convert data_folder/ output.slaf --skip-validation
slaf append new_data.h5ad existing.slaf --skip-validation
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

# Skip validation (if already validated)
slaf convert data_folder/ output.slaf --skip-validation
slaf append new_data.h5ad existing.slaf --skip-validation

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
- **Preserves all metadata** including:
  - Cell and gene annotations (`obs` and `var` columns)
  - Alternative expression matrices (`layers` like `spliced`, `unspliced`, `counts`)
  - Multi-dimensional arrays (`obsm` like UMAP coordinates, PCA embeddings)
  - Gene-level embeddings (`varm` like PCA loadings)
  - Unstructured metadata (`uns` like analysis parameters)

## Workflow Examples

### Multi-File Workflow

```bash
# 1. Validate files first (recommended)
slaf validate-input-files batch1/ batch2/ batch3/

# 2. Convert all batches to single SLAF
slaf convert batch1/ initial.slaf

# 3. Append additional batches
slaf append batch2/ initial.slaf
slaf append batch3/ initial.slaf

# 4. Explore the combined dataset
slaf info initial.slaf
```

### Incremental Data Collection

```bash
# Start with first batch
slaf convert batch_001/ dataset.slaf

# Add new batches as they arrive
slaf append batch_002/ dataset.slaf
slaf append batch_003/ dataset.slaf
slaf append batch_004/ dataset.slaf

# Each append maintains data integrity and source tracking
```

### Quality Control Workflow

```bash
# 1. Validate all files before conversion
slaf validate-input-files all_batches/

# 2. Convert with validation (automatic)
slaf convert all_batches/ combined.slaf

# 3. Check source file tracking
slaf query combined.slaf "SELECT source_file, COUNT(*) FROM cells GROUP BY source_file"
```

## Next Steps

After converting your data:

1. **Explore**: `slaf info output.slaf`
2. **Query**: `slaf query output.slaf "SELECT * FROM expression LIMIT 10"`
3. **Use in Python**: `import slaf; data = slaf.SLAFArray("output.slaf")`
4. **Check Source Files**: `slaf query output.slaf "SELECT DISTINCT source_file FROM cells"`

See the [Getting Started](../getting-started/quickstart.md) guide for more examples.
