# Migrating to SLAF

This guide covers how to load data into SLAF from various formats and sources.

## Supported Formats

SLAF supports loading data from multiple single-cell data formats:

- **AnnData** (.h5ad files)
- **10x Genomics** (Cell Ranger output)
- **CSV/TSV** files
- **Parquet** files
- **HDF5** files

## Basic Data Loading

### From AnnData

```python
import slaf
from anndata import read_h5ad

# Load existing AnnData
adata = read_h5ad("data.h5ad")

# Convert to SLAF
slaf_array = slaf.from_anndata(adata, "output.slaf")
```

### From 10x Genomics

```python
import slaf

# Load from 10x directory
slaf_array = slaf.from_10x("path/to/filtered_feature_bc_matrix/", "output.slaf")
```

### From CSV/TSV

```python
import slaf
import pandas as pd

# Load expression matrix
expression = pd.read_csv("expression.csv", index_col=0)

# Load cell metadata
cell_meta = pd.read_csv("cell_metadata.csv", index_col=0)

# Load gene metadata
gene_meta = pd.read_csv("gene_metadata.csv", index_col=0)

# Create SLAF array
slaf_array = slaf.from_dataframes(
    expression=expression,
    cell_metadata=cell_meta,
    gene_metadata=gene_meta,
    output_path="output.slaf"
)
```

## Advanced Loading Options

### Custom Data Types

```python
import slaf

# Specify data types for memory optimization
slaf_array = slaf.from_anndata(
    adata,
    "output.slaf",
    cell_dtypes={
        "cell_type": "category",
        "batch": "category",
        "total_counts": "float32"
    },
    gene_dtypes={
        "gene_type": "category",
        "chromosome": "category"
    }
)
```

### Chunked Loading

For very large datasets, you can load data in chunks:

```python
import slaf

# Load in chunks to manage memory
slaf_array = slaf.from_anndata_chunked(
    "data.h5ad",
    "output.slaf",
    chunk_size=10000  # cells per chunk
)
```

### Validation

SLAF performs automatic validation during loading:

```python
import slaf

# Load with validation
slaf_array = slaf.from_anndata(
    adata,
    "output.slaf",
    validate=True,  # Default
    check_duplicates=True,
    verify_expression=True
)
```

## Loading from Multiple Sources

### Merging Datasets

```python
import slaf

# Load multiple datasets
arrays = []
for path in ["data1.h5ad", "data2.h5ad", "data3.h5ad"]:
    array = slaf.from_anndata(path, f"temp_{path}.slaf")
    arrays.append(array)

# Merge into single SLAF array
merged = slaf.merge(arrays, "merged.slaf")
```

### Batch Integration

```python
import slaf

# Load with batch information
slaf_array = slaf.from_anndata_with_batch(
    ["batch1.h5ad", "batch2.h5ad"],
    batch_names=["batch1", "batch2"],
    output_path="integrated.slaf"
)
```

## Performance Considerations

### Memory Usage

- Use appropriate data types (e.g., `float32` instead of `float64`)
- Load data in chunks for very large datasets
- Use categorical types for string columns

### Storage Optimization

```python
import slaf

# Optimize storage
slaf_array = slaf.from_anndata(
    adata,
    "output.slaf",
    compression="zstd",  # or "lz4", "gzip"
    compression_level=3,
    optimize_storage=True
)
```

## Error Handling

### Common Issues

1. **Memory Errors**: Use chunked loading
2. **Type Errors**: Check data types and convert if needed
3. **Missing Data**: Handle NaN values appropriately
4. **Duplicate Indices**: Use `check_duplicates=True`

### Debugging

```python
import slaf

# Enable verbose logging
slaf.set_log_level("DEBUG")

# Load with detailed error reporting
try:
    slaf_array = slaf.from_anndata(adata, "output.slaf")
except slaf.LoadError as e:
    print(f"Loading failed: {e}")
    print(f"Details: {e.details}")
```

## Best Practices

1. **Always validate** your input data before loading
2. **Use appropriate data types** to save memory and storage
3. **Handle missing data** explicitly
4. **Test with small datasets** first
5. **Keep original data** as backup
6. **Document your loading process** for reproducibility
