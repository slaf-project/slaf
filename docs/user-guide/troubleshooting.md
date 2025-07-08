# Troubleshooting

This guide helps you resolve common issues when using SLAF.

## Common Issues

### Memory Errors

**Problem**: `MemoryError` or `OutOfMemoryError` when loading large datasets.

**Solutions**:

1. **Use chunked loading**:

   ```python
   import slaf

   # Load in chunks
   slaf_array = slaf.from_anndata_chunked(
       "large_data.h5ad",
       "output.slaf",
       chunk_size=5000  # Reduce chunk size
   )
   ```

2. **Reduce memory usage**:

   ```python
   # Use smaller data types
   slaf_array = slaf.from_anndata(
       adata,
       "output.slaf",
       cell_dtypes={"total_counts": "float32"},
       gene_dtypes={"gene_type": "category"}
   )
   ```

3. **Close other applications** to free up memory

### Slow Query Performance

**Problem**: SQL queries are running slowly.

**Solutions**:

1. **Check indexes**:

   ```python
   # View available indexes
   indexes = slaf_array.get_indexes()
   print(indexes)

   # Create missing indexes
   slaf_array.create_index("cells", "cell_type")
   ```

2. **Optimize queries**:

   ```python
   # Use EXPLAIN to see query plan
   plan = slaf_array.explain("""
       SELECT * FROM cells WHERE cell_type = 'T cells'
   """)
   print(plan)
   ```

3. **Add WHERE clauses** to filter data early

### File Not Found Errors

**Problem**: `FileNotFoundError` when trying to load data.

**Solutions**:

1. **Check file paths**:

   ```python
   import os

   # Verify file exists
   if not os.path.exists("data.h5ad"):
       print("File not found!")
   ```

2. **Use absolute paths**:

   ```python
   import os

   # Use absolute path
   abs_path = os.path.abspath("data.h5ad")
   slaf_array = slaf.from_anndata(abs_path, "output.slaf")
   ```

### Type Errors

**Problem**: `TypeError` during data loading or queries.

**Solutions**:

1. **Check data types**:

   ```python
   import pandas as pd

   # Check column types
   print(adata.obs.dtypes)
   print(adata.var.dtypes)
   ```

2. **Convert data types**:
   ```python
   # Convert problematic columns
   adata.obs['cell_type'] = adata.obs['cell_type'].astype('category')
   adata.obs['total_counts'] = adata.obs['total_counts'].astype('float32')
   ```

### Missing Dependencies

**Problem**: `ImportError` or `ModuleNotFoundError`.

**Solutions**:

1. **Install missing packages**:

   ```bash
   pip install anndata scanpy pandas numpy
   ```

2. **Check installation**:

   ```bash
   pip list | grep slaf
   ```

3. **Reinstall SLAF**:
   ```bash
   pip uninstall slaf
   pip install slaf
   ```

## Debugging

### Enable Debug Logging

```python
import slaf
import logging

# Set debug level
slaf.set_log_level("DEBUG")

# Or use Python logging
logging.basicConfig(level=logging.DEBUG)
```

### Check Data Integrity

```python
import slaf

# Validate SLAF array
slaf_array = slaf.SLAFArray("data.slaf")

# Check basic properties
print(f"Number of cells: {slaf_array.n_cells}")
print(f"Number of genes: {slaf_array.n_genes}")
print(f"File size: {slaf_array.file_size}")

# Check for issues
issues = slaf_array.validate()
if issues:
    print("Validation issues found:")
    for issue in issues:
        print(f"  - {issue}")
```

### Test with Small Data

```python
import slaf

# Test with subset of data
small_adata = adata[:1000, :1000]  # First 1000 cells and genes
slaf_array = slaf.from_anndata(small_adata, "test.slaf")

# Test queries
results = slaf_array.query("SELECT COUNT(*) FROM cells")
print(results)
```

## Performance Issues

### Slow Data Loading

**Solutions**:

1. **Use SSD storage** for better I/O performance
2. **Increase chunk size** if memory allows:

   ```python
   slaf_array = slaf.from_anndata_chunked(
       "data.h5ad",
       "output.slaf",
       chunk_size=10000  # Increase chunk size
   )
   ```

3. **Use compression**:
   ```python
   slaf_array = slaf.from_anndata(
       adata,
       "output.slaf",
       compression="lz4"  # Faster compression
   )
   ```

### Slow Queries

**Solutions**:

1. **Use appropriate indexes**:

   ```python
   # Create indexes for frequently queried columns
   slaf_array.create_index("cells", "cell_type")
   slaf_array.create_index("cells", "batch")
   slaf_array.create_index("genes", "gene_name")
   ```

2. **Optimize query structure**:

   ```python
   # Good: Filter early
   results = slaf_array.query("""
       SELECT * FROM cells
       WHERE cell_type = 'T cells'  # Filter first
       AND total_counts > 1000
   """)

   # Bad: No filtering
   results = slaf_array.query("SELECT * FROM cells")  # Loads everything
   ```

## Getting Help

### Before Asking for Help

1. **Check this troubleshooting guide**
2. **Search existing issues** on GitHub
3. **Try with minimal example** to isolate the problem
4. **Include error messages** and stack traces
5. **Provide system information**:

   ```python
   import sys
   import slaf

   print(f"Python version: {sys.version}")
   print(f"SLAF version: {slaf.__version__}")
   print(f"Platform: {sys.platform}")
   ```

### Reporting Issues

When reporting issues, include:

1. **Complete error message** and stack trace
2. **Minimal reproducible example**
3. **System information** (OS, Python version, etc.)
4. **Expected vs actual behavior**
5. **Steps to reproduce**

### Useful Commands

```python
# Check SLAF version
import slaf
print(slaf.__version__)

# Check system info
import sys
print(sys.version)
print(sys.platform)

# Check memory usage
import psutil
print(f"Memory usage: {psutil.virtual_memory().percent}%")

# Profile performance
import time
start = time.time()
# Your code here
print(f"Time taken: {time.time() - start:.2f}s")
```

## Best Practices

1. **Always validate** your input data
2. **Test with small datasets** first
3. **Use appropriate data types** to save memory
4. **Create indexes** for frequently queried columns
5. **Handle errors gracefully** in production code
6. **Monitor memory usage** for large datasets
7. **Keep backups** of your original data
