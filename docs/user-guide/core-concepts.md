# Core Concepts

This page explains the fundamental concepts behind SLAF and how it works.

## What is SLAF?

**SLAF** (Sparse Lance Array Format) is a high-performance format for single-cell data that combines the power of SQL with lazy evaluation. It's designed to solve the performance and scalability challenges of traditional single-cell data formats.

## Key Design Principles

### 1. SQL-First Architecture

SLAF stores data in a relational format that can be queried directly with SQL:

- **`cells` table**: Cell metadata and QC metrics
- **`genes` table**: Gene metadata and annotations
- **`expression` table**: Sparse expression matrix data

This design enables:

- Complex queries with joins across tables
- Aggregations and statistical operations
- Filtering and subsetting with SQL WHERE clauses
- Integration with existing SQL tools and workflows

### 2. Lazy Evaluation

SLAF implements lazy evaluation, meaning data is only loaded when explicitly requested:

```python
# No data loaded yet
adata = read_slaf("data.slaf")
subset = adata[adata.obs.cell_type == "T cells", :]

# Data loaded only when .compute() is called
expression = subset.X.compute()
```

Benefits:

- **Memory efficient**: Only load what you need
- **Fast operations**: Metadata operations are instant
- **Scalable**: Handle datasets larger than memory

### 3. High-Performance Storage

SLAF uses [Lance](https://lancedb.github.io/lance/) as its underlying storage format:

- **100x faster** random access than Parquet
- **Zero-copy schema evolution**: Add/drop columns without copying data
- **Vector search**: Find nearest neighbors in under 1ms
- **Cloud storage support**: Works with S3, GCS, and local filesystems

## Data Model

### Database Schema

```sql
-- Cells table
CREATE TABLE cells (
    cell_id VARCHAR PRIMARY KEY,
    cell_integer_id INTEGER,
    cell_type VARCHAR,
    batch VARCHAR,
    total_counts INTEGER,
    n_genes_by_counts INTEGER,
    high_mito BOOLEAN,
    -- ... additional metadata columns
);

-- Genes table
CREATE TABLE genes (
    gene_id VARCHAR PRIMARY KEY,
    gene_integer_id INTEGER,
    highly_variable BOOLEAN,
    -- ... additional metadata columns
);

-- Expression table
CREATE TABLE expression (
    cell_id VARCHAR,
    gene_id VARCHAR,
    cell_integer_id INTEGER,
    gene_integer_id INTEGER,
    value FLOAT,
    -- ... additional expression data
);
```

### Key Features

- **Integer IDs**: String IDs are mapped to integers for faster queries
- **Sparse storage**: Only non-zero expression values are stored
- **Indexed columns**: Frequently queried columns are indexed
- **Compressed storage**: Data is automatically compressed

## Query Optimization

SLAF includes a query optimizer that:

- **Rewrites queries** for better performance
- **Pushes filters** down to storage level
- **Uses indexes** automatically when available
- **Optimizes joins** between tables

### Example Query Optimization

```python
# This query gets optimized automatically
results = slaf_array.query("""
    SELECT c.cell_type, AVG(e.value) as avg_expr
    FROM cells c
    JOIN expression e ON c.cell_id = e.cell_id
    JOIN genes g ON e.gene_id = g.gene_id
    WHERE g.highly_variable = true
    AND c.batch = 'batch1'
    GROUP BY c.cell_type
""")
```

The optimizer:

1. Pushes the `batch` filter to the cells table
2. Pushes the `highly_variable` filter to the genes table
3. Uses indexes on `cell_integer_id` and `gene_integer_id` for joins
4. Performs the aggregation in memory

## Integration with Existing Tools

### Scanpy Compatibility

SLAF provides drop-in compatibility with Scanpy:

```python
from slaf.integrations import read_slaf

# Load as AnnData
adata = read_slaf("data.slaf")

# Use familiar Scanpy operations
import scanpy as sc
sc.pp.normalize_total(adata)
sc.pp.log1p(adata)
sc.pp.highly_variable_genes(adata)
```

### Machine Learning Ready

SLAF includes utilities for ML training:

```python
from slaf.ml import SLAFTokenizer

# Tokenize data for transformer models
tokenizer = SLAFTokenizer.from_slaf(slaf_array)
tokens = tokenizer.encode_batch(cell_ids)
```

## Performance Characteristics

### Speed

- **Random access**: 100x faster than Parquet
- **Filtering**: 10x faster than HDF5
- **SQL queries**: Near-native DuckDB performance
- **Lazy operations**: Instant metadata operations

### Memory Usage

- **Lazy loading**: Only load requested data
- **Efficient storage**: Compressed sparse format
- **Streaming**: Process data in chunks
- **Garbage collection**: Automatic cleanup of unused data

### Scalability

- **Large datasets**: Handle datasets larger than RAM
- **Parallel processing**: Support for distributed operations
- **Cloud storage**: Native support for S3/GCS
- **Incremental updates**: Add new data without rewriting

## Comparison with Other Formats

| Feature            | SLAF | HDF5 | Parquet | Zarr |
| ------------------ | ---- | ---- | ------- | ---- |
| SQL queries        | ✅   | ❌   | ✅      | ❌   |
| Lazy evaluation    | ✅   | ❌   | ❌      | ✅   |
| Random access      | ✅   | ✅   | ❌      | ✅   |
| Schema evolution   | ✅   | ❌   | ❌      | ✅   |
| Cloud storage      | ✅   | ❌   | ✅      | ✅   |
| Scanpy integration | ✅   | ✅   | ❌      | ✅   |

## Best Practices

### Data Loading

```python
# Good: Use lazy loading
adata = read_slaf("data.slaf")
subset = adata[adata.obs.cell_type == "T cells", :]
expression = subset.X.compute()  # Load only when needed

# Avoid: Loading everything at once
adata = read_slaf("data.slaf")
expression = adata.X.compute()  # May use too much memory
```

### Query Optimization

```python
# Good: Use SQL for complex operations
results = slaf_array.query("""
    SELECT cell_type, COUNT(*) as count
    FROM cells
    WHERE batch = 'batch1'
    GROUP BY cell_type
""")

# Avoid: Python loops for aggregations
cells = slaf_array.filter_cells(batch='batch1')
cell_types = cells.groupby('cell_type').size()  # Slower
```

### Memory Management

```python
# Good: Process in batches
for batch in slaf_array.obs.batch.unique():
    subset = slaf_array.filter_cells(batch=batch)
    expression = subset.X.compute()
    # Process batch...

# Avoid: Loading everything
expression = slaf_array.X.compute()  # May not fit in memory
```

## Next Steps

- Learn about [Data Loading](../user-guide/data-loading.md)
- Explore [SQL Queries](../user-guide/sql-queries.md)
- See [Examples](../examples/getting-started.md) for real-world usage
