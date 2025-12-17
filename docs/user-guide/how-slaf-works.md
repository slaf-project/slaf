# How SLAF Works

**SLAF** (Sparse Lazy Array Format) is a high-performance format for single-cell data that combines the power of SQL with lazy evaluation. It's designed to solve the performance and scalability challenges of traditional single-cell data formats.

## Key Design Principles

### 1. SQL-Native Design with Relational Schema

SLAF is built on a **SQL-native relational schema** that enables direct SQL queries while providing lazy AnnData/Scanpy equivalences for seamless migration:

#### Relational Schema

SLAF stores single-cell data in core tables with an extensible schema:

- **`cells` table**: Cell metadata, QC metrics, and annotations with `cell_id` (string) and `cell_integer_id` (integer). Also stores multi-dimensional arrays (obsm) like UMAP coordinates and PCA embeddings as `FixedSizeListArray` columns.
- **`genes` table**: Gene metadata, annotations, and feature information with `gene_id` (string) and `gene_integer_id` (integer). Also stores multi-dimensional arrays (varm) like PCA loadings as `FixedSizeListArray` columns.
- **`expression` table**: Sparse expression matrix with `cell_integer_id`, `gene_integer_id`, and `value` columns
- **`layers` table** (optional): Alternative expression matrices (e.g., `spliced`, `unspliced`, `counts`) stored in wide format with one column per layer, sharing the same `cell_integer_id` and `gene_integer_id` structure as the expression table

The expression and layers tables use integer IDs for efficiency, so you need to JOIN with metadata tables to get string identifiers. Multi-dimensional arrays (obsm/varm) are stored alongside scalar metadata in the same tables using Arrow's native `FixedSizeListArray` type for efficient vector operations.

This relational design enables **direct SQL queries** for everything:

```python
# Direct SQL for complex aggregations
results = slaf.query("""
    SELECT cell_type,
           COUNT(*) as cell_count,
           AVG(total_counts) as avg_counts,
           SUM(e.value) as total_expression
    FROM cells c
    JOIN expression e ON c.cell_integer_id = e.cell_integer_id
    WHERE batch = 'batch1' AND n_genes_by_counts >= 500
    GROUP BY cell_type
    ORDER BY cell_count DESC
""")

# Window functions for advanced analysis
ranked_genes = slaf.query("""
    SELECT g.gene_id,
           c.cell_type,
           e.value,
           ROW_NUMBER() OVER (
               PARTITION BY c.cell_type
               ORDER BY e.value DESC
           ) as rank
    FROM expression e
    JOIN cells c ON e.cell_integer_id = c.cell_integer_id
    JOIN genes g ON e.gene_integer_id = g.gene_integer_id
    WHERE g.gene_id IN ('MS4A1', 'CD3D', 'CD8A')
""")
```

#### Lazy AnnData/Scanpy Equivalences

For users migrating from Scanpy, SLAF provides **drop-in lazy equivalents**:

```python
# Load as LazyAnnData (no data loaded yet)
adata = read_slaf("data.slaf")

# Use familiar Scanpy-style operations
subset = adata[adata.obs.cell_type == "T cells", :] # This is lazy

# Access expression data lazily
expression = subset.X.compute()  # Only loads the subset

# Use Scanpy preprocessing (lazy)
from slaf.scanpy import pp
pp.normalize_total(adata, target_sum=1e4, inplace=True)
pp.log1p(adata)
pp.highly_variable_genes(adata)
expression = adata[cell_ids, gene_ids].X.compute()
```

#### Seamless Switching Between Interfaces

You can **switch between SQL and AnnData interfaces** as needed:

```python
# Start with AnnData interface
lazy_adata = read_slaf("data.slaf")
t_cells = lazy_adata[lazy_adata.obs.cell_type == "T cells", :]

# Switch to SQL for complex operations
t_cells_slaf = t_cells.slaf  # Access underlying SLAFArray object
complex_query_result = t_cells_slaf.query("""
    SELECT g.gene_id,
           COUNT(*) as expressing_cells,
           AVG(e.value) as mean_expression
    FROM expression e
    JOIN genes g ON e.gene_integer_id = g.gene_integer_id
    WHERE e.cell_integer_id IN (
        SELECT cell_integer_id FROM cells
        WHERE cell_type = 'T cells'
    )
    GROUP BY g.gene_id
    HAVING expressing_cells >= 10
    ORDER BY mean_expression DESC
""")

# Back to AnnData for visualization
import scanpy as sc
t_cells_as_adata = t_cells.compute()  # Convert to native scanpy
sc.pl.umap(t_cells_as_adata, color='leiden')
```

Benefits:

- **SQL-native**: Direct access to relational database capabilities
- **SQL-native**: Complex aggregations and window functions
- **Migration-friendly**: Drop-in replacement for existing Scanpy workflows
- **Flexible**: Switch between SQL and AnnData interfaces as needed

### 2. Polars-Like: OLAP Database with Pushdown Filters

SLAF leverages modern OLAP databases and pushdown filters on optimized storage formats rather than in-memory operations:

- **`cells` table**: Cell metadata, QC metrics, and multi-dimensional arrays (obsm)
- **`genes` table**: Gene metadata, annotations, and multi-dimensional arrays (varm)
- **`expression` table**: Sparse expression matrix data
- **`layers` table**: Alternative expression matrices (optional, wide format)

Like Polars, SLAF pushes complex operations down to the query engine:

```python
# Metadata-only filtering without loading expression data
filtered_cells = slaf.filter_cells(n_genes_by_counts=">=500")
high_quality = slaf.filter_cells(
    n_genes_by_counts=">=1000",
    pct_counts_mt="<=10"
)
```

Benefits:

- **Memory efficient**: Only load metadata when filtering
- **Faster** metadata filtering vs h5ad as datasets scale

### 3. Zarr-Like: Lazy Loading of Sparse Matrices

SLAF provides lazy loading of sparse matrices from cloud storage with concurrent access patterns:

```python
# No data loaded yet - just metadata
adata = read_slaf("data.slaf")

# Lazy slicing like Zarr chunked arrays
subset = adata[adata.obs.cell_type == "T cells", :]
single_cell = adata.get_cell_expression("AAACCTGAGAAACCAT-1")
gene_expression = adata.get_gene_expression("MS4A1")

# Data loaded only when .compute() is called
expression = subset.X.compute()
```

Benefits:

- **Memory efficient** for submatrix operations
- **Concurrent access**: Multiple readers can access different slices
- **Cloud-native**: Direct access to data in S3/GCS without downloading
- **Chunked processing**: Handle datasets larger than RAM

### 4. Dask-Like: Lazy Computation Graphs

SLAF enables building computational graphs of operations that execute lazily on demand:

```python
# Build lazy computation graph
adata = LazyAnnData("data.slaf")

# Each operation is lazy - no data loaded yet
pp.calculate_qc_metrics(adata, inplace=True)
pp.filter_cells(adata, min_counts=500, min_genes=200, inplace=True)
pp.normalize_total(adata, target_sum=1e4, inplace=True)
pp.log1p(adata)

# Execute only on the slice of interest
expression = adata.X[cell_ids, gene_ids].compute()
```

Benefits:

- **Complex pipelines**: Build preprocessing workflows impossible with eager processing
- **Composable**: Chain operations without intermediate materialization
- **Memory efficient**: Only process the slice you need
- **Scalable**: Handle datasets that would cause memory explosions

### 5. Advanced Query Optimization

SLAF includes sophisticated query optimization to overcome current limitations of LanceDB:

```python
# Adaptive batching for large scattered ID sets
batched_query = QueryOptimizer.build_optimized_query(
    entity_ids=large_id_list,
    entity_type="cell",
    use_adaptive_batching=True
)

# CTE optimization for complex queries
cte_query = QueryOptimizer.build_cte_query(
    entity_ids=scattered_ids,
    entity_type="gene"
)
```

Key optimizations:

- **Submatrix optimization**: Efficient slicing for complex selectors
- **Adaptive batching**: Optimize query patterns based on ID distribution
- **Range vs IN optimization**: Choose BETWEEN vs IN clauses intelligently

### 6. Foundation Model Training Support

SLAF's versatile SQL combined with OLAP-optimized query engine enables window function queries that directly support tokenization and streaming dataloaders:

```python
# Streaming tokenization for transformer models
from slaf.ml.dataloaders import SLAFDataLoader

# Create production-ready DataLoader
dataloader = SLAFDataLoader(
    slaf_array=slaf_array,
    tokenizer_type="geneformer",
    batch_size=32,
    max_genes=2048,
    vocab_size=50000
)

# Stream batches for training
for batch in dataloader:
    input_ids = batch["input_ids"]      # Already tokenized
    attention_mask = batch["attention_mask"]
    cell_ids = batch["cell_ids"]
    # Your training code here

# High-throughput dataloading
# 15k cells/sec peak throughput
# 30M tokens/sec for large batches
```

Benefits:

- **Streaming architecture**: Supports asynchronous pre-fetching and concurrent streaming
- **GPU-optimized**: Batch sizes up to 2048 cells with high GPU utilization
- **Multi-node ready**: Shard-aware streaming for distributed training
- **Foundation model support**: Direct integration with scGPT, Geneformer, etc.

## Comparison with Other Formats

| Feature                | SLAF | H5AD | Zarr | TileDB SOMA |
| ---------------------- | ---- | ---- | ---- | ----------- |
| **Storage**            |      |      |      |             |
| Cloud-Native           | ✅   | ❌   | ✅   | ✅          |
| Sparse Arrays          | ✅   | ✅   | ❌   | ✅          |
| Chunked Reads          | ✅   | ❌   | ✅   | ✅          |
| Schema Evolution       | ✅   | ❌   | ✅   | ✅          |
| **Compute**            |      |      |      |             |
| SQL Queries            | ✅   | ❌   | ❌   | ✅          |
| Optimized Query Engine | ✅   | ❌   | ❌   | ✅          |
| Random Access          | ✅   | ✅   | ✅   | ✅          |
| **Use Cases**          |      |      |      |             |
| Scanpy Integration     | ✅   | ✅   | ✅   | ❌          |
| Lazy Computation       | ✅   | ❌   | ❌   | ❌          |
| Tokenizers             | ✅   | ❌   | ❌   | ❌          |
| Dataloaders            | ✅   | ❌   | ❌   | ❌          |
| Embeddings Support     | 🔄   | ❌   | ❌   | ❌          |
| Visualization Support  | 🔄   | ✅   | ❌   | ❌          |

**Legend:**

- ✅ = Supported
- ❌ = Not supported
- 🔄 = Coming soon

## Next Steps

- Learn about [Migrating to SLAF](../user-guide/migrating-to-slaf.md)
- Explore [SQL Queries](../examples/sql-queries.md)
- See [Examples](../examples/getting-started.md) for real-world usage
