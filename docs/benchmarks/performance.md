# Performance Benchmarks

SLAF delivers **capability expansion** for single-cell analysis - enabling workflows that are impractical or impossible with traditional tools due to memory constraints and performance limitations.

## **Metadata Filtering & Quality Control**

SLAF provides **efficient metadata-only queries** that avoid loading expression data when only cell/gene metadata is needed, similar to using Polars or DuckDB for structured data queries.

### Traditional Approach (Load Everything)

```python
# Load entire dataset into memory - including expression matrix
adata = sc.read_h5ad("data.h5ad")  # 7.8 MB for PBMC3K (metadata + expression)

# Filter cells using pandas boolean indexing on metadata
filtered_cells = adata.obs[adata.obs.n_genes_by_counts >= 500]

# Complex filtering with multiple conditions
high_quality = adata.obs[
    (adata.obs.n_genes_by_counts >= 1000) &
    (adata.obs.pct_counts_mt <= 10)
]

# Cluster-based filtering
cluster_cells = adata.obs[adata.obs.leiden.isin(["0", "1", "2"])]
```

### SLAF Approach (Metadata-Only Loading)

```python
# Load only metadata into memory
slaf = SLAFArray("data.slaf")  # Only loads obs/var metadata

# Direct filtering with SQL optimization
filtered_cells = slaf.filter_cells(n_genes_by_counts=">=500")

# Complex filtering with multiple conditions
high_quality = slaf.filter_cells(
    n_genes_by_counts=">=1000",
    pct_counts_mt="<=10"
)

# Cluster-based filtering
cluster_cells = slaf.filter_cells(leiden=["0", "1", "2"])
```

### Performance Results

| Scenario                | Traditional Total (ms) | SLAF Total (ms) | Total Speedup | Memory Efficiency |
| ----------------------- | ---------------------- | --------------- | ------------- | ----------------- |
| Cells with ≥500 genes   | 39.7                   | 9.9             | **4.0x**      | **4.9x**          |
| Cells with ≤15% mt      | 22.4                   | 8.5             | **2.6x**      | **4.6x**          |
| Cells in clusters 0,1,2 | 19.6                   | 9.1             | **2.1x**      | **5.0x**          |
| High-quality cells      | 18.5                   | 8.3             | **2.2x**      | **6.1x**          |

**Key Insight**: The speedup comes from **faster metadata loading** (SLAF loads only metadata vs h5ad loading everything), while memory efficiency comes from **loading only the data you need**. This is similar to using Polars/DuckDB for structured data queries instead of pandas.

## **Lazy Slicing (Expression Analysis)**

SLAF provides **lazy submatrix extraction** that loads only the cells and genes of interest, similar to Zarr's chunked array access patterns.

### Traditional Approach (Load Everything, Slice Later)

```python
# Must load entire dataset
adata = sc.read_h5ad("data.h5ad")

# Single-cell expression
cell_id = "AAACCTGAGAAACCAT-1"
cell_idx = adata.obs.index.get_loc(cell_id)
result = adata.X[cell_idx, :]

# Single-gene expression
gene_id = "MS4A1"
gene_idx = adata.var.index.get_loc(gene_id)
result = adata.X[:, gene_idx]

# Submatrix extraction
cell_start, cell_end = 0, 100
gene_start, gene_end = 0, 50
result = adata.X[cell_start:cell_end, gene_start:gene_end]
```

### SLAF Approach (Lazy Submatrix Loading)

```python
# No full dataset loading
slaf = SLAFArray("data.slaf")

# Single-cell expression
result = slaf.get_cell_expression("AAACCTGAGAAACCAT-1")

# Single-gene expression
result = slaf.get_gene_expression("MS4A1")

# Submatrix extraction
result = slaf.get_submatrix(
    cell_range=(0, 100),
    gene_range=(0, 50)
)
```

### Performance Results

| Scenario                 | Traditional Total (ms) | SLAF Total (ms) | Total Speedup | Memory Efficiency |
| ------------------------ | ---------------------- | --------------- | ------------- | ----------------- |
| Single cell expression   | 18.9                   | 10.7            | **1.8x**      | **6.4x**          |
| Two cells                | 18.5                   | 10.6            | **1.7x**      | **6.0x**          |
| Single gene across cells | 21.4                   | 21.5            | 1.0x          | **6.4x**          |
| 100×50 submatrix         | 18.4                   | 44.4            | 0.4x          | **6.0x**          |
| 500×500 submatrix        | 18.3                   | 52.0            | 0.4x          | **1.2x**          |

**Key Insight**: The primary advantage is **memory efficiency** - SLAF loads only the slice of interest rather than the entire dataset. Speed benefits depend on slice size vs dataset size. This is similar to Zarr's chunked array access patterns.

## **Lazy Computation (Preprocessing Pipelines)**

SLAF enables **lazy computation graphs** that build complex preprocessing pipelines and only execute them on the slice of interest, similar to Dask's delayed computation patterns.

### Traditional Approach (Eager Processing)

```python
# Each step loads data into memory
adata = sc.read_h5ad("data.h5ad")

# QC metrics calculation
sc.pp.calculate_qc_metrics(adata, inplace=True)

# Cell filtering with min_counts and min_genes
sc.pp.filter_cells(adata, min_counts=500, min_genes=200, inplace=True)

# Gene filtering
sc.pp.filter_genes(adata, min_counts=10, min_cells=5, inplace=True)

# Normalization
sc.pp.normalize_total(adata, target_sum=1e4, inplace=True)

# Log transformation
sc.pp.log1p(adata)

# Each operation processes the entire dataset
expression = adata.X[cell_ids, gene_ids]
```

### SLAF Approach (Lazy Computation Graph)

```python
# Build lazy computation graph
adata = LazyAnnData("data.slaf")  # LazyAnnData object

# QC metrics calculation (lazy)
pp.calculate_qc_metrics(adata, inplace=True)

# Cell filtering (lazy)
pp.filter_cells(adata, min_counts=500, min_genes=200, inplace=True)

# Gene filtering (lazy)
pp.filter_genes(adata, min_counts=10, min_cells=5, inplace=True)

# Normalization (lazy)
pp.normalize_total(adata, target_sum=1e4, inplace=True)

# Log transformation (lazy)
pp.log1p(adata)

# Only execute the computation on the slice of interest
expression = adata.X[cell_ids, gene_ids].compute()  # LazyExpressionMatrix.compute()
```

### Performance Results

| Operation                     | Traditional Total (ms) | SLAF Total (ms) | Total Speedup | Memory Efficiency |
| ----------------------------- | ---------------------- | --------------- | ------------- | ----------------- |
| Calculate QC metrics          | 115.4                  | 55.0            | **2.1x**      | **13.3x**         |
| Filter cells (min_counts=500) | 23.6                   | 233.8           | 0.1x          | **1.0x**          |
| Filter genes (min_counts=10)  | 32.7                   | 232.8           | 0.1x          | **1.7x**          |
| Normalize total               | 25.2                   | 419.8           | 0.1x          | **1.7x**          |
| Log1p transformation          | 23.1                   | 208.8           | 0.1x          | **0.6x**          |

**Key Insight**: Lazy computation enables **complex preprocessing pipelines** that would cause memory explosions with traditional tools. The computation cost is paid when materializing results, but the memory efficiency enables workflows impossible with eager processing. This is similar to Dask's delayed computation patterns.

> **Note**: The current benchmarks show the "worst case" for lazy computation on small datasets. On larger datasets, SLAF should show significant speedups as the cost of processing only the slice of interest becomes much lower than processing the entire dataset.

## **High-Throughput Dataloading for GPU Training**

SLAF provides **streaming tokenization** that converts single-cell data into training-ready sequences for transformer models like scGPT and Geneformer.

### Performance Results

| Configuration                  | Cells/sec | Tokens/sec | Batch Size | Max Genes | GPU Utilization |
| ------------------------------ | --------- | ---------- | ---------- | --------- | --------------- |
| scGPT small batch (32 cells, 512 genes)            |    1,861 |  1,909,606 |         32 |        512 | ~10% |
| scGPT medium batch (128 cells, 1024 genes)         |    5,195 |  5,329,646 |        128 |       1024 | ~20% |
| scGPT large batch (512 cells, 1024 genes)          |    9,250 |  9,490,393 |        512 |       1024 | ~60% |
| scGPT xlarge batch (2048 cells, 1024 genes)        |   11,146 | 11,435,873 |       2048 |       1024 | ~210% |
| Geneformer small batch (32 cells, 1024 genes)      |    1,933 |  1,979,732 |         32 |       1024 | ~10% |
| Geneformer medium batch (128 cells, 2048 genes)    |    5,140 | 10,527,183 |        128 |       2048 | ~20% |
| Geneformer large batch (512 cells, 2048 genes)     |   10,889 | 22,300,658 |        512 |       2048 | ~60% |
| Geneformer xlarge batch (2048 cells, 2048 genes)   |   14,536 | 29,769,583 |       2048 |       2048 | ~210% |


**Key Insights:**

- **Peak throughput**: ~14,536 cells/sec (Geneformer xlarge batch)
- **Token generation**: Up to 30M tokens/sec for large batches
- **Batch size scaling**: Throughput improves with larger batches up to ~2048 cells
- **GPU utilization**: Conservative estimates based on memory usage patterns

### Real-World Training Considerations

**Typical Training Requirements:**

- **scGPT (1.4B params)**: Batch size 32-64 cells, ~50 ms training time per step
- **Throughput requirement**: 32 cells × 20 batches/sec x 8 GPUs / node = 5120 cells/sec
- **SLAF performance**: 11K cells/sec (> 2x over requirement)

SLAF's high-throughput streaming architecture enables dataloading approaches that would make multi-node training truly efficient by leveraging asynchronous pre-fetching, shard-aware and concurrent streaming.
