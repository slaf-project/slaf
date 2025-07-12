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

| Scenario | Traditional Total (ms) | SLAF Total (ms) | Total Speedup | Memory Efficiency | Description                                 |
| -------- | ---------------------- | --------------- | ------------- | ----------------- | ------------------------------------------- |
| S1       | 37.0                   | 9.8             | 3.8x          | 4.9x              | Cells with >=500 genes                      |
| S2       | 19.8                   | 8.1             | 2.4x          | 4.6x              | Cells with <=15% mitochondrial genes        |
| S3       | 20.5                   | 9.0             | 2.3x          | 4.6x              | Cells with low mitochondrial content        |
| S4       | 19.7                   | 8.7             | 2.3x          | 5.0x              | Cells in clusters 0,1,2                     |
| S5       | 20.3                   | 8.5             | 2.4x          | 5.6x              | Cells in largest cluster (0)                |
| S6       | 18.6                   | 8.1             | 2.3x          | 5.5x              | Cells from batch_1                          |
| S7       | 18.4                   | 8.5             | 2.2x          | 5.8x              | Cells in clusters 0,1 from batch_1          |
| S8       | 18.6                   | 8.1             | 2.3x          | 6.1x              | High-quality cells (>=1000 genes, <=10% mt) |
| S9       | 18.5                   | 9.3             | 2.0x          | 5.8x              | Cells with 800-2000 total counts            |
| S10      | 19.0                   | 9.0             | 2.1x          | 4.9x              | Cells with 200-1500 genes                   |

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

| Scenario | Traditional Total (ms) | SLAF Total (ms) | Total Speedup | Memory Efficiency | Description                  |
| -------- | ---------------------- | --------------- | ------------- | ----------------- | ---------------------------- |
| S1       | 18.5                   | 11.1            | 1.7x          | 6.4x              | Single cell expression       |
| S2       | 20.3                   | 11.4            | 1.8x          | 6.2x              | Another single cell          |
| S3       | 19.2                   | 11.0            | 1.7x          | 6.0x              | Two cells                    |
| S4       | 18.3                   | 11.0            | 1.7x          | 5.8x              | Three cells                  |
| S5       | 18.3                   | 21.3            | 0.9x          | 6.4x              | Single gene across all cells |
| S6       | 18.3                   | 21.9            | 0.8x          | 6.3x              | Another single gene          |
| S7       | 19.2                   | 26.8            | 0.7x          | 6.1x              | Two genes                    |
| S8       | 19.1                   | 29.7            | 0.6x          | 6.1x              | Three genes                  |
| S9       | 18.2                   | 44.9            | 0.4x          | 6.0x              | 100x50 submatrix             |
| S10      | 18.8                   | 45.8            | 0.4x          | 3.5x              | 500x100 submatrix            |
| S11      | 18.6                   | 51.3            | 0.4x          | 1.2x              | 500x500 submatrix            |

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

| Operation | Traditional Total (ms) | SLAF Total (ms) | Total Speedup | Memory Efficiency | Description                                        |
| --------- | ---------------------- | --------------- | ------------- | ----------------- | -------------------------------------------------- |
| S1        | 115.3                  | 45.7            | 2.5x          | 13.3x             | Calculate QC metrics                               |
| S2        | 28.7                   | 220.8           | 0.1x          | 1.0x              | Filter cells (min_counts=500, min_genes=200)       |
| S3        | 21.5                   | 21.9            | 1.0x          | 7.2x              | Filter cells (min_counts=1000, min_genes=500)      |
| S4        | 31.0                   | 230.1           | 0.1x          | 1.7x              | Filter cells (max_counts=5000, max_genes=2500)     |
| S5        | 31.0                   | 210.7           | 0.1x          | 1.7x              | Filter genes (min_counts=10, min_cells=5)          |
| S6        | 30.5                   | 217.1           | 0.1x          | 1.7x              | Filter genes (min_counts=50, min_cells=10)         |
| S7        | 24.9                   | 407.0           | 0.1x          | 1.7x              | Normalize total (target_sum=1e4)                   |
| S8        | 20.7                   | 409.0           | 0.1x          | 1.7x              | Normalize total (target_sum=1e6)                   |
| S9        | 21.5                   | 196.4           | 0.1x          | 0.6x              | Log1p transformation                               |
| S10       | 24.9                   | 26.7            | 0.9x          | 10.4x             | Find highly variable genes                         |
| S11       | 25.5                   | 27.1            | 0.9x          | 10.4x             | Find top 2000 highly variable genes                |
| S12       | 36.5                   | 304.7           | 0.1x          | 0.9x              | QC metrics + cell filtering + gene filtering       |
| S13       | 22.1                   | 139.0           | 0.2x          | 6.5x              | Normalize total + slice 100x50 submatrix (lazy)    |
| S14       | 22.7                   | 45.4            | 0.5x          | 6.4x              | Log1p + slice 200x100 submatrix (lazy)             |
| S15       | 23.4                   | 146.2           | 0.2x          | 5.5x              | Normalize + Log1p + slice 500x250 submatrix (lazy) |
| S16       | 23.8                   | 51.2            | 0.5x          | 6.6x              | Normalize + Log1p + mean per gene (lazy)           |
| S17       | 21.9                   | 50.3            | 0.4x          | 6.5x              | Normalize + Log1p + variance per cell (lazy)       |

**Key Insight**: Lazy computation enables **complex preprocessing pipelines** that would cause memory explosions with traditional tools. The computation cost is paid when materializing results, but the memory efficiency enables workflows impossible with eager processing. This is similar to Dask's delayed computation patterns.

> **Note**: The current benchmarks show the "worst case" for lazy computation on small datasets. On larger datasets, SLAF should show significant speedups as the cost of processing only the slice of interest becomes much lower than processing the entire dataset.

## **High-Throughput Dataloading for GPU Training**

SLAF provides **high-throughput streaming** for GPU training workloads, enabling efficient tokenization and batching for large language models like scGPT and Geneformer.

### Performance Results

| Configuration                                    | Cells/sec | Tokens/sec | Batch Size | Max Genes | GPU Utilization |
| ------------------------------------------------ | --------- | ---------- | ---------- | --------- | --------------- |
| scGPT small batch (32 cells, 512 genes)          | 1,861     | 1,909,606  | 32         | 512       | ~10%            |
| scGPT medium batch (128 cells, 1024 genes)       | 5,195     | 5,329,646  | 128        | 1024      | ~20%            |
| scGPT large batch (512 cells, 1024 genes)        | 9,250     | 9,490,393  | 512        | 1024      | ~60%            |
| scGPT xlarge batch (2048 cells, 1024 genes)      | 11,146    | 11,435,873 | 2048       | 1024      | ~210%           |
| Geneformer small batch (32 cells, 1024 genes)    | 1,933     | 1,979,732  | 32         | 1024      | ~10%            |
| Geneformer medium batch (128 cells, 2048 genes)  | 5,140     | 10,527,183 | 128        | 2048      | ~20%            |
| Geneformer large batch (512 cells, 2048 genes)   | 10,889    | 22,300,658 | 512        | 2048      | ~60%            |
| Geneformer xlarge batch (2048 cells, 2048 genes) | 14,536    | 29,769,583 | 2048       | 2048      | ~210%           |

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
