# Performance Benchmarks

SLAF delivers **capability expansion** for single-cell analysis - enabling workflows that are impractical or impossible with traditional tools due to memory constraints and performance limitations.

## **Benchmark Setup**

### Dataset

- **Dataset**: A processed version of PBMC3K (2,695 cells × 1,863 genes)
- **Size**: ~23 MB (h5ad format)

```shell
⚡  slaf info ../slaf-datasets/pbmc3k_processed.slaf
SLAF Dataset
  Shape: 2695 cells × 1863 genes
  Format version: 0.1
  Cell metadata columns: 9
    n_genes, n_genes_by_counts, total_counts, leiden, batch...
  Gene metadata columns: 12
    gene_ids, n_cells, mt, n_cells_by_counts, mean_counts...
  Record counts:
    Cells: 2,695
    Genes: 1,863
    Expression records: 415,134
  Optimizations:
    use_integer_keys: True
```

### Hardware Configuration

- **Machine**: Apple MacBook Pro with M1 Max
- **Memory**: 32 GB RAM
- **Storage**: NVMe SSD
- **OS**: macOS 14.0

### Test Environment

- **Storage Type**: Local disk (not cloud object storage)
- **Python**: 3.12.0

> **Note**: These benchmarks represent performance on a high-end laptop. Production deployments on dedicated servers with faster storage and more memory may show different performance characteristics.

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

| Scenario                | Traditional Total (ms) | SLAF Total (ms) | Total Speedup | Memory Efficiency | Description          |
| ----------------------- | ---------------------- | --------------- | ------------- | ----------------- | -------------------- |
| S1       |   33.8 |    15.9 |     2.1x |     4.9x | Cells with >=500 genes |
| S2       |   23.7 |    13.7 |     1.7x |     4.6x | Cells with <=15% mitochondrial genes |
| S3       |   22.1 |    14.7 |     1.5x |     4.6x | Cells with low mitochondrial content |
| S4       |   21.7 |    14.4 |     1.5x |     5.0x | Cells in clusters 0,1,2 |
| S5       |   20.9 |    14.0 |     1.5x |     5.6x | Cells in largest cluster (0) |
| S6       |   22.3 |    15.6 |     1.4x |     5.5x | Cells from batch_1 |
| S7       |   19.7 |    13.5 |     1.5x |     5.8x | Cells in clusters 0,1 from batch_1 |
| S8       |   19.2 |    14.6 |     1.3x |     6.1x | High-quality cells (>=1000 genes, <=10% mt) |
| S9       |   19.9 |    15.5 |     1.3x |     5.8x | Cells with 800-2000 total counts |
| S10       |   19.4 |    63.7 |     0.3x |     4.9x | Cells with 200-1500 genes |


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

| Scenario                 | Traditional Total (ms) | SLAF Total (ms) | Total Speedup | Memory Efficiency | Description          |
| ------------------------ | ---------------------- | --------------- | ------------- | ----------------- | -------------------- |
| S1       |   19.5 |    16.1 |     1.2x |     6.4x | Single cell expression |
| S2       |   18.4 |    17.6 |     1.0x |     6.2x | Another single cell |
| S3       |   18.4 |    17.3 |     1.1x |     6.0x | Two cells |
| S4       |   19.1 |    16.9 |     1.1x |     5.8x | Three cells |
| S5       |   19.7 |    28.5 |     0.7x |     6.4x | Single gene across all cells |
| S6       |   21.1 |    30.9 |     0.7x |     6.3x | Another single gene |
| S7       |   19.0 |    34.1 |     0.6x |     6.1x | Two genes |
| S8       |   18.8 |    32.5 |     0.6x |     6.1x | Three genes |
| S9       |   19.1 |    50.1 |     0.4x |     6.0x | 100x50 submatrix |
| S10       |   20.4 |    52.1 |     0.4x |     3.5x | 500x100 submatrix |
| S11       |   18.9 |    58.3 |     0.3x |     1.2x | 500x500 submatrix |


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

| Operation                     | Traditional Total (ms) | SLAF Total (ms) | Total Speedup | Memory Efficiency | Description          |
| ----------------------------- | ---------------------- | --------------- | ------------- | ----------------- | -------------------- |
| S1       |   133.0 |    56.0 |     2.4x |     13.3x | Calculate QC metrics |
| S2       |   22.8 |    227.4 |     0.1x |     1.0x | Filter cells (min_counts=500, min_genes=200) |
| S3       |   22.2 |    29.4 |     0.8x |     7.2x | Filter cells (min_counts=1000, min_genes=500) |
| S4       |   31.4 |    235.1 |     0.1x |     1.7x | Filter cells (max_counts=5000, max_genes=2500) |
| S5       |   35.5 |    234.2 |     0.2x |     1.7x | Filter genes (min_counts=10, min_cells=5) |
| S6       |   32.3 |    229.1 |     0.1x |     1.7x | Filter genes (min_counts=50, min_cells=10) |
| S7       |   26.5 |    806.4 |     0.0x |     1.7x | Normalize total (target_sum=1e4) |
| S8       |   22.1 |    787.7 |     0.0x |     1.7x | Normalize total (target_sum=1e6) |
| S9       |   23.9 |    214.0 |     0.1x |     0.6x | Log1p transformation |
| S10       |   27.6 |    35.6 |     0.8x |     10.4x | Find highly variable genes |
| S11       |   28.2 |    33.9 |     0.8x |     10.4x | Find top 2000 highly variable genes |
| S12       |   38.3 |    285.0 |     0.1x |     0.9x | QC metrics + cell filtering + gene filtering |
| S13       |   22.7 |    132.9 |     0.2x |     6.5x | Normalize total + slice 100x50 submatrix (lazy) |
| S14       |   22.8 |    52.2 |     0.4x |     6.4x | Log1p + slice 200x100 submatrix (lazy) |
| S15       |   24.2 |    140.7 |     0.2x |     5.5x | Normalize + Log1p + slice 500x250 submatrix (lazy) |
| S16       |   23.9 |    59.4 |     0.4x |     6.4x | Normalize + Log1p + mean per gene (lazy) |
| S17       |   23.2 |    58.9 |     0.4x |     6.3x | Normalize + Log1p + variance per cell (lazy) |


**Key Insight**: Lazy computation enables **complex preprocessing pipelines** that would cause memory explosions with traditional tools. The computation cost is paid when materializing results, but the memory efficiency enables workflows impossible with eager processing. This is similar to Dask's delayed computation patterns.

> **Note**: The current benchmarks show the "worst case" for lazy computation on small datasets. On larger datasets, SLAF should show significant speedups as the cost of processing only the slice of interest becomes much lower than processing the entire dataset.

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
