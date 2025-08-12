# Bioinformatics Benchmarks

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
- **Storage**: 1 TB NVMe SSD
- **OS**: macOS 13.6.1

### Test Environment

- **Storage Type**: Local disk (not cloud object storage)
- **Python**: 3.12.0

> **Note**: These benchmarks represent performance on a high-end laptop. Production deployments on dedicated servers with faster storage and more memory may show different performance characteristics.

## **Metadata Filtering & Quality Control**

SLAF provides **efficient metadata-only queries** that avoid loading expression data when only cell/gene metadata is needed, similar to using Polars or DuckDB for structured data queries.

### Traditional Approach (Load Everything)

```python
# Load entire dataset into memory - including expression matrix
adata = sc.read_h5ad("data.h5ad", backed="r")  # 7.8 MB for PBMC3K (metadata + expression)

# Filter cells using polars boolean indexing on metadata
filtered_cells = adata.obs.filter(pl.col("n_genes_by_counts") >= 500)

# Complex filtering with multiple conditions
high_quality = adata.obs.filter(
    (pl.col("n_genes_by_counts") >= 1000) &
    (pl.col("pct_counts_mt") <= 10)
)

# Cluster-based filtering
cluster_cells = adata.obs.filter(pl.col("leiden").is_in(["0", "1", "2"]))
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
| S1       | 29.6                   | 2.2             | 13.5x         | 9.5x              | Cells with >=500 genes                      |
| S2       | 20.1                   | 1.7             | 11.6x         | 9.4x              | Cells with <=15% mitochondrial genes        |
| S3       | 20.1                   | 2.0             | 10.2x         | 9.4x              | Cells with low mitochondrial content        |
| S4       | 20.8                   | 2.4             | 8.8x          | 9.7x              | Cells in clusters 0,1,2                     |
| S5       | 19.2                   | 1.8             | 10.7x         | 11.0x             | Cells in largest cluster (0)                |
| S6       | 20.1                   | 1.9             | 10.9x         | 10.8x             | Cells from batch_1                          |
| S7       | 20.2                   | 2.1             | 9.6x          | 11.4x             | Cells in clusters 0,1 from batch_1          |
| S8       | 22.4                   | 2.1             | 10.5x         | 12.0x             | High-quality cells (>=1000 genes, <=10% mt) |
| S9       | 20.7                   | 2.3             | 8.9x          | 11.4x             | Cells with 800-2000 total counts            |
| S10      | 19.9                   | 2.4             | 8.4x          | 9.4x              | Cells with 200-1500 genes                   |

**Key Insight**: The speedup comes from **faster metadata loading** (SLAF loads only metadata vs h5ad loading everything), while memory efficiency comes from **loading only the data you need**. This is similar to using Polars/DuckDB for structured data queries instead of pandas.

## **Lazy Slicing (Expression Analysis)**

SLAF provides **lazy submatrix extraction** that loads only the cells and genes of interest, similar to Zarr's chunked array access patterns.

### Traditional Approach (Load Everything, Slice Later)

```python
# Must load entire dataset
adata = sc.read_h5ad("data.h5ad", backed="r")

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
| S1       | 20.1                   | 8.9             | 2.3x          | 12.9x             | Single cell expression       |
| S2       | 21.4                   | 8.5             | 2.5x          | 12.8x             | Another single cell          |
| S3       | 19.3                   | 8.5             | 2.3x          | 12.6x             | Two cells                    |
| S4       | 20.9                   | 9.5             | 2.2x          | 12.4x             | Three cells                  |
| S5       | 20.4                   | 13.4            | 1.5x          | 12.9x             | Single gene across all cells |
| S6       | 20.2                   | 14.1            | 1.4x          | 12.8x             | Another single gene          |
| S7       | 20.2                   | 14.3            | 1.4x          | 12.7x             | Two genes                    |
| S8       | 21.1                   | 14.6            | 1.4x          | 12.7x             | Three genes                  |
| S9       | 18.7                   | 42.3            | 0.4x          | 12.6x             | 100x50 submatrix             |
| S10      | 19.1                   | 45.6            | 0.4x          | 9.8x              | 500x100 submatrix            |
| S11      | 19.1                   | 56.3            | 0.3x          | 4.9x              | 500x500 submatrix            |

**Key Insight**: The primary advantage is **memory efficiency** - SLAF loads only the slice of interest rather than the entire dataset. Speed benefits depend on slice size vs dataset size. This is similar to Zarr's chunked array access patterns.

## **Lazy Computation (Preprocessing Pipelines)**

SLAF enables **lazy computation graphs** that build complex preprocessing pipelines and only execute them on the slice of interest, similar to Dask's delayed computation patterns.

### Traditional Approach (Eager Processing)

```python
# Each step loads data into memory
adata = sc.read_h5ad("data.h5ad", backed="r")

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
| S1        | 134.8                  | 34.4            | 3.9x          | 15.8x             | Calculate QC metrics                               |
| S2        | 23.5                   | 230.2           | 0.1x          | 3.6x              | Filter cells (min_counts=500, min_genes=200)       |
| S3        | 33.5                   | 226.3           | 0.1x          | 15.3x             | Filter cells (min_counts=100, min_genes=50)        |
| S4        | 31.8                   | 214.3           | 0.1x          | 1.0x              | Filter cells (max_counts=10000, max_genes=3000)    |
| S5        | 31.7                   | 218.2           | 0.1x          | 1.0x              | Filter genes (min_counts=10, min_cells=5)          |
| S6        | 32.4                   | 215.4           | 0.2x          | 1.0x              | Filter genes (min_counts=20, min_cells=5)          |
| S7        | 25.8                   | 213.1           | 0.1x          | 1.0x              | Normalize total (target_sum=1e4)                   |
| S8        | 22.0                   | 218.1           | 0.1x          | 1.0x              | Normalize total (target_sum=1e6)                   |
| S9        | 22.4                   | 205.7           | 0.1x          | 0.2x              | Log1p transformation                               |
| S10       | 27.7                   | 124.3           | 0.2x          | 9.7x              | Find highly variable genes                         |
| S11       | 32.1                   | 22.4            | 1.4x          | 9.7x              | Find top 2000 highly variable genes                |
| S12       | 39.4                   | 296.3           | 0.1x          | 2.3x              | QC metrics + cell filtering + gene filtering       |
| S13       | 21.9                   | 49.4            | 0.4x          | 2.3x              | Normalize total + slice 100x50 submatrix (lazy)    |
| S14       | 23.0                   | 41.4            | 0.6x          | 2.2x              | Log1p + slice 200x100 submatrix (lazy)             |
| S15       | 25.0                   | 59.3            | 0.4x          | 2.0x              | Normalize + Log1p + slice 500x250 submatrix (lazy) |
| S16       | 23.1                   | 15.0            | 1.5x          | 2.2x              | Normalize + Log1p + mean per gene (lazy)           |
| S17       | 22.5                   | 16.0            | 1.4x          | 2.1x              | Normalize + Log1p + variance per cell (lazy)       |

**Key Insight**: Lazy computation enables **complex preprocessing pipelines** that would cause memory explosions with traditional tools. The computation cost is paid when materializing results, but the memory efficiency enables workflows impossible with eager processing. This is similar to Dask's delayed computation patterns.

> **Note**: The current benchmarks show the "worst case" for lazy computation on small datasets. On larger datasets, SLAF should show significant speedups as the cost of processing only the slice of interest becomes much lower than processing the entire dataset.
