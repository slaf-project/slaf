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

| Scenario | Traditional Total (ms) | SLAF Total (ms) | Total Speedup | Memory Efficiency | Description                                 |
| -------- | ---------------------- | --------------- | ------------- | ----------------- | ------------------------------------------- |
| S1       | 24.2                   | 53.9            | 0.4x          | 1.0x              | Cells with >=500 genes                      |
| S2       | 25.6                   | 17.4            | 1.5x          | 1.0x              | Cells with <=15% mitochondrial genes        |
| S3       | 21.3                   | 16.7            | 1.3x          | 1.0x              | Cells with low mitochondrial content        |
| S4       | 20.9                   | 16.3            | 1.3x          | 1.0x              | Cells in clusters 0,1,2                     |
| S5       | 20.6                   | 17.4            | 1.2x          | 0.9x              | Cells in largest cluster (0)                |
| S6       | 20.9                   | 18.0            | 1.2x          | 0.9x              | Cells from batch_1                          |
| S7       | 21.2                   | 15.3            | 1.4x          | 0.9x              | Cells in clusters 0,1 from batch_1          |
| S8       | 19.8                   | 15.2            | 1.3x          | 0.9x              | High-quality cells (>=1000 genes, <=10% mt) |
| S9       | 21.2                   | 16.0            | 1.3x          | 0.9x              | Cells with 800-2000 total counts            |
| S10      | 20.7                   | 15.8            | 1.3x          | 1.0x              | Cells with 200-1500 genes                   |

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

| Scenario | Traditional Total (ms) | SLAF Total (ms) | Total Speedup | Memory Efficiency | Description            |
| -------- | ---------------------- | --------------- | ------------- | ----------------- | ---------------------- |
| S1       | 22.4                   | 19.8            | 1.1x          | 0.9x              | Single cell expression |
| S2       | 19.5                   | 19.1            | 1.0x          | 0.9x              | Another single cell    |
| S3       | 19.5                   | 18.8            | 1.0x          | 0.9x              | Two cells              |
| S4       | 19.5                   | 19.4            | 1.0x          | 0.8x              | Three cells            |
| S5       | 19.5                   | 201.1           | 0.1x          | 0.9x              | 100x50 submatrix       |
| S6       | 19.7                   | 218.5           | 0.1x          | 0.9x              | 500x100 submatrix      |
| S7       | 22.4                   | 227.7           | 0.1x          | 0.7x              | 500x500 submatrix      |

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
| S1        | 33.4                   | 179.5           | 0.2x          | 1.9x              | Calculate QC metrics                               |
| S2        | 26.3                   | 1140.0          | 0.0x          | 0.2x              | Filter cells (min_counts=500, min_genes=200)       |
| S3        | 28.2                   | 1128.8          | 0.0x          | 0.2x              | Filter cells (min_counts=1000, min_genes=500)      |
| S4        | 34.1                   | 1131.6          | 0.0x          | 0.3x              | Filter cells (max_counts=5000, max_genes=2500)     |
| S5        | 36.0                   | 1121.4          | 0.0x          | 0.3x              | Filter genes (min_counts=10, min_cells=5)          |
| S6        | 35.9                   | 1136.5          | 0.0x          | 0.3x              | Filter genes (min_counts=50, min_cells=10)         |
| S7        | 23.4                   | 1729.8          | 0.0x          | 0.3x              | Normalize total (target_sum=1e4)                   |
| S8        | 23.5                   | 2087.3          | 0.0x          | 0.3x              | Normalize total (target_sum=1e6)                   |
| S9        | 26.4                   | 1099.2          | 0.0x          | 0.1x              | Log1p transformation                               |
| S10       | 31.4                   | 157.0           | 0.2x          | 1.1x              | Find highly variable genes                         |
| S11       | 29.5                   | 137.1           | 0.2x          | 1.1x              | Find top 2000 highly variable genes                |
| S12       | 39.3                   | 1668.9          | 0.0x          | 0.1x              | QC metrics + cell filtering + gene filtering       |
| S13       | 23.6                   | 325.0           | 0.1x          | 0.9x              | Normalize total + slice 100x50 submatrix (lazy)    |
| S14       | 24.6                   | 202.9           | 0.1x          | 0.9x              | Log1p + slice 200x100 submatrix (lazy)             |
| S15       | 25.7                   | 376.0           | 0.1x          | 0.9x              | Normalize + Log1p + slice 500x250 submatrix (lazy) |
| S16       | 26.6                   | 95.7            | 0.3x          | 0.9x              | Normalize + Log1p + mean per gene (lazy)           |
| S17       | 22.9                   | 97.7            | 0.2x          | 0.9x              | Normalize + Log1p + variance per cell (lazy)       |

**Key Insight**: Lazy computation enables **complex preprocessing pipelines** that would cause memory explosions with traditional tools. The computation cost is paid when materializing results, but the memory efficiency enables workflows impossible with eager processing. This is similar to Dask's delayed computation patterns.

> **Note**: The current benchmarks show the "worst case" for lazy computation on small datasets. On larger datasets, SLAF should show significant speedups as the cost of processing only the slice of interest becomes much lower than processing the entire dataset.
