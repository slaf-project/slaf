# Bioinformatics Benchmarks

SLAF delivers **capability expansion** for single-cell analysis - enabling workflows that are impractical or impossible with traditional tools due to memory constraints and performance limitations.

## **Benchmark Setup**

### Dataset

- **Dataset**: A synthetic dataset (49,955 cells × 25,000 genes)
- **Size**: ~722 MB (h5ad format)

```shell
SLAF Dataset
  Shape: 49,955 cells × 25,000 genes
  Format version: 0.1
  Cell metadata columns: 11
    cell_type, batch, n_genes_by_counts, total_counts, pct_counts_mt...
  Gene metadata columns: 10
    gene_symbol, highly_variable, n_cells, mt, n_cells_by_counts...
  Record counts:
    Cells: 49,955
    Genes: 25,000
    Expression records: computing...
    Expression records: 62,443,678
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

SLAF provides **efficient metadata-only queries** that avoid loading expression data when only cell/gene metadata is needed, similar to using Polars for structured data queries.

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
| S1       | 610.6                  | 4.1             | 150.1x        | 97.3x             | Cells with >=500 genes                      |
| S2       | 199.6                  | 2.1             | 95.9x         | 97.8x             | Cells with <=15% mitochondrial genes        |
| S3       | 183.9                  | 1.6             | 115.2x        | 102.8x            | Cells with low mitochondrial content        |
| S4       | 190.4                  | 2.2             | 86.7x         | 144.1x            | Cells in clusters 0,1,2                     |
| S5       | 190.4                  | 1.7             | 112.3x        | 152.9x            | Cells in largest cluster (0)                |
| S6       | 264.1                  | 2.0             | 130.2x        | 114.7x            | Cells from batch_1                          |
| S7       | 187.9                  | 2.4             | 78.7x         | 151.9x            | Cells in clusters 0,1 from batch_1          |
| S8       | 185.1                  | 2.1             | 86.3x         | 102.8x            | High-quality cells (>=1000 genes, <=10% mt) |
| S9       | 187.1                  | 2.2             | 86.7x         | 135.2x            | Cells with 800-2000 total counts            |
| S10      | 181.0                  | 3.0             | 60.7x         | 97.3x             | Cells with 200-1500 genes                   |

**Key Insight**: The speedup comes from **faster metadata loading** (SLAF loads only metadata vs h5ad loading everything), while memory efficiency comes from **loading only the data you need**. This is similar to using Polars for structured data queries instead of pandas.

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
| S1       | 650.5                  | 17.3            | 37.6x         | 156.8x            | Single cell expression       |
| S2       | 189.6                  | 13.8            | 13.7x         | 156.8x            | Another single cell          |
| S3       | 184.6                  | 12.8            | 14.4x         | 155.9x            | Two cells                    |
| S4       | 183.8                  | 12.3            | 15.0x         | 155.0x            | Three cells                  |
| S5       | 219.5                  | 622.4           | 0.4x          | 155.5x            | Single gene across all cells |
| S6       | 217.2                  | 502.3           | 0.4x          | 155.5x            | Another single gene          |
| S7       | 227.6                  | 416.8           | 0.5x          | 153.4x            | Two genes                    |
| S8       | 236.5                  | 2054.4          | 0.1x          | 151.4x            | Three genes                  |
| S9       | 414.6                  | 20.8            | 19.9x         | 157.4x            | 100x50 submatrix             |
| S10      | 186.6                  | 73.4            | 2.5x          | 155.6x            | 500x100 submatrix            |
| S11      | 187.4                  | 72.6            | 2.6x          | 147.6x            | 500x500 submatrix            |

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
| S1        | 1782.1                 | 2396.0          | 0.7x          | 160.3x            | Calculate QC metrics                               |
| S2        | 1355.6                 | 2509.5          | 0.5x          | 1.5x              | Filter cells (min_counts=500, min_genes=200)       |
| S3        | 817.7                  | 2225.3          | 0.4x          | 1.5x              | Filter cells (min_counts=100, min_genes=50)        |
| S4        | 801.2                  | 2181.5          | 0.4x          | 1.5x              | Filter cells (max_counts=10000, max_genes=3000)    |
| S5        | 1236.9                 | 2247.8          | 0.6x          | 1.5x              | Filter genes (min_counts=10, min_cells=5)          |
| S6        | 1235.9                 | 3349.3          | 0.4x          | 1.5x              | Filter genes (min_counts=20, min_cells=5)          |
| S7        | 517.6                  | 2091.9          | 0.2x          | 1.5x              | Normalize total (target_sum=1e4)                   |
| S8        | 523.7                  | 2284.0          | 0.2x          | 1.5x              | Normalize total (target_sum=1e6)                   |
| S9        | 685.0                  | 1702.9          | 0.4x          | 0.0x              | Log1p transformation                               |
| S10       | 696.4                  | 1452.9          | 0.5x          | 104.2x            | Find highly variable genes                         |
| S11       | 918.6                  | 1297.2          | 0.7x          | 104.2x            | Find top 2000 highly variable genes                |
| S12       | 2516.4                 | 6654.4          | 0.4x          | 1.5x              | QC metrics + cell filtering + gene filtering       |
| S13       | 344.4                  | 2103.6          | 0.2x          | 2.1x              | Normalize total + slice 100x50 submatrix (lazy)    |
| S14       | 560.3                  | 1849.5          | 0.3x          | 2.1x              | Log1p + slice 200x100 submatrix (lazy)             |
| S15       | 589.4                  | 3338.0          | 0.2x          | 2.1x              | Normalize + Log1p + slice 500x250 submatrix (lazy) |
| S16       | 2416.3                 | 1817.6          | 1.3x          | 2.1x              | Normalize + Log1p + mean per gene (lazy)           |
| S17       | 629.2                  | 1650.3          | 0.4x          | 2.0x              | Normalize + Log1p + variance per cell (lazy)       |

**Key Insight**: Lazy computation enables **complex preprocessing pipelines** that would cause memory explosions with traditional tools. The computation cost is paid when materializing results, but the memory efficiency enables workflows impossible with eager processing. This is similar to Dask's delayed computation patterns.

> **Note**: The current benchmarks show the "worst case" for lazy computation on small datasets. On larger datasets, SLAF should show significant speedups as the cost of processing only the slice of interest becomes much lower than processing the entire dataset.
