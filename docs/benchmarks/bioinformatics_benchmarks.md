# Bioinformatics Benchmarks: SLAF vs Traditional Formats

SLAF provides **dramatic performance improvements** over traditional single-cell data formats for common bioinformatics operations. This document presents comprehensive benchmarks comparing SLAF against h5ad (AnnData) and TileDB SOMA across realistic bioinformatics workflows.

## **Overview**

These benchmarks demonstrate SLAF's performance advantages in three key areas:

1. **Metadata Filtering** - Cell and gene filtering operations
2. **Expression Queries** - Retrieving expression data for specific cells/genes
3. **Preprocessing Pipelines** - Scanpy-based preprocessing workflows

### **Test Dataset: synthetic_50k_processed**

- **Cells**: 49,955 cells
- **Genes**: 25,000 genes
- **Input File Size**: ~722MB h5ad file

### **Hardware Configuration**

- **Machine**: Apple MacBook Pro with M1 Max
- **Memory**: 32 GB RAM
- **Storage**: 1 TB NVMe SSD (local disk)
- **OS**: macOS 13.6.1
- **Python**: 3.12.0

## **Cell Filtering Benchmarks**

Cell filtering is a fundamental operation in single-cell analysis, used for quality control, cell type selection, and data subsetting.

### **Performance Results**

| Scenario | h5ad Total (ms) | SLAF Total (ms) | TileDB Total (ms) | SLAF vs h5ad | SLAF vs TileDB | Description                                 |
| -------- | --------------- | --------------- | ----------------- | ------------ | -------------- | ------------------------------------------- |
| S1       | 530.0           | 2.9             | 20.9              | **183.6x**   | **7.3x**       | Cells with >=500 genes                      |
| S2       | 169.7           | 2.0             | 21.3              | **83.3x**    | **10.5x**      | High UMI count (total_counts > 2000)        |
| S3       | 170.7           | 1.9             | 21.3              | **92.2x**    | **11.5x**      | Mitochondrial fraction < 0.1                |
| S4       | 177.1           | 2.0             | 18.6              | **86.7x**    | **9.1x**       | Complex multi-condition filter              |
| S5       | 186.6           | 2.8             | 18.1              | **67.2x**    | **6.5x**       | Cell type annotation filter                 |
| S6       | 171.4           | 2.0             | 20.9              | **86.3x**    | **10.5x**      | Cells from batch_1                          |
| S7       | 207.0           | 2.3             | 23.8              | **89.4x**    | **10.3x**      | Cells in clusters 0,1 from batch_1          |
| S8       | 170.9           | 2.1             | 23.0              | **79.6x**    | **10.7x**      | High-quality cells (>=1000 genes, <=10% mt) |
| S9       | 172.1           | 2.5             | 19.3              | **70.2x**    | **7.9x**       | Cells with 800-2000 total counts            |
| S10      | 173.5           | 2.1             | 20.8              | **84.6x**    | **10.1x**      | Cells with 200-1500 genes                   |

**Average Performance:**

- **SLAF vs h5ad**: **92.3x faster**
- **SLAF vs TileDB**: **9.4x faster**
- **Memory Usage**: SLAF uses 115.7x less memory than h5ad

### **Key Insights**

!!! success "Dramatic Performance Advantage"

    SLAF achieves **92.3x average speedup** over h5ad for cell filtering operations, demonstrating the massive performance benefits of modern columnar storage and optimized querying.

!!! info "Columnar Format Efficiency"

    Both SLAF and TileDB (Arrow-interoperable formats) significantly outperform h5ad, with SLAF providing an additional 9.4x advantage over TileDB through its optimized streaming architecture.

## **Gene Filtering Benchmarks**

Gene filtering operations are essential for feature selection, quality control, and differential expression in single-cell analysis.

### **Performance Results**

| Scenario | h5ad Total (ms) | SLAF Total (ms) | TileDB Total (ms) | SLAF vs h5ad | SLAF vs TileDB | Description                                 |
| -------- | --------------- | --------------- | ----------------- | ------------ | -------------- | ------------------------------------------- |
| S1       | 43.4            | 3.0             | 22.3              | **14.6x**    | **7.5x**       | Genes expressed in >=10 cells               |
| S2       | 32.3            | 1.7             | 19.5              | **19.4x**    | **11.7x**      | Genes with >=100 total counts               |
| S3       | 32.1            | 1.8             | 9.3               | **17.4x**    | **5.0x**       | Genes with mean expression >=0.1            |
| S4       | 31.1            | 1.6             | 15.8              | **19.9x**    | **10.1x**      | Exclude mitochondrial genes                 |
| S5       | 32.7            | 1.7             | 16.9              | **19.7x**    | **10.2x**      | Highly variable genes                       |
| S6       | 31.7            | 2.1             | 15.8              | **15.4x**    | **7.7x**       | Non-highly variable genes                   |
| S7       | 31.5            | 2.0             | 18.2              | **15.8x**    | **9.1x**       | Genes in >=50 cells with >=500 total counts |
| S8       | 31.7            | 1.9             | 19.8              | **17.0x**    | **10.6x**      | Genes with 100-10000 total counts           |
| S9       | 33.2            | 2.0             | 11.3              | **16.4x**    | **5.6x**       | Genes in 5-1000 cells                       |

**Average Performance:**

- **SLAF vs h5ad**: **17.3x faster**
- **SLAF vs TileDB**: **8.6x faster**
- **Memory Usage**: SLAF uses 2.2x less memory than h5ad

### **Key Insights**

!!! success "Consistent High Performance"

    SLAF maintains consistent 14x+ speedups across all gene filtering scenarios, demonstrating robust optimization of Polars operations and modern storage formats.

!!! info "Memory Efficiency"

    Gene filtering operations show moderate memory efficiency gains, with SLAF using 2.2x less memory than h5ad for equivalent operations.

## **Expression Queries Benchmarks**

Expression queries retrieve specific expression data for cells or genes, supporting analysis workflows that require targeted data access.

### **Performance Results**

| Scenario | h5ad Total (ms) | SLAF Total (ms) | TileDB Total (ms) | SLAF vs h5ad | SLAF vs TileDB | Description                  |
| -------- | --------------- | --------------- | ----------------- | ------------ | -------------- | ---------------------------- |
| S1       | 484.5           | 16.1            | 63.8              | **30.1x**    | **4.0x**       | Single cell expression       |
| S2       | 251.3           | 13.9            | 19.3              | **18.1x**    | **1.4x**       | Another single cell          |
| S3       | 328.3           | 14.2            | 18.1              | **23.1x**    | **1.3x**       | Two cells                    |
| S4       | 233.2           | 15.5            | 19.0              | **15.1x**    | **1.2x**       | Three cells                  |
| S5       | 232.7           | 523.7           | 150.8             | **0.4x**     | **0.3x**       | Single gene across all cells |
| S6       | 203.4           | 442.6           | 84.2              | **0.5x**     | **0.2x**       | Another single gene          |
| S7       | 256.1           | 303.0           | 97.7              | **0.8x**     | **0.3x**       | Two genes                    |
| S8       | 212.0           | 655.9           | 83.3              | **0.3x**     | **0.1x**       | Three genes                  |
| S9       | 221.4           | 22.5            | 9.9               | **9.9x**     | **0.4x**       | 100x50 submatrix             |
| S10      | 168.3           | 61.9            | 12.1              | **2.7x**     | **0.2x**       | 500x100 submatrix            |
| S11      | 212.2           | 63.2            | 10.8              | **3.4x**     | **0.2x**       | 500x500 submatrix            |

**Average Performance:**

- **SLAF vs h5ad**: **9.5x faster**
- **SLAF vs TileDB**: **0.9x faster**
- **Memory Usage**: SLAF uses 154.6x less memory than h5ad

### **Key Insights**

!!! success "Query Optimization"

    SLAF's expression query performance demonstrates efficient sparse matrix operations and optimized data access patterns, achieving 9.5x average speedup over h5ad.

!!! info "TileDB's Expression Query Strengths"

    TileDB demonstrates impressive performance for gene expression queries and submatrix operations, often outperforming both SLAF and h5ad. For single gene queries across all cells (S5-S8), TileDB shows 2.5-10x speedup over SLAF, highlighting its optimized columnar access patterns for gene-centric operations.

!!! info "SLAF's Cell-Centric Advantages"

    SLAF maintains strong performance for cell-centric queries (S1-S4, S9-S11), achieving 15-30x speedup over h5ad for single cell and submatrix operations, while TileDB shows competitive or superior performance for larger submatrices.

!!! info "Mixed Performance Profile"

    The benchmarks reveal a nuanced performance landscape: SLAF excels at cell-centric operations and metadata filtering, while TileDB demonstrates superior performance for gene-centric expression queries and large submatrix operations. This suggests different systems may be optimal for different analysis workflows.

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
| S1        | 1150.4                 | 2577.0          | 0.4x          | 148.1x            | Calculate QC metrics                               |
| S2        | 941.0                  | 2045.1          | 0.5x          | 1.5x              | Filter cells (min_counts=500, min_genes=200)       |
| S3        | 929.7                  | 2058.9          | 0.5x          | 1.5x              | Filter cells (min_counts=100, min_genes=50)        |
| S4        | 862.1                  | 2184.9          | 0.4x          | 1.5x              | Filter cells (max_counts=10000, max_genes=3000)    |
| S5        | 1208.9                 | 2586.7          | 0.5x          | 1.5x              | Filter genes (min_counts=10, min_cells=5)          |
| S6        | 1183.5                 | 2516.4          | 0.5x          | 1.5x              | Filter genes (min_counts=20, min_cells=5)          |
| S7        | 436.2                  | 3675.7          | 0.1x          | 1.5x              | Normalize total (target_sum=1e4)                   |
| S8        | 410.9                  | 2843.9          | 0.1x          | 1.5x              | Normalize total (target_sum=1e6)                   |
| S9        | 628.2                  | 2083.7          | 0.3x          | 0.0x              | Log1p transformation                               |
| S10       | 653.9                  | 1660.6          | 0.4x          | 98.9x             | Find highly variable genes                         |
| S11       | 695.6                  | 1642.6          | 0.4x          | 98.9x             | Find top 2000 highly variable genes                |
| S12       | 2359.9                 | 6418.5          | 0.4x          | 1.5x              | QC metrics + cell filtering + gene filtering       |
| S13       | 338.8                  | 3417.2          | 0.1x          | 2.0x              | Normalize total + slice 100x50 submatrix (lazy)    |
| S14       | 617.9                  | 2174.4          | 0.3x          | 2.0x              | Log1p + slice 200x100 submatrix (lazy)             |
| S15       | 627.9                  | 3510.9          | 0.2x          | 2.0x              | Normalize + Log1p + slice 500x250 submatrix (lazy) |
| S16       | 922.6                  | 3061.3          | 0.3x          | 1.9x              | Normalize + Log1p + mean per gene (lazy)           |
| S17       | 620.4                  | 3174.6          | 0.2x          | 1.8x              | Normalize + Log1p + variance per cell (lazy)       |

---

**Key Insight**: Lazy computation enables **complex preprocessing pipelines** that would cause memory explosions with traditional tools. The computation cost is paid when materializing results, but the memory efficiency enables workflows impossible with eager processing. This is similar to Dask's delayed computation patterns.

_For detailed migration guides, see [SLAF vs h5ad Benchmarks](slaf_vs_h5ad_benchmarks.md) and [SLAF vs TileDB Benchmarks](slaf_vs_tiledb_benchmarks.md)._
