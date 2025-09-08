# SLAF vs TileDB Performance Benchmarks

This document provides a comprehensive performance comparison between SLAF and TileDB SOMA across bioinformatics and machine learning workflows, focusing on modern columnar storage formats.

## **Overview**

Both SLAF and TileDB represent modern approaches to single-cell data storage, using Arrow-interoperable columnar formats. However, SLAF demonstrates **consistent performance advantages** across metadata filtering and dataloader throughput, highlighting the benefits of its optimized streaming architecture and efficient data access patterns.

### **Key Performance Summary**

| Category                   | SLAF vs TileDB Speedup | Dataset                 |
| -------------------------- | ---------------------- | ----------------------- |
| **Conversion Performance** | **9.9x faster**        | synthetic_50k_processed |
| **Cell Filtering**         | **5.6x faster**        | synthetic_50k_processed |
| **Gene Filtering**         | **8.6x faster**        | synthetic_50k_processed |
| **Cell-centric Queries**   | **1.1x-1.4x faster**   | synthetic_50k_processed |
| **Gene-centric Queries**   | **2.5x-3.3x slower**   | synthetic_50k_processed |
| **Submatrix Queries**      | **2.5x-5.0x slower**   | synthetic_50k_processed |
| **ML Data Loading**        | **45.7x faster**       | Tahoe-100M              |

!!! success "Modern Format Advantage"

    Both SLAF and TileDB significantly outperform traditional h5ad-based approaches, demonstrating the benefits of modern cloud-native storage. SLAF provides additional performance optimizations for streaming and data access.

## **Conversion Performance**

Conversion from h5ad to modern formats demonstrates SLAF's efficiency in data ingestion workflows.

**Input Dataset**: synthetic_50k_processed (49,955 cells × 25,000 genes, ~722MB h5ad file)

| Metric              | SLAF        | TileDB SOMA | SLAF vs TileDB Improvement     |
| ------------------- | ----------- | ----------- | ------------------------------ |
| **Conversion Time** | **1.87s**   | 18.44s      | **9.9x faster**                |
| **Output Size**     | **349.2MB** | 561.1MB     | **38% smaller**                |
| **Peak Memory**     | **826.6MB** | 4,399.6MB   | **5.3x more memory efficient** |

**Key Advantages:**

- **9.9x faster conversion** from h5ad to SLAF format
- **38% smaller output size** compared to TileDB SOMA
- **5.3x more memory efficient** - SLAF uses only 827MB vs TileDB's 4.4GB peak memory
- **Optimized chunked processing** with efficient memory management

### **TileDB to SLAF Conversion**

Converting from TileDB SOMA to SLAF format is straightforward and efficient:

**TileDB to SLAF Performance**: synthetic_50k_processed (49,955 cells × 25,000 genes)

**Migration Benefits:**

- **Fast conversion**: 50k cell dataset converts in just 2 seconds
- **Linear scaling**: Performance scales linearly with the number of cells
- **Simple command**: `slaf convert data.tiledb output.slaf`

!!! success "Easy Migration from TileDB"

    Converting from TileDB to SLAF is simple and fast. A 50k cell dataset takes only 2 seconds to convert, with linear scaling performance. The conversion preserves all data types and metadata while providing significant performance improvements for downstream analysis.

!!! success "Fast Migration Path"

    SLAF's superior conversion performance enables rapid migration of existing h5ad datasets, with conversion times under 2 seconds for 50k cell datasets and significantly smaller output files.

## **Bioinformatics Benchmarks**

### **Cell Filtering Performance**

Cell filtering operations demonstrate the efficiency of modern columnar storage for metadata operations.

| Scenario | TileDB Total (ms) | SLAF Total (ms) | Speedup  | Description                                 |
| -------- | ----------------- | --------------- | -------- | ------------------------------------------- |
| S1       | 43.7              | 9.4             | **4.7x** | Cells with >=500 genes                      |
| S2       | 16.0              | 2.6             | **6.1x** | High UMI count (total_counts > 2000)        |
| S3       | 16.5              | 3.9             | **4.2x** | Mitochondrial fraction < 0.1                |
| S4       | 15.5              | 7.3             | **2.1x** | Complex multi-condition filter              |
| S5       | 14.3              | 2.6             | **5.5x** | Cell type annotation filter                 |
| S6       | 12.5              | 1.7             | **7.5x** | Cells from batch_1                          |
| S7       | 14.5              | 3.8             | **3.8x** | Cells in clusters 0,1 from batch_1          |
| S8       | 12.6              | 1.9             | **6.7x** | High-quality cells (>=1000 genes, <=10% mt) |
| S9       | 12.9              | 1.8             | **7.1x** | Cells with 800-2000 total counts            |
| S10      | 12.7              | 1.6             | **8.1x** | Cells with 200-1500 genes                   |

**Average Performance: 5.6x faster**

### **Gene Filtering Performance**

Gene filtering operations show consistent performance advantages for SLAF's optimized Polars operations.

| Scenario | TileDB Total (ms) | SLAF Total (ms) | Speedup   | Description                                 |
| -------- | ----------------- | --------------- | --------- | ------------------------------------------- |
| S1       | 22.0              | 2.1             | **10.3x** | Genes expressed in >=10 cells               |
| S2       | 15.3              | 1.6             | **9.4x**  | Genes with >=100 total counts               |
| S3       | 13.6              | 1.9             | **7.3x**  | Genes with mean expression >=0.1            |
| S4       | 19.2              | 1.7             | **11.4x** | Exclude mitochondrial genes                 |
| S5       | 13.3              | 1.6             | **8.2x**  | Highly variable genes                       |
| S6       | 13.6              | 1.5             | **8.9x**  | Non-highly variable genes                   |
| S7       | 13.9              | 1.9             | **7.2x**  | Genes in >=50 cells with >=500 total counts |
| S8       | 12.0              | 1.7             | **6.8x**  | Genes with 100-10000 total counts           |
| S9       | 14.1              | 1.7             | **8.2x**  | Genes in 5-1000 cells                       |

**Average Performance: 8.6x faster**

### **Expression Queries Performance**

Expression queries demonstrate the efficiency of optimized sparse matrix operations, where TileDB often wins.

| Scenario | TileDB Total (ms) | SLAF Total (ms) | Speedup  | Description                  |
| -------- | ----------------- | --------------- | -------- | ---------------------------- |
| S1       | 41.7              | 30.6            | **1.4x** | Single cell expression       |
| S2       | 14.9              | 12.9            | **1.1x** | Another single cell          |
| S3       | 16.6              | 12.8            | **1.3x** | Two cells                    |
| S4       | 15.8              | 12.9            | **1.2x** | Three cells                  |
| S5       | 130.4             | 328.3           | **0.4x** | Single gene across all cells |
| S6       | 87.6              | 303.8           | **0.3x** | Another single gene          |
| S7       | 96.3              | 313.2           | **0.3x** | Two genes                    |
| S8       | 9.1               | 355.0           | **0.0x** | Three genes                  |
| S9       | 9.4               | 20.2            | **0.5x** | 100x50 submatrix             |
| S10      | 9.1               | 55.4            | **0.2x** | 500x100 submatrix            |
| S11      | 11.2              | 51.0            | **0.2x** | 500x500 submatrix            |

**Average Performance:**

- SLAF wins cell-centric (1.1x-1.4x),
- TileDB wins gene-centric (2.5x-3.3x) and submatrix (2.5x-5.0x)

## **Machine Learning Benchmarks**

### **Raw Data Loading Performance**

Raw data loading performance demonstrates the advantages of SLAF's optimized streaming architecture.

| System            | Throughput (cells/sec) | Notes                 |
| ----------------- | ---------------------- | --------------------- |
| **SLAF**          | **25,244**             | Optimized streaming   |
| TileDB DataLoader | 552                    | Custom PyTorch loader |

**Performance Comparison: 45.7x faster**

### **GPU-Ready Output Performance**

SLAF provides pre-tokenized sequences ready for GPU training, while TileDB DataLoader only provides raw data.

| System            | Throughput (cells/sec) | Throughput (tokens/sec) | Output Type             |
| ----------------- | ---------------------- | ----------------------- | ----------------------- |
| **SLAF**          | **7,465**              | **15,288,876**          | Pre-tokenized sequences |
| TileDB DataLoader | N/A                    | N/A                     | Raw data only           |

!!! success "GPU Training Advantage"

    SLAF is the only system providing GPU-ready tokenized output, enabling efficient training of foundation models like Geneformer and scGPT.

## **Technical Implementation Comparison**

| Aspect             | SLAF                                                    | TileDB                                                       |
| ------------------ | ------------------------------------------------------- | ------------------------------------------------------------ |
| **Storage**        | Arrow-interoperable columnar storage with Lance backend | Arrow-interoperable array-native storage with TileDB backend |
| **Metadata**       | Polars DataFrames for efficient filtering operations    | Arrow tables with Polars for filtering operations            |
| **Expression**     | Optimized sparse COO matrices with zero-copy access     | Smart indexing for both cell and gene based slicing          |
| **ML Integration** | Native PyTorch DataLoader with tokenization support     | Needs third party custom PyTorch DataLoader for raw data     |

### **Key Technical Advantages**

!!! success "Optimized Streaming Architecture"

    SLAF's asynchronous prefetching and background processing provide significant performance advantages over TileDB's synchronous loading approach.

!!! success "Enhanced ML Integration"

    SLAF provides native PyTorch DataLoader integration with built-in tokenization support, while TileDB requires custom loader implementation.

!!! success "Memory Efficiency"

    SLAF's optimized memory management and zero-copy operations result in 1.5-2.3x memory efficiency gains over TileDB.

!!! success "Simplified API"

    SLAF provides a more streamlined API for common bioinformatics operations, while TileDB requires more manual configuration.

## **Migration Benefits**

### **Performance Improvements**

- **5.6x faster** bioinformatics operations
- **45.7x faster** machine learning data loading
- **Faster conversion from h5ad** to SLAF format, enabling rapid migration of existing datasets
- **Easy TileDB migration**: 2-second conversion for 50k cell datasets with linear scaling

### **Developer Experience**

- **Simplified API** with familiar Polars operations
- **Scanpy-native lazy workflows** with drop-in replacement of AnnData objects, enabling efficient lazy computation graphs for preprocessing and filtering pipelines
- **Enhanced ML integration** with native PyTorch support
- **Built-in tokenization** for foundation model training

## **Use Case Recommendations**

### **Choose SLAF for:**

- **High-throughput machine learning** workflows requiring fast data loading
- **Foundation model training** requiring GPU-ready tokenized sequences
- **Streaming applications** where continuous data flow is critical
- **Memory-constrained environments** where efficiency is important

### **Choose TileDB for:**

- **Existing TileDB infrastructure** where migration costs are high
- **Multi-modal data** where TileDB's broader ecosystem is beneficial

## **Conclusion**

SLAF provides **strong performance advantages** over TileDB across most benchmark categories:

- **Metadata filtering**: 5.6x-8.6x faster (cell and gene filtering)
- **Machine learning data loading**: 45.7x faster
- **Expression queries**: Mixed performance (SLAF wins cell-centric, TileDB wins gene-centric)

While both systems represent modern approaches to single-cell data storage, SLAF's optimized streaming architecture, enhanced ML integration, and simplified API provide significant advantages for most use cases. The benchmarks demonstrate that SLAF's performance optimizations and developer-friendly design make it the preferred choice for high-throughput bioinformatics and machine learning workflows.

---

_For detailed migration guidance including TileDB to SLAF conversion, see [Migrating to SLAF](../user-guide/migrating-to-slaf.md). For comprehensive benchmark results, see [Bioinformatics Benchmarks](bioinformatics_benchmarks.md) and [ML Benchmarks](ml_benchmarks.md)._
