# ML Benchmarks: SLAF vs State-of-the-Art Dataloaders

SLAF provides state-of-the-art (SOTA) performance in data loading throughput for machine learning workflows, reaching **2.6x speedups** relative to current SOTA, particularly for training transformer-based single-cell foundation models. What follows are comprehensive benchmarks comparing SLAF against state-of-the-art dataloaders including scDataset, BioNeMo SCDL, AnnDataLoader, AnnLoader, and TileDB DataLoader.

## **Motivation**

The goal of these benchmarks is to demonstrate that SLAF can stream tokens to modern GPUs at a rate sufficient to prevent idle time between training loops. For a 1B parameter model like scGPT, _fast enough_ means delivering training batches **within 50 ms** to keep the GPU utilization high. This benchmark establishes SLAF's ability to meet the throughput requirements for efficient foundation model training on massive single-cell datasets.

## **Dataset and Hardware**

### **Dataset: Tahoe-100M**

We downloaded one of the 7 h5ad files comprising the [Tahoe-100M dataset](https://www.biorxiv.org/content/10.1101/2025.02.20.639398v1) made accessible by [ARC Institute](https://github.com/ArcInstitute/arc-virtual-cell-atlas/tree/main). This slice of the dataset contains 5,481,420 cells and 62,710 genes, with approximately 8B non-zero expression values. All benchmarks reported below used this dataset except for the TileDB dataloader since we couldn't successfully convert a 5M-cell dataset to the Tile DB SOMA Experiment format with 32G RAM. For the TileDB DataLoader alone, we report numbers on a smaller 50k-cell synthetic dataset.

### **Conversion and Optimization**

We used the [SLAF converter](../api/data/#slafconverter) (see [Migrating to SLAF](../user-guide/migrating-to-slaf.md)) to convert the h5ad file to SLAF format. The Lance table fragments (Lance's term for partitions) were optimized for compression/query tradeoffs, with 5-10M non-zeros (rows) per fragment in the expression table. While inherently parallelizable, conversion is currently single process, and took about 10 minutes for this dataset.

### **Hardware Configuration**

- **Machine**: Apple MacBook Pro with M1 Max
- **Memory**: 32 GB RAM
- **Storage**: 1 TB NVMe SSD
- **OS**: macOS 13.6.1

!!! note "Note"

      These benchmarks represent performance on a high-end laptop. Production deployments on dedicated servers with faster storage may show different performance characteristics. Likewise, performance from object storage to non-colocated compute might be worse.

## **Internal Benchmarks**

### **Methodology**

We used a batch size of 32 with an enhanced warmup and measurement procedure to ensure accurate and consistent results, especially for the Mixture of Scanners (MoS) strategy:

- **Initial Warmup**: 15 batches to initialize the dataloader
- **Extended Warmup**: 10 seconds to allow MoS to fully stabilize all fragment generators
- **Measurement Period**: 40 seconds of pure performance measurement (excluding warmup time)
- **Total Runtime**: 50 seconds per benchmark (10s warmup + 40s measurement)

This methodology ensures that all dataloader strategies reach steady-state performance before measurement begins, eliminating variance from incomplete initialization.

### **Tokenization Strategy Comparison**

We benchmarked different tokenization strategies to understand the performance impact of various preprocessing options:

| Tokenization Strategy                   | Throughput (cells/sec) | Throughput (tokens/sec) |
| --------------------------------------- | ---------------------- | ----------------------- |
| scGPT with binning                      | 5,350                  | 10,967,687              |
| scGPT without binning                   | 5,157                  | 10,572,411              |
| Geneformer with percentile filtering    | 6,999                  | 14,335,137              |
| Geneformer without percentile filtering | 6,494                  | 13,300,337              |
| Raw mode (no tokenization)              | 22,323                 | N/A                     |

**Tokenization Strategy Comparison:**

```
Throughput (cells/sec)
─────────────────────────────────────────────────────────
Raw mode              ████████████████████████ 22,323
Geneformer (filter)   ███████ 6,999
Geneformer (no filt)  ██████ 6,494
scGPT (binning)       █████ 5,350
scGPT (no binning)    █████ 5,157
─────────────────────────────────────────────────────────
                      0    5K   10K   15K   20K   25K
```

!!! success "Strategy Insights"

    - **Geneformer strategies** show ~30% higher throughput than scGPT strategies
    - **Binning and filtering** have minimal performance impact (~7% difference)
    - **Raw mode** provides 3.4x higher throughput than tokenized modes, demonstrating the tokenization overhead

### **Raw Mode Performance Scaling**

Raw mode bypasses tokenization and returns Polars DataFrames that have the exact schema as sparse CSR tensors, demonstrating SLAF's base data loading performance.

| Batch Size | Throughput (cells/sec) | Total Cells | Measurement Time (s) |
| ---------- | ---------------------- | ----------- | -------------------- |
| 32         | 23,783                 | 713,577     | 30.0                 |
| 64         | 25,259                 | 765,957     | 30.3                 |
| 128        | 28,079                 | 842,394     | 30.0                 |
| 256        | 28,169                 | 850,146     | 30.2                 |

**Raw Mode Batch Size Scaling:**

```
Throughput (cells/sec)
─────────────────────────────────────────────────────────
Batch 256    ████████████████████████████████ 28,169
Batch 128    ██████████████████████████████ 28,079
Batch 64     ███████████████████████████ 25,259
Batch 32     ██████████████████████████ 23,783
─────────────────────────────────────────────────────────
             0    5K   10K   15K   20K   25K   30K
```

!!! success "Optimization Validation"

    Raw mode throughput shows **1.2x improvement** from batch size 32 to 256, demonstrating that SLAF's data loading pipeline scales efficiently with larger batch sizes while maintaining high performance.

### **Fragment vs Batch Loading Comparison**

SLAF supports two loading strategies: fragment-based and batch-based loading. Fragment-based loading processes entire Lance fragments at once, while batch-based loading processes multiple Lance batches sequentially.

| Strategy               | Throughput (cells/sec) | Total Cells | Total Batches |
| ---------------------- | ---------------------- | ----------- | ------------- |
| Fragment-Based Loading | 22,472                 | 229,669     | 7,180         |
| Batch-Based Loading    | 24,354                 | 243,554     | 8,038         |

!!! note "Fragment Strategy Performance"

    Batch-based loading shows modestly higher throughput than fragment-based loading in this benchmark, but test-retest repeatability shows high variance. The performance difference should not be overinterpreted as it may vary significantly across different runs and hardware configurations.

!!! info "Strategy Selection"

    Mixture of Scanners (MoS) is the default strategy in SLAF for foundation model training, providing 88% of random entropy with only 3.2% throughput penalty. Sequential loading is available for maximum throughput by setting `use_mixture_of_scanners=False, by_fragment=False` to the SLAFDataLoader for users who prioritize speed over entropy.

### **Tokenized Mode: Tokens/sec Scaling**

Tokenized mode provides pre-tokenized sequences ready for GPU training, demonstrating SLAF's end-to-end pipeline performance.

| Batch Size | Throughput (cells/sec) | Throughput (tokens/sec) | Total Cells | Measurement Time (s) |
| ---------- | ---------------------- | ----------------------- | ----------- | -------------------- |
| 32         | 7,141                  | 14,624,846              | 215,990     | 30.2                 |
| 64         | 7,147                  | 14,637,356              | 223,872     | 31.3                 |
| 128        | 7,309                  | 14,969,420              | 224,663     | 30.7                 |
| 256        | 7,269                  | 14,885,945              | 224,511     | 30.9                 |

**Tokenized Mode Throughput Scaling:**

```
Throughput (tokens/sec, millions)
─────────────────────────────────────────────────────────
Batch 128    ████████████████████████████████ 14.97M
Batch 256   ███████████████████████████████ 14.89M
Batch 64    ███████████████████████████████ 14.64M
Batch 32    ███████████████████████████████ 14.62M
─────────────────────────────────────────────────────────
             0    3M    6M    9M   12M   15M   18M
```

!!! success "Tokenization Efficiency"

    Token throughput remains remarkably constant across batch sizes (1.0x scaling), demonstrating that SLAF's tokenization pipeline is well-optimized and not the bottleneck. This validates that tokens/sec is the meaningful metric for GPU training workloads.

### **Entropy Measurement: Training Batch Randomness**

To ensure models don't converge to local minima due to biased and highly correlated training batches, we want to make training batches as random as possible. However, random reads are more expensive than sequential reads, so we need to balance randomness with performance.

To address this challenge, we developed a novel dataloader strategy called the **Mixture of Scanners (MoS)** approach, which randomly tasks a small randomized group of scanners to populate a queue of training batches by reading from different starting points of the dataset. A deeper dive into our approach to optimize dataloaders is available [here](../blog/blazing-fast-dataloaders.md) and a more detailed write up of the MoS dataloader is [here](../blog/blazing-fast-dataloaders-2.md).

To measure entropy without using metadata, we simulate random cell IDs and measure L1 distance between pairs of cell IDs both within and across adjacent training batches for our different dataloaders to show how each dataloader strategy performs relative to a purely sequential (lowerbound) vs a truly random approach (upperbound).

We ran a test on 10,000 batches with a batch_size of 32 from a 5.4M cell dataset and found these results:

**Entropy Measurement Results:**

| Strategy   | Within-Batch L1 | Across-Batch L1 |
| ---------- | --------------- | --------------- |
| sequential | 94.1            | 104.5           |
| fragment   | 1,643.5         | 1,672.6         |
| mos        | 1,608,648.2     | 1,642,829.9     |
| random     | 1,828,595.2     | 1,824,468.9     |

**Normalized Entropy Scores [0=Sequential, 1=Random]:**

| Strategy   | Within-Batch L1 | Across-Batch L1 |
| ---------- | --------------- | --------------- |
| sequential | 0.000           | 0.000           |
| fragment   | 0.001           | 0.001           |
| mos        | 0.880           | 0.900           |

**Throughput Performance Results:**

| Strategy   | Throughput (cells/sec) | Total Cells | Total Batches |
| ---------- | ---------------------- | ----------- | ------------- |
| sequential | 23,728                 | 711,990     | 23,509        |
| fragment   | 26,769                 | 803,072     | 25,216        |
| mos        | 22,972                 | 689,234     | 21,546        |

!!! success "Entropy Strategy Performance"

    - **Sequential loading** provides the lowest entropy (0.000), with contiguous cell IDs from Lance batches
    - **Fragment-based loading** shows minimal improvement (0.001), processing complete Lance fragments for slightly higher entropy
    - **Mixture of Scanners (MoS)** achieves near-random entropy (0.88+), demonstrating effective randomization while maintaining high throughput
    - **MoS approach** provides **88% of the entropy** of truly random sampling while maintaining the performance benefits of structured data access

!!! success "Throughput Performance Analysis"

    - **Fragment-based loading** achieves the highest throughput (26,769 cells/sec), showing **12.8% improvement** over sequential loading
    - **MoS approach** maintains competitive throughput (22,972 cells/sec), only **3.2% slower** than sequential loading despite providing 88% random entropy
    - **Performance-entropy trade-off**: MoS successfully balances high entropy (0.88) with minimal throughput penalty (3.2% vs sequential)
    - **All strategies** maintain excellent throughput (>22K cells/sec), demonstrating SLAF's efficient data loading architecture

!!! info "Entropy Interpretation Guide"

    - **Within-Batch**: How random are the cells within each batch
    - **Across-Batch**: How much batch composition changes between batches
    - **L1 Distance**: Mean absolute difference between cell ID pairs
    - **Scores closer to 0** = more sequential, **closer to 1** = more random

!!! info "MoS Implementation Benefits"

    The Mixture of Scanners approach successfully balances the competing demands of training batch randomness and data loading performance. By using multiple scanners reading from different dataset locations, MoS achieves 88% of the entropy of truly random sampling without creating pre-randomized copies of datasets. The approach maintains 96.8% of sequential loading throughput while providing near-random batch composition, making it ideal for training foundation models that require both high throughput and effective batch randomization.

## **External Benchmarks**

### **Alternate Dataloaders**

We compared SLAF against six state-of-the-art dataloaders:

1. **[annbatch](https://annbatch.readthedocs.io/en/stable/index.html)** - High-performance data loader for minibatching on-disk AnnData, co-developed by lamin and scverse
2. **[AnnLoader](https://anndata.readthedocs.io/en/latest/generated/anndata.experimental.AnnLoader.html)** - Experimental PyTorch DataLoader for AnnData objects from `anndata.experimental`
3. **[AnnDataLoader](https://docs.scvi-tools.org/en/stable/api/reference/scvi.dataloaders.AnnDataLoader.html)** - From [scvi-tools](https://docs.scvi-tools.org/en/stable/index.html), designed for training variational autoencoder (VAE)-style models
4. **[scDataset](https://github.com/Kidara/scDataset/tree/main)** - Recently released high-performance dataloader with multiprocessing support
5. **[TileDB DataLoader](https://tiledbsoma.readthedocs.io/)** - An internal custom PyTorch DataLoader for TileDB SOMA experiments
6. **[BioNeMo SCDL](https://docs.nvidia.com/bionemo-framework/2.0/user-guide/developer-guide/bionemo-scdl/bionemo-scdl-Overview/)** - NVIDIA's single-cell data loading framework for scalable training of foundation models

### **Methodology**

To match the benchmarks from the [scDataset paper](https://arxiv.org/pdf/2506.01883) as closely as possible, we used a `batch_size=64` across all comparisons. For scDataset itself, we used the optimal parameters in our hardware (`block_size=8`, `fetch_factor=64`, which were different from the ones found to be optimal in the paper). However, we couldn't use `num_workers=12` out of the box because h5ad datasets aren't pickle-able and PyTorch DataLoaders expect this since they use multiprocessing.

**Enhanced Measurement Procedure**: All external benchmarks now use the same enhanced measurement procedure as internal benchmarks for fair comparison:

- **Initial Warmup**: 15 batches to initialize each dataloader
- **Extended Warmup**: 10 seconds to allow all systems to reach steady state
- **Measurement Period**: 30 seconds of pure performance measurement (excluding warmup time)
- **Total Runtime**: 40 seconds per benchmark (10s warmup + 30s measurement)

This ensures fair and consistent performance comparisons across all dataloader systems.

### **Tier 1: Raw Data Loading Comparison**

Raw data loading performance measures the base throughput of each system without any tokenization overhead. All benchmarks use `batch_size=64` for consistent comparison.

| System                  | Throughput (cells/sec) |
| ----------------------- | ---------------------- |
| **annbatch**            | **68,867**             |
| **SLAF**                | **22,399**             |
| BioNeMo SCDL            | 2,976                  |
| scDataset               | 2,550                  |
| TileDB DataLoader (MoS) | 601                    |
| AnnDataLoader           | 411                    |
| AnnLoader               | 251                    |

**Throughput Comparison Chart:**

```
Throughput (cells/sec)
─────────────────────────────────────────────────────────
annbatch          ████████████████████████████████████ 68,867
SLAF              ████████████ 22,399
BioNeMo SCDL      ██ 2,976
scDataset         ██ 2,550
TileDB (MoS)      █ 601
AnnDataLoader      █ 411
AnnLoader          █ 251
─────────────────────────────────────────────────────────
                  0   10K   20K   30K   40K   50K   60K   70K
```

!!! success "SOTA Performance"

    SLAF achieves **8.8x higher throughput** than scDataset, **7.5x higher throughput** than BioNeMo SCDL, **37.3x higher throughput** than TileDB DataLoader, **54.5x higher throughput** than AnnDataLoader, and **89.2x higher throughput** than AnnLoader in raw data loading.

!!! info "annbatch Performance Analysis"

    annbatch demonstrates exceptional raw data loading performance, achieving **68,867 cells/sec**—**3.1x higher** than SLAF's throughput. This performance advantage stems from fundamental storage format differences: annbatch uses **CSC (Compressed Sparse Column)** format via zarr, which is optimized specifically for row-wise batch loading operations common in ML training workflows.

    SLAF, in contrast, uses **COO (Coordinate)** format, which provides superior flexibility across multiple use cases. This design choice reflects SLAF's broader mission: to serve as a single unified format that efficiently supports (1) low-latency cell and gene queries, (2) batch processing operations, and (3) ML training workloads—all from the same stored representation without data duplication. The COO format enables SLAF to maintain a "store once, query in place" philosophy across these diverse workloads, trading some raw loading throughput for greater versatility and query performance.

    For users whose primary use case is high-throughput ML training on pre-shuffled datasets, annbatch's CSC-based approach provides excellent performance. For users requiring a single format that supports both training and analytical queries, SLAF's COO-based architecture offers a more balanced solution.

!!! info "scDataset Performance Analysis"

    Our comprehensive benchmarks reveal that scDataset achieves **2,550 cells/sec** with optimized parameters (`block_size=8`, `fetch_factor=64`). This performance is consistent with the system's design, though lower than our initial expectations. Note that these benchmarks use different hardware (M1 Max) than the scDataset paper's reported results (NVIDIA DGX CPU), which may account for some performance differences.

    However, we found significant limitations with multiprocessing due to pickling issues with h5py-backed AnnData objects. See our [detailed scDataset benchmarks](scdataset_benchmarks.md) for complete analysis including parameter scaling and multiprocessing limitations.

!!! info "Parameter Scaling Validation"

    Our parameter sweeps confirm scDataset's strong scaling behavior: **23.1x improvement** from worst to best configuration. The `fetch_factor` parameter shows the strongest scaling (20x+ improvement), while `block_size` shows more moderate effects. This validates the design approach described in their paper, though optimal parameters may vary by hardware.

!!! info "Multiprocessing Limitations"

    We were unable to test `num_workers > 0` due to pickling errors with h5py objects. We're still working with the scDataset team to figure out implementation differences.

### **Tier 2: GPU-Ready Output Comparison**

Raw data loading benchmarks are great, provided that we intend to train on gene expression counts directly. However, for modern foundation models like Geneformer, scGPT, Transcriptformer, or STATE, cell sentences are constructed using tokens that represent gene identity and expression bins. A lot of these workflows require dataframe-friendly operations like sorting, windowing, ranking, and filtering. Our view is that it much better to situate these computations within the (typically) CPU-bound dataloader, rather than expect the GPU in the training loop to do the heavy lifting. Accordingly, SLAF dataloaders take a tokenizer and transform raw data into training-ready token sequences.

This GPU-ready throughput measures end-to-end performance including tokenization (that includes windowing, ranking, vocabulary mapping and padding), which is critical for training workflows involving models that turn cells into sentences.

Even though SLAF's tokenizing dataloaders do more work (tokenization), we find that their throughput remains competitive with scDataset's raw-data dataloader, achieving comparable performance despite the additional processing overhead.

| System   | Throughput (cells/sec) | Throughput (tokens/sec) |
| -------- | ---------------------- | ----------------------- |
| **SLAF** | **7,487**              | **15,332,896**          |

!!! success "GPU-Ready Cell Sentences"

    SLAF dataloaders provide the only GPU-ready input among the available alternatives.

## **Some Discrepancies**

!!! info "Tokens/sec is better than Cells/sec"

      Cells/sec can be a misleading metric for GPU training workloads. A better measure of throughput to minimize GPU idle time is tokens/sec, since pre-made token sequences are ready for matrix multiplication on the GPU. SLAF's tokenized mode demonstrates this principle: while cells/sec decreases due to tokenization overhead, relative to the raw mode, the constant tokens/sec across batch sizes shows that the tokenization pipeline is well-optimized across scales.

!!! info "Scaling behaviors reveal hidden optimization opportunities"

    SLAF's constant scaling with batch size suggests that the loading and processing are impedance matched: loading more data per batch does not slow down throughput. Constant dataloader throughput for larger batch sizes implies that the bottleneck to batch size is not dataloading but GPU memory. In contrast, we observed that scDataset's throughput scales linearly with batch size (not shown in these results), suggesting that it is doing more work than needed at small batch sizes, and could achieve better performance with optimizations like async prefetching.

!!! info "In-memory formats matter"

    The performance difference between AnnDataLoader (422 cells/sec) and scDataset (9,550 cells/sec) is dramatic. While scDataset is smarter at batching and randomization, since our benchmark tests them on loading from h5ad, it's important to compare apples to apples dataloader outputs. AnnDataLoader and AnnLoader return `torch.sparse_csr` tensors whereas scDataset returns `scipy.sparse.csr_matrix`, and these format inter-conversions represent non-zero overhead.

    In our work, we noticed different overheads for conversion from polars dataframe (SLAF's preferred format for raw data) to torch and scipy sparse formats, and ultimately decided to keep raw outputs in polars. The performance of AnnLoader and AnnDataLoader relative to scDataset is almost certainly due to the overhead of conversion from scipy sparse arrays to torch arrays and worth benchmarking more carefully to identify low-hanging fruit for optimizations in both AnnLoader and AnnDataLoader.

## **Conclusion**

### **Cloud-Native Architecture**

While these benchmarks use local SSD, the Lance format is native to cloud storage. Early tests suggest that latency between S3 and EC2 in the same region is not appreciably different from local storage. This opens up a cloud-native, store-once, query-multiple-times zero-copy architecture that eliminates data duplication.

### **GPU Throughput Requirements**

What's a good enough cells/sec rate to keep an 8×H100 node at $2/hr busy? Assuming 50 ms per training loop for a model like scGPT:

- **8 GPUs × 32 cells/batch × 20 batches/sec = 5,120 cells/sec** would maximize GPU utilization
- **Tahoe-100M training**: 100M cells ÷ 5,120 cells/sec = ~5.4 hours per epoch ~ $86 / epoch
- **Anything faster than 5,120 cells/sec** opens up multi-node training possibilities, trading off cost and wall clock time.

This raises the question: can we build towards a $100 scGPT model through efficient multi-node training enabled by high-throughput data loading? More on this soon!

### **High Concurrency and Multi-User Training**

The Lance format's high concurrency, optimized for production multimodal data lakes with high QPS, enables not only multi-node training but multiple users training multiple models simultaneously without their own copies of the dataset. This contrasts with h5ad, which requires:

1. **Local storage**: The dataset must be local to the CPU instance loading it for attached GPUs
2. **Non-concurrent access**: One copy of the dataset per user

SLAF, with Lance under the hood, enables a truly scalable architecture for foundation model training on massive single-cell datasets.

---

_These benchmarks demonstrate SLAF's position as the leading solution for high-performance single-cell data loading, enabling efficient training of foundation models on massive datasets with minimal resource requirements and maximum scalability._

_SLAF is a young project with a bus factor of 1. You can help improve that by using it and contributing to it. Read about the [SLAF vision in this blog post](blog/introducing-slaf.md) and contribute at [github.com/slaf-project/slaf](https://github.com/slaf-project/slaf)._
