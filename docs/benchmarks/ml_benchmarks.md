# ML Benchmarks: SLAF vs State-of-the-Art Dataloaders

SLAF provides state-of-the-art (SOTA) performance in data loading throughput for machine learning workflows, reaching **2x speedups** relative to current standards, particularly for training transformer-based single-cell foundation models. What follows are comprehensive benchmarks comparing SLAF against state-of-the-art dataloaders including scDataset, AnnDataLoader, and AnnLoader.

## **Motivation**

The goal of these benchmarks is to demonstrate that SLAF can stream tokens to modern GPUs at a rate sufficient to prevent idle time between training loops. For a 1B parameter model like scGPT, _fast enough_ means delivering training batches **within 50 ms** to keep the GPU utilization high. This benchmark establishes SLAF's ability to meet the throughput requirements for efficient foundation model training on massive single-cell datasets.

## **Dataset and Hardware**

### **Dataset: Tahoe-100M**

We downloaded one of the 7 h5ad files comprising the [Tahoe-100M dataset](https://www.biorxiv.org/content/10.1101/2025.02.20.639398v1) made accessible by [ARC Institute](https://github.com/ArcInstitute/arc-virtual-cell-atlas/tree/main). This slice of the dataset contains 5,481,420 cells and 62,710 genes, with approximately 8B non-zero expression values. All benchmarks reported below used this dataset unless indicated otherwise.

### **Conversion and Optimization**

We used the [SLAF converter](../api/data/#slafconverter) (see [Migrating to SLAF](../user-guide/migrating-to-slaf.md)) to convert the h5ad file to SLAF format. The Lance table fragments (Lance's term for partitions) were optimized for compression/query tradeoffs, with 50M non-zeros (rows) per fragment in the expression table. While inherently parallelizable, conversion is currently single process, and took about 10 minutes for this dataset.

### **Hardware Configuration**

- **Machine**: Apple MacBook Pro with M1 Max
- **Memory**: 32 GB RAM
- **Storage**: 1 TB NVMe SSD
- **OS**: macOS 13.6.1

!!! note "Note"

      These benchmarks represent performance on a high-end laptop. Production deployments on dedicated servers with faster storage may show different performance characteristics. Likewise, performance from object storage to non-colocated compute might be worse.

## **Internal Benchmarks**

### **Methodology**

We used a batch size of 32, 3 warmup batches, and a measurement duration of 10 seconds for all internal benchmarks.

### **Tokenization Strategy Comparison**

We benchmarked different tokenization strategies to understand the performance impact of various preprocessing options:

| Tokenization Strategy                   | Throughput (cells/sec) | Throughput (tokens/sec) |
| --------------------------------------- | ---------------------- | ----------------------- |
| scGPT with binning                      | 6,633                  | 13,598,202              |
| scGPT without binning                   | 6,750                  | 13,839,536              |
| Geneformer with percentile filtering    | 9,246                  | 18,937,785              |
| Geneformer without percentile filtering | 9,496                  | 19,448,877              |
| Raw mode (no tokenization)              | 25,184                 | N/A                     |

!!! success "Strategy Insights"

    - **Geneformer strategies** show ~40% higher throughput than scGPT strategies
    - **Binning and filtering** have minimal performance impact (~2% difference)
    - **Raw mode** provides 2.8x higher throughput than tokenized modes, demonstrating the tokenization overhead

### **Raw Mode Performance Scaling**

Raw mode bypasses tokenization and returns Polars DataFrames that have the exact schema as sparse CSR tensors, demonstrating SLAF's base data loading performance.

| Batch Size | Throughput (cells/sec) | Total Cells | Measurement Time (s) |
| ---------- | ---------------------- | ----------- | -------------------- |
| 32         | 24,140                 | 241,412     | 10.0                 |
| 64         | 28,084                 | 280,851     | 10.0                 |
| 128        | 29,545                 | 295,637     | 10.0                 |
| 256        | 30,575                 | 305,770     | 10.0                 |

!!! success "Optimization Validation"

    Raw mode throughput shows **1.3x improvement** from batch size 32 to 256, demonstrating that SLAF's data loading pipeline scales efficiently with larger batch sizes while maintaining high performance.

### **Fragment vs Batch Loading Comparison**

SLAF supports two loading strategies: fragment-based and batch-based loading. Fragment-based loading processes entire Lance fragments at once, while batch-based loading processes multiple Lance batches sequentially.

| Strategy               | Throughput (cells/sec) | Total Cells | Total Batches |
| ---------------------- | ---------------------- | ----------- | ------------- |
| Fragment-Based Loading | 21,735                 | 229,669     | 7,180         |
| Batch-Based Loading    | 19,512                 | 195,356     | 6,441         |

!!! note "Fragment Strategy Performance"

    Fragment-based loading shows modestly higher throughput than batch-based loading in this benchmark, but test-retest repeatability shows high variance. The performance difference should not be overinterpreted as it may vary significantly across different runs and hardware configurations.

!!! info "Strategy Selection"

    Batch-based loading is the default strategy in SLAF as it has lower memory overhead. Fragment-based loading is available as an alternative with just a single additional argument (`by_fragment=True`) to the SLAFDataLoader for users who prefer processing larger data chunks.

### **Tokenized Mode: Tokens/sec Scaling**

Tokenized mode provides pre-tokenized sequences ready for GPU training, demonstrating SLAF's end-to-end pipeline performance.

| Batch Size | Throughput (cells/sec) | Throughput (tokens/sec) | Total Cells | Measurement Time (s) |
| ---------- | ---------------------- | ----------------------- | ----------- | -------------------- |
| 32         | 9,334                  | 19,115,469              | 93,604      | 10.0                 |
| 64         | 9,296                  | 19,037,797              | 93,025      | 10.0                 |
| 128        | 9,254                  | 18,951,337              | 92,766      | 10.0                 |
| 256        | 9,387                  | 19,225,378              | 94,119      | 10.0                 |

!!! success "Tokenization Efficiency"

    Token throughput remains remarkably constant across batch sizes (1.0x scaling), demonstrating that SLAF's tokenization pipeline is well-optimized and not the bottleneck. This validates that tokens/sec is the meaningful metric for GPU training workloads.

## **External Benchmarks**

### **Alternate Dataloaders**

We compared SLAF against three state-of-the-art dataloaders:

1. **[AnnLoader](https://anndata.readthedocs.io/en/latest/generated/anndata.experimental.AnnLoader.html)** - Experimental PyTorch DataLoader for AnnData objects from `anndata.experimental`
2. **[AnnDataLoader](https://docs.scvi-tools.org/en/stable/api/reference/scvi.dataloaders.AnnDataLoader.html)** - From [scvi-tools](https://docs.scvi-tools.org/en/stable/index.html), designed for training variational autoencoder (VAE)-style models
3. **[scDataset](https://github.com/Kidara/scDataset/tree/main)** - Recently released high-performance dataloader with multiprocessing support

!!! question "Help"

    At the time of writing, we couldn't find a submodule called scdl from NVIDIA BioNeMo's PyPI package that implements [the scdl dataloader](https://docs.nvidia.com/bionemo-framework/2.0/user-guide/developer-guide/bionemo-scdl/bionemo-scdl-Overview/); it seems to have been deprecated.

### **Methodology**

To match the benchmarks from the [scDataset paper](https://arxiv.org/pdf/2506.01883) as closely as possible, we used a `batch_size=64` across all comparisons. For scDataset itself, we used the optimal parameters in our hardware (`block_size=8`, `fetch_factor=64`, which were different from the ones found to be optimal in the paper). However, we couldn't use `num_workers=12` out of the box because h5ad datasets aren't pickle-able and PyTorch DataLoaders expect this since they use multiprocessing.

### **Tier 1: Raw Data Loading Comparison**

Raw data loading performance measures the base throughput of each system without any tokenization overhead.

| System        | Throughput (cells/sec) |
| ------------- | ---------------------- |
| **SLAF**      | **22,451**             |
| scDataset     | 10,785                 |
| AnnDataLoader | 392                    |
| AnnLoader     | 240                    |

!!! success "SOTA Performance"

    SLAF achieves **2.1x higher throughput** than scDataset and **57x higher throughput** than AnnDataLoader in raw data loading.

!!! info "scDataset Performance Analysis"

    Our comprehensive benchmarks reveal that scDataset can achieve excellent performance with proper parameter tuning. We observed **10,785 cells/sec** with optimized parameters, which is **5.4x higher** than the paper's reported ~2,000 cells/sec, even without using multiprocessing. Note that these are completely different systems though (M1 Max vs NVIDIA DGX CPU).

    However, we found significant limitations with multiprocessing due to pickling issues with h5py-backed AnnData objects. See our [detailed scDataset benchmarks](scdataset_benchmarks.md) for complete analysis including parameter scaling and multiprocessing limitations.

!!! info "Parameter Scaling Validation"

    Our parameter sweeps confirm scDataset's strong scaling behavior: **23.1x improvement** from worst to best configuration. The `fetch_factor` parameter shows the strongest scaling (20x+ improvement), while `block_size` shows more moderate effects. This validates the design approach described in their paper, though optimal parameters may vary by hardware.

!!! info "Multiprocessing Limitations"

    We were unable to test `num_workers > 0` due to pickling errors with h5py objects. We're still working with the scDataset team to figure out implementation differences.

### **Tier 2: GPU-Ready Output Comparison**

Raw data loading benchmarks are great, provided that we intend to train on gene expression counts directly. However, for modern foundation models like Geneformer, scGPT, Transcriptformer, or STATE, cell sentences are constructed using tokens that represent gene identity and expression bins. A lot of these workflows require dataframe-friendly operations like sorting, windowing, ranking, and filtering. Our view is that it much better to situate these computations within the (typically) CPU-bound dataloader, rather than expect the GPU in the training loop to do the heavy lifting. Accordingly, SLAF dataloaders take a tokenizer and transform raw data into training-ready token sequences.

This GPU-ready throughput measures end-to-end performance including tokenization (that includes windowing, ranking, vocabulary mapping and padding), which is critical for training workflows involving models that turn cells into sentences.

Even though SLAF's tokenizing dataloaders do more work, we find that their throughput exceeds scDataset's raw-data dataloader by 1.8x.

| System   | Throughput (cells/sec) | Throughput (tokens/sec) |
| -------- | ---------------------- | ----------------------- |
| **SLAF** | **8,415**              | **17,234,044**          |

!!! success "GPU-Ready Cell Sentences"

    SLAF dataloaders provide the only GPU-ready input among the available alternatives.

## **Some Discrepancies**

!!! info "Tokens/sec is better than Cells/sec"

      Cells/sec can be a misleading metric for GPU training workloads. A better measure of throughput to minimize GPU idle time is tokens/sec, since pre-made token sequences are ready for matrix multiplication on the GPU. SLAF's tokenized mode demonstrates this principle: while cells/sec decreases due to tokenization overhead, relative to the raw mode, the constant tokens/sec across batch sizes shows that the tokenization pipeline is well-optimized across scales.

!!! info "Scaling behaviors reveal hidden optimization opportunities"

    SLAF's constant scaling with batch size suggests that the loading and processing are impedance matched: loading more data per batch does not slow down throughput. Constant dataloader throughput for larger batch sizes implies that the bottleneck to batch size is not dataloading but GPU memory. In contrast, we observed that scDataset's throughput scales linearly with batch size (not shown in these results), suggesting that it is doing more work than needed at small batch sizes, and could achieve better performance with optimizations like async prefetching.

!!! info "In-memory formats matter"

    The performance difference between AnnDataLoader (392 cells/sec) and scDataset (10,785 cells/sec) is dramatic. While scDataset is smarter at batching and randomization, since our benchmark tests them on loading from h5ad, it's important to compare apples to apples dataloader outputs. AnnDataLoader and AnnLoader return `torch.sparse_csr` tensors whereas scDataset returns `scipy.sparse.csr_matrix`.

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
