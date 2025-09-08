# SLAF Blog

Welcome to the SLAF blog, where we share insights, technical deep dives, and updates about the Sparse Lazy Array Format for single-cell genomics.

## Latest Posts

### [Blazing Fast Dataloaders #2: Ignatius takes a trip to the Library of Congress](blazing-fast-dataloaders-2.md)

_Last updated: September 8, 2025_

Remember Ignatius J Reilly from _A Confederacy of Dunces_? Voracious, impatient, impressionable, perambulatorily challenged? That's modern neural network pretraining on GPUs. In this post, we explore how SLAF's mixture of scanners approach achieves near-perfect randomization (88-90% of theoretical maximum) while maintaining 97% of sequential throughput performance. We dive deep into the "Library of Congress" metaphor to explain how our contraption delivers randomized books at high throughput without reorganizing the library.

### [6.4x Faster DataLoaders: Deconstructing PyTorch for Single-Cell Genomics](blazing-fast-dataloaders.md)

_Last updated: August 22, 2025_

Single-cell transcriptomics datasets have reached escape velocity, with modern experiments yielding counts for upwards of 5M cells × 20k genes. This technical deep dive explores how we achieved 6.4x performance improvement over standard PyTorch DataLoaders, reaching 28,207 cells/second through five key innovations: contiguous reads, single-threaded prefetching, vectorized window functions, block shuffling, and vectorized tokenization.

### [Introducing SLAF: The Single-Cell Data Format for the Virtual Cell Era](introducing-slaf.md)

_Last updated: August 14, 2025_

Single-cell datasets have grown from 50k to 100M cells in less than a decade, creating a fundamental mismatch between our tools and our needs. This introduction to SLAF (Sparse Lazy Array Format) explores how we're solving 2025 problems with modern technology, combining the best ideas from Zarr, Dask, Lance, and Polars into a cloud-native, SQL-powered format designed for the modern single-cell era.

---

## About SLAF

SLAF (Sparse Lazy Array Format) is a cloud-native single-cell storage format for the virtual cell era:

- **Zarr-inspired**: Zero-copy, query-in-place access to cloud storage
- **Dask-inspired**: Lazy computation graphs that optimize before execution
- **Lance + Polars-inspired**: OLAP-powered SQL with pushdown optimization
- **Scanpy-compatible**: Drop-in replacement for existing workflows

## Get Started

Ready to try SLAF? Check out our [quickstart guide](../getting-started/quickstart/) or explore the [API documentation](../api/). Find it on [Github](https://github.com/slaf-project/slaf). Deep dive into [benchmarks](../benchmarks/).

---

_Have questions or want to contribute? We'd love to hear from you!_
