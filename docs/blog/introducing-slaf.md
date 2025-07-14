# Introducing SLAF: The Single-Cell Data Format for the Virtual Cell Era

---

Single-cell datasets have grown from 50k to 100M cells in less than a decade. What used to fit comfortably in memory now requires out-of-core, distributed computing. What used to transfer in minutes now takes days. What used to be a simple analysis pipeline for a bioinformatician now requires a team of infrastructure engineers.

The single-cell data explosion has created a fundamental mismatch between our tools and our needs. We're trying to solve 2025 problems with 2015 technology. But it doesn't have to be that way.

This is the story of **SLAF** (Sparse Lazy Array Format) --- a cloud-native, SQL-powered format designed for the modern single-cell era.

---

## A 10-year lookback at storage and compute revolutions

Before diving into single-cell data, I thought it would be worth a historical detour through some key innovations over the past decade that directly inspired SLAF's architecture.

!!! tip "The Big Picture"

    We'll explore six key innovations that revolutionized data storage and compute:

    1. **Compressed, chunked storage** with optimized I/O
    2. **Cloud-native chunked storage** enabling concurrent I/O and computation
    3. **Lazy computation** for complex workflows (Dask, Polars)
    4. **Skip what you don't need** (Parquet's predicate/projection pushdown)
    5. **Zero-infrastructure embedded databases on disk** (Lance, DuckDB)
    6. **High-performance query engines** (Polars, DuckDB, Data Fusion)

### The Zarr Revolution: Compressed, Chunked Storage

!!! info "The Problem"

    In 2016, [Alistair Miles](https://alimanfoo.github.io/), a genomics scientist at the Wellcome Trust Sanger Institute, was struggling with genetic variation data containing around **20 billion elements** - too large for memory, too small for distributed computing. Traditional compute would OOM-kill his research before it began.

Miles discovered HDF5 with the h5py Python library, which provided a great solution for storing multi-dimensional arrays. The arrays were divided into chunks, each compressed, enabling efficient storage and fast access. But there was a problem: HDF5 was still too slow for interactive work.

He then discovered [Bcolz](https://bcolz.readthedocs.io/), which used the Blosc compression library. Blosc could use multiple threads internally, worked well with CPU cache architecture, and was much faster than HDF5. In his [benchmarks](https://alimanfoo.github.io/2016/04/14/to-hdf5-and-beyond.html), Bcolz was more than 10 times faster at storing data than HDF5.

But Bcolz had a limitation: it could only be chunked along the first dimension. Taking slices of other dimensions required reading and decompressing the entire array.

So Miles created [Zarr](https://zarr.readthedocs.io/), which like Bcolz used Blosc internally but supported chunking along multiple dimensions. This enabled better performance for multiple data access patterns.

!!! success "Key Innovation #1"

    **Compressed, chunked storage with optimized encoding/decoding for speed of I/O.**

From these early beginnings, Zarr has grown far beyond genomics. Today it powers massive array formats across climate science, geospatial data, medical imaging, and microscopy --- any domain dealing with large, multi-dimensional arrays that need efficient cloud-native access.

### The Zarr/Dask Collaboration: Remote Storage & Concurrency

!!! info "The Problem"

    [Matt Rocklin](https://matthewrocklin.com/) (then, at the Anaconda foundation) was tackling a similar challenge: how to work with arrays too big for memory in numpy for embarassingly parallel computations.

Miles created [Zarr](https://zarr.readthedocs.io/) for compressed, chunked storage. Rocklin built [Dask](https://dask.org/) for lazy, out-of-core computation ([early ideas](https://blog.dask.org/2014/12/27/Towards-OOC), [consolidation](https://blog.dask.org/2015/05/19/State-of-Dask)).

But there were two separate technical challenges to solve for true cloud-native array access:

**Challenge 1: Remote Storage Access**
Zarr could read chunks efficiently from local storage, but accessing remote storage (like S3) required a different approach. Traditional file I/O was designed for local filesystems, not HTTP-based object storage with different latency characteristics. Zarr solved this by implementing HTTP-based chunk access that could work directly with cloud object storage.

**Challenge 2: The Python GIL Bottleneck**
The fundamental problem was the Python Global Interpreter Lock (GIL). As [Alistair Miles documented](https://alimanfoo.github.io/2016/05/16/cpu-blues.html), h5py doesn't release the GIL during I/O operations, which means other threads cannot run while h5py is doing anything - even if those threads want to do something unrelated to HDF5 I/O. This serialized all parallel operations, limiting CPU utilization to just over 1 core (~130% CPU).

The breakthrough came when Zarr was designed to **release the GIL during compression and decompression**. This simple but crucial design decision meant other threads could carry on working in parallel. This GIL-aware design enabled true concurrent I/O and computation --- a fundamental requirement for cloud-native array access.

When they [collaborated](https://alimanfoo.github.io/2016/05/16/cpu-blues.html), combining Zarr's remote chunked storage with Dask's process-based distributed computation, they finally achieved distributed computation on remote arrays for truly concurrent cloud-native array access.

!!! success "Key Innovation #2"

    **Cloud-native chunked storage enabling concurrent I/O and computation.**

### The Lazy Computation Revolution

!!! info "The Problem"

    Traditional eager computation loads data into memory immediately and processes it step by step. This works fine for small datasets, but becomes impossible when your data is larger than RAM. You can't even start the computation because the first step fails with "Out of Memory".

As [Matt Rocklin described](https://blog.dask.org/2015/05/19/State-of-Dask), Dask started as "a parallel on-disk array" that solved a fundamental problem: **how do you compute on data that's too big for memory?**

The key insight was **lazy computation graphs**. Instead of immediately loading and processing data, Dask builds a recipe (a directed acyclic graph) of all the operations you want to perform. This graph is just functions, dicts, and tuples - lightweight and easy to understand. Only when you call `.compute()` does Dask actually execute the operations, and it can optimize the entire graph before execution.

For example, if you want to sum a trillion numbers, Dask breaks this into a million smaller sums that each fit in memory, then sums the sums. A previously impossible task becomes a million and one easy ones.

[Polars](https://polars.rs/) took this to the next level for dataframes, enabling complex query chains that only touch the data they need. The entire query is optimized before any data is loaded.

!!! success "Key Innovation #3"

    **Lazy computation graphs that optimize before execution.**

### The Parquet Revolution: Skip What You Don't Need

!!! info "The Problem"

    The Zarr/Dask combination was fantastic for embarrassingly parallel work across _entire_ datasets. But what if you only wanted to analyze a subset of data that met certain criteria?

Meanwhile, [Parquet](https://parquet.apache.org/) didn't solve the partial compute problem but its design taught us how we might. Parquet was born at Twitter and Cloudera during the early days of the transition from OLTP to OLAP databases, and from row-based to [columnar storage](https://en.wikipedia.org/wiki/Column-oriented_DBMS). The insight was simple: analytical queries often access only a few columns, so storing data column-by-column enables much better compression and faster queries.

Parquet stores data in compressed columnar chunks called "row groups" and stores metadata about these chunks. At query time, you only materialize the metadata in memory, then use it to decide:

- Which row groups to skip reading (predicate pushdown)
- Which columns to skip reading (projection pushdown)

!!! success "Key Innovation #4"

    **Don't just make I/O faster, skip most I/O you don't need to do.**

### The Lance Revolution: Zero-Infrastructure Embedded Databases

!!! info "The Problem"

    Parquet was revolutionary for analytical queries, but it had fundamental limitations that became apparent as data workloads evolved:

    - **Database features missing**: Schema evolution, partitioning, time travel, and ACID transactions weren't first-class citizens. [Apache Iceberg](https://iceberg.apache.org/) emerged to address these limitations, but locked into Parquet as a format.

    - **AI-era data types**: Parquet preceded the AI era, so vector embeddings, multimodal data, and vector search weren't native. Modern AI workloads need to store and query embeddings alongside structured data.

    - **Metadata overhead**: Parquet's metadata can become expensive and suboptimal for random access patterns, especially as datasets grow.

    - **Infrastructure complexity**: Traditional databases require servers, configuration, and ongoing maintenance --- a barrier for many use cases.

[Lance](https://lancedb.github.io/lance/) directly addressed these limitations by creating a table format that provides:

- **Database features**: Schema evolution, time travel, partitioning, and ACID transactions as first-class citizens (parity with Iceberg)
- **AI-native design**: Native support for vector embeddings, multimodal data, and vector search
- **Optimized metadata**: Efficient random access patterns with minimal metadata overhead
- **Zero infrastructure**: A file on disk is a full database, like the next generation SQLite

[DuckDB](https://duckdb.org/) takes a different but complementary approach: an embedded OLAP database written in C++ that provides a full database experience with zero infrastructure. While Lance focuses on the table format, DuckDB focuses on both its own storate format and a highly extensible query engine. Both eliminate the need for servers and configuration.

!!! success "Key Innovation #5"

    **Zero-infrastructure embedded databases on disk.**

### The OLAP Query Engine Revolution: Fast Languages + Lazy Execution

The final piece was query engines written in faster languages than Python. [Polars](https://polars.rs/) is written in Rust and executes chained query workloads lazily. [DuckDB](https://duckdb.org/) is an embedded OLAP database and query engine written in C++ that can push down complex queries to the storage layer.

By combining high-performance query engines with metadata-rich table formats like Parquet and Lance, you get superpowers: complex queries that only touch the data they need.

!!! success "Key Innovation #6"

    **High-performance query engines that can skip unnecessary computation.**

### The Single-Cell Gap

The single-cell community evolved from simple [MTX format](https://math.nist.gov/MatrixMarket/formats.html) (essentially TSV files) to [H5AD](https://anndata.readthedocs.io/en/stable/) (HDF5-based [AnnData](https://anndata.readthedocs.io/en/stable/)). Innovation happened on the metadata storage side with AnnData, and computational workflows with [Scanpy](https://scanpy.readthedocs.io/) leveraging scipy sparse matrices and numpy. But storage hasn't leveraged the amazing innovations in cloud-native, out-of-core, query-optimized, lazy, metadata-rich workflows that revolutionized other domains.

A format like SLAF can bring these proven innovations to single-cell data.

---

## The Evolution of Single-Cell Data Storage

Single-cell data storage has evolved from simple [MTX format](https://math.nist.gov/MatrixMarket/formats.html) (TSV files) to [H5AD](https://anndata.readthedocs.io/en/stable/) (HDF5-based [AnnData](https://anndata.readthedocs.io/en/stable/)) as the de facto standard. While H5AD brought compression and [Scanpy](https://scanpy.readthedocs.io/) integration, it has fundamental limitations: single-file format (no concurrent access), in-memory operations (everything must fit in RAM), and no cloud-native access.

Recent attempts like [SOMA](https://github.com/single-cell-data/SOMA) (TileDB/CZI collaboration) and [cellxgene-census](https://github.com/chanzuckerberg/cellxgene-census) recognize these limitations but have different focuses: SOMA is broad-scope sparse arrays, while cellxgene-census is tied to specific datasets. The gap remains: we need cloud-native storage that works seamlessly with existing bioinformatics tools while expanding to AI use cases.

---

## The Scale Problem: Why Current Tools Are Breaking

Let's paint a picture of what happens when you try to analyze a 100M-cell dataset with current tools.

<div class="grid" markdown cols="2">

<div markdown>

:fontawesome-solid-server: **Infrastructure Bottlenecks**

A typical 100M-cell dataset might be 500GB-1TB. With H5AD: download takes hours to days, must fit entire dataset in memory (impossible on most machines), all operations happen in memory.

**You can't analyze data larger than your RAM.**

</div>

<div markdown>

:fontawesome-solid-users: **Human Bottlenecks**

Bioinformatician wants to analyze large dataset → requests infrastructure changes → waits weeks → downloads massive dataset → runs into memory issues → requests more RAM → waits again.

**Every scale-up requires infrastructure engineers.**

</div>

<div markdown>

:fontawesome-solid-copy: **Data Duplication**

Team of 5 bioinformaticians × 500GB dataset × 5 experiments = 12.5TB for what should be 500GB. Each researcher needs their own copy, each experiment needs a copy.

**This data multiplication problem is unsustainable at scale.**

</div>

<div markdown>

:fontawesome-solid-brain: **AI Workload Mismatch**

Traditional workflows: filtering, normalization, PCA, clustering. New AI-native workflows: nearest neighbor search on embeddings, gene-gene relationship ranking, transformer training, distributed training.

**Current tools weren't designed for these workloads.**

</div>

</div>

---

## Who is SLAF For?

SLAF is designed for the modern single-cell ecosystem facing scale challenges:

| Persona                                  | Problem                                                                               | Current Reality                                                          | How SLAF Helps                                                                           |
| ---------------------------------------- | ------------------------------------------------------------------------------------- | ------------------------------------------------------------------------ | ---------------------------------------------------------------------------------------- |
| **Bioinformaticians**                    | OOM errors on 10M+ cell datasets, can't do self-service analysis                      | Need infrastructure engineers, wait weeks for provisioning               | Lazy evaluation eliminates OOM errors, enables 100M-cell analysis on laptop              |
| **Foundation Model Builders**            | Spending more time on data engineering than model training, data transfer bottlenecks | Copy 500GB datasets to each node, experiments bottlenecked by throughput | Cloud-native streaming eliminates duplication, SQL queries enable efficient tokenization |
| **Tech Leaders & Architects**            | Storage costs exploding, data multiplication problem                                  | 5 bioinformaticians × 500GB × 5 experiments = 12.5TB for 500GB data      | Zero-copy, query-in-place storage means one copy serves everyone                         |
| **Tool Builders** (cellxgene developers) | Can't provide interactive experiences on massive datasets                             | Users wait minutes, need expensive infrastructure                        | Concurrent, cloud-scale access with high QPS, commodity web services                     |
| **Atlas Builders** (CZI, etc.)           | Can't serve massive datasets to research community                                    | Datasets too large to download, expensive serving infrastructure         | Cloud-native, zero-copy storage enables global access without downloads                  |
| **Data Integrators** (Elucidata, etc.)   | Complex data integration workflows don't scale                                        | Multiple data transfers, expensive compute for joins/aggregations        | SQL-native design enables complex integration with pushdown optimization                 |

## How SLAF Works

SLAF combines the best ideas from multiple domains into a unified solution:

- **Zarr-inspired**: Zero-copy, query-in-place access to cloud storage
- **Dask-inspired**: Lazy computation graphs that optimize before execution
- **Lance + DuckDB-inspired**: OLAP-powered SQL with pushdown optimization
- **Scanpy-compatible**: Drop-in replacement for existing workflows

For a detailed technical deep dive into SLAF's architecture, including SQL-native relational schema, lazy evaluation, cloud-native concurrent access, and foundation model training support, see [How SLAF Works](../user-guide/how-slaf-works.md).

---

## Limitations and Future Work

SLAF is a significant step forward, but it's an early project with a bus factor of 1.

### Current Limitations

- **Visualization**: Limited support for interactive visualization (coming soon)
- **Embeddings**: No native support for cell/gene embeddings (in development)
- **Migration**: Better tools to migrate from mtx / h5ad to SLAF (in progress)
- **Feature Incomplete**: Doesn't have full parity with scanpy
- **Ecosystem**: Limited integration with other single-cell tools

### Future Work

- **Embeddings Support**: Native storage and querying of cell/gene embeddings
- **Visualization**: Integration with cellxgene and other visualization tools
- **Distributed Computing**: Better support for distributed analysis workflows
- **Schema Evolution**: Support for evolving data schemas over time

---

## Getting Started

SLAF is designed to be easy to adopt. Here's how to get started:

### Installation

```bash
# Using uv (recommended)
uv add slaf

# Or pip
pip install slaf
```

### Basic Usage

```python
from slaf import SLAFArray

# Load a SLAF dataset
slaf_array = SLAFArray("path/to/dataset.slaf")

# Query with SQL
results = slaf_array.query("""
    SELECT cell_type, COUNT(*) as count
    FROM cells
    GROUP BY cell_type
""")

# Use with Scanpy
from slaf.integrations import read_slaf
adata = read_slaf("path/to/dataset.slaf")
```

### Migration from H5AD

```python
# Convert existing H5AD to SLAF
from slaf.data import SLAFConverter

# Convert h5ad file to SLAF format
converter = SLAFConverter()
converter.convert("input.h5ad", "output.slaf")

# Or convert from AnnData object
import scanpy as sc
adata = sc.read_h5ad("input.h5ad")
converter.convert_anndata(adata, "output.slaf")
```

---

If you've read this far, thank you! You're definitely invested in the space.

The single-cell data explosion isn't going to slow down. If anything, it's accelerating. We need tools that can keep up. SLAF is an early attempt to build the modern companion for bioinformaticians turned AI engineers. It's designed to handle the 100M-cell era while maintaining the familiar numpy-like slicing and Scanpy idioms that researchers have built their workflows around.

I'm excited to see how you adopt and extend SLAF. Together, we can build the infrastructure needed for the next decade of single-cell research.

---

_Want to learn more? Check out the [SLAF documentation](https://slaf-project.github.io/slaf/) or join the conversation on [GitHub](https://github.com/slaf-project/slaf)._
