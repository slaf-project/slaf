# SLAF Documentation

<div class="grid cards" markdown>

- :fontawesome-solid-rocket: **[Quick Start](getting-started/quickstart.md)**

  Get up and running with SLAF in minutes. Learn the basics of loading data and running your first queries.

- :fontawesome-solid-book: **[User Guide](user-guide/core-concepts.md)**

  Deep dive into SLAF concepts, data loading, SQL queries, and advanced filtering techniques.

- :fontawesome-solid-code: **[API Reference](api/core.md)**

  Complete API documentation for all SLAF modules and functions.

- :fontawesome-solid-flask: **[Examples](examples/getting-started.md)**

  Interactive examples and tutorials showing real-world usage patterns.

</div>

## What is SLAF?

**SLAF** (Sparse Lazy Array Format) is a high-performance format for single-cell data that combines the power of SQL with lazy evaluation. Built on top of [Lance](https://lancedb.github.io/lance/) and [DuckDB](https://duckdb.org/), SLAF provides:

- 🚀 **Lightning Fast Queries**: SQL-level performance for data operations
- 💾 **Memory Efficient**: Lazy evaluation, only load what you need
- 🔍 **SQL Native**: Direct SQL queries on your single-cell data
- 🧬 **Scanpy Compatible**: Drop-in replacement for AnnData workflows
- ⚡ **Production Ready**: Built for large-scale single-cell analysis
- ⚙️ **ML Ready**: Ready for ML training with efficient tokenization

## Key Features

### High-Performance SQL Queries

```python
import slaf

# Load your data
slaf_array = slaf.SLAFArray("path/to/data.slaf")

# Run SQL queries directly
results = slaf_array.query("""
    SELECT cell_type, AVG(total_counts) as avg_counts
    FROM cells
    WHERE batch = 'batch1'
    GROUP BY cell_type
    ORDER BY avg_counts DESC
""")
```

### Lazy Evaluation with Scanpy Integration

```python
from slaf.integrations import read_slaf

# Load as lazy AnnData
adata = read_slaf("path/to/data.slaf")

# Operations are lazy until you call .compute()
subset = adata[adata.obs.cell_type == "T cells", :]
expression = subset.X.compute()  # Only now is data loaded
```

### Efficient ML Training

```python
from slaf.ml import SLAFTokenizer

# Tokenize your data for ML
tokenizer = SLAFTokenizer.from_slaf(slaf_array)
tokens = tokenizer.encode_batch(cell_ids)
```

## Quick Installation

```bash
pip install slaf
```

For development dependencies:

```bash
pip install slaf[docs,dev]
```

## Architecture

SLAF stores single-cell data in three optimized tables:

- **`cells`**: Cell metadata and QC metrics
- **`genes`**: Gene metadata and annotations
- **`expression`**: Sparse expression matrix data

All tables are stored in the [Lance format](https://lancedb.github.io/lance/) for maximum performance and can be queried directly with SQL via DuckDB.

## Performance Benchmarks

SLAF provides significant performance improvements over traditional formats:

- **100x faster** random access than Parquet
- **10x faster** filtering operations than HDF5
- **Memory efficient** lazy evaluation for large datasets
- **SQL-native** queries for complex operations

See our [benchmarks](benchmarks/performance.md) for detailed performance comparisons.

## Getting Help

- 📖 **Documentation**: This site contains comprehensive guides and API docs
- 💬 **GitHub Issues**: Report bugs or request features on [GitHub](https://github.com/slaf-project/slaf)
- 📧 **Email**: Contact pavan.ramkumar@gmail.com for questions

## Contributing

We welcome contributions! See our [contributing guide](development/contributing.md) for details on how to get started.

---

<div class="grid" markdown>

- :fontawesome-solid-heart:{ .heart } **Made with love for the single-cell community**

- :fontawesome-solid-code:{ .code } **[View on GitHub](https://github.com/slaf-project/slaf){target=\_blank}**

</div>
