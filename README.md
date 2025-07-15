# SLAF (Sparse Lazy Array Format)

[![Python](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![Tests](https://img.shields.io/github/actions/workflow/status/slaf-project/slaf/ci.yml?branch=main&label=tests)](https://github.com/slaf-project/slaf/actions)
[![Coverage](https://img.shields.io/endpoint?url=https://gist.githubusercontent.com/pavanramkumar/33e8b97f85afdc956a71edc623f5c2ba/raw/slaf-coverage.json)](https://github.com/slaf-project/slaf/actions)
[![Code style](https://img.shields.io/badge/code%20style-ruff-black.svg)](https://github.com/astral-sh/ruff)

<!-- Uncomment when published to PyPI: [![PyPI](https://img.shields.io/badge/PyPI-0.2.0-blue.svg)](https://pypi.org/project/slaf/) -->

**SLAF** is a high-performance format for single-cell data that combines the power of SQL with lazy evaluation. Built for large-scale single-cell analysis with memory efficiency and production-ready ML capabilities.

**Be Lazy** (lazy APIs for AnnData and Scanpy) • **Write SQL** (arbitrary SQL to query the tables) • **Train Foundation Models** (with tokenizers and dataloaders)

## 🚀 Key Features

- **⚡ Fast**: SQL-level performance for data operations
- **💾 Memory Efficient**: Lazy evaluation, only load what you need
- **🔍 SQL Native**: Direct SQL queries on your data
- **🧬 Scanpy Compatible**: Drop-in replacement for AnnData workflows
- **⚙️ ML Ready**: Ready for ML training with efficient tokenization
- **🔧 Production Ready**: Built for large-scale single-cell analysis

## 📦 Installation

```bash
# Using uv (recommended)
uv add slafdb

# Or pip
pip install slafdb

# Or conda
conda install -c conda-forge slafdb

# Development installation
git clone https://github.com/slaf-project/slaf.git
cd slaf
uv sync --dev
```

## 🚀 Quick Start

### Basic Usage

```python
from slaf import SLAFArray

# Load a SLAF dataset
slaf = SLAFArray("path/to/dataset.slaf")

# Describe the dataset
print(slaf.info())

# Execute SQL queries directly
results = slaf.query("""
    SELECT batch, COUNT(*) as count
    FROM cells
    GROUP BY batch
    ORDER BY count DESC
""")
print(results)
```

### Filtering Data

```python
# Filter cells by metadata
filtered_cells = slaf.filter_cells(
    batch="batch1",
    total_counts=">1000"
)

# Filter genes
filtered_genes = slaf.filter_genes(
    highly_variable=True
)

# Get expression submatrix
expression = slaf.get_submatrix(
    cell_selector=filtered_cells,
    gene_selector=filtered_genes
)
```

## 🦥 Be Lazy - Lazy AnnData & Scanpy Integration

SLAF provides lazy versions of AnnData and Scanpy operations that only compute when needed:

```python
from slaf.integrations.anndata import read_slaf
import scanpy as sc

# Load as lazy AnnData
adata = read_slaf("path/to/dataset.slaf")
print(f"Type: {type(adata)}")  # LazyAnnData
print(f"Expression matrix type: {type(adata.X)}")  # LazyExpressionMatrix

# Apply scanpy operations (lazy)
sc.pp.normalize_total(adata)
sc.pp.log1p(adata)
sc.pp.highly_variable_genes(adata)

# Still lazy - no computation yet
print(f"Still lazy: {type(adata.X)}")

# Compute when needed
adata.compute()  # Now it's a real AnnData object
```

### Lazy Computation Control

```python
# Compute specific parts
expression_matrix = adata.X.compute()  # Just the expression matrix
cell_metadata = adata.obs              # Cell metadata
gene_metadata = adata.var              # Gene metadata

# Or compute everything at once
real_adata = adata.compute()
```

### Lazy Slicing

```python
# All slicing operations are lazy
subset = adata[:100, :50]  # Lazy slice
filtered = adata[adata.obs['n_genes_by_counts'] > 1000]  # Lazy filtering
```

## 🔍 Write SQL - Direct Database Access

SLAF stores data in three main tables that you can query directly with SQL:

### Database Schema

- **`cells`**: Cell metadata and QC metrics
- **`genes`**: Gene metadata and annotations
- **`expression`**: Sparse expression matrix data

### SQL Queries

```python
# Get expression data for specific cells
cell_expression = slaf.query("""
    SELECT
        c.cell_id,
        c.total_counts,
        COUNT(e.gene_id) as genes_expressed,
        AVG(e.value) as avg_expression
    FROM cells c
    JOIN expression e ON c.cell_integer_id = e.cell_integer_id
    WHERE c.batch = 'batch1'
    GROUP BY c.cell_id, c.total_counts
    ORDER BY genes_expressed DESC
    LIMIT 10
""")

# Find highly expressed genes
high_expr_genes = slaf.query("""
    SELECT
        g.gene_id,
        COUNT(e.cell_id) as cells_expressing,
        AVG(e.value) as avg_expression
    FROM genes g
    JOIN expression e ON g.gene_integer_id = e.gene_integer_id
    GROUP BY g.gene_id
    HAVING cells_expressing > 100
    ORDER BY avg_expression DESC
    LIMIT 10
""")
```

## 🧠 Train Foundation Models - ML Training

SLAF provides efficient tokenization and dataloaders for training foundation models:

### Tokenization

```python
from slaf.ml import SLAFTokenizer

# Create tokenizer
tokenizer = SLAFTokenizer(
    slaf_array=slaf,
    vocab_size=50000,
    n_expression_bins=10,
    chunk_size=2048
)

# Geneformer tokenization (gene sequence)
geneformer_tokens = tokenizer.tokenize_geneformer(
    cell_integer_id_range=(0, 100),
    max_genes=2048,
    min_percentile=10
)

# scGPT tokenization (gene-expression pairs)
scgpt_tokens = tokenizer.tokenize_scgpt(
    cell_integer_id_range=(0, 100),
    max_genes=1024,
    use_sql_binning=True
)
```

### DataLoader for Training

```python
from slaf.ml import SLAFDataLoader

# Create DataLoader
dataloader = SLAFDataLoader(
    slaf_array=slaf,
    tokenizer_type="geneformer",  # or "scgpt"
    batch_size=32,
    max_genes=2048,
    num_workers=4
)

# Use with PyTorch training
for batch in dataloader:
    input_ids = batch["input_ids"]
    attention_mask = batch["attention_mask"]
    cell_ids = batch["cell_ids"]

    # Your training loop here
    loss = model(input_ids, attention_mask=attention_mask)
    loss.backward()
```

## 🛠️ Command Line Interface

### Data Conversion

```bash
# Convert AnnData to SLAF
slaf convert input.h5ad output.slaf

# Convert HDF5 to SLAF
slaf convert input.h5 output.slaf --format hdf5
```

### Data Querying

```bash
# Execute SQL query
slaf query dataset.slaf "SELECT * FROM cells LIMIT 10"

# Save results to CSV
slaf query dataset.slaf "SELECT * FROM cells" --output cells.csv
```

### Dataset Information

```bash
slaf info dataset.slaf
```

## 📚 Documentation

- [SLAF Documentation](https://slaf-project.github.io/slaf/)
- [Quickstart](https://slaf-project.github.io/slaf/getting-started/quickstart/)
- [API Reference](https://slaf-project.github.io/slaf/api/)
- [Examples](https://slaf-project.github.io/slaf/examples/getting-started/)
- [User Guide](https://slaf-project.github.io/slaf/user-guide/how-slaf-works/)
- [Developers Guide](https://slaf-project.github.io/slaf/development/contributing/)
- [Maintainers Guide](https://slaf-project.github.io/slaf/development/maintaining/)

## 🙏 Acknowledgments

Built on top of

- [DuckDB](https://duckdb.org/) for fast SQL queries
- [Lance](https://lancedb.github.io/lance/) for cloud-native, efficient columnar storage
- [Scanpy](https://scanpy.readthedocs.io/) and [Anndata](https://github.com/scverse/anndata) ecosystem
