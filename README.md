# SLAF (Sparse Lazy Array Format)

<div align="center">
  <img src="docs/assets/slaf-logo-light.svg" alt="SLAF Logo" width="400"/>
</div>

[![Python](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![Tests](https://img.shields.io/github/actions/workflow/status/slaf-project/slaf/ci.yml?branch=main&label=tests)](https://github.com/slaf-project/slaf/actions)
[![Coverage](https://img.shields.io/endpoint?url=https://gist.githubusercontent.com/pavanramkumar/33e8b97f85afdc956a71edc623f5c2ba/raw/slaf-coverage.json)](https://github.com/slaf-project/slaf/actions)
[![Code style](https://img.shields.io/badge/code%20style-ruff-black.svg)](https://github.com/astral-sh/ruff)
[![PyPI](https://img.shields.io/badge/PyPI-0.2.0-blue.svg)](https://pypi.org/project/slafdb/)
[![PyPI Downloads](https://static.pepy.tech/badge/slafdb)](https://pepy.tech/projects/slafdb)

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

### Default Installation (Batteries Included)

The default installation includes core functionality, CLI tools, and data conversion capabilities:

```bash
# Using uv (recommended)
uv add slafdb

# Or pip
pip install slafdb
```

**What's included by default:**

- ✅ Core SLAF functionality (SQL queries, data structures)
- ✅ CLI tools (`slaf convert`, `slaf query`, etc.)
- ✅ Data conversion tools (scanpy, h5py for h5ad files)
- ✅ Rich console output and progress bars
- ✅ Cross-platform compatibility

**What's NOT included by default:**

Dependencies for:

- ❌ Machine learning features (PyTorch tokenizers)
- ❌ Advanced single-cell tools (igraph, leidenalg)

### Platform-Specific Notes

**Polars Compatibility:**

- **Linux/Windows**: Works with standard `polars`
- **macOS (Apple Silicon)**: May require `polars-lts-cpu` for compatibility

If you encounter polars-related issues on macOS, you have several options:

**Option 1: Manual platform-specific installation**

```bash
# For macOS Apple Silicon
pip install "polars-lts-cpu>=1.31.0"
pip install slafdb

# For Linux/Windows
pip install slafdb
```

**Option 2: Use uv with manual polars specification**

```bash
# For macOS Apple Silicon
uv add "polars-lts-cpu>=1.31.0"
uv add slafdb

# For Linux/Windows
uv add slafdb
```

**Note**: Package managers don't automatically choose between `polars` and `polars-lts-cpu` - you may need to specify the correct version for your platform.

### Optional Dependencies

Add specific features as needed:

**Using uv:**

```bash
uv add "slafdb[ml]"
uv add "slafdb[advanced]"
uv add "slafdb[full]"
uv add "slafdb[dev]"
```

**Using pip:**

```bash
pip install slafdb[ml]
pip install slafdb[advanced]
pip install slafdb[full]
pip install slafdb[dev]
```

### Development Installation

```bash
git clone https://github.com/slaf-project/slaf.git
cd slaf
uv add --extra dev --extra test --extra docs
```

## 🚀 Quick Start

### Converting Your Data

Convert your existing single-cell data to SLAF format - **no extra dependencies required!**

```bash
# Convert AnnData (.h5ad) to SLAF
slaf convert input.h5ad output.slaf

# Convert HDF5 to SLAF
slaf convert input.h5 output.slaf

# Convert 10x Genomics data
slaf convert path/to/10x/filtered_feature_bc_matrix output.slaf
```

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

# Create tokenizer for GeneFormer style tokenization
tokenizer = SLAFTokenizer(
    slaf_array=slaf,
    tokenizer_type="geneformer",
    vocab_size=50000,
    n_expression_bins=10
)

# Geneformer tokenization (gene sequence only)
gene_sequences = [[1, 2, 3], [4, 5, 6]]  # Example gene IDs
input_ids, attention_mask = tokenizer.tokenize(
    gene_sequences,
    max_genes=2048
)

# Create tokenizer for scGPT style tokenization
tokenizer = SLAFTokenizer(
    slaf_array=slaf,
    tokenizer_type="scgpt",
    vocab_size=50000,
    n_expression_bins=10
)

# scGPT tokenization (gene-expression pairs)
gene_sequences = [[1, 2, 3], [4, 5, 6]]  # Gene IDs
expr_sequences = [[0.5, 0.8, 0.2], [0.9, 0.1, 0.7]]  # Expression values
input_ids, attention_mask = tokenizer.tokenize(
    gene_sequences,
    expr_sequences=expr_sequences,
    max_genes=1024
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
    max_genes=2048
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
# Convert AnnData to SLAF (included by default)
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

- [Lance](https://lancedb.github.io/lance/) for cloud-native, efficient columnar storage
- [DuckDB](https://duckdb.org/) for lazy, composable OLAP queries pushed down to Lance
- [Polars](https://pola.rs/) for lazy, composable, in-memory, zero-copy data processing
