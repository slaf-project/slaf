# Quick Start

Get up and running with SLAF in minutes! Experience the joy of lightning-fast SQL queries on your single-cell data.

## Installation

Install SLAF using your preferred package manager:

```bash
# Using uv (recommended)
uv add slafdb

# Or pip
pip install slafdb

# Or conda
conda install -c conda-forge slafdb
```

For development dependencies (including documentation):

```bash
# Using uv (recommended)
uv pip install -e ".[docs,dev,test]"

# Or using pip
pip install slafdb[docs,dev,test]
```

## Your First SLAF Experience

Let's simulate a single-cell dataset similar to what you'd get from a real experiment.

### Option 1: Create a Synthetic Dataset (Recommended for First Time)

This creates a realistic single-cell dataset from scratch:

```python
import numpy as np
import pandas as pd
import scanpy as sc
from scipy.sparse import csr_matrix
from slaf.data import SLAFConverter

# Set random seed for reproducible results
np.random.seed(42)

# Create a realistic single-cell dataset
n_cells, n_genes = 1000, 2000

# Create sparse matrix with realistic sparsity
density = 0.1  # 10% sparsity - typical for single-cell data
n_nonzero = int(n_cells * n_genes * density)

# Generate realistic expression data (log-normal distribution)
data = np.random.lognormal(0, 1, n_nonzero).astype(np.float32)
row_indices = np.random.randint(0, n_cells, n_nonzero)
col_indices = np.random.randint(0, n_genes, n_nonzero)

# Create sparse matrix
X = csr_matrix((data, (row_indices, col_indices)), shape=(n_cells, n_genes))

# Create cell metadata
obs = pd.DataFrame({
    "cell_type": np.random.choice(["T_cell", "B_cell", "NK_cell", "Monocyte"], n_cells),
    "batch": np.random.choice(["batch1", "batch2", "batch3"], n_cells),
    "total_counts": X.sum(axis=1).A1,
    "n_genes_by_counts": (X > 0).sum(axis=1).A1,
    "high_mito": np.random.choice([True, False], n_cells, p=[0.1, 0.9]),
}, index=pd.Index([f"cell_{i:04d}" for i in range(n_cells)]))

# Create gene metadata
var = pd.DataFrame({
    "gene_symbol": [f"GENE_{i:06d}" for i in range(n_genes)],
    "highly_variable": np.random.choice([True, False], n_genes, p=[0.2, 0.8]),
    "total_counts": X.sum(axis=0).A1,
    "n_cells_by_counts": (X > 0).sum(axis=0).A1,
}, index=pd.Index([f"ENSG_{i:08d}" for i in range(n_genes)]))

# Create AnnData object
adata = sc.AnnData(X=X, obs=obs, var=var)

print(f"✅ Created dataset: {adata.n_obs:,} cells × {adata.n_vars:,} genes")
print(f"   Sparsity: {1 - adata.X.nnz / (adata.n_obs * adata.n_vars):.1%}")
```

### Option 2: Use a Real Dataset (PBMC3K)

For a more authentic experience, you can use the popular PBMC3K dataset:

```python
import scanpy as sc
from slaf.data import SLAFConverter

# Download PBMC3K dataset (this will take a moment)
print("Downloading PBMC3K dataset...")
adata = sc.datasets.pbmc3k()
adata.var_names_make_unique()

print(f"✅ Downloaded PBMC3K: {adata.n_obs:,} cells × {adata.n_vars:,} genes")
print(f"   Sparsity: {1 - adata.X.nnz / (adata.n_obs * adata.n_vars):.1%}")

# Add some basic metadata if not present
if "total_counts" not in adata.obs.columns:
    adata.obs["total_counts"] = adata.X.sum(axis=1).A1
if "n_genes_by_counts" not in adata.obs.columns:
    adata.obs["n_genes_by_counts"] = (adata.X > 0).sum(axis=1).A1
if "batch" not in adata.obs.columns:
    adata.obs["batch"] = "pbmc3k"
```

### Step 2: Convert to SLAF Format

```python
# Convert to SLAF format
converter = SLAFConverter()
slaf_path = "my_dataset.slaf"
converter.convert_anndata(adata, slaf_path)

print(f"✅ Converted to SLAF format: {slaf_path}")
```

### Step 3: Load and Explore Your Data

```python
import slaf

# Load your SLAF dataset
slaf_array = slaf.SLAFArray(slaf_path)

# Check basic info
print(f"Dataset shape: {slaf_array.shape}")
print(f"Number of cells: {slaf_array.shape[0]:,}")
print(f"Number of genes: {slaf_array.shape[1]:,}")

# View cell metadata
print("\nCell metadata columns:")
print(list(slaf_array.obs.columns))

# View gene metadata
print("\nGene metadata columns:")
print(list(slaf_array.var.columns))
```

## Your First SQL Queries

Now experience the power of SQL on your single-cell data!

### Basic Queries

```python
# Count cells by cell type
cell_types = slaf_array.query("""
    SELECT cell_type, COUNT(*) as count
    FROM cells
    GROUP BY cell_type
    ORDER BY count DESC
""")
print("Cell type distribution:")
print(cell_types)

# Find cells with high total counts
high_count_cells = slaf_array.query("""
    SELECT cell_id, cell_type, total_counts
    FROM cells
    WHERE total_counts > 1000
    ORDER BY total_counts DESC
    LIMIT 5
""")
print("\nCells with high total counts:")
print(high_count_cells)
```

### Advanced Queries with Joins

```python
# Find highly variable genes with their expression stats
variable_genes = slaf_array.query("""
    SELECT
        g.gene_id,
        g.gene_symbol,
        g.total_counts as gene_total_counts,
        COUNT(e.value) as cells_expressed,
        AVG(e.value) as avg_expression,
        MAX(e.value) as max_expression
    FROM genes g
    LEFT JOIN expression e ON g.gene_id = e.gene_id
    WHERE g.highly_variable = true
    GROUP BY g.gene_id, g.gene_symbol, g.total_counts
    ORDER BY g.total_counts DESC
    LIMIT 10
""")
print("Top highly variable genes:")
print(variable_genes)
```

### Complex Analysis

```python
# Analyze expression patterns by cell type
cell_type_analysis = slaf_array.query("""
    SELECT
        c.cell_type,
        COUNT(DISTINCT c.cell_id) as num_cells,
        AVG(c.total_counts) as avg_total_counts,
        AVG(c.n_genes_by_counts) as avg_genes_per_cell,
        COUNT(CASE WHEN c.high_mito = true THEN 1 END) as high_mito_cells
    FROM cells c
    GROUP BY c.cell_type
    ORDER BY num_cells DESC
""")
print("Cell type analysis:")
print(cell_type_analysis)
```

## Scanpy Integration

SLAF works seamlessly with Scanpy for familiar workflows:

```python
from slaf.integrations import read_slaf

# Load as lazy AnnData
adata = read_slaf(slaf_path)

print(f"✅ Loaded as LazyAnnData: {adata.shape[0]:,} cells × {adata.shape[1]:,} genes")
print(f"   Type: {type(adata)}")
print(f"   Expression matrix type: {type(adata.X)}")

# Use familiar AnnData operations
print(f"\nCell metadata: {adata.obs.columns.tolist()}")
print(f"Gene metadata: {adata.var.columns.tolist()}")

# Lazy slicing - no data loaded yet
t_cells = adata[adata.obs.cell_type == "T_cell", :]
print(f"\nT cells subset: {t_cells.shape}")

# Only load data when you need it
expression_matrix = t_cells.X.compute()
print(f"Loaded expression matrix: {expression_matrix.shape}")
```

## Using the SLAF CLI

SLAF includes a powerful command-line interface for common data operations:

### Getting Help

```bash
# Show general help
slaf --help

# Show help for specific command
slaf convert --help
slaf query --help
```

### Basic Commands

```bash
# Check SLAF version
slaf version

# Show info about your dataset
slaf info my_dataset.slaf

# Execute a SQL query
slaf query my_dataset.slaf "SELECT COUNT(*) FROM cells"

# Convert other formats to SLAF
slaf convert data.h5ad output.slaf
```

### Common Use Cases

**Converting Datasets:**

```bash
# Convert an AnnData file
slaf convert pbmc3k.h5ad pbmc3k.slaf

# Convert with verbose output to see details
slaf convert pbmc3k.h5ad pbmc3k.slaf --verbose
```

**Exploring Datasets:**

```bash
# Get basic info about a dataset
slaf info pbmc3k.slaf

# Run a simple query
slaf query pbmc3k.slaf "SELECT cell_type, COUNT(*) FROM cells GROUP BY cell_type"

# Export query results to CSV
slaf query pbmc3k.slaf "SELECT * FROM cells WHERE cell_type = 'T cells'" --output t_cells.csv
```

**Data Analysis Pipeline:**

```bash
# Convert input data
slaf convert input.h5ad output.slaf

# Verify conversion
slaf info output.slaf

# Run analysis queries
slaf query output.slaf "SELECT cell_type, AVG(total_counts) FROM cells GROUP BY cell_type"
```

## Next Steps

- Read the [User Guide](../user-guide/how-slaf-works.md) for detailed concepts
- Explore [Examples](../examples/getting-started.md) for real-world use cases
- Check the [API Reference](../api/core.md) for complete documentation
- Report bugs or make feature requests on [GitHub](https://github.com/slaf-project/slaf)
