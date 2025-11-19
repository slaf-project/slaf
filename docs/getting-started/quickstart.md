# Quick Start

Get up and running with SLAF in minutes! Experience the joy of lightning-fast SQL queries on your single-cell data.

## Installation

Install SLAF using your preferred package manager:

```bash
# Using uv (recommended)
uv add slafdb

# Or pip
pip install slafdb
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
import polars as pl
import scanpy as sc
from scipy.sparse import csr_matrix
from slaf.data.converter import SLAFConverter

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
obs = pl.DataFrame({
    "cell_id": [f"cell_{i:04d}" for i in range(n_cells)],
    "cell_type": np.random.choice(["T_cell", "B_cell", "NK_cell", "Monocyte"], n_cells),
    "batch": np.random.choice(["batch1", "batch2", "batch3"], n_cells),
    "total_counts": X.sum(axis=1).A1,
    "n_genes_by_counts": (X > 0).sum(axis=1).A1,
    "high_mito": np.random.choice([True, False], n_cells, p=[0.1, 0.9]),
})

# Create gene metadata
var = pl.DataFrame({
    "gene_id": [f"ENSG_{i:08d}" for i in range(n_genes)],
    "gene_symbol": [f"GENE_{i:06d}" for i in range(n_genes)],
    "highly_variable": np.random.choice([True, False], n_genes, p=[0.2, 0.8]),
    "total_counts": X.sum(axis=0).A1,
    "n_cells_by_counts": (X > 0).sum(axis=0).A1,
})

# Create AnnData object
adata = sc.AnnData(X=X, obs=obs, var=var)

print(f"✅ Created dataset: {adata.n_obs:,} cells × {adata.n_vars:,} genes")
print(f"   Sparsity: {1 - adata.X.nnz / (adata.n_obs * adata.n_vars):.1%}")
```

### Option 2: Use a Real Dataset (PBMC3K)

For a more authentic experience, you can use the popular PBMC3K dataset:

```python
import scanpy as sc
from slaf.data.converter import SLAFConverter

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
    LEFT JOIN expression e ON g.gene_integer_id = e.gene_integer_id
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

## Lazy Evaluation with Anndata Integration

SLAF works seamlessly with Scanpy for familiar
workflows with lazy evaluation:

```python
from slaf.integrations.anndata import read_slaf

# Load as lazy AnnData
adata = read_slaf(slaf_path)
print(f"✅ Loaded: {adata.shape[0]:,} cells × "
      f"{adata.shape[1]:,} genes")

# Lazy slicing - no data loaded yet
t_cells = adata[adata.obs.cell_type == "T_cell", :]

# Only load data when you need it
expression_matrix = t_cells.X.compute()
print(f"Computed expression: {expression_matrix.shape}")
```

## Lazy Scanpy Preprocessing

SLAF provides lazy versions of scanpy preprocessing functions
that work efficiently on large datasets:

```python
from slaf.integrations import scanpy as slaf_scanpy
from slaf.integrations.anndata import read_slaf

# Load as lazy AnnData
adata = read_slaf(slaf_path)

# Calculate QC metrics (lazy SQL aggregation)
cell_qc, gene_qc = slaf_scanpy.pp.calculate_qc_metrics(
    adata, inplace=False
)
print("Cell QC metrics:")
print(cell_qc.head())

# Filter cells (lazy - no data loaded)
adata_filtered = slaf_scanpy.pp.filter_cells(
    adata,
    min_genes=200,
    min_counts=1000,
    inplace=False
)
print(f"After filtering: {adata_filtered.shape}")

# Filter genes (lazy)
adata_filtered = slaf_scanpy.pp.filter_genes(
    adata_filtered,
    min_cells=30,
    inplace=False
)
print(f"After gene filtering: {adata_filtered.shape}")

# Normalize total (lazy transformation)
adata_norm = slaf_scanpy.pp.normalize_total(
    adata_filtered,
    target_sum=1e4,
    inplace=False
)

# Apply log1p transformation (lazy)
adata_log = slaf_scanpy.pp.log1p(
    adata_norm, inplace=False
)

# Find highly variable genes (lazy SQL)
hvg_stats = slaf_scanpy.pp.highly_variable_genes(
    adata_log,
    min_mean=0.0125,
    max_mean=3,
    min_disp=0.5,
    inplace=False
)
print(f"Highly variable genes: {hvg_stats['highly_variable'].sum()}")

# All operations are lazy - compute when needed
expression = adata_log.X.compute()
print(f"Computed expression: {expression.shape}")
```

### Chaining Transformations

You can chain lazy transformations efficiently:

```python
# Build complete preprocessing pipeline
adata_processed = slaf_scanpy.pp.normalize_total(
    adata, target_sum=1e4, inplace=False
)
adata_processed = slaf_scanpy.pp.log1p(
    adata_processed, inplace=False
)

# Slice after transformations
subset = adata_processed[:1000, :500]

# Compute only when needed
result = subset.X.compute()
print(f"Processed subset: {result.shape}")
```

### Key Benefits

- **Memory Efficient**: Operations stored as instructions
- **SQL Performance**: QC metrics use SQL aggregation
- **Composable**: Chain transformations easily
- **Lazy**: Compute only when you call `.compute()`

## Efficient Filtering

SLAF provides optimized filtering methods:

```python
# Filter cells by metadata
t_cells = slaf_array.filter_cells(cell_type="T_cell", total_counts=">1000")
print(f"Found {len(t_cells)} T cells with high counts")

# Filter genes
variable_genes = slaf_array.filter_genes(highly_variable=True)
print(f"Found {len(variable_genes)} highly variable genes")

# Get expression submatrix
expression = slaf_array.get_submatrix(
    cell_selector=t_cells,
    gene_selector=variable_genes
)
print(f"Expression submatrix: {expression.shape}")
```

## ML Training with Dataloaders

SLAF provides efficient tokenization and dataloaders for training foundation models:

### Tokenization

```python
from slaf.ml import SLAFTokenizer

# Create tokenizer for GeneFormer style tokenization
tokenizer = SLAFTokenizer(
    slaf_array=slaf_array,
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
    slaf_array=slaf_array,
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

# Create DataLoader (uses MoS by default for high entropy)
dataloader = SLAFDataLoader(
    slaf_array=slaf_array,
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
