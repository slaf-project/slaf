# Quick Start

Get up and running with SLAF in minutes! This guide will walk you through the basics of loading data and running your first queries.

## Installation

Install SLAF using pip:

```bash
# Using uv (recommended)
uv add slaf

# Or using pip
pip install slaf
```

For development dependencies (including documentation):

```bash
# Using uv (recommended)
uv pip install -e ".[docs,dev]"

# Or using pip
pip install slaf[docs,dev]
```

## Using the SLAF CLI

SLAF includes a command-line interface for common tasks:

```bash
# Show version and help
slaf version
slaf --help

# List available examples
slaf examples --list

# Export examples to HTML for documentation
slaf examples --export

# Show info about a dataset
slaf info path/to/your/data.slaf

# Execute a SQL query
slaf query path/to/your/data.slaf "SELECT COUNT(*) FROM cells"
```

## Your First SLAF Dataset

### Loading Data

```python
import slaf

# Load a SLAF dataset
slaf_array = slaf.SLAFArray("path/to/your/data.slaf")

# Check basic info
print(f"Dataset shape: {slaf_array.shape}")
print(f"Number of cells: {slaf_array.shape[0]:,}")
print(f"Number of genes: {slaf_array.shape[1]:,}")
```

### Exploring the Data

SLAF stores data in three main tables that you can query directly:

```python
# View cell metadata
print("Cell metadata:")
print(slaf_array.obs.head())

# View gene metadata
print("\nGene metadata:")
print(slaf_array.var.head())

# Check what columns are available
print(f"\nCell metadata columns: {list(slaf_array.obs.columns)}")
print(f"Gene metadata columns: {list(slaf_array.var.columns)}")
```

## Your First SQL Query

SLAF gives you direct SQL access to your data:

```python
# Count cells by cell type
cell_types = slaf_array.query("""
    SELECT cell_type, COUNT(*) as count
    FROM cells
    GROUP BY cell_type
    ORDER BY count DESC
""")
print(cell_types)
```

## Filtering Cells and Genes

Use convenient filtering methods:

```python
# Filter cells by cell type
t_cells = slaf_array.filter_cells(cell_type="T cells")
print(f"Found {len(t_cells)} T cells")

# Filter genes by expression
highly_variable = slaf_array.filter_genes(highly_variable=True)
print(f"Found {len(highly_variable)} highly variable genes")
```

## Getting Expression Data

```python
# Get expression for specific cells
cell_ids = ["cell_001", "cell_002", "cell_003"]
expression = slaf_array.get_cell_expression(cell_ids)
print(expression.head())

# Get expression for specific genes
gene_ids = ["GENE1", "GENE2", "GENE3"]
expression = slaf_array.get_gene_expression(gene_ids)
print(expression.head())
```

## Scanpy Integration

SLAF works seamlessly with Scanpy:

```python
from slaf.integrations import read_slaf

# Load as lazy AnnData
adata = read_slaf("path/to/your/data.slaf")

# Use familiar AnnData operations
print(f"AnnData shape: {adata.shape}")
print(f"Cell metadata: {adata.obs.columns.tolist()}")
print(f"Gene metadata: {adata.var.columns.tolist()}")

# Lazy slicing - no data loaded yet
subset = adata[adata.obs.cell_type == "T cells", :]

# Only load data when you need it
expression_matrix = subset.X.compute()
print(f"Loaded expression matrix: {expression_matrix.shape}")
```

## Advanced SQL Queries

SLAF supports complex SQL operations:

```python
# Find cells with high expression of specific genes
high_expr_cells = slaf_array.query("""
    SELECT DISTINCT c.cell_id, c.cell_type, c.total_counts
    FROM cells c
    JOIN expression e ON c.cell_id = e.cell_id
    JOIN genes g ON e.gene_id = g.gene_id
    WHERE g.gene_id IN ('GENE1', 'GENE2', 'GENE3')
    AND e.value > 10
    ORDER BY c.total_counts DESC
    LIMIT 10
""")
print(high_expr_cells)
```

## Performance Tips

- **Use integer IDs**: SLAF automatically maps string IDs to integers for faster queries
- **Leverage SQL**: Complex operations are often faster with SQL than Python loops
- **Lazy evaluation**: Use `.compute()` only when you need the actual data
- **Batch operations**: Process data in batches for large datasets

## Next Steps

- Read the [User Guide](../user-guide/core-concepts.md) for detailed concepts
- Explore [Examples](../examples/getting-started.md) for real-world use cases
- Check the [API Reference](../api/core.md) for complete documentation

## Getting Help

- 📖 **Documentation**: This site contains comprehensive guides
- 💬 **GitHub Issues**: Report bugs on [GitHub](https://github.com/pavanramkumar/slaf)
- 📧 **Email**: Contact pavan.ramkumar@gmail.com for questions
