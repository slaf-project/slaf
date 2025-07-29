# SQL Queries

SLAF provides native SQL querying capabilities for single-cell data, allowing you to perform complex analyses using familiar SQL syntax.

## Database Schema

SLAF stores data in three main tables:

- **`cells`**: Cell metadata with `cell_id` (string) and `cell_integer_id` (integer)
- **`genes`**: Gene metadata with `gene_id` (string) and `gene_integer_id` (integer)
- **`expression`**: Sparse expression data with `cell_integer_id`, `gene_integer_id`, and `value`

The expression table uses integer IDs for efficiency, so you need to JOIN with metadata tables to get string identifiers.

## Basic SQL Queries

### Simple SELECT Queries

```python
import slaf

# Load your data
slaf_array = slaf.SLAFArray("data.slaf")

# Basic cell query
results = slaf_array.query("""
    SELECT cell_id, cell_type, total_counts
    FROM cells
    WHERE batch = 'batch1'
    LIMIT 10
""")
```

### Gene Expression Queries

```python
# Query gene expression data
results = slaf_array.query("""
    SELECT
        c.cell_id,
        c.cell_type,
        g.gene_id,
        e.value
    FROM cells c
    JOIN expression e ON c.cell_integer_id = e.cell_integer_id
    JOIN genes g ON e.gene_integer_id = g.gene_integer_id
    WHERE g.gene_id IN ('CD3D', 'CD3E', 'CD3G')
    AND c.cell_type = 'T cells'
""")
```

## Aggregation and Grouping

### Cell Type Analysis

```python
# Analyze by cell type
results = slaf_array.query("""
    SELECT
        cell_type,
        COUNT(*) as cell_count,
        AVG(total_counts) as avg_counts,
        STDDEV(total_counts) as std_counts
    FROM cells
    GROUP BY cell_type
    ORDER BY cell_count DESC
""")
```

### Gene Expression Statistics

```python
# Gene expression statistics
results = slaf_array.query("""
    SELECT
        g.gene_id,
        COUNT(e.value) as expressed_cells,
        AVG(e.value) as mean_expression,
        MAX(e.value) as max_expression
    FROM genes g
    JOIN expression e ON g.gene_integer_id = e.gene_integer_id
    GROUP BY g.gene_id
    HAVING expressed_cells > 100
    ORDER BY mean_expression DESC
""")
```

## Advanced Queries

### Subqueries

```python
# Find cells with high expression of specific genes
results = slaf_array.query("""
    SELECT DISTINCT c.cell_id, c.cell_type
    FROM cells c
    WHERE c.cell_integer_id IN (
        SELECT e.cell_integer_id
        FROM expression e
        JOIN genes g ON e.gene_integer_id = g.gene_integer_id
        WHERE g.gene_id = 'CD3D'
        AND e.value > 10
    )
""")
```

### Window Functions

```python
# Rank genes by expression within each cell type
results = slaf_array.query("""
    SELECT
        c.cell_type,
        g.gene_id,
        e.value,
        RANK() OVER (
            PARTITION BY c.cell_type
            ORDER BY e.value DESC
        ) as rank_in_cell_type
    FROM cells c
    JOIN expression e ON c.cell_integer_id = e.cell_integer_id
    JOIN genes g ON e.gene_integer_id = g.gene_integer_id
""")
```

### CTEs (Common Table Expressions)

```python
# Using CTEs for complex analysis
results = slaf_array.query("""
    WITH cell_stats AS (
        SELECT
            cell_type,
            AVG(total_counts) as avg_counts
        FROM cells
        GROUP BY cell_type
    ),
    high_expression_genes AS (
        SELECT DISTINCT g.gene_integer_id, g.gene_id
        FROM genes g
        JOIN expression e ON g.gene_integer_id = e.gene_integer_id
        WHERE e.value > 5
    )
    SELECT
        c.cell_type,
        g.gene_id,
        e.value,
        cs.avg_counts
    FROM cells c
    JOIN expression e ON c.cell_integer_id = e.cell_integer_id
    JOIN high_expression_genes g ON e.gene_integer_id = g.gene_integer_id
    JOIN cell_stats cs ON c.cell_type = cs.cell_type
    WHERE e.value > cs.avg_counts
""")
```

## Performance Tips

### Use Integer IDs for Joins

For better performance, use integer IDs in JOIN conditions:

```python
# Efficient: Use integer IDs for joins
results = slaf_array.query("""
    SELECT c.cell_id, g.gene_id, e.value
    FROM expression e
    JOIN cells c ON e.cell_integer_id = c.cell_integer_id
    JOIN genes g ON e.gene_integer_id = g.gene_integer_id
    WHERE e.value > 0
""")

# Less efficient: Using string IDs in WHERE clauses
results = slaf_array.query("""
    SELECT c.cell_id, g.gene_id, e.value
    FROM expression e
    JOIN cells c ON e.cell_integer_id = c.cell_integer_id
    JOIN genes g ON e.gene_integer_id = g.gene_integer_id
    WHERE c.cell_id = 'cell_001'  -- Requires string comparison
""")
```

### Optimize for Large Datasets

```python
# For large datasets, filter early and use integer IDs
results = slaf_array.query("""
    SELECT
        c.cell_id,
        g.gene_id,
        e.value
    FROM expression e
    JOIN cells c ON e.cell_integer_id = c.cell_integer_id
    JOIN genes g ON e.gene_integer_id = g.gene_integer_id
    WHERE e.cell_integer_id BETWEEN 0 AND 1000  -- Early filtering
    AND e.value > 0
    ORDER BY e.cell_integer_id, e.gene_integer_id
""")
```

## Next Steps

- Explore the [User Guide](../user-guide/how-slaf-works.md) for detailed concepts
- Check the [API Reference](../api/core.md) for complete documentation
