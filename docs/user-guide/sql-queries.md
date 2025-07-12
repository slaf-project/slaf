# SQL Queries

SLAF provides native SQL querying capabilities for single-cell data, allowing you to perform complex analyses using familiar SQL syntax.

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
        g.gene_name,
        e.expression
    FROM cells c
    JOIN expression e ON c.cell_id = e.cell_id
    JOIN genes g ON e.gene_id = g.gene_id
    WHERE g.gene_name IN ('CD3D', 'CD3E', 'CD3G')
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
        g.gene_name,
        COUNT(e.expression) as expressed_cells,
        AVG(e.expression) as mean_expression,
        MAX(e.expression) as max_expression
    FROM genes g
    JOIN expression e ON g.gene_id = e.gene_id
    GROUP BY g.gene_name
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
    WHERE c.cell_id IN (
        SELECT e.cell_id
        FROM expression e
        JOIN genes g ON e.gene_id = g.gene_id
        WHERE g.gene_name = 'CD3D'
        AND e.expression > 10
    )
""")
```

### Window Functions

```python
# Rank genes by expression within each cell type
results = slaf_array.query("""
    SELECT
        c.cell_type,
        g.gene_name,
        e.expression,
        RANK() OVER (
            PARTITION BY c.cell_type
            ORDER BY e.expression DESC
        ) as rank_in_cell_type
    FROM cells c
    JOIN expression e ON c.cell_id = e.cell_id
    JOIN genes g ON e.gene_id = g.gene_id
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
        SELECT DISTINCT g.gene_id, g.gene_name
        FROM genes g
        JOIN expression e ON g.gene_id = e.gene_id
        WHERE e.expression > 5
    )
    SELECT
        c.cell_type,
        g.gene_name,
        e.expression,
        cs.avg_counts
    FROM cells c
    JOIN expression e ON c.cell_id = e.cell_id
    JOIN high_expression_genes g ON e.gene_id = g.gene_id
    JOIN cell_stats cs ON c.cell_type = cs.cell_type
    WHERE e.expression > cs.avg_counts
""")
```
