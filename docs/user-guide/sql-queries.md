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
    WHERE e.expression > 0
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
    WHERE e.expression > 0
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

## Performance Optimization

### Indexing

SLAF automatically creates indexes for efficient querying:

```python
# Check available indexes
indexes = slaf_array.get_indexes()
print(indexes)

# Create custom index if needed
slaf_array.create_index("cells", "cell_type")
slaf_array.create_index("genes", "gene_name")
```

### Query Optimization

```python
# Use EXPLAIN to see query plan
plan = slaf_array.explain("""
    SELECT cell_type, COUNT(*)
    FROM cells
    WHERE batch = 'batch1'
    GROUP BY cell_type
""")
print(plan)
```

## Data Export

### Export Query Results

```python
# Export to various formats
results = slaf_array.query("SELECT * FROM cells WHERE cell_type = 'T cells'")

# To pandas DataFrame
df = results.to_pandas()

# To CSV
results.to_csv("t_cells.csv")

# To Parquet
results.to_parquet("t_cells.parquet")
```

### Batch Processing

```python
# Process large queries in batches
for batch in slaf_array.query_batched("""
    SELECT * FROM cells
    WHERE batch = 'batch1'
""", batch_size=1000):
    # Process each batch
    process_batch(batch)
```

## Common Query Patterns

### Cell Filtering

```python
# Filter by multiple criteria
results = slaf_array.query("""
    SELECT *
    FROM cells
    WHERE cell_type IN ('T cells', 'B cells')
    AND total_counts BETWEEN 1000 AND 10000
    AND n_genes > 500
    AND batch = 'batch1'
""")
```

### Gene Set Analysis

```python
# Analyze specific gene sets
gene_set = ['CD3D', 'CD3E', 'CD3G', 'CD4', 'CD8A']

results = slaf_array.query(f"""
    SELECT
        c.cell_id,
        c.cell_type,
        SUM(e.expression) as total_expression
    FROM cells c
    JOIN expression e ON c.cell_id = e.cell_id
    JOIN genes g ON e.gene_id = g.gene_id
    WHERE g.gene_name IN ({','.join([f"'{g}'" for g in gene_set])})
    GROUP BY c.cell_id, c.cell_type
    HAVING total_expression > 0
""")
```

### Quality Control

```python
# QC metrics
results = slaf_array.query("""
    SELECT
        batch,
        COUNT(*) as total_cells,
        AVG(total_counts) as avg_counts,
        AVG(n_genes) as avg_genes,
        COUNT(CASE WHEN total_counts < 1000 THEN 1 END) as low_quality_cells
    FROM cells
    GROUP BY batch
""")
```

## Best Practices

1. **Use appropriate WHERE clauses** to filter data early
2. **Limit results** when exploring data
3. **Use indexes** for frequently queried columns
4. **Test queries** on small subsets first
5. **Use EXPLAIN** to understand query performance
6. **Batch large queries** to manage memory usage
