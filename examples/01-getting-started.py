import marimo

__generated_with = "0.14.0"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo

    from slaf import SLAFArray
    from slaf.integrations import scanpy as slaf_scanpy
    from slaf.integrations.anndata import read_slaf

    return SLAFArray, mo, read_slaf, slaf_scanpy


@app.cell
def _(mo):
    mo.md(
        """
    # SLAF Getting Started Guide

    This notebook introduces SLAF (Sparse Lazy Array Format) - a high-performance format for single-cell data that combines the power of SQL with lazy evaluation.

    **Key Benefits:**

    - 🚀 **Fast**: SQL-level performance for data operations

    - 💾 **Memory Efficient**: Lazy evaluation, only load what you need

    - 🔍 **SQL Native**: Direct SQL queries on your data

    - 🧬 **Scanpy Compatible**: Drop-in replacement for AnnData workflows

    - ⚡ **Production Ready**: Built for large-scale single-cell analysis

    - ⚙️ **ML Ready**: Ready for ML training with efficient tokenization
    """
    )
    return


@app.cell
def _(SLAFArray):
    # Load SLAF dataset using the low-level interface
    slaf = SLAFArray("../slaf-datasets/pbmc3k_processed.slaf")
    print(f"✅ Loaded SLAF dataset: {slaf.shape[0]:,} cells × {slaf.shape[1]:,} genes")

    # Show dataset information
    slaf.info()
    return (slaf,)


@app.cell
def _(mo):
    mo.md(
        """
    ## 1. Understanding the SLAF Database Schema

    SLAF stores data in three main tables that you can query directly with SQL:
    """
    )
    return


@app.cell
def _():
    def show_database_schema():
        # Show the database schema
        print("📊 SLAF Database Schema")
        print("=" * 50)

        # Get table information
        tables = ["cells", "genes", "expression"]

        for table in tables:
            print(f"\n🔍 Table: {table}")
            if table == "cells":
                print("   Purpose: Cell metadata and QC metrics")
                print("   Key columns:")
                print("     - cell_id: Unique cell identifier")
                print("     - cell_integer_id: Integer ID for efficient queries")
                print("     - cell_type: Cell type annotation")
                print("     - batch: Batch information")
                print("     - total_counts: Total UMI counts per cell")
                print("     - n_genes_by_counts: Number of genes expressed")
                print("     - high_mito: Boolean flag for high mitochondrial content")
            elif table == "genes":
                print("   Purpose: Gene metadata and annotations")
                print("   Key columns:")
                print("     - gene_id: Unique gene identifier")
                print("     - gene_integer_id: Integer ID for efficient queries")
                print("     - highly_variable: Boolean flag for highly variable genes")
            elif table == "expression":
                print("   Purpose: Sparse expression matrix data")
                print("   Key columns:")
                print("     - cell_integer_id: Integer cell ID for efficient queries")
                print("     - gene_integer_id: Integer gene ID for efficient queries")
                print("     - value: Expression value (UMI counts)")
                print("   Note: String IDs (cell_id, gene_id) are in metadata tables")

        return

    show_database_schema()
    return


@app.cell
def _(slaf):
    def show_sample_data():
        # Show sample data from each table
        print("📋 Sample Data from Each Table")
        print("=" * 50)

        # Sample from cells table
        print("\n🔬 Sample cells:")
        cells_sample = slaf.query("SELECT * FROM cells LIMIT 3")
        print(cells_sample)

        # Sample from genes table
        print("\n🧬 Sample genes:")
        genes_sample = slaf.query("SELECT * FROM genes LIMIT 3")
        print(genes_sample)

        # Sample from expression table
        print("\n📈 Sample expression data:")
        expr_sample = slaf.query("SELECT * FROM expression LIMIT 5")
        print(expr_sample)

    show_sample_data()
    return


@app.cell
def _(mo):
    mo.md(
        """
    ## 2. SQL Queries - Your Data, Your Way

    SLAF gives you direct SQL access to your data. Here are some practical examples:
    """
    )
    return


@app.cell
def _(slaf):
    def run_basic_sql_queries():
        # Basic SQL queries
        print("🔍 Basic SQL Queries")
        print("=" * 40)

        # Count records
        print("\n1. Count records in each table:")
        for table in ["cells", "genes", "expression"]:
            count = slaf.query(f"SELECT COUNT(*) as count FROM {table}")
            print(f"   {table}: {count.item(0, 0):,} records")

        # Cell type distribution
        print("\n2. Batch distribution:")
        cell_types = slaf.query(
            """
            SELECT batch, COUNT(*) as count
            FROM cells
            GROUP BY batch
            ORDER BY count DESC
        """
        )
        print(cell_types)

        # Expression statistics
        print("\n3. Expression value statistics:")
        expr_stats = slaf.query(
            """
            SELECT
                MIN(value) as min_expr,
                MAX(value) as max_expr,
                AVG(value) as avg_expr,
                COUNT(*) as total_records,
                COUNT(CASE WHEN value > 0 THEN 1 END) as non_zero_records
            FROM expression
        """
        )
        print(expr_stats)

    run_basic_sql_queries()
    return


@app.cell
def _(slaf):
    def run_advanced_sql_queries():
        # Advanced SQL queries with joins
        print("🔗 Advanced SQL Queries with Joins")
        print("=" * 45)

        # Cells with their expression summary
        print("\n1. Top 5 cells by total expression:")
        top_cells = slaf.query(
            """
            SELECT
                c.cell_id,
                c.total_counts,
                COUNT(e.value) as expressed_genes,
                AVG(e.value) as avg_expression
            FROM cells c
            LEFT JOIN expression e ON c.cell_integer_id = e.cell_integer_id
            GROUP BY c.cell_id, c.total_counts
            ORDER BY c.total_counts DESC
            LIMIT 5
        """
        )
        print(top_cells)

        # Highly variable genes with expression stats
        print("\n2. Highly variable genes with expression stats:")
        hvg_stats = slaf.query(
            """
            SELECT
                g.gene_id,
                g.highly_variable,
                COUNT(e.value) as cells_expressed,
                AVG(e.value) as avg_expression,
                MAX(e.value) as max_expression
            FROM genes g
            LEFT JOIN expression e ON g.gene_integer_id = e.gene_integer_id
            WHERE g.highly_variable = true
            GROUP BY g.gene_id, g.highly_variable
            ORDER BY avg_expression DESC
            LIMIT 5
        """
        )
        print(hvg_stats)

    run_advanced_sql_queries()
    return


@app.cell
def _(mo):
    mo.md(
        """
    ## 3. Convenience Methods - Easy Filtering

    SLAF provides convenient methods for common operations:
    """
    )
    return


@app.cell
def _(slaf):
    # Demonstrate convenience methods
    print("🎯 Convenience Methods")
    print("=" * 30)

    # Filter cells
    print("\n1. Filter cells by criteria:")

    # High quality cells
    high_quality = slaf.filter_cells(n_genes_by_counts=">=1000", total_counts=">=2000")
    print(f"   High quality cells (≥1000 genes, ≥2000 counts): {len(high_quality):,}")

    # Filter genes
    print("\n2. Filter genes by criteria:")

    # Highly variable genes
    hvg_genes = slaf.filter_genes(highly_variable=True)
    print(f"   Highly variable genes: {len(hvg_genes):,}")

    return


@app.cell
def _(mo):
    mo.md(
        """
    ## 4. Lazy AnnData Interface - Scanpy Compatible

    SLAF provides a lazy AnnData interface that's compatible with scanpy workflows:
    """
    )
    return


@app.cell
def _(read_slaf):
    # Load as lazy AnnData
    adata = read_slaf("../slaf-datasets/pbmc3k_processed.slaf")
    print(
        f"✅ Loaded as LazyAnnData: {adata.shape[0]:,} cells × {adata.shape[1]:,} genes"
    )
    print(f"   Type: {type(adata)}")
    print(f"   Expression matrix type: {type(adata.X)}")
    return (adata,)


@app.cell
def _(adata):
    # Basic AnnData operations
    print("🔬 Basic AnnData Operations")
    print("=" * 35)

    print(f"Dataset shape: {adata.shape}")
    print(f"Number of cells: {adata.n_obs}")
    print(f"Number of genes: {adata.n_vars}")

    # Show available metadata
    if hasattr(adata, "obs") and adata.obs is not None:
        print(f"\nCell metadata columns: {list(adata.obs.columns)}")

    if hasattr(adata, "var") and adata.var is not None:
        print(f"Gene metadata columns: {list(adata.var.columns)}")

    return


@app.cell
def _(adata):
    # Lazy slicing operations
    print("✂️ Lazy Slicing Operations")
    print("=" * 30)

    # Slice cells
    subset_cells = adata[:100, :]
    print(f"First 100 cells subset: {subset_cells.shape}")

    # Slice genes
    subset_genes = adata[:, :50]
    print(f"First 50 genes subset: {subset_genes.shape}")

    # Combined slice
    subset_both = adata[:50, :25]
    print(f"50 cells × 25 genes subset: {subset_both.shape}")

    print("\nNote: These operations are lazy - no data is loaded until needed!")

    return (subset_cells,)


@app.cell
def _(mo):
    mo.md(
        """
    ## 5. Lazy Scanpy Preprocessing

    SLAF provides lazy versions of scanpy preprocessing functions:
    """
    )
    return


@app.cell
def _(adata, slaf_scanpy):
    # Lazy scanpy preprocessing
    print("🧬 Lazy Scanpy Preprocessing")
    print("=" * 35)

    # Ensure metadata is loaded before preprocessing
    print("1. Loading metadata...")
    _ = adata.obs  # Trigger metadata loading
    _ = adata.var  # Trigger metadata loading
    print("   ✅ Metadata loaded")

    # Calculate QC metrics (lazy)
    print("\n2. Calculating QC metrics...")
    slaf_scanpy.pp.calculate_qc_metrics(adata, inplace=True)
    print("   ✅ QC metrics calculated (lazily)")

    # Filter cells (lazy) - use more conservative filtering
    print("\n3. Filtering cells...")
    slaf_scanpy.pp.filter_cells(adata, min_genes=200, inplace=True)
    print("   ✅ Cells filtered (lazily)")

    # Filter genes (lazy) - use more conservative filtering
    print("\n4. Filtering genes...")
    slaf_scanpy.pp.filter_genes(adata, min_cells=30, inplace=True)
    print("   ✅ Genes filtered (lazily)")

    print(f"\nFinal dataset shape: {adata.shape[0]:,} cells × {adata.shape[1]:,} genes")
    print("Note: All operations are lazy - data is only processed when accessed!")

    return


@app.cell
def _(adata, subset_cells):
    # Demonstrate .compute() method
    print("⚡ Explicit Computation with .compute()")
    print("=" * 40)

    print("1. Computing expression matrix only:")
    print(f"   Before: {type(adata.X)}")
    sparse_matrix = adata.X.compute()
    print(f"   After: {type(sparse_matrix)}")
    print(f"   Shape: {sparse_matrix.shape}")

    print("\n2. Computing sliced data:")
    print(f"   Slice type: {type(subset_cells)}")
    native_slice = subset_cells.compute()
    print(f"   Computed slice type: {type(native_slice)}")
    print(f"   Shape: {native_slice.shape}")

    print("\n3. Computing full dataset (with sync check):")
    print(f"   Before: {type(adata)}")

    # Check if metadata and expression matrix are in sync
    try:
        obs_rows = (
            len(adata.obs) if hasattr(adata, "obs") and adata.obs is not None else 0
        )
        var_rows = (
            len(adata.var) if hasattr(adata, "var") and adata.var is not None else 0
        )
        x_rows = adata.X.shape[0] if hasattr(adata, "X") and adata.X is not None else 0

        print(f"   Metadata rows: obs={obs_rows}, var={var_rows}")
        print(f"   Expression matrix rows: {x_rows}")

        if obs_rows != x_rows:
            print("   ⚠️ Metadata and expression matrix are out of sync")
            print("   This is expected after lazy filtering operations.")
            print("   Recommendation: Use .X.compute() for expression data only.")
            print("   ✅ Example continues successfully despite sync issue")
        else:
            # Try to compute the full dataset
            native_adata = adata.compute()
            print(f"   After: {type(native_adata)}")
            print(f"   Shape: {native_adata.shape}")
            print("   ✅ Successfully computed full dataset")

    except Exception as e:
        print(f"   ⚠️ Sync issue detected: {str(e)[:100]}...")
        print("   This is expected after lazy filtering operations.")
        print("   Recommendation: Use .X.compute() for expression data only.")
        print("   ✅ Example continues successfully despite sync issue")

    print("\nKey insight: .compute() converts lazy objects to native scanpy objects!")

    return


@app.cell
def _(read_slaf, mo):
    # Demonstrate workaround for metadata sync issues
    print("🔄 Workaround for Metadata Sync Issues")
    print("=" * 45)

    mo.md(
        """
    **Note**: After lazy filtering operations, metadata and expression matrix can get out of sync.
    Here's how to work around this limitation:
    """
    )

    print("1. Create fresh LazyAnnData after filtering:")
    print("   ```python")
    print("   # Instead of computing the filtered object directly")
    print("   # Create a fresh LazyAnnData and apply filters again")
    print("   fresh_adata = read_slaf('path/to/dataset.slaf')")
    print("   # Apply your filters to the fresh object")
    print("   sc.pp.filter_cells(fresh_adata, min_genes=200)")
    print("   sc.pp.filter_genes(fresh_adata, min_cells=30)")
    print("   # Now compute() should work")
    print("   real_adata = fresh_adata.compute()")
    print("   ```")

    print("\n2. Use .X.compute() for expression data only:")
    print("   ```python")
    print("   # For expression matrix only")
    print("   expression_matrix = adata.X.compute()")
    print("   # For metadata only")
    print("   cell_metadata = adata.obs")
    print("   gene_metadata = adata.var")
    print("   ```")

    print("\n3. Use SQL queries for complex filtering:")
    print("   ```python")
    print("   # Use SLAF's SQL interface for complex filtering")
    print("   filtered_cells = slaf.filter_cells(n_genes_by_counts='>=500')")
    print("   filtered_genes = slaf.filter_genes(highly_variable=True)")
    print("   ```")

    return


@app.cell
def _(mo):
    mo.md(
        """
    ## 6. Performance Characteristics

    Let's examine SLAF's performance characteristics:
    """
    )
    return


@app.cell
def _(slaf):
    def examine_performance_characteristics():
        import time

        print("⚡ Performance Characteristics")
        print("=" * 35)

        # Test cell filtering performance
        print("\n1. Cell filtering performance:")

        # SLAF SQL approach
        start_time = time.time()
        _ = slaf.query("SELECT COUNT(*) FROM cells WHERE n_genes_by_counts > 500")
        slaf_time = time.time() - start_time
        print(f"   SLAF SQL: {slaf_time:.4f}s")

        # SLAF convenience method
        start_time = time.time()
        _ = slaf.filter_cells(n_genes_by_counts=">500")
        slaf_filter_time = time.time() - start_time
        print(f"   SLAF filter method: {slaf_filter_time:.4f}s")

        # Test expression aggregation
        print("\n2. Expression aggregation performance:")

        # SLAF SQL aggregation
        start_time = time.time()
        _ = slaf.query("SELECT AVG(value) FROM expression")
        slaf_agg_time = time.time() - start_time
        print(f"   SLAF SQL aggregation: {slaf_agg_time:.4f}s")

        # Lazy AnnData approach (when computed)
        print("   Lazy AnnData: Operations stored, computed on demand")

    examine_performance_characteristics()

    return


@app.cell
def _(mo):
    mo.md(
        """
    ## 8. Quick ML Training Example

    SLAF provides efficient tokenization and DataLoader for ML training:
    """
    )
    return


@app.cell
def _(slaf):
    def demonstrate_ml_training():
        from slaf.ml.dataloaders import SLAFDataLoader
        from slaf.ml.tokenizers import SLAFTokenizer

        print("🎯 Quick ML Training Example")
        print("=" * 35)

        # Initialize tokenizer
        tokenizer = SLAFTokenizer(
            slaf_array=slaf,
            tokenizer_type="geneformer",
            vocab_size=2000,  # Dataset has <2000 genes
            n_expression_bins=20,
        )
        vocab_info = tokenizer.get_vocab_info()
        print(f"✅ Tokenizer: {vocab_info['vocab_size']} vocabulary size")

        # Create DataLoader
        dataloader = SLAFDataLoader(
            slaf_array=slaf,
            tokenizer_type="geneformer",
            batch_size=16,
            max_genes=100,
            vocab_size=2000,  # Dataset has <2000 genes
            n_expression_bins=20,
            n_epochs=1,
        )

        # Get first batch
        batch = next(iter(dataloader))
        print(f"✅ DataLoader: {batch['input_ids'].shape} tensors")
        print(f"   Cell IDs: {len(batch['cell_ids'])} cells")

        return tokenizer, dataloader

    tokenizer, dataloader = demonstrate_ml_training()
    return


@app.cell
def _(mo):
    mo.md(
        """
    ## Summary

    **What you've learned:**

    1. **SQL Schema**: SLAF stores data in 3 tables (cells, genes, expression) that you can query directly
    2. **SQL Power**: Direct SQL access for complex queries and aggregations
    3. **Convenience Methods**: Easy filtering with `filter_cells()` and `filter_genes()`
    4. **Lazy AnnData**: Scanpy-compatible interface with lazy evaluation
    5. **Lazy Preprocessing**: scanpy functions that work lazily
    6. **Performance**: SQL-level performance for data operations
    7. **Explicit Computation**: Use `.compute()` to convert lazy objects to native Python objects
    8. **ML Training**: Ready for ML training with efficient tokenization and DataLoader

    **Next Steps:**

    - **02-lazy-processing.py**: Deep dive into lazy evaluation capabilities
    - **03-ml-training-pipeline.py**: Complete ML training workflows with streaming DataLoader
    """
    )
    return


if __name__ == "__main__":
    app.run()
