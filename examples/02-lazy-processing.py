import marimo

__generated_with = "0.17.0"
app = marimo.App(width="medium")


@app.cell
def _():
    import time

    import marimo as mo
    import numpy as np
    from scipy.sparse import csr_matrix

    from slaf.integrations import scanpy as slaf_scanpy
    from slaf.integrations.anndata import read_slaf

    return csr_matrix, mo, np, read_slaf, slaf_scanpy, time


@app.cell
def _(mo):
    mo.md(
        """
    # SLAF Lazy Processing Deep Dive

    This notebook explores SLAF's lazy evaluation capabilities in detail. You'll learn how to:

    - Build complex analysis pipelines without loading data
    - Apply multiple transformations efficiently
    - Use different slicing patterns
    - Control when computation happens
    - Understand performance benefits

    **Key Concept**: Lazy evaluation means operations are stored as instructions and only executed when you explicitly request the results.

    **Key Benefits:**
    - 🚀 **Instant Pipeline Building**: No waiting for data loading
    - 💾 **Memory Efficient**: Only load what you need
    - 🔄 **Composable**: Operations can be combined and preserved
    - ⚡ **SQL Performance**: Leverage database-level optimizations
    - 🧬 **Scanpy Compatible**: Familiar interface with performance benefits
    """
    )
    return


@app.cell
def _(read_slaf):
    # Load data for lazy processing examples
    adata = read_slaf("../slaf-datasets/pbmc3k_processed.slaf")
    print(f"✅ Loaded dataset: {adata.shape[0]:,} cells × {adata.shape[1]:,} genes")
    print(f"   Type: {type(adata)}")
    print(f"   Expression matrix type: {type(adata.X)}")
    return (adata,)


@app.cell
def _(mo):
    mo.md(
        """
    ## 1. Understanding Lazy Objects

    SLAF provides two main lazy object types:

    - **LazyAnnData**: Lazy version of AnnData with scanpy compatibility

    - **LazyExpressionMatrix**: Lazy version of the expression matrix
    """
    )
    return


@app.cell
def _(adata):
    # Demonstrate lazy object types
    print("🔍 Lazy Object Types")
    print("=" * 30)

    print(f"1. LazyAnnData type: {type(adata)}")
    print(f"   - Shape: {adata.shape}")
    print(
        f"   - Obs columns: {list(adata.obs.columns) if hasattr(adata, 'obs') and adata.obs is not None else 'Not loaded'}"
    )
    print(
        f"   - Var columns: {list(adata.var.columns) if hasattr(adata, 'var') and adata.var is not None else 'Not loaded'}"
    )

    print(f"\n2. LazyExpressionMatrix type: {type(adata.X)}")
    print(f"   - Shape: {adata.X.shape}")
    print(
        f"   - Parent: {type(adata.X.parent_adata) if hasattr(adata.X, 'parent_adata') else 'None'}"
    )

    print("\n3. Key insight: These objects store operations, not data!")
    print("   - No data is loaded until you call .compute()")
    print("   - Operations are composed efficiently")
    print("   - Memory usage stays low")
    return


@app.cell
def _(mo):
    mo.md(
        """
    ## 2. Explicit Computation Control

    You control when data is actually computed using these methods:
    """
    )
    return


@app.cell
def _(adata, time):
    def demonstrate_computation_control():
        # Demonstrate explicit computation control
        print("🎛️ Explicit Computation Control")
        print("=" * 35)

        print("Available computation methods:")
        print("1. adata.compute() → native AnnData object")
        print("2. adata.X.compute() → scipy.sparse.csr_matrix")
        print("3. adata.obs → pandas.DataFrame (cell metadata)")
        print("4. adata.var → pandas.DataFrame (gene metadata)")

        print("\nLet's demonstrate:")

        # Compute just the expression matrix
        print("\n1. Computing expression matrix...")
        start_time = time.time()
        sparse_matrix = adata.X.compute()
        compute_time = time.time() - start_time
        print(f"   ✅ Computed in {compute_time:.4f}s")
        print(f"   Type: {type(sparse_matrix)}")
        print(f"   Shape: {sparse_matrix.shape}")
        print(f"   Memory: {sparse_matrix.data.nbytes / 1024 / 1024:.1f} MB")

        # Access cell metadata
        print("\n2. Accessing cell metadata...")
        start_time = time.time()
        obs_df = adata.obs
        obs_time = time.time() - start_time
        print(f"   ✅ Accessed in {obs_time:.4f}s")
        print(f"   Type: {type(obs_df)}")
        print(f"   Shape: {obs_df.shape}")

        return (sparse_matrix, obs_df)

    sparse_matrix, obs_df = demonstrate_computation_control()
    return


@app.cell
def _(mo):
    mo.md(
        """
    ## 3. Slicing Patterns - All the Ways to Slice

    SLAF supports multiple slicing patterns, all of which are lazy:
    """
    )
    return


@app.cell
def _(adata):
    # Demonstrate different slicing patterns
    def demonstrate_slicing_patterns(adata):
        print("✂️ Slicing Patterns")
        print("=" * 25)

        print("1. Basic integer slicing:")
        slice1 = adata[:100, :50]
        print(f"   adata[:100, :50] → {type(slice1)} with shape {slice1.shape}")

        print("\n2. Expression matrix slicing:")
        slice2 = adata.X[:100, :50]
        print(f"   adata.X[:100, :50] → {type(slice2)} with shape {slice2.shape}")

        print("\n3. Boolean indexing (after QC metrics are available):")
        # First add some QC metrics
        if (
            hasattr(adata, "obs")
            and adata.obs is not None
            and "n_genes_by_counts" in adata.obs.columns
        ):
            high_quality_mask = adata.obs["n_genes_by_counts"] > 1000
            slice3 = adata[high_quality_mask, :]
            print(
                f"   adata[high_quality_mask, :] → {type(slice3)} with shape {slice3.shape}"
            )
        else:
            print(
                "   (QC metrics not available yet - will be computed in preprocessing section)"
            )

        print("\n4. Mixed indexing:")
        slice4 = adata[:100, adata.var.index[:50]]
        print(
            f"   adata[:100, adata.var.index[:50]] → {type(slice4)} with shape {slice4.shape}"
        )

        print("\nKey insight: All slicing returns lazy objects!")

    demonstrate_slicing_patterns(adata)
    return


@app.cell
def _(mo):
    mo.md(
        """
    ## 4. Transformation Patterns

    SLAF supports lazy transformations that are stored and applied when needed:
    """
    )
    return


@app.cell
def _(read_slaf):
    # Load fresh data for transformation examples
    adata_fresh = read_slaf("../slaf-datasets/pbmc3k_processed.slaf")
    print("✅ Loaded fresh dataset for transformations")
    return (adata_fresh,)


@app.cell
def _(adata_fresh, slaf_scanpy):
    # Demonstrate transformation patterns
    print("🔄 Transformation Patterns")
    print("=" * 30)

    print("1. Single transformation:")
    adata_norm = slaf_scanpy.pp.normalize_total(
        adata_fresh, target_sum=1e4, inplace=False
    )
    print(f"   normalize_total() → {type(adata_norm)}")
    print(
        f"   Transformations stored: {list(adata_norm._transformations.keys()) if hasattr(adata_norm, '_transformations') else 'None'}"
    )

    print("\n2. Chained transformations:")
    adata_processed = slaf_scanpy.pp.normalize_total(
        adata_fresh, target_sum=1e4, inplace=False
    )
    adata_processed = slaf_scanpy.pp.log1p(adata_processed, inplace=False)
    print(f"   normalize_total().log1p() → {type(adata_processed)}")
    print(
        f"   Transformations stored: {list(adata_processed._transformations.keys()) if hasattr(adata_processed, '_transformations') else 'None'}"
    )

    print("\n3. Transformation on sliced data:")
    # First slice, then apply transformation (safer pattern)
    slice_data = adata_fresh[:100, :50]
    slice_transformed = slaf_scanpy.pp.normalize_total(
        slice_data, target_sum=1e4, inplace=False
    )
    print(f"   adata[:100, :50].normalize_total() → {type(slice_transformed)}")
    print(f"   Shape: {slice_transformed.shape}")

    print("\n4. Multiple transformations on slice:")
    # First slice, then apply transformations (safer pattern)
    slice_data = adata_fresh[:100, :50]
    slice_multi = slaf_scanpy.pp.normalize_total(
        slice_data, target_sum=1e4, inplace=False
    )
    slice_multi = slaf_scanpy.pp.log1p(slice_multi, inplace=False)
    print(f"   Multiple transformations on slice → {type(slice_multi)}")
    print(
        f"   Transformations: {list(slice_multi._transformations.keys()) if hasattr(slice_multi, '_transformations') else 'None'}"
    )
    return adata_processed, slice_multi, slice_transformed


@app.cell
def _(adata_processed, slice_multi, slice_transformed, time):
    def demonstrate_transformation_application():
        # Demonstrate transformation application
        print("⚡ Applying Transformations")
        print("=" * 30)

        print("1. Computing transformed data:")
        start_time = time.time()
        native_processed = adata_processed.compute()
        process_time = time.time() - start_time
        print(f"   ✅ Computed in {process_time:.4f}s")
        print(f"   Type: {type(native_processed)}")
        print(f"   Shape: {native_processed.shape}")

        print("\n2. Computing transformed slice:")
        start_time = time.time()
        # Use .X.compute() to avoid the metadata mismatch issue
        native_slice_matrix = slice_transformed.X.compute()
        slice_time = time.time() - start_time
        print(f"   ✅ Computed in {slice_time:.4f}s")
        print(f"   Type: {type(native_slice_matrix)}")
        print(f"   Shape: {native_slice_matrix.shape}")

        print("\n3. Computing multi-transformed slice:")
        start_time = time.time()
        # Use .X.compute() to avoid the metadata mismatch issue
        native_multi_matrix = slice_multi.X.compute()
        multi_time = time.time() - start_time
        print(f"   ✅ Computed in {multi_time:.4f}s")
        print(f"   Type: {type(native_multi_matrix)}")
        print(f"   Shape: {native_multi_matrix.shape}")

        return (native_processed, native_slice_matrix, native_multi_matrix)

    demonstrate_transformation_application()
    return


@app.cell
def _(mo):
    mo.md(
        """
    ## 5. Transformation Preservation Through Operations

    Transformations are preserved through slicing and other operations:
    """
    )
    return


@app.cell
def _(read_slaf):
    # Load fresh data for preservation examples
    adata_preserve = read_slaf("../slaf-datasets/pbmc3k.slaf")
    return (adata_preserve,)


@app.cell
def _(adata_preserve, np, slaf_scanpy):
    # Demonstrate transformation preservation
    print("🔄 Transformation Preservation")
    print("=" * 35)

    print("1. Apply transformations first, then slice:")
    adata_transformed = slaf_scanpy.pp.normalize_total(
        adata_preserve, target_sum=1e4, inplace=False
    )
    adata_transformed = slaf_scanpy.pp.log1p(adata_transformed, inplace=False)
    slice_after = adata_transformed[:100, :50]
    print(
        f"   Original transformations: {list(adata_transformed._transformations.keys()) if hasattr(adata_transformed, '_transformations') else 'None'}"
    )
    print(
        f"   Slice transformations: {list(slice_after._transformations.keys()) if hasattr(slice_after, '_transformations') else 'None'}"
    )

    print("\n2. Slice first, then apply transformations:")
    slice_before = adata_preserve[:100, :50]
    transformed_slice = slaf_scanpy.pp.normalize_total(
        slice_before, target_sum=1e4, inplace=False
    )
    transformed_slice = slaf_scanpy.pp.log1p(transformed_slice, inplace=False)
    print(f"   Transformed slice: {type(transformed_slice)}")
    print(
        f"   Transformations: {list(transformed_slice._transformations.keys()) if hasattr(transformed_slice, '_transformations') else 'None'}"
    )

    print("\n3. Complex slicing patterns preserve transformations:")
    print(
        "   Note: Chained slicing (e.g., adata[:200, :100][:50, :25]) is not supported."
    )
    print("   Use single-step slicing instead: adata[50:250, 25:125]")

    # Single-step slicing (equivalent to nested slicing)
    single_step_slice = adata_transformed[50:250, 25:125]
    print(f"   Single-step slice: {type(single_step_slice)}")
    print(
        f"   Transformations preserved: {list(single_step_slice._transformations.keys()) if hasattr(single_step_slice, '_transformations') else 'None'}"
    )

    # Boolean mask slicing
    cell_mask = np.zeros(adata_transformed.shape[0], dtype=bool)
    cell_mask[50:250] = True
    gene_mask = np.zeros(adata_transformed.shape[1], dtype=bool)
    gene_mask[25:125] = True
    boolean_slice = adata_transformed[cell_mask, gene_mask]
    print(f"   Boolean mask slice: {type(boolean_slice)}")
    print(
        f"   Transformations preserved: {list(boolean_slice._transformations.keys()) if hasattr(boolean_slice, '_transformations') else 'None'}"
    )

    # Step slicing
    step_slice = adata_transformed[::4, ::2]  # Every 4th cell, every 2nd gene
    print(f"   Step slice: {type(step_slice)}")
    print(
        f"   Transformations preserved: {list(step_slice._transformations.keys()) if hasattr(step_slice, '_transformations') else 'None'}"
    )
    return single_step_slice, slice_after, transformed_slice


@app.cell
def _(single_step_slice, slice_after, time, transformed_slice):
    def verify_transformation_preservation():
        # Verify transformation preservation by computing
        print("✅ Verifying Transformation Preservation")
        print("=" * 40)

        print("1. Computing slice with preserved transformations:")
        start_time = time.time()
        # Use .X.compute() to avoid the metadata mismatch issue
        result1 = slice_after.X.compute()
        time1 = time.time() - start_time
        print(f"   ✅ Computed in {time1:.4f}s")
        print(f"   Type: {type(result1)}")
        print(f"   Shape: {result1.shape}")

        print("\n2. Computing slice with applied transformations:")
        start_time = time.time()
        # Use .X.compute() to avoid the metadata mismatch issue
        result2 = transformed_slice.X.compute()
        time2 = time.time() - start_time
        print(f"   ✅ Computed in {time2:.4f}s")
        print(f"   Type: {type(result2)}")
        print(f"   Shape: {result2.shape}")

        print("\n3. Computing single-step slice with preserved transformations:")
        start_time = time.time()
        # Use .X.compute() to avoid the metadata mismatch issue
        result3 = single_step_slice.X.compute()
        time3 = time.time() - start_time
        print(f"   ✅ Computed in {time3:.4f}s")
        print(f"   Type: {type(result3)}")
        print(f"   Shape: {result3.shape}")

    verify_transformation_preservation()
    return


@app.cell
def _(mo):
    mo.md(
        """
    ## 6. Performance Benefits - Building Complex Pipelines

    Let's see how lazy evaluation enables efficient complex pipelines:
    """
    )
    return


@app.cell
def _(adata, slaf_scanpy, time):
    # Demonstrate performance benefits

    def demonstrate_complex_pipeline(adata):
        print("⚡ Performance Benefits")
        print("=" * 25)

        print("1. Building complex pipeline (no computation yet):")
        start_time = time.time()

        # Build a complex pipeline
        pipeline = slaf_scanpy.pp.normalize_total(adata, target_sum=1e4, inplace=False)
        pipeline = slaf_scanpy.pp.log1p(pipeline, inplace=False)

        # Note: highly_variable_genes returns a DataFrame, not a LazyAnnData object
        # so it can't be chained in the pipeline like other transformations

        # Slice the processed data
        final_slice = pipeline[:500, :200]

        build_time = time.time() - start_time
        print(f"   ✅ Pipeline built in {build_time:.4f}s")
        print(f"   Final object: {type(final_slice)}")
        print(
            "   Expected shape: (500, 200)"
        )  # Avoid accessing .shape on transformed slice
        print(
            f"   Transformations: {list(final_slice._transformations.keys()) if hasattr(final_slice, '_transformations') else 'None'}"
        )

        print("\n2. Computing the final result:")
        start_time = time.time()
        # Use .X.compute() to avoid the metadata mismatch issue
        final_result = final_slice.X.compute()
        compute_time = time.time() - start_time
        print(f"   ✅ Computed in {compute_time:.4f}s")
        print(f"   Type: {type(final_result)}")
        print(f"   Shape: {final_result.shape}")

        print(f"\n3. Total time: {build_time + compute_time:.4f}s")
        print(
            "   Key insight: Pipeline building is instant, computation happens only when needed!"
        )

    demonstrate_complex_pipeline(adata)
    return


@app.cell
def _(mo):
    mo.md(
        """
    ## 7. Memory Efficiency Comparison

    Let's compare memory usage between lazy and eager approaches:
    """
    )
    return


@app.cell
def _(read_slaf):
    # Memory efficiency comparison
    print("💾 Memory Efficiency Comparison")
    print("=" * 35)

    import gc

    import psutil

    def get_memory_usage():
        """Get current memory usage in MB"""
        process = psutil.Process()
        return process.memory_info().rss / 1024 / 1024

    # Load fresh data
    adata_mem = read_slaf("../slaf-datasets/pbmc3k_processed.slaf")

    print("1. Memory after loading lazy data:")
    gc.collect()
    lazy_memory = get_memory_usage()
    print(f"   Lazy loading: {lazy_memory:.1f} MB")

    print("\n2. Memory after computing full dataset:")
    gc.collect()
    start_memory = get_memory_usage()
    _ = adata_mem.compute()
    end_memory = get_memory_usage()
    print(f"   Eager loading: {end_memory:.1f} MB")
    print(f"   Memory increase: {end_memory - start_memory:.1f} MB")

    print("\n3. Memory after computing small slice:")
    gc.collect()
    slice_memory_before = get_memory_usage()
    _ = adata_mem[:100, :50].compute()
    slice_memory_after = get_memory_usage()
    print(f"   Small slice: {slice_memory_after:.1f} MB")
    print(f"   Memory increase: {slice_memory_after - slice_memory_before:.1f} MB")

    print(
        f"\nKey insight: Lazy loading uses {lazy_memory:.1f} MB vs eager loading {end_memory:.1f} MB"
    )
    print(f"Memory savings: {((end_memory - lazy_memory) / end_memory * 100):.1f}%")
    return


@app.cell
def _(mo):
    mo.md(
        """
    ## 8. Advanced Slicing Patterns

    Let's explore more advanced slicing patterns:
    """
    )
    return


@app.cell
def _(read_slaf, slaf_scanpy):
    # Advanced slicing patterns
    print("🔬 Advanced Slicing Patterns")
    print("=" * 35)

    # Load data and add QC metrics
    adata_advanced = read_slaf("../slaf-datasets/pbmc3k_processed.slaf")
    slaf_scanpy.pp.calculate_qc_metrics(adata_advanced, inplace=True)

    print("1. Boolean indexing with QC metrics:")
    high_quality_mask = adata_advanced.obs["n_genes_by_counts"] > 1000
    high_quality_cells = adata_advanced[high_quality_mask, :]
    print(f"   High quality cells: {high_quality_cells.shape}")

    print("\n2. Gene-based filtering:")
    if (
        hasattr(adata_advanced, "var")
        and adata_advanced.var is not None
        and "highly_variable" in adata_advanced.var.columns
    ):
        hvg_mask = adata_advanced.var["highly_variable"]
        hvg_genes = adata_advanced[:, hvg_mask]
        print(f"   Highly variable genes: {hvg_genes.shape}")
    else:
        print("   (Highly variable genes not available yet)")

    print("\n3. Combined cell and gene filtering:")
    combined = adata_advanced[high_quality_mask, :100]
    print(f"   Combined filtering: {combined.shape}")

    print("\n4. Expression-based filtering:")
    # Get cells with high total counts
    high_counts_mask = adata_advanced.obs["total_counts"] > 2000
    high_counts_cells = adata_advanced[high_counts_mask, :]
    print(f"   High count cells: {high_counts_cells.shape}")
    return


@app.cell
def _(mo):
    mo.md(
        """
    ## 9. Lazy vs Eager Performance Comparison

    Let's compare the performance of lazy vs eager approaches:
    """
    )
    return


@app.cell
def _(adata, slaf_scanpy, time):
    # Performance comparison
    def compare_lazy_vs_eager_performance(adata):
        print("⚡ Lazy vs Eager Performance")
        print("=" * 35)

        # Test scenario: Apply transformations and slice
        print("Scenario: normalize_total → log1p → slice[:100, :50]")

        # Lazy approach
        print("\n1. Lazy approach:")
        adata_lazy = adata

        start_time = time.time()
        lazy_pipeline = slaf_scanpy.pp.normalize_total(
            adata_lazy, target_sum=1e4, inplace=False
        )
        lazy_pipeline = slaf_scanpy.pp.log1p(lazy_pipeline, inplace=False)
        lazy_slice = lazy_pipeline[:100, :50]
        lazy_build_time = time.time() - start_time

        start_time = time.time()
        # Use .X.compute() to avoid the metadata mismatch issue
        _ = lazy_slice.X.compute()
        lazy_compute_time = time.time() - start_time

        print(f"   Build time: {lazy_build_time:.4f}s")
        print(f"   Compute time: {lazy_compute_time:.4f}s")
        print(f"   Total time: {lazy_build_time + lazy_compute_time:.4f}s")

        # Eager approach (simulated)
        print("\n2. Eager approach (simulated):")

        start_time = time.time()
        _ = adata.compute()
        eager_load_time = time.time() - start_time

        # Simulate eager transformations (this would be done in memory)
        print(f"   Load time: {eager_load_time:.4f}s")
        print("   Transformations would be done in memory (slower for large datasets)")

        print("\n3. Key benefits:")
        print("   - Lazy: Build complex pipelines instantly")
        print("   - Lazy: Only compute what you need")
        print("   - Lazy: Memory efficient")
        print("   - Lazy: SQL-level performance for operations")

    compare_lazy_vs_eager_performance(adata)
    return


@app.cell
def _(mo):
    mo.md(
        """
    ## 10. Best Practices and Tips

    Here are some best practices for using SLAF's lazy evaluation:
    """
    )
    return


@app.cell
def _():
    # Best practices
    print("💡 Best Practices and Tips")
    print("=" * 30)

    print("1. Pipeline Building:")
    print("   ✅ Build complete pipelines before computing")
    print("   ✅ Chain transformations: adata.normalize_total().log1p()")
    print("   ✅ Slice after transformations for efficiency")

    print("\n2. Computation Control:")
    print("   ✅ Use .compute() only when you need the data")
    print("   ✅ Use .obs or .var for metadata")
    print("   ✅ Use .X.compute() for expression matrix only")

    print("\n3. Memory Management:")
    print("   ✅ Keep lazy objects for intermediate results")
    print("   ✅ Compute only final results")
    print("   ✅ Use slicing to reduce memory usage")

    print("\n4. Performance Optimization:")
    print("   ✅ Leverage SQL-level operations")
    print("   ✅ Use boolean indexing for filtering")
    print("   ✅ Combine operations in single queries when possible")

    print("\n5. Debugging:")
    print("   ✅ Check object types: type(adata)")
    print("   ✅ Check transformations: adata._transformations")
    print("   ✅ Use .info() for dataset overview")
    return


@app.cell
def _(mo):
    mo.md(
        """
    ## 11. Working with Layers

    SLAF supports AnnData layers (alternative expression matrices) stored in the `layers.lance` table.

    Let's create a synthetic dataset with layers, convert it to SLAF, and demonstrate layer operations:
    """
    )
    return


@app.cell
def _(csr_matrix, np):
    # Create synthetic AnnData with layers, obsm, varm, and uns for demonstration
    import scanpy as sc

    # Create a small synthetic dataset
    n_cells, n_genes = 200, 500
    np.random.seed(42)

    # Main expression matrix
    X = csr_matrix(np.random.lognormal(0, 1, (n_cells, n_genes)).astype(np.float32))

    # Create AnnData object
    adata_synthetic = sc.AnnData(X=X)

    # Add obs metadata
    adata_synthetic.obs["cell_type"] = np.random.choice(
        ["T_cell", "B_cell", "NK_cell"], n_cells
    )
    adata_synthetic.obs["total_counts"] = X.sum(axis=1).A1

    # Add var metadata
    adata_synthetic.var["highly_variable"] = np.random.choice(
        [True, False], n_genes, p=[0.2, 0.8]
    )

    # Add layers (alternative expression matrices)
    adata_synthetic.layers["spliced"] = csr_matrix(
        np.random.lognormal(0, 1, (n_cells, n_genes)).astype(np.float32)
    )
    adata_synthetic.layers["unspliced"] = csr_matrix(
        np.random.lognormal(0, 1, (n_cells, n_genes)).astype(np.float32)
    )

    # Add obsm (cell embeddings)
    adata_synthetic.obsm["X_umap"] = np.random.randn(n_cells, 2).astype(np.float32)
    adata_synthetic.obsm["X_pca"] = np.random.randn(n_cells, 50).astype(np.float32)

    # Add varm (gene embeddings)
    adata_synthetic.varm["PCs"] = np.random.randn(n_genes, 50).astype(np.float32)

    # Add uns (unstructured metadata)
    adata_synthetic.uns["neighbors"] = {
        "params": {"n_neighbors": 15, "metric": "euclidean"}
    }
    adata_synthetic.uns["pca"] = {
        "variance_ratio": np.random.rand(50).tolist(),
        "variance": np.random.rand(50).tolist(),
    }

    print(
        f"✅ Created synthetic AnnData: {adata_synthetic.shape[0]} cells × {adata_synthetic.shape[1]} genes"
    )
    print(f"   Layers: {list(adata_synthetic.layers.keys())}")
    print(f"   Obsm: {list(adata_synthetic.obsm.keys())}")
    print(f"   Varm: {list(adata_synthetic.varm.keys())}")
    print(f"   Uns: {list(adata_synthetic.uns.keys())}")
    return (adata_synthetic,)


@app.cell
def _(adata_synthetic):
    # Convert synthetic AnnData to SLAF format
    import os
    import tempfile

    from slaf.data.converter import SLAFConverter

    # Create temporary directory for SLAF dataset
    temp_dir = tempfile.mkdtemp()
    slaf_path = os.path.join(temp_dir, "synthetic_with_metadata.slaf")

    # Convert to SLAF (use chunked=False for small in-memory AnnData objects)
    converter = SLAFConverter(chunked=False)
    converter.convert_anndata(adata_synthetic, slaf_path)
    print(f"✅ Converted synthetic dataset to SLAF: {slaf_path}")
    return (slaf_path,)


@app.cell
def _(mo):
    mo.md(
        """
    ## 12. Working with Metadata Views

    SLAF provides lazy views for all AnnData metadata objects (obs, var, obsm, varm, uns):
    """
    )
    return


@app.cell
def _(read_slaf, slaf_path):
    # Load the converted SLAF dataset with all metadata
    adata_metadata = read_slaf(slaf_path)
    print(
        f"✅ Loaded SLAF dataset: {adata_metadata.shape[0]} cells × {adata_metadata.shape[1]} genes"
    )
    return (adata_metadata,)


@app.cell
def _(adata_metadata, csr_matrix):
    def demonstrate_layers():
        print("📚 Working with Layers")
        print("=" * 30)

        print("1. Check available layers:")
        available_layers = list(adata_metadata.layers.keys())
        print(f"   Available layers: {available_layers}")
        print(f"   Number of layers: {len(available_layers)}")

        print("\n2. Access a layer:")
        if available_layers:
            layer_name = available_layers[0]
            layer_matrix = adata_metadata.layers[layer_name]
            print(f"   Layer '{layer_name}' type: {type(layer_matrix)}")
            print(f"   Layer shape: {layer_matrix.shape}")
            print(f"   Layer is lazy: {hasattr(layer_matrix, 'compute')}")

        print("\n3. Compute a layer (lazy evaluation):")
        if available_layers:
            layer_name = available_layers[0]
            layer_matrix = adata_metadata.layers[layer_name]
            layer_data = layer_matrix.compute()
            print(
                f"   Computed '{layer_name}': {type(layer_data)}, shape {layer_data.shape}"
            )

        print("\n4. Create a new layer:")
        # Create a simple normalized layer as example
        normalized = adata_metadata.X.compute()
        # Simple normalization example
        normalized = normalized / normalized.sum(axis=1) * 10000
        adata_metadata.layers["normalized"] = csr_matrix(normalized)
        print("   Created 'normalized' layer")
        print(f"   Updated layers: {list(adata_metadata.layers.keys())}")

        print("\n5. Layer operations:")
        print("   - Dictionary-like interface: layers['name'], 'name' in layers")
        print("   - Lazy evaluation: layers['name'].compute()")
        print("   - Immutability: Converted layers are protected from deletion")
        print("   - Wide format: Stored in layers.lance table (one column per layer)")

    demonstrate_layers()
    return


@app.cell
def _(adata_metadata):
    def demonstrate_metadata_views():
        print("📊 Working with Metadata Views")
        print("=" * 35)

        print("1. Obs (cell metadata) - DataFrame-like interface:")
        print(f"   Available columns: {list(adata_metadata.obs.columns)[:5]}...")
        print(f"   Number of columns: {len(adata_metadata.obs.columns)}")
        print("   - DataFrame-like: obs.columns, obs.head(), obs['col']")
        print("   - Dict-like: 'col' in obs, len(obs)")
        print("   - Mutations: obs['new_col'] = values")

        print("\n2. Var (gene metadata) - DataFrame-like interface:")
        print(f"   Available columns: {list(adata_metadata.var.columns)[:5]}...")
        print(f"   Number of columns: {len(adata_metadata.var.columns)}")
        print("   - Same interface as obs")
        print("   - Mutations: var['new_col'] = values")

        print("\n3. Obsm (cell embeddings) - Dictionary-like interface:")
        obsm_keys = list(adata_metadata.obsm.keys())
        if obsm_keys:
            print(f"   Available keys: {obsm_keys}")
            for key in obsm_keys[:2]:  # Show first 2
                arr = adata_metadata.obsm[key]
                print(f"   - {key}: shape {arr.shape}, type {type(arr)}")
        else:
            print("   No obsm keys found (can add: adata.obsm['X_umap'] = coords)")

        print("\n4. Varm (gene embeddings) - Dictionary-like interface:")
        varm_keys = list(adata_metadata.varm.keys())
        if varm_keys:
            print(f"   Available keys: {varm_keys}")
            for key in varm_keys[:2]:  # Show first 2
                arr = adata_metadata.varm[key]
                print(f"   - {key}: shape {arr.shape}, type {type(arr)}")
        else:
            print("   No varm keys found (can add: adata.varm['PCs'] = loadings)")

        print("\n5. Uns (unstructured metadata) - Dictionary-like interface:")
        uns_keys = list(adata_metadata.uns.keys())
        if uns_keys:
            print(f"   Available keys: {uns_keys}")
            print(f"   Example: {list(uns_keys)[:3]}")
        else:
            print("   No uns keys found (can add: adata.uns['analysis'] = {...})")

        print("\n6. Key features:")
        print("   - Lazy evaluation: Metadata accessed on-demand")
        print("   - Immutability: Converted columns/keys are protected")
        print("   - Subsetting: Views respect cell/gene selectors")
        print("   - Persistence: Changes are saved to Lance tables")

    demonstrate_metadata_views()
    return


@app.cell
def _(adata_metadata, np):
    def demonstrate_metadata_mutations():
        print("✏️ Metadata Mutations Example")
        print("=" * 35)

        print("1. Create obs column:")
        cluster_labels = np.random.choice(["A", "B", "C"], adata_metadata.n_obs)
        adata_metadata.obs["clusters"] = cluster_labels
        print(
            f"   ✅ Created 'clusters' column with {len(np.unique(cluster_labels))} unique values"
        )
        print("   Column is immediately saved to cells.lance")

        print("\n2. Create var column:")
        hvg_flags = np.random.choice([True, False], adata_metadata.n_vars, p=[0.3, 0.7])
        adata_metadata.var["is_hvg"] = hvg_flags
        print(
            f"   ✅ Created 'is_hvg' column: {hvg_flags.sum()} genes marked as highly variable"
        )
        print("   Column is immediately saved to genes.lance")

        print("\n3. Store obsm (cell embeddings):")
        # Create a new embedding (different from existing X_umap)
        new_embedding = np.random.randn(adata_metadata.n_obs, 3).astype(np.float32)
        adata_metadata.obsm["X_custom"] = new_embedding
        print(f"   ✅ Created 'X_custom' embedding: shape {new_embedding.shape}")
        print("   Stored as FixedSizeListArray in cells.lance")
        print(f"   Available obsm keys: {list(adata_metadata.obsm.keys())}")

        print("\n4. Store varm (gene embeddings):")
        # Create a new gene embedding (different from existing PCs)
        new_loadings = np.random.randn(adata_metadata.n_vars, 20).astype(np.float32)
        adata_metadata.varm["custom_loadings"] = new_loadings
        print(f"   ✅ Created 'custom_loadings' embedding: shape {new_loadings.shape}")
        print("   Stored as FixedSizeListArray in genes.lance")
        print(f"   Available varm keys: {list(adata_metadata.varm.keys())}")

        print("\n5. Store uns (unstructured metadata):")
        adata_metadata.uns["custom_analysis"] = {
            "params": {"n_neighbors": 15, "resolution": 0.5}
        }
        print("   ✅ Created 'custom_analysis' key in uns")
        print("   Stored in uns.json file")
        print(f"   Available uns keys: {list(adata_metadata.uns.keys())}")

        print("\n6. Delete mutable columns/keys:")
        # Delete the columns/keys we just created (they're mutable)
        del adata_metadata.obs["clusters"]
        print("   ✅ Deleted 'clusters' column (mutable)")
        del adata_metadata.var["is_hvg"]
        print("   ✅ Deleted 'is_hvg' column (mutable)")
        del adata_metadata.obsm["X_custom"]
        print("   ✅ Deleted 'X_custom' embedding (mutable)")
        del adata_metadata.varm["custom_loadings"]
        print("   ✅ Deleted 'custom_loadings' embedding (mutable)")
        del adata_metadata.uns["custom_analysis"]
        print("   ✅ Deleted 'custom_analysis' key from uns")
        print(
            "\n   Note: Immutable columns/keys (converted from h5ad) cannot be deleted"
        )

    demonstrate_metadata_mutations()
    return


@app.cell
def _(mo):
    mo.md(
        """
    ## Summary

    **What you've learned about SLAF's lazy processing:**

    1. **Lazy Objects**: LazyAnnData and LazyExpressionMatrix store operations, not data
    2. **Explicit Control**: Use .compute() methods to control when data is processed
    3. **Slicing Patterns**: Multiple slicing patterns, all lazy and composable
    4. **Transformations**: Lazy transformations that are preserved through operations
    5. **Performance**: Build complex pipelines instantly, compute only when needed
    6. **Memory Efficiency**: Significant memory savings compared to eager loading
    7. **Best Practices**: Guidelines for optimal lazy evaluation usage
    8. **Layers**: Alternative expression matrices stored in layers.lance (wide format)
    9. **Metadata Views**: Full support for obs, var, obsm, varm, and uns with lazy evaluation

    **Next Steps:**
    - **03-ml-training-pipeline.py**: Complete ML training workflows with tokenizers and dataloaders
    """
    )
    return


if __name__ == "__main__":
    app.run()
