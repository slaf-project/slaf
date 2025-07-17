import marimo

__generated_with = "0.14.6"
app = marimo.App(width="medium")


@app.cell
def _():
    import time

    import marimo as mo

    from slaf.integrations import scanpy as slaf_scanpy
    from slaf.integrations.anndata import read_slaf

    return mo, read_slaf, slaf_scanpy, time


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
def _(adata_fresh, slaf_scanpy):
    # Demonstrate new preprocessing functions
    print("🆕 New Preprocessing Functions")
    print("=" * 35)

    print("1. Scale transformation (z-score normalization):")
    adata_scaled = slaf_scanpy.pp.scale(
        adata_fresh, zero_center=True, max_value=10, inplace=False
    )
    print(f"   Scale applied: {type(adata_scaled)}")
    print(
        f"   Transformations: {list(adata_scaled._transformations.keys()) if hasattr(adata_scaled, '_transformations') else 'None'}"
    )

    print("\n2. Sample transformation (random sampling):")
    adata_sampled = slaf_scanpy.pp.sample(
        adata_fresh, n_obs=100, n_vars=50, random_state=42, inplace=False
    )
    print(f"   Sample applied: {type(adata_sampled)}")
    print(
        f"   Transformations: {list(adata_sampled._transformations.keys()) if hasattr(adata_sampled, '_transformations') else 'None'}"
    )

    print("\n3. Downsample counts transformation:")
    adata_downsampled = slaf_scanpy.pp.downsample_counts(
        adata_fresh, counts_per_cell=1000, random_state=42, inplace=False
    )
    print(f"   Downsample applied: {type(adata_downsampled)}")
    print(
        f"   Transformations: {list(adata_downsampled._transformations.keys()) if hasattr(adata_downsampled, '_transformations') else 'None'}"
    )

    print("\n4. Combined new transformations:")
    adata_combined = slaf_scanpy.pp.scale(adata_fresh, zero_center=True, inplace=False)
    adata_combined = slaf_scanpy.pp.sample(
        adata_combined, n_obs=50, n_vars=25, inplace=False
    )
    adata_combined = slaf_scanpy.pp.downsample_counts(
        adata_combined, counts_per_cell=500, inplace=False
    )
    print(f"   Combined transformations: {type(adata_combined)}")
    print(
        f"   All transformations: {list(adata_combined._transformations.keys()) if hasattr(adata_combined, '_transformations') else 'None'}"
    )

    return adata_scaled, adata_sampled, adata_downsampled, adata_combined


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
def _(adata_preserve, slaf_scanpy):
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
    import numpy as np

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
    ## Summary

    **What you've learned about SLAF's lazy processing:**

    1. **Lazy Objects**: LazyAnnData and LazyExpressionMatrix store operations, not data
    2. **Explicit Control**: Use .compute() methods to control when data is processed
    3. **Slicing Patterns**: Multiple slicing patterns, all lazy and composable
    4. **Transformations**: Lazy transformations that are preserved through operations
    5. **Performance**: Build complex pipelines instantly, compute only when needed
    6. **Memory Efficiency**: Significant memory savings compared to eager loading
    7. **Best Practices**: Guidelines for optimal lazy evaluation usage

    **Next Steps:**
    - **03-ml-training-pipeline.py**: Complete ML training workflows with tokenizers and dataloaders
    """
    )
    return


if __name__ == "__main__":
    app.run()
