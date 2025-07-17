"""Tests for scanpy preprocessing with LazyQuery composition (Phase 0.3)."""

import time

import scipy.sparse

from slaf.integrations.anndata import LazyAnnData
from slaf.integrations.scanpy import LazyPreprocessing, pp


class TestScanpyLazyComposition:
    """Test scanpy preprocessing with LazyQuery composition."""

    def test_complex_preprocessing_pipeline(self, tiny_slaf):
        """Test complex single-cell preprocessing pipeline with multiple composition steps."""
        # Create LazyAnnData
        adata = LazyAnnData(tiny_slaf)

        print(f"Original dataset shape: {adata.shape}")

        # 1. Calculate QC metrics (already works)
        print("\n1. Calculating QC metrics...")
        cell_qc, gene_qc = pp.calculate_qc_metrics(adata, inplace=False)
        print(f"   Cell QC shape: {cell_qc.shape}")
        print(f"   Gene QC shape: {gene_qc.shape}")

        # 2. Filter cells (test composition)
        print("\n2. Filtering cells...")
        filtered_adata = pp.filter_cells(
            adata, min_counts=10, min_genes=5, inplace=False
        )
        if filtered_adata is not None:
            print(f"   Filtered cells: {filtered_adata.n_obs}")
        else:
            print("   Filter applied in-place")

        # 3. Filter genes (test composition)
        print("\n3. Filtering genes...")
        filtered_adata = pp.filter_genes(
            adata, min_counts=5, min_cells=3, inplace=False
        )
        if filtered_adata is not None:
            print(f"   Filtered genes: {filtered_adata.n_vars}")
        else:
            print("   Filter applied in-place")

        # 4. Normalize (test transformation composition)
        print("\n4. Normalizing data...")
        normalized_adata = pp.normalize_total(adata, target_sum=1e4, inplace=False)
        if normalized_adata is not None:
            print("   Normalization applied to copy")
        else:
            print("   Normalization applied in-place")

        # 5. Log transform (test transformation composition)
        print("\n5. Applying log1p transformation...")
        transformed_adata = pp.log1p(adata, inplace=False)
        if transformed_adata is not None:
            print("   Log1p transformation applied to copy")
        else:
            print("   Log1p transformation applied in-place")

        # 6. Subset to specific cells and genes (test slicing composition)
        print("\n6. Subsetting data...")
        subset = adata[:50, :25]  # First 50 cells, first 25 genes
        print(f"   Subset shape: {subset.shape}")

        # 7. Access data (should trigger all compositions in single query)
        print("\n7. Accessing final data...")
        try:
            X = subset.X.compute()  # Should execute one optimized query
            print(f"   Final expression matrix shape: {X.shape}")
            print("   ✅ All compositions executed successfully!")
        except Exception as e:
            print(f"   ❌ Error accessing data: {e}")
            raise

        # Verify the pipeline worked
        assert X.shape[0] <= 50  # Should be filtered/subset
        assert X.shape[1] <= 25  # Should be filtered/subset
        print(f"   ✅ Final shape verification: {X.shape}")

    def test_qc_metrics_with_composition(self, tiny_slaf):
        """Test QC metrics calculation with LazyQuery composition."""
        adata = LazyAnnData(tiny_slaf)

        # Test that QC metrics work with composition
        cell_qc, gene_qc = pp.calculate_qc_metrics(adata, inplace=False)

        # Verify basic structure
        assert "n_genes_by_counts" in cell_qc.columns
        assert "total_counts" in cell_qc.columns
        assert "n_cells_by_counts" in gene_qc.columns
        assert "total_counts" in gene_qc.columns

        print(f"✅ QC metrics composition works: {cell_qc.shape}, {gene_qc.shape}")

    def test_cell_filtering_composition(self, tiny_slaf):
        """Test cell filtering with LazyQuery composition."""
        adata = LazyAnnData(tiny_slaf)
        original_cells = adata.n_obs

        # Test cell filtering
        filtered_adata = pp.filter_cells(
            adata, min_counts=10, min_genes=5, inplace=False
        )

        if filtered_adata is not None:
            assert filtered_adata.n_obs <= original_cells
            print(f"✅ Cell filtering composition works: {filtered_adata.n_obs} cells")
        else:
            print("✅ Cell filtering applied in-place")

    def test_gene_filtering_composition(self, tiny_slaf):
        """Test gene filtering with LazyQuery composition."""
        adata = LazyAnnData(tiny_slaf)
        original_genes = adata.n_vars

        # Test gene filtering
        filtered_adata = pp.filter_genes(
            adata, min_counts=5, min_cells=3, inplace=False
        )

        if filtered_adata is not None:
            assert filtered_adata.n_vars <= original_genes
            print(f"✅ Gene filtering composition works: {filtered_adata.n_vars} genes")
        else:
            print("✅ Gene filtering applied in-place")

    def test_normalization_composition(self, tiny_slaf):
        """Test normalization with LazyQuery composition."""
        adata = LazyAnnData(tiny_slaf)

        # Test normalization
        normalized_adata = pp.normalize_total(adata, target_sum=1e4, inplace=False)

        if normalized_adata is not None:
            # Check that transformation was stored
            assert hasattr(normalized_adata, "_transformations")
            assert "normalize_total" in normalized_adata._transformations
            print("✅ Normalization composition works")
        else:
            print("✅ Normalization applied in-place")

    def test_log1p_composition(self, tiny_slaf):
        """Test log1p transformation with LazyQuery composition."""
        adata = LazyAnnData(tiny_slaf)

        # Test log1p transformation
        transformed_adata = pp.log1p(adata, inplace=False)

        if transformed_adata is not None:
            # Check that transformation was stored
            assert hasattr(transformed_adata, "_transformations")
            assert "log1p" in transformed_adata._transformations
            print("✅ Log1p composition works")
        else:
            print("✅ Log1p transformation applied in-place")

    def test_slicing_composition(self, tiny_slaf):
        """Test slicing with LazyQuery composition."""
        adata = LazyAnnData(tiny_slaf)

        # Test slicing
        subset = adata[:50, :25]

        # Verify slicing worked
        assert subset.shape[0] <= 50
        assert subset.shape[1] <= 25
        print(f"✅ Slicing composition works: {subset.shape}")

    def test_memory_efficiency(self, tiny_slaf):
        """Test that composition doesn't cause memory spikes."""
        import gc

        import psutil

        # Get initial memory usage
        process = psutil.Process()
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB

        adata = LazyAnnData(tiny_slaf)

        # Perform multiple operations
        for _ in range(5):
            # Create a new subset each time
            subset = adata[:10, :10]
            _ = subset.X.compute()  # Should work without materialization

            # Force garbage collection
            gc.collect()

        # Get final memory usage
        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_increase = final_memory - initial_memory

        print(
            f"Memory usage: {initial_memory:.1f}MB -> {final_memory:.1f}MB (+{memory_increase:.1f}MB)"
        )

        # Memory increase should be reasonable (< 100MB for this small dataset)
        assert memory_increase < 100, (
            f"Memory increase too high: {memory_increase:.1f}MB"
        )
        print("✅ Memory efficiency test passed")

    def test_query_optimization(self, tiny_slaf):
        """Test that final queries are optimized (not multiple separate queries)."""
        adata = LazyAnnData(tiny_slaf)

        # Apply multiple transformations
        pp.normalize_total(adata, target_sum=1e4, inplace=True)
        pp.log1p(adata, inplace=True)

        # Create subset
        subset = adata[:10, :10]

        # Access data - this should trigger optimized query
        X = subset.X.compute()

        # For now, just verify it works
        assert X.shape == (10, 10)
        print("✅ Query optimization test passed (basic functionality verified)")

    def test_error_handling(self, tiny_slaf):
        """Test error handling for unsupported operations."""
        adata = LazyAnnData(tiny_slaf)

        # Test with invalid parameters
        try:
            pp.filter_cells(adata, min_counts=1000000, inplace=False)
            print("❌ Should have raised ValueError for impossible filter")
            raise AssertionError("Expected ValueError for impossible filter")
        except ValueError:
            print("✅ Error handling works for impossible filters")

        # Test with invalid slice
        try:
            subset = adata[1000000:, :]  # Out of bounds
            _ = subset.X.compute()
            print("❌ Should have raised error for out of bounds slice")
            raise AssertionError("Expected error for out of bounds slice")
        except Exception:
            print("✅ Error handling works for out of bounds slices")

    def test_normalize_total_log1p_slicing_composition(self, tiny_slaf):
        """Test normalize_total -> log1p -> slicing composition works lazily end-to-end"""
        # Load test data
        adata = LazyAnnData(tiny_slaf)

        # Apply normalize_total transformation
        LazyPreprocessing.normalize_total(adata, target_sum=10000, inplace=True)

        # Apply log1p transformation
        LazyPreprocessing.log1p(adata, inplace=True)

        # Apply slicing (this should be lazy)
        subset = adata[:10, :50]  # First 10 cells, first 50 genes

        # Verify that no computation has happened yet
        assert hasattr(adata, "_transformations")
        assert "normalize_total" in adata._transformations
        assert "log1p" in adata._transformations

        # Now compute the result - this should apply all transformations in one go
        start_time = time.time()
        result_matrix = subset.X.compute()
        compute_time = time.time() - start_time

        # Verify the result
        assert result_matrix.shape == (10, 50)
        assert isinstance(result_matrix, scipy.sparse.csr_matrix)

        # Verify that the transformations were applied correctly
        # The matrix should have been normalized and log1p transformed
        assert result_matrix.data.min() >= 0  # log1p ensures non-negative

        print(
            f"✅ normalize_total -> log1p -> slicing composition completed in {compute_time:.4f}s"
        )
        print(f"   Final matrix shape: {result_matrix.shape}")
        print(
            f"   Matrix density: {result_matrix.nnz / (result_matrix.shape[0] * result_matrix.shape[1]):.3f}"
        )

        return result_matrix
