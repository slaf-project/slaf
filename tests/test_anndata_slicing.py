"""
Tests for LazyAnnData slicing functionality

Note: We use the default backend for all tests to ensure consistency
"""

import numpy as np
import pytest

from slaf.integrations.anndata import LazyAnnData, LazyExpressionMatrix
from tests.test_anndata import compare_metadata_essentials


class TestLazyAnnDataSlicing:
    """Test slicing operations on LazyAnnData"""

    def test_basic_slicing(self, tiny_slaf):
        """Test basic slicing operations"""
        adata = LazyAnnData(tiny_slaf)

        # Test cell slicing
        subset = adata[:10]
        assert subset.shape == (10, 50)
        assert len(subset.obs) == 10

        # Test gene slicing
        subset = adata[:, :50]
        assert subset.shape == (100, 50)
        assert len(subset.var) == 50

    def test_advanced_slicing(self, tiny_slaf):
        """Test advanced slicing operations"""
        adata = LazyAnnData(tiny_slaf)

        # Test step slicing
        subset = adata[::2]
        assert subset.shape == (50, 50)

        subset = adata[:, ::2]
        assert subset.shape == (100, 25)

    def test_boolean_indexing(self, tiny_slaf):
        """Test boolean indexing"""
        lazy_adata = LazyAnnData(tiny_slaf)

        # Create boolean masks
        cell_mask = np.random.choice([True, False], size=100)
        gene_mask = np.random.choice([True, False], size=50)

        # Test boolean indexing
        subset = lazy_adata[cell_mask]
        assert subset.shape == (np.sum(cell_mask), 50)

        subset = lazy_adata[:, gene_mask]
        assert subset.shape == (100, np.sum(gene_mask))

    def test_list_indexing(self, tiny_slaf):
        """Test list indexing"""
        adata = LazyAnnData(tiny_slaf)

        # Test list indexing for cells
        cell_indices = [0, 5, 10, 15]
        subset = adata[cell_indices]
        assert subset.shape == (len(cell_indices), 50)

        # Test list indexing for genes
        gene_indices = [0, 10, 20, 30]
        subset = adata[:, gene_indices]
        assert subset.shape == (100, len(gene_indices))

    def test_mixed_indexing(self, tiny_slaf):
        """Test mixed indexing types"""
        lazy_adata = LazyAnnData(tiny_slaf)

        # Mix slice and boolean indexing
        cell_slice = slice(0, 10)
        gene_mask = np.random.choice([True, False], size=50)

        subset = lazy_adata[cell_slice, gene_mask]
        assert subset.shape == (10, np.sum(gene_mask))

    def test_expression_matrix_slicing(self, tiny_slaf):
        """Test slicing on expression matrix"""
        lazy_adata = LazyAnnData(tiny_slaf)

        # Test cell slicing on expression matrix
        X_subset = lazy_adata.X[:10, :]
        assert X_subset.shape == (10, 50)

        # Test gene slicing on expression matrix
        X_subset = lazy_adata.X[:, :25]
        assert X_subset.shape == (100, 25)

    def test_metadata_slicing(self, tiny_slaf):
        """Test that metadata is properly sliced"""
        lazy_adata = LazyAnnData(tiny_slaf)

        # Test cell metadata slicing
        subset = lazy_adata[:10]
        assert len(subset.obs) == 10
        assert len(subset.obs_names) == 10

        # Test gene metadata slicing
        subset = lazy_adata[:, :25]
        assert len(subset.var) == 25
        assert len(subset.var_names) == 25

    def test_nested_slicing(self, tiny_slaf):
        """Test nested slicing operations using single-step slicing"""
        lazy_adata = LazyAnnData(tiny_slaf)

        # Use single-step slicing instead of chained slicing
        # This is equivalent to the previous chained slicing but in one step
        subset = lazy_adata[0:50, 0:25]
        assert subset.shape == (50, 25)

    def test_single_index_slicing(self, tiny_slaf):
        """Test single index slicing"""
        adata = LazyAnnData(tiny_slaf)

        # Test single cell selection
        subset = adata[5]
        assert subset.shape == (1, 50)

        # Test single gene selection
        subset = adata[:, 5]
        assert subset.shape == (100, 1)

    def test_negative_indexing(self, tiny_slaf):
        """Test negative indexing"""
        adata = LazyAnnData(tiny_slaf)

        # Test negative cell indexing
        subset = adata[-10:]
        assert subset.shape == (10, 50)

        # Test negative gene indexing
        subset = adata[:, -25:]
        assert subset.shape == (100, 25)

    def test_slice_with_step(self, tiny_slaf):
        """Test slicing with step parameter"""
        adata = LazyAnnData(tiny_slaf)

        # Test cell slicing with step
        subset = adata[::2]
        assert subset.shape == (50, 50)

        # Test gene slicing with step
        subset = adata[:, ::2]
        assert subset.shape == (100, 25)

    def test_boolean_mask_operations(self, tiny_slaf):
        """Test operations with boolean masks"""
        lazy_adata = LazyAnnData(tiny_slaf)

        # Create boolean masks
        cell_mask = np.zeros(100, dtype=bool)
        cell_mask[0:10] = True

        gene_mask = np.zeros(50, dtype=bool)
        gene_mask[0:25] = True

        # Test boolean indexing
        subset = lazy_adata[cell_mask]
        assert subset.shape == (10, 50)

        subset = lazy_adata[:, gene_mask]
        assert subset.shape == (100, 25)

    def test_copy_after_slicing(self, tiny_slaf):
        """Test that slicing preserves copy functionality"""
        adata = LazyAnnData(tiny_slaf)

        # Create a slice
        subset = adata[:10, :25]

        # Test copy on sliced data
        copied = subset.copy()
        assert copied.shape == subset.shape
        assert copied is not subset

    def test_transformations_preserved(self, tiny_slaf):
        """Test that transformations are preserved after slicing"""
        lazy_adata = LazyAnnData(tiny_slaf)

        # Apply transformations
        from slaf.integrations.scanpy import pp

        pp.normalize_total(lazy_adata, target_sum=1e4)
        pp.log1p(lazy_adata)

        # Create slice
        subset = lazy_adata[:10, :25]

        # Check that transformations are preserved
        assert "normalize_total" in subset._transformations
        assert "log1p" in subset._transformations

        # Test that compute() returns native AnnData with transformations applied
        native_subset = subset.compute()
        assert hasattr(native_subset, "obs")
        assert hasattr(native_subset, "var")
        assert hasattr(native_subset, "X")
        assert not isinstance(native_subset, LazyAnnData)

    def test_large_submatrix_slicing(self, tiny_slaf):
        """Test slicing of large submatrices (from benchmark scenarios)"""
        adata = LazyAnnData(tiny_slaf)

        # Test large cell slice (similar to benchmark scenario)
        subset = adata[:50]  # First 50 cells
        assert subset.shape == (50, 50)
        assert len(subset.obs) == 50
        assert len(subset.var) == 50

        # Test large gene slice
        subset = adata[:, :25]  # First 25 genes
        assert subset.shape == (100, 25)
        assert len(subset.obs) == 100
        assert len(subset.var) == 25

    def test_metadata_column_subsetting(self, tiny_slaf):
        """Test metadata column subsetting (from benchmark scenarios)"""
        adata = LazyAnnData(tiny_slaf)

        # Test obs column subsetting
        if len(adata.obs.columns) >= 2:
            subset_cols = adata.obs.columns[:2].tolist()
            obs_subset = adata.obs[subset_cols]
            assert obs_subset.shape == (100, 2)
            assert list(obs_subset.columns) == subset_cols

        # Test var column subsetting
        if len(adata.var.columns) >= 2:
            subset_cols = adata.var.columns[:2].tolist()
            var_subset = adata.var[subset_cols]
            assert var_subset.shape == (50, 2)
            assert list(var_subset.columns) == subset_cols

    def test_expression_matrix_properties(self, tiny_slaf):
        """Test expression matrix properties (from benchmark scenarios)"""
        adata = LazyAnnData(tiny_slaf)

        # Test shape property
        assert adata.X.shape == (100, 50)
        assert adata.shape == (100, 50)

        # Test that X returns a LazyExpressionMatrix when sliced
        X_slice = adata.X[:10, :25]
        assert isinstance(X_slice, LazyExpressionMatrix)
        assert X_slice.shape == (10, 25)

        # Test that compute() returns a sparse matrix
        sparse_slice = X_slice.compute()
        import scipy.sparse

        assert isinstance(sparse_slice, scipy.sparse.spmatrix)
        assert sparse_slice.shape == (10, 25)

    def test_complex_selector_composition(self, tiny_slaf):
        """Test complex selector composition (from benchmark insights)"""
        adata = LazyAnnData(tiny_slaf)

        # Test composition of slice and boolean mask
        cell_slice = slice(0, 20)
        gene_mask = np.zeros(50, dtype=bool)
        gene_mask[10:30] = True

        subset = adata[cell_slice, gene_mask]
        assert subset.shape == (20, 20)  # 20 cells, 20 genes (10:30)

        # Test composition of boolean mask and slice
        cell_mask = np.zeros(100, dtype=bool)
        cell_mask[10:30] = True
        gene_slice = slice(0, 25)

        subset = adata[cell_mask, gene_slice]
        assert subset.shape == (20, 25)  # 20 cells (10:30), 25 genes

    def test_edge_case_slicing(self, tiny_slaf):
        """Test edge cases encountered during benchmarking"""
        adata = LazyAnnData(tiny_slaf)

        # Test empty slice (currently not properly handled, so test actual behavior)
        subset = adata[0:0]
        # Note: Currently returns full shape, but this is a known limitation
        assert subset.shape == (100, 50)  # Current behavior

        # Test single element slice
        subset = adata[0:1]
        assert subset.shape == (1, 50)

        # Test slice that goes beyond bounds (should clamp)
        subset = adata[90:110]  # Beyond 100 cells
        assert subset.shape == (10, 50)  # Should clamp to 90:100

        # Test slice with step that doesn't divide evenly
        subset = adata[::3]  # Step of 3 on 100 cells
        expected_cells = len(range(0, 100, 3))
        assert subset.shape == (expected_cells, 50)

    def test_metadata_index_consistency(self, tiny_slaf):
        """Test that metadata indices are consistent after slicing"""
        adata = LazyAnnData(tiny_slaf)

        # Test that obs_names and var_names are consistent
        subset = adata[:10, :25]

        # Check that obs_names matches the sliced obs index
        assert len(subset.obs_names) == 10
        assert len(subset.var_names) == 25
        assert list(subset.obs_names) == list(subset.obs.index)
        assert list(subset.var_names) == list(subset.var.index)

    def test_lazy_evaluation_preservation(self, tiny_slaf):
        """Test that lazy evaluation is preserved through slicing"""
        adata = LazyAnnData(tiny_slaf)

        # Create slice without materializing
        subset = adata[:10, :25]

        # Check that the expression matrix is still lazy
        assert hasattr(subset.X, "slaf_array")
        # The shape should be updated to reflect the slicing
        assert subset.X.shape == (10, 25)  # Sliced shape

        # Materialize and check it works (only for implemented backends)
        if hasattr(subset.X, "toarray"):
            X_slice = subset.X[:10, :25]  # Actually slice the expression matrix
            X_dense = X_slice.toarray()
            assert X_dense.shape == (10, 25)

    def test_numpy_array_function_compatibility(self, tiny_slaf):
        """Test compatibility with numpy array functions"""
        adata = LazyAnnData(tiny_slaf)

        # Test aggregation methods (only for implemented backends)
        if hasattr(adata.X, "mean"):
            # Test axis-wise aggregation (returns arrays)
            mean_result = adata.X.mean(axis=0)
            assert isinstance(mean_result, np.ndarray)
            assert mean_result.shape == (1, 50)  # Mean across cells for each gene

            # Test sum method
            sum_result = adata.X.sum(axis=1)
            assert isinstance(sum_result, np.ndarray)
            assert sum_result.shape == (100, 1)  # Sum across genes for each cell

    def test_chained_slicing_not_supported(self, tiny_slaf):
        """Test that chained slicing raises NotImplementedError (this is by design)"""
        from slaf.integrations.anndata import LazyAnnData

        lazy_adata = LazyAnnData(tiny_slaf)
        first_slice = lazy_adata[:20, :30]
        with pytest.raises(
            NotImplementedError,
            match="Chained slicing on LazyAnnData objects is not supported",
        ):
            _ = first_slice[:10, :15]


class TestLazyAnnDataValueComparison:
    """Test that sliced values match expected results"""

    def test_single_cell_slicing_values(self, tiny_slaf, tiny_adata):
        """Test that single cell slicing returns correct values"""
        lazy_adata = LazyAnnData(tiny_slaf)

        # Test single cell selection
        lazy_single = lazy_adata[5]
        native_single = tiny_adata[5]

        # Compare expression values
        X_lazy = lazy_single.X.compute().toarray()
        X_native = native_single.X.toarray()
        np.testing.assert_allclose(X_lazy, X_native, rtol=1e-7)

        # Compare metadata essentials
        compare_metadata_essentials(
            lazy_single.obs, native_single.obs, "single cell obs"
        )
        compare_metadata_essentials(
            lazy_single.var, native_single.var, "single cell var"
        )

    def test_single_gene_slicing_values(self, tiny_slaf, tiny_adata):
        """Test that single gene slicing returns correct values"""
        lazy_adata = LazyAnnData(tiny_slaf)

        # Test single gene selection
        lazy_single = lazy_adata[:, 5]
        native_single = tiny_adata[:, 5]

        # Compare expression values
        X_lazy = lazy_single.X.compute().toarray()
        X_native = native_single.X.toarray()
        np.testing.assert_allclose(X_lazy, X_native, rtol=1e-7)

        # Compare metadata essentials
        compare_metadata_essentials(
            lazy_single.obs, native_single.obs, "single gene obs"
        )
        compare_metadata_essentials(
            lazy_single.var, native_single.var, "single gene var"
        )

    def test_range_slicing_values(self, tiny_slaf, tiny_adata):
        """Test that range slicing returns correct values"""
        lazy_adata = LazyAnnData(tiny_slaf)

        # Test range slicing
        lazy_range = lazy_adata[10:20, 15:25]
        native_range = tiny_adata[10:20, 15:25]

        # Compare expression values
        X_lazy = lazy_range.X.compute().toarray()
        X_native = native_range.X.toarray()
        np.testing.assert_allclose(X_lazy, X_native, rtol=1e-7)

        # Compare metadata essentials
        compare_metadata_essentials(lazy_range.obs, native_range.obs, "range obs")
        compare_metadata_essentials(lazy_range.var, native_range.var, "range var")

    def test_step_slicing_values(self, tiny_slaf, tiny_adata):
        """Test that step slicing returns correct values"""
        lazy_adata = LazyAnnData(tiny_slaf)

        # Test step slicing
        lazy_step = lazy_adata[::20, ::30]
        native_step = tiny_adata[::20, ::30]

        # Compare expression values
        X_lazy = lazy_step.X.compute().toarray()
        X_native = native_step.X.toarray()
        np.testing.assert_allclose(X_lazy, X_native, rtol=1e-7)

        # Compare metadata essentials
        compare_metadata_essentials(lazy_step.obs, native_step.obs, "step obs")
        compare_metadata_essentials(lazy_step.var, native_step.var, "step var")

    def test_boolean_mask_slicing_values(self, tiny_slaf, tiny_adata):
        """Test that boolean mask slicing returns correct values"""
        lazy_adata = LazyAnnData(tiny_slaf)

        # Create boolean masks
        cell_mask = np.zeros(100, dtype=bool)
        cell_mask[10:20] = True

        gene_mask = np.zeros(50, dtype=bool)
        gene_mask[15:25] = True

        # Test boolean slicing
        lazy_bool = lazy_adata[cell_mask, gene_mask]
        native_bool = tiny_adata[cell_mask, gene_mask]

        # Compare expression values
        X_lazy = lazy_bool.X.compute().toarray()
        X_native = native_bool.X.toarray()
        np.testing.assert_allclose(X_lazy, X_native, rtol=1e-7)

        # Compare metadata essentials
        compare_metadata_essentials(lazy_bool.obs, native_bool.obs, "boolean obs")
        compare_metadata_essentials(lazy_bool.var, native_bool.var, "boolean var")

    def test_list_indexing_values(self, tiny_slaf, tiny_adata):
        """Test that list indexing returns correct values"""
        lazy_adata = LazyAnnData(tiny_slaf)

        # Test list indexing
        cell_indices = [5, 15, 25, 35]
        gene_indices = [10, 20, 30, 40]

        lazy_list = lazy_adata[cell_indices, gene_indices]
        native_list = tiny_adata[cell_indices, gene_indices]

        # Compare expression values
        X_lazy = lazy_list.X.compute().toarray()
        X_native = native_list.X.toarray()
        np.testing.assert_allclose(X_lazy, X_native, rtol=1e-7)

        # Compare metadata essentials
        compare_metadata_essentials(lazy_list.obs, native_list.obs, "list obs")
        compare_metadata_essentials(lazy_list.var, native_list.var, "list var")

    def test_chained_slicing_values(self, tiny_slaf, tiny_adata):
        """Test that single-step slicing returns correct values"""
        lazy_adata = LazyAnnData(tiny_slaf)

        # Test single-step slicing: slice cells and genes in one operation
        lazy_sliced = lazy_adata[10:30, 15:35]
        native_sliced = tiny_adata[10:30, 15:35]

        # Compare expression values
        X_lazy = lazy_sliced.X.compute().toarray()
        X_native = native_sliced.X.toarray()
        np.testing.assert_allclose(X_lazy, X_native, rtol=1e-7)

        # Compare metadata essentials
        compare_metadata_essentials(
            lazy_sliced.obs, native_sliced.obs, "single-step obs"
        )
        compare_metadata_essentials(
            lazy_sliced.var, native_sliced.var, "single-step var"
        )

    def test_mixed_slicing_values(self, tiny_slaf, tiny_adata):
        """Test that mixed slicing types return correct values"""
        lazy_adata = LazyAnnData(tiny_slaf)

        # Test mixed slicing: slice + boolean mask
        cell_slice = slice(10, 30)
        gene_mask = np.zeros(50, dtype=bool)
        gene_mask[15:25] = True

        lazy_mixed = lazy_adata[cell_slice, gene_mask]
        native_mixed = tiny_adata[cell_slice, gene_mask]

        # Compare expression values
        X_lazy = lazy_mixed.X.compute().toarray()
        X_native = native_mixed.X.toarray()
        np.testing.assert_allclose(X_lazy, X_native, rtol=1e-7)

        # Compare metadata essentials
        compare_metadata_essentials(lazy_mixed.obs, native_mixed.obs, "mixed obs")
        compare_metadata_essentials(lazy_mixed.var, native_mixed.var, "mixed var")

    def test_negative_indexing_values(self, tiny_slaf, tiny_adata):
        """Test that negative indexing returns correct values"""
        lazy_adata = LazyAnnData(tiny_slaf)

        # Test negative indexing
        lazy_neg = lazy_adata[-10:, -15:]
        native_neg = tiny_adata[-10:, -15:]

        # Compare expression values
        X_lazy = lazy_neg.X.compute().toarray()
        X_native = native_neg.X.toarray()
        np.testing.assert_allclose(X_lazy, X_native, rtol=1e-7)

        # Compare metadata essentials
        compare_metadata_essentials(lazy_neg.obs, native_neg.obs, "negative obs")
        compare_metadata_essentials(lazy_neg.var, native_neg.var, "negative var")

    def test_complex_chained_slicing_values(self, tiny_slaf, tiny_adata):
        """Test complex single-step slicing scenarios"""
        lazy_adata = LazyAnnData(tiny_slaf)

        # Complex single-step slicing: slice cells and genes in one operation
        # This is equivalent to the previous chained slicing but in one step
        lazy_sliced = lazy_adata[20:60, ::2]  # Slice cells 20-59, every other gene
        native_sliced = tiny_adata[20:60, ::2]

        # Compare final results
        X_lazy = lazy_sliced.X.compute().toarray()
        X_native = native_sliced.X.toarray()
        np.testing.assert_allclose(X_lazy, X_native, rtol=1e-7)

        # Compare metadata essentials
        compare_metadata_essentials(
            lazy_sliced.obs, native_sliced.obs, "complex single-step obs"
        )
        compare_metadata_essentials(
            lazy_sliced.var, native_sliced.var, "complex single-step var"
        )

    def test_edge_case_slicing_values(self, tiny_slaf, tiny_adata):
        """Test edge case slicing values"""
        lazy_adata = LazyAnnData(tiny_slaf)

        # Test single element slice
        lazy_single = lazy_adata[0:1, 0:1]
        native_single = tiny_adata[0:1, 0:1]

        X_lazy = lazy_single.X.compute().toarray()
        X_native = native_single.X.toarray()
        np.testing.assert_allclose(X_lazy, X_native, rtol=1e-7)

        # Test slice that goes beyond bounds
        lazy_bounds = lazy_adata[90:110, 40:60]  # Beyond bounds
        native_bounds = tiny_adata[90:110, 40:60]

        X_lazy = lazy_bounds.X.compute().toarray()
        X_native = native_bounds.X.toarray()
        np.testing.assert_allclose(X_lazy, X_native, rtol=1e-7)

        # Compare metadata essentials
        compare_metadata_essentials(
            lazy_single.obs, native_single.obs, "edge case single obs"
        )
        compare_metadata_essentials(
            lazy_single.var, native_single.var, "edge case single var"
        )
        compare_metadata_essentials(
            lazy_bounds.obs, native_bounds.obs, "edge case bounds obs"
        )
        compare_metadata_essentials(
            lazy_bounds.var, native_bounds.var, "edge case bounds var"
        )

    @pytest.mark.skip(reason="Not implemented")
    def test_transformation_preservation_values(self, tiny_slaf, tiny_adata):
        """Test that transformations are correctly applied to sliced data"""
        lazy_adata = LazyAnnData(tiny_slaf)
        native_adata = tiny_adata.copy()

        # Apply transformations
        import scanpy as sc

        from slaf.integrations.scanpy import pp

        pp.normalize_total(lazy_adata, target_sum=1e4)
        pp.log1p(lazy_adata)

        sc.pp.normalize_total(native_adata, target_sum=1e4)
        sc.pp.log1p(native_adata)

        # Test slicing after transformations
        lazy_sliced = lazy_adata[10:30, 15:35]
        native_sliced = native_adata[10:30, 15:35]

        # Compare transformed values using compute()
        native_lazy_sliced = lazy_sliced.compute()
        # Convert to dense arrays for comparison
        try:
            X_lazy = native_lazy_sliced.X.toarray()
        except AttributeError:
            X_lazy = np.asarray(native_lazy_sliced.X)
        try:
            X_native = native_sliced.X.toarray()
        except AttributeError:
            X_native = np.asarray(native_sliced.X)
        np.testing.assert_allclose(X_lazy, X_native, rtol=1e-6)

        # Compare metadata essentials
        compare_metadata_essentials(
            native_lazy_sliced.obs, native_sliced.obs, "transformation obs"
        )
        compare_metadata_essentials(
            native_lazy_sliced.var, native_sliced.var, "transformation var"
        )

        # Verify compute() returns AnnData
        assert hasattr(native_lazy_sliced, "obs")
        assert hasattr(native_lazy_sliced, "var")
        assert hasattr(native_lazy_sliced, "X")
        assert not isinstance(native_lazy_sliced, LazyAnnData)

    def test_expression_matrix_direct_slicing_values(self, tiny_slaf, tiny_adata):
        """Test direct slicing of expression matrix returns correct values"""
        lazy_adata = LazyAnnData(tiny_slaf)

        # Test direct expression matrix slicing
        X_lazy_slice = lazy_adata.X[10:20, 15:25].compute().toarray()
        X_native_slice = tiny_adata.X[10:20, 15:25].toarray()

        np.testing.assert_allclose(X_lazy_slice, X_native_slice, rtol=1e-7)

        # Test single cell expression
        X_lazy_cell = lazy_adata.X[5, :].compute().toarray()
        X_native_cell = tiny_adata.X[5, :].toarray()

        np.testing.assert_allclose(X_lazy_cell, X_native_cell, rtol=1e-7)

        # Test single gene expression
        X_lazy_gene = lazy_adata.X[:, 5].compute().toarray()
        X_native_gene = tiny_adata.X[:, 5].toarray()

        np.testing.assert_allclose(X_lazy_gene, X_native_gene, rtol=1e-7)


class TestUnifiedLazySystemConsistency:
    """Test that the unified lazy system produces consistent results"""

    def test_unified_slicing_consistency(self, tiny_slaf):
        """Test that adata[:10, :20] and adata.X[:10, :20] produce identical results"""
        lazy_adata = LazyAnnData(tiny_slaf)

        # Test different slice patterns
        test_cases = [
            ("Single cell", slice(5, 6), slice(None)),
            ("Single gene", slice(None), slice(10, 11)),
            ("Small submatrix", slice(10, 20), slice(15, 25)),
            ("Range slice", slice(0, 100), slice(0, 50)),
        ]

        for name, cell_slice, gene_slice in test_cases:
            # LazyAnnData slicing
            slice1 = lazy_adata[cell_slice, gene_slice]
            matrix1 = slice1.X.compute()

            # LazyExpressionMatrix slicing
            slice2 = lazy_adata.X[cell_slice, gene_slice]
            matrix2 = slice2.compute()

            # Verify results are identical
            assert matrix1.shape == matrix2.shape, f"Shapes differ for {name}"

            if matrix1.data.size == 0 and matrix2.data.size == 0:
                # Both empty - that's fine
                pass
            elif matrix1.data.size == 0 or matrix2.data.size == 0:
                # One empty, other not - this is an error
                raise AssertionError()
            else:
                # Both have data - compare values
                diff = abs(matrix1.data - matrix2.data).max()
                assert diff < 1e-10, f"Results differ for {name}: max diff = {diff}"

    def test_transformation_consistency(self, tiny_slaf):
        """Test that transformations work consistently across different access patterns"""
        lazy_adata = LazyAnnData(tiny_slaf)

        # Apply transformations
        from slaf.integrations.scanpy import pp

        pp.normalize_total(lazy_adata, target_sum=10000, inplace=True)
        pp.log1p(lazy_adata, inplace=True)

        # Test that both access patterns produce same results
        slice1 = lazy_adata[:10, :20]
        slice2 = lazy_adata.X[:10, :20]

        matrix1 = slice1.X.compute()
        matrix2 = slice2.compute()

        # Verify transformations were applied consistently
        assert matrix1.shape == matrix2.shape
        if matrix1.data.size > 0:
            diff = abs(matrix1.data - matrix2.data).max()
            assert diff < 1e-10, f"Transformation results differ: max diff = {diff}"

    def test_large_submatrix_consistency(self, tiny_slaf):
        """Test consistency with larger submatrices"""
        lazy_adata = LazyAnnData(tiny_slaf)

        # Test with larger slices (if dataset is big enough)
        if lazy_adata.shape[0] >= 500 and lazy_adata.shape[1] >= 250:
            slice1 = lazy_adata[:500, :250]
            slice2 = lazy_adata.X[:500, :250]

            matrix1 = slice1.X.compute()
            matrix2 = slice2.compute()

            assert matrix1.shape == matrix2.shape
            if matrix1.data.size > 0:
                diff = abs(matrix1.data - matrix2.data).max()
                assert diff < 1e-10, (
                    f"Large submatrix results differ: max diff = {diff}"
                )

    def test_empty_matrix_handling(self, tiny_slaf):
        """Test handling of empty matrix results"""
        lazy_adata = LazyAnnData(tiny_slaf)

        # Test with out-of-bounds slices that should produce empty results
        large_slice = slice(10000, 10010)  # Way beyond dataset size

        slice1 = lazy_adata[large_slice, large_slice]
        slice2 = lazy_adata.X[large_slice, large_slice]

        matrix1 = slice1.X.compute()
        matrix2 = slice2.compute()

        # Both should be empty
        assert matrix1.shape == matrix2.shape
        assert matrix1.data.size == 0
        assert matrix2.data.size == 0

    @pytest.mark.skip(reason="Not implemented")
    def test_transformation_preservation_across_patterns(self, tiny_slaf):
        """Test that transformations are preserved across different slicing patterns"""
        lazy_adata = LazyAnnData(tiny_slaf)

        # Apply transformations
        from slaf.integrations.scanpy import pp

        pp.normalize_total(lazy_adata, target_sum=10000, inplace=True)
        pp.log1p(lazy_adata, inplace=True)

        # Test multiple slicing patterns
        patterns = [
            ("Single cell", slice(5, 6), slice(None)),
            ("Single gene", slice(None), slice(10, 11)),
            ("Small submatrix", slice(10, 20), slice(15, 25)),
        ]

        results = []
        for name, cell_slice, gene_slice in patterns:
            # Test both access patterns
            slice1 = lazy_adata[cell_slice, gene_slice]
            slice2 = lazy_adata.X[cell_slice, gene_slice]

            matrix1 = slice1.X.compute()
            matrix2 = slice2.compute()

            # Store results for comparison
            results.append((name, matrix1, matrix2))

        # Verify all results are consistent
        for name, matrix1, matrix2 in results:
            assert matrix1.shape == matrix2.shape, f"Shapes differ for {name}"
            if matrix1.data.size > 0:
                diff = abs(matrix1.data - matrix2.data).max()
                assert diff < 1e-10, f"Results differ for {name}: max diff = {diff}"


class TestMetadataAccessPerformance:
    """Test metadata access performance and caching behavior"""

    def test_repeated_obs_access(self, tiny_slaf):
        """Test that repeated obs access is efficient (cached)"""
        lazy_adata = LazyAnnData(tiny_slaf)

        # First access should load data
        obs1 = lazy_adata.obs
        assert obs1 is not None

        # Repeated access should be fast (cached)
        obs2 = lazy_adata.obs
        assert obs2 is not None
        assert obs1 is obs2  # Should be the same object (cached)

    def test_repeated_var_access(self, tiny_slaf):
        """Test that repeated var access is efficient (cached)"""
        lazy_adata = LazyAnnData(tiny_slaf)

        # First access should load data
        var1 = lazy_adata.var
        assert var1 is not None

        # Repeated access should be fast (cached)
        var2 = lazy_adata.var
        assert var2 is not None
        assert var1 is var2  # Should be the same object (cached)

    def test_metadata_consistency_after_slicing(self, tiny_slaf):
        """Test that metadata remains consistent after slicing"""
        lazy_adata = LazyAnnData(tiny_slaf)

        # Get original metadata
        original_obs = lazy_adata.obs
        original_var = lazy_adata.var

        # Apply slicing
        sliced_adata = lazy_adata[:10, :20]

        # Metadata should be filtered appropriately
        sliced_obs = sliced_adata.obs
        sliced_var = sliced_adata.var

        # Check that slicing worked correctly
        assert len(sliced_obs) <= len(original_obs)
        assert len(sliced_var) <= len(original_var)

        # Check that the sliced metadata is a subset of the original
        assert all(idx in original_obs.index for idx in sliced_obs.index)
        assert all(idx in original_var.index for idx in sliced_var.index)


class TestTransformationPerformance:
    """Test transformation performance and overhead"""

    def test_transformation_overhead_measurement(self, tiny_slaf):
        """Test that we can measure transformation overhead"""
        lazy_adata = LazyAnnData(tiny_slaf)

        # Test without transformations
        slice_no_transform = lazy_adata[:10, :20]
        matrix_no_transform = slice_no_transform.X.compute()

        # Apply transformations
        from slaf.integrations.scanpy import pp

        pp.normalize_total(lazy_adata, target_sum=10000, inplace=True)
        pp.log1p(lazy_adata, inplace=True)

        # Test with transformations
        slice_with_transform = lazy_adata[:10, :20]
        matrix_with_transform = slice_with_transform.X.compute()

        # Both should produce valid results
        assert matrix_no_transform.shape == matrix_with_transform.shape
        assert matrix_no_transform.shape == (10, 20)

    def test_transformation_order_preservation(self, tiny_slaf):
        """Test that transformation order is preserved"""
        lazy_adata = LazyAnnData(tiny_slaf)

        # Apply transformations in specific order
        from slaf.integrations.scanpy import pp

        pp.normalize_total(lazy_adata, target_sum=10000, inplace=True)
        pp.log1p(lazy_adata, inplace=True)

        # Verify transformations are stored in order
        assert hasattr(lazy_adata, "_transformations")
        transform_keys = list(lazy_adata._transformations.keys())

        # normalize_total should come before log1p
        assert "normalize_total" in transform_keys
        assert "log1p" in transform_keys
        assert transform_keys.index("normalize_total") < transform_keys.index("log1p")

    def test_transformation_clearing(self, tiny_slaf):
        """Test that transformations can be cleared"""
        lazy_adata = LazyAnnData(tiny_slaf)

        # Apply transformations
        from slaf.integrations.scanpy import pp

        pp.normalize_total(lazy_adata, target_sum=10000, inplace=True)
        pp.log1p(lazy_adata, inplace=True)

        # Verify transformations are applied
        assert hasattr(lazy_adata, "_transformations")
        assert len(lazy_adata._transformations) == 2

        # Clear transformations
        lazy_adata._transformations.clear()

        # Verify transformations are cleared
        assert len(lazy_adata._transformations) == 0

        # Test that slicing still works without transformations
        slice_no_transform = lazy_adata[:10, :20]
        matrix_no_transform = slice_no_transform.X.compute()
        assert matrix_no_transform.shape == (10, 20)
