"""
Tests for LazyAnnData functionality

Note: We use the default backend for all tests to ensure consistency
"""

import numpy as np
import pandas as pd
import pytest
import scanpy as sc

from slaf.integrations.anndata import LazyAnnData, LazyExpressionMatrix, read_slaf
from slaf.integrations.scanpy import pp

# Move align_dtypes to module level


def compare_metadata_essentials(lazy_df, native_df, description=""):
    """
    Compare essential metadata properties instead of entire DataFrames.

    This is much simpler and more robust than trying to align entire DataFrames.
    """
    # Check basic properties
    assert len(lazy_df) == len(native_df), f"{description}: Length mismatch"
    assert list(lazy_df.columns) == list(native_df.columns), (
        f"{description}: Column mismatch"
    )

    # Check index
    pd.testing.assert_index_equal(
        lazy_df.index, native_df.index, f"{description}: Index mismatch"
    )

    # For categorical columns, check categories and values
    for col in lazy_df.columns:
        if isinstance(native_df[col].dtype, pd.CategoricalDtype):
            # Check that both are categorical
            assert isinstance(lazy_df[col].dtype, pd.CategoricalDtype), (
                f"{description}: Column {col} should be categorical"
            )
            # Check categories match
            pd.testing.assert_index_equal(
                lazy_df[col].cat.categories,
                native_df[col].cat.categories,
                f"{description}: Categories mismatch for column {col}",
            )
            # Check values match (ignoring unused categories)
            assert list(lazy_df[col]) == list(native_df[col]), (
                f"{description}: Values mismatch for categorical column {col}"
            )
        else:
            # For non-categorical columns, check values
            np.testing.assert_array_equal(
                lazy_df[col].values,
                native_df[col].values,
                f"{description}: Values mismatch for column {col}",
            )


class TestLazyAnnDataCorrectness:
    """Test that LazyAnnData provides correct results compared to native scanpy"""

    def test_basic_properties(self, tiny_slaf, tiny_adata):
        """Test basic properties of LazyAnnData match native AnnData"""
        lazy_adata = LazyAnnData(tiny_slaf)

        # Test shape matches
        assert lazy_adata.shape == tiny_adata.shape
        assert lazy_adata.n_obs == tiny_adata.n_obs
        assert lazy_adata.n_vars == tiny_adata.n_vars

        # Test that X is accessible
        assert hasattr(lazy_adata, "X")
        assert lazy_adata.X.shape == tiny_adata.X.shape

    def test_expression_matrix_access(self, tiny_slaf, tiny_adata):
        """Test that expression matrix data matches native AnnData"""
        lazy_adata = LazyAnnData(tiny_slaf)

        # Test full matrix access
        X_lazy = lazy_adata.X.compute().toarray()
        X_native = tiny_adata.X.toarray()

        np.testing.assert_allclose(X_lazy, X_native, rtol=1e-7)

        # Test that it's a LazyExpressionMatrix object
        assert isinstance(lazy_adata.X, LazyExpressionMatrix)

    def test_metadata_access(self, tiny_slaf, tiny_adata):
        """Test that cell and gene metadata matches native AnnData"""
        lazy_adata = LazyAnnData(tiny_slaf)

        # Compare metadata essentials
        compare_metadata_essentials(lazy_adata.obs, tiny_adata.obs, "obs")
        compare_metadata_essentials(lazy_adata.var, tiny_adata.var, "var")

        # Test obs_names and var_names match
        pd.testing.assert_index_equal(lazy_adata.obs_names, tiny_adata.obs_names)
        pd.testing.assert_index_equal(lazy_adata.var_names, tiny_adata.var_names)

    def test_aggregation_operations(self, tiny_slaf, tiny_adata):
        """Test that aggregation operations match native AnnData"""
        lazy_adata = LazyAnnData(tiny_slaf)

        # Test mean aggregation
        gene_means_lazy = lazy_adata.X.mean(axis=0)
        gene_means_native = tiny_adata.X.mean(axis=0)
        np.testing.assert_allclose(
            gene_means_lazy, gene_means_native, rtol=1e-6, atol=1e-8
        )

        cell_means_lazy = lazy_adata.X.mean(axis=1)
        cell_means_native = tiny_adata.X.mean(axis=1)
        np.testing.assert_allclose(
            cell_means_lazy, cell_means_native, rtol=1e-6, atol=1e-8
        )

        # Test sum aggregation
        gene_sums_lazy = lazy_adata.X.sum(axis=0)
        gene_sums_native = tiny_adata.X.sum(axis=0)
        np.testing.assert_allclose(
            gene_sums_lazy, gene_sums_native, rtol=1e-6, atol=1e-8
        )

        cell_sums_lazy = lazy_adata.X.sum(axis=1)
        cell_sums_native = tiny_adata.X.sum(axis=1)
        np.testing.assert_allclose(
            cell_sums_lazy, cell_sums_native, rtol=1e-6, atol=1e-8
        )

    def test_copy_operation(self, tiny_slaf, tiny_adata):
        """Test that copy operation preserves data correctly"""
        lazy_adata = LazyAnnData(tiny_slaf)

        # Test copy
        copied = lazy_adata.copy()
        assert copied.shape == lazy_adata.shape
        assert copied is not lazy_adata

        # Test that copy has same data as original
        np.testing.assert_allclose(
            lazy_adata.X.compute().toarray(), copied.X.compute().toarray(), rtol=1e-7
        )

        # Test that copy matches native AnnData copy
        native_copy = tiny_adata.copy()
        np.testing.assert_allclose(
            copied.X.compute().toarray(), native_copy.X.toarray(), rtol=1e-7
        )


class TestLazyAnnDataBackends:
    """Test different backend options"""

    def test_scipy_backend(self, tiny_slaf):
        """Test scipy backend"""
        lazy_adata = LazyAnnData(tiny_slaf, backend="scipy")
        assert lazy_adata.shape == (100, 50)

    def test_auto_backend(self, tiny_slaf):
        """Test auto backend selection"""
        lazy_adata = LazyAnnData(tiny_slaf, backend="auto")
        assert lazy_adata.shape == (100, 50)

    def test_invalid_backend(self, tiny_slaf):
        """Test that invalid backend raises error"""
        with pytest.raises(ValueError, match="Unknown backend"):
            LazyAnnData(tiny_slaf, backend="invalid")


class TestLazyAnnDataTransformations:
    """Test transformation operations"""

    def test_normalize_total(self, tiny_slaf, tiny_adata):
        """Test normalize_total transformation matches native scanpy"""
        lazy_adata = LazyAnnData(tiny_slaf)
        native_adata = tiny_adata.copy()

        # Apply normalization to both
        pp.normalize_total(lazy_adata, target_sum=1e4)
        sc.pp.normalize_total(native_adata, target_sum=1e4)

        # Check that transformation was applied
        assert "normalize_total" in lazy_adata._transformations

        # Compare results - compute() returns native AnnData
        native_lazy = lazy_adata.compute()
        # Convert to dense arrays for comparison
        try:
            X_lazy = native_lazy.X.toarray()
        except AttributeError:
            X_lazy = np.asarray(native_lazy.X)
        try:
            X_native = native_adata.X.toarray()
        except AttributeError:
            X_native = np.asarray(native_adata.X)
        np.testing.assert_allclose(X_lazy, X_native, rtol=1e-6)

        # Verify compute() returns AnnData
        assert hasattr(native_lazy, "obs")
        assert hasattr(native_lazy, "var")
        assert hasattr(native_lazy, "X")

    def test_log1p(self, tiny_slaf, tiny_adata):
        """Test log1p transformation matches native scanpy"""
        lazy_adata = LazyAnnData(tiny_slaf)
        native_adata = tiny_adata.copy()

        # Apply log1p to both
        pp.log1p(lazy_adata)
        sc.pp.log1p(native_adata)

        # Check that transformation was applied
        assert "log1p" in lazy_adata._transformations

        # Compare results - compute() returns native AnnData
        native_lazy = lazy_adata.compute()
        # Convert to dense arrays for comparison
        try:
            X_lazy = native_lazy.X.toarray()
        except AttributeError:
            X_lazy = np.asarray(native_lazy.X)
        try:
            X_native = native_adata.X.toarray()
        except AttributeError:
            X_native = np.asarray(native_adata.X)
        np.testing.assert_allclose(X_lazy, X_native, rtol=1e-6)

        # Verify compute() returns AnnData
        assert hasattr(native_lazy, "obs")
        assert hasattr(native_lazy, "var")
        assert hasattr(native_lazy, "X")

    def test_multiple_transformations(self, tiny_slaf, tiny_adata):
        """Test multiple transformations match native scanpy"""
        lazy_adata = LazyAnnData(tiny_slaf)
        native_adata = tiny_adata.copy()

        # Apply multiple transformations to both
        pp.normalize_total(lazy_adata, target_sum=1e4)
        pp.log1p(lazy_adata)

        sc.pp.normalize_total(native_adata, target_sum=1e4)
        sc.pp.log1p(native_adata)

        # Check that both transformations were applied
        assert "normalize_total" in lazy_adata._transformations
        assert "log1p" in lazy_adata._transformations

        # Compare results - compute() returns native AnnData
        native_lazy = lazy_adata.compute()
        # Convert to dense arrays for comparison
        try:
            X_lazy = native_lazy.X.toarray()
        except AttributeError:
            X_lazy = np.asarray(native_lazy.X)
        try:
            X_native = native_adata.X.toarray()
        except AttributeError:
            X_native = np.asarray(native_adata.X)
        np.testing.assert_allclose(X_lazy, X_native, rtol=1e-7)

        # Verify compute() returns AnnData
        assert hasattr(native_lazy, "obs")
        assert hasattr(native_lazy, "var")
        assert hasattr(native_lazy, "X")


class TestComputeInterface:
    """Test the new compute() interface"""

    def test_compute_returns_ann_data(self, tiny_slaf):
        """Test that compute() returns native AnnData object"""
        lazy_adata = LazyAnnData(tiny_slaf)
        native_adata = lazy_adata.compute()

        # Verify it's a native AnnData object
        assert hasattr(native_adata, "obs")
        assert hasattr(native_adata, "var")
        assert hasattr(native_adata, "X")
        assert hasattr(native_adata, "shape")

        # Verify it's not a LazyAnnData
        assert not isinstance(native_adata, LazyAnnData)

    def test_x_compute_returns_sparse_matrix(self, tiny_slaf):
        """Test that X.compute() returns scipy.sparse.csr_matrix"""
        lazy_adata = LazyAnnData(tiny_slaf)
        sparse_matrix = lazy_adata.X.compute()

        # Verify it's a sparse matrix
        assert hasattr(sparse_matrix, "toarray")
        assert hasattr(sparse_matrix, "shape")
        assert hasattr(sparse_matrix, "data")

        # Verify it's not a LazyExpressionMatrix
        assert not isinstance(sparse_matrix, LazyExpressionMatrix)


class TestReadSlafFunction:
    """Test read_slaf function"""

    def test_read_slaf_basic(self, tiny_slaf_path):
        """Test basic read_slaf functionality"""
        adata = read_slaf(tiny_slaf_path)
        assert isinstance(adata, LazyAnnData)
        assert adata.shape == (100, 50)

    def test_read_slaf_with_backend(self, tiny_slaf_path):
        """Test read_slaf with backend parameter"""
        adata = read_slaf(tiny_slaf_path, backend="scipy")
        assert isinstance(adata, LazyAnnData)
        assert adata.shape == (100, 50)
