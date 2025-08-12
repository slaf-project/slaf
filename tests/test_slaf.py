import threading
import time

import polars as pl
import pytest

from slaf.core.slaf import SLAFArray


class TestSLAFArray:
    """Test SLAFArray functionality"""

    def test_slaf_array_initialization(self, small_slaf):
        """Test that SLAFArray initializes correctly"""
        assert isinstance(small_slaf, SLAFArray)
        assert small_slaf.shape == (10, 5)
        assert small_slaf.config is not None

    def test_info_method(self, small_slaf):
        """Test the info method"""
        # This should not raise any exceptions
        small_slaf.info()

    def test_lazy_import_performance(self):
        """Test that lazy imports work correctly and are fast"""
        # Test core import is fast
        start_time = time.time()
        from slaf import SLAFArray

        import_time = time.time() - start_time

        # Core import should be fast (< 1 second)
        assert import_time < 1.0
        assert SLAFArray is not None

    def test_lazy_import_functions(self):
        """Test that lazy import functions work correctly"""
        from slaf import get_converter, get_integrations, get_ml_components

        # Test converter lazy import
        start_time = time.time()
        converter = get_converter()
        converter_time = time.time() - start_time

        # Should work and not be too slow
        assert converter is not None
        assert converter_time < 5.0  # Allow some time for heavy imports

        # Test integrations lazy import
        start_time = time.time()
        LazyAnnData, LazyExpressionMatrix, pp = get_integrations()
        integrations_time = time.time() - start_time

        # Should work and be relatively fast
        assert LazyAnnData is not None
        assert LazyExpressionMatrix is not None
        assert pp is not None
        assert integrations_time < 2.0

        # Test ML components lazy import
        start_time = time.time()
        SLAFDataLoader, SLAFTokenizer, TokenizerType, create_shuffle, create_window = (
            get_ml_components()
        )
        ml_time = time.time() - start_time

        # Should work and not be too slow
        assert SLAFDataLoader is not None
        assert SLAFTokenizer is not None
        assert TokenizerType is not None
        assert create_shuffle is not None
        assert create_window is not None
        assert ml_time < 5.0  # Allow some time for heavy imports

    def test_async_metadata_loading(self, small_slaf):
        """Test that async metadata loading works correctly"""
        # Check initial state
        assert hasattr(small_slaf, "_metadata_loaded")
        assert hasattr(small_slaf, "_metadata_loading")
        assert hasattr(small_slaf, "_metadata_thread")

        # Check metadata status methods exist
        assert hasattr(small_slaf, "is_metadata_ready")
        assert hasattr(small_slaf, "is_metadata_loading")
        assert hasattr(small_slaf, "wait_for_metadata")

        # Test status methods
        assert isinstance(small_slaf.is_metadata_ready(), bool)
        assert isinstance(small_slaf.is_metadata_loading(), bool)

    def test_immediate_capabilities(self, small_slaf):
        """Test that immediate capabilities work without metadata loading"""
        # Shape should be available immediately
        assert small_slaf.shape == (10, 5)

        # Config should be available immediately
        assert small_slaf.config is not None
        assert "array_shape" in small_slaf.config

        # SQL queries should work immediately
        result = small_slaf.query("SELECT COUNT(*) as count FROM cells")
        assert isinstance(result, pl.DataFrame)
        assert "count" in result.columns
        assert result.item(0, "count") == 10

    def test_lazy_metadata_access(self, small_slaf):
        """Test that metadata access triggers lazy loading"""
        # Initially metadata might not be loaded
        # Note: We don't use initial_ready since it might be True if metadata was already loaded

        # Access metadata (should trigger loading if not already loaded)
        obs_columns = small_slaf.obs.columns
        var_columns = small_slaf.var.columns

        # Should have metadata after access
        assert len(obs_columns) > 0
        assert len(var_columns) > 0
        assert small_slaf.is_metadata_ready()

        # Subsequent access should be instant
        start_time = time.time()
        _ = small_slaf.obs.columns  # Access again to test caching
        access_time = time.time() - start_time

        # Should be very fast (cached)
        assert access_time < 0.1

    def test_metadata_loading_thread_safety(self, small_slaf):
        """Test that metadata loading is thread-safe"""

        # Access metadata from multiple threads
        def access_metadata():
            return small_slaf.obs.columns, small_slaf.var.columns

        # Create multiple threads
        threads = []
        results = []

        for _ in range(3):
            thread = threading.Thread(target=lambda: results.append(access_metadata()))
            threads.append(thread)
            thread.start()

        # Wait for all threads to complete
        for thread in threads:
            thread.join()

        # All threads should get the same result
        assert len(results) == 3
        for obs_cols, var_cols in results:
            assert len(obs_cols) > 0
            assert len(var_cols) > 0

    def test_wait_for_metadata(self, small_slaf):
        """Test the wait_for_metadata method"""
        # If metadata is already loaded, should return immediately
        if small_slaf.is_metadata_ready():
            start_time = time.time()
            small_slaf.wait_for_metadata()
            wait_time = time.time() - start_time
            assert wait_time < 0.1  # Should be very fast if already loaded

        # Test with timeout
        start_time = time.time()
        small_slaf.wait_for_metadata(timeout=1.0)
        wait_time = time.time() - start_time
        assert wait_time < 1.1  # Should not exceed timeout significantly

    def test_metadata_properties(self, small_slaf):
        """Test that obs and var properties work correctly"""
        # Test obs property
        obs = small_slaf.obs
        assert isinstance(obs, pl.DataFrame)
        assert len(obs) == 10  # All cells
        assert "cell_id" in obs.columns

        # Test var property
        var = small_slaf.var
        assert isinstance(var, pl.DataFrame)
        assert len(var) == 5  # All genes
        assert "gene_id" in var.columns

    def test_metadata_loading_optimization(self, small_slaf):
        """Test that metadata loading uses optimized paths"""
        # Trigger metadata loading first
        _ = small_slaf.obs.columns
        _ = small_slaf.var.columns

        # Check that column caches are available
        assert hasattr(small_slaf, "_obs_columns")
        assert hasattr(small_slaf, "_var_columns")

    def test_initialization_performance(self, tmp_path):
        """Test that initialization is fast"""
        # Create a minimal test dataset
        import json

        import lance

        # Create test data
        test_dir = tmp_path / "test_slaf"
        test_dir.mkdir()

        # Create config
        config = {
            "array_shape": [100, 50],
            "tables": {
                "cells": "cells.lance",
                "genes": "genes.lance",
                "expression": "expression.lance",
            },
        }

        with open(test_dir / "config.json", "w") as f:
            json.dump(config, f)

        # Create minimal Lance datasets
        cells_data = {
            "cell_id": [f"cell_{i}" for i in range(100)],
            "cell_integer_id": list(range(100)),
        }
        genes_data = {
            "gene_id": [f"gene_{i}" for i in range(50)],
            "gene_integer_id": list(range(50)),
        }

        lance.write_dataset(cells_data, test_dir / "cells.lance")
        lance.write_dataset(genes_data, test_dir / "genes.lance")

        # Create empty expression dataset with proper schema
        import pyarrow as pa

        expression_schema = pa.schema(
            [
                pa.field("cell_integer_id", pa.int32()),
                pa.field("gene_integer_id", pa.int32()),
                pa.field("value", pa.float32()),
            ]
        )
        lance.write_dataset([], test_dir / "expression.lance", schema=expression_schema)

        # Test initialization performance
        start_time = time.time()
        slaf_array = SLAFArray(str(test_dir))
        init_time = time.time() - start_time

        # Should be very fast (< 1 second)
        assert init_time < 1.0
        assert slaf_array.shape == (100, 50)

    def test_filter_cells_basic(self, small_slaf):
        """Test basic cell filtering"""
        # Filter cells by cell type
        result = small_slaf.filter_cells(cell_type="A")

        # Check that we got a polars DataFrame
        assert isinstance(result, pl.DataFrame)

        # Check that it has the expected columns
        assert "cell_id" in result.columns
        assert "cell_type" in result.columns
        assert "total_counts" in result.columns

        # Check that all cells have the expected cell type
        if len(result) > 0:
            assert all(result["cell_type"] == "A")

    def test_filter_cells_range_queries(self, small_slaf):
        """Test cell filtering with range queries"""
        # Filter cells with total counts > 5
        result = small_slaf.filter_cells(total_counts=">5")

        # Check that we got a polars DataFrame
        assert isinstance(result, pl.DataFrame)

        # Check that it has the expected columns
        assert "cell_id" in result.columns
        assert "cell_type" in result.columns
        assert "total_counts" in result.columns

        # Check that all cells have total counts > 5
        if len(result) > 0:
            assert all(result["total_counts"] > 5)

        # Test multiple range conditions
        result2 = small_slaf.filter_cells(total_counts=">=3")

        # Check that we got a polars DataFrame
        assert isinstance(result2, pl.DataFrame)

        # Check that all cells have total counts in range
        if len(result2) > 0:
            assert all(result2["total_counts"] >= 3)

    def test_filter_cells_list_values(self, small_slaf):
        """Test cell filtering with list values"""
        # Filter cells by multiple cell types
        result = small_slaf.filter_cells(cell_type=["A", "B"])

        # Check that we got a polars DataFrame
        assert isinstance(result, pl.DataFrame)

        # Check that all cells have one of the expected cell types
        if len(result) > 0:
            assert all(result["cell_type"].is_in(["A", "B"]))

    def test_filter_cells_no_filters(self, small_slaf):
        """Test cell filtering with no filters (should return all cells)"""
        result = small_slaf.filter_cells()

        # Check that we got a polars DataFrame
        assert isinstance(result, pl.DataFrame)
        assert len(result) == 10  # All cells

    def test_filter_genes_basic(self, small_slaf):
        """Test basic gene filtering"""
        # Filter genes by gene type
        result = small_slaf.filter_genes(gene_type="protein_coding")

        # Check that we got a polars DataFrame
        assert isinstance(result, pl.DataFrame)

        # Check that it has the expected columns
        assert "gene_id" in result.columns
        assert "gene_type" in result.columns
        assert "highly_variable" in result.columns

        # Check that all genes have the expected gene type
        if len(result) > 0:
            assert all(result["gene_type"] == "protein_coding")

    def test_filter_genes_range_queries(self, small_slaf):
        """Test gene filtering with range queries"""
        # Filter genes by expression mean
        result = small_slaf.filter_genes(expression_mean=">0.5")

        # Check that we got a polars DataFrame
        assert isinstance(result, pl.DataFrame)

        # Check that it has the expected columns
        assert "gene_id" in result.columns
        assert "gene_type" in result.columns
        assert "expression_mean" in result.columns

        # Check that all genes have expression mean > 0.5
        if len(result) > 0:
            assert all(result["expression_mean"] > 0.5)

        # Test multiple range conditions
        result2 = small_slaf.filter_genes(expression_mean=">=0.1")

        # Check that we got a polars DataFrame
        assert isinstance(result2, pl.DataFrame)

        # Check that all genes have expression mean in range
        if len(result2) > 0:
            assert all(result2["expression_mean"] >= 0.1)

    def test_filter_genes_no_filters(self, small_slaf):
        """Test gene filtering with no filters (should return all genes)"""
        result = small_slaf.filter_genes()

        # Check that we got a polars DataFrame
        assert isinstance(result, pl.DataFrame)
        assert len(result) == 5  # All genes

    def test_get_cell_expression(self, small_slaf):
        """Test getting expression data for specific cells"""
        # Wait for metadata to be loaded
        small_slaf.wait_for_metadata()

        # Get expression for first cell
        cell_id = small_slaf.obs["cell_id"][0]
        result = small_slaf.get_cell_expression(cell_id)

        # Check that we got a polars DataFrame
        assert isinstance(result, pl.DataFrame)

        # Check that it has the expected columns
        assert "cell_id" in result.columns
        assert "gene_id" in result.columns
        assert "value" in result.columns

        # Check that all rows have the expected cell ID
        if len(result) > 0:
            assert all(result["cell_id"] == cell_id)

    def test_get_gene_expression(self, small_slaf):
        """Test getting expression data for specific genes"""
        # Wait for metadata to be loaded
        small_slaf.wait_for_metadata()

        # Get expression for first gene
        gene_id = small_slaf.var["gene_id"][0]
        result = small_slaf.get_gene_expression(gene_id)

        # Check that we got a polars DataFrame
        assert isinstance(result, pl.DataFrame)

        # Check that it has the expected columns
        assert "cell_id" in result.columns
        assert "gene_id" in result.columns
        assert "value" in result.columns

        # Check that all rows have the expected gene ID
        if len(result) > 0:
            assert all(result["gene_id"] == gene_id)

    def test_get_cell_expression_multiple(self, small_slaf):
        """Test getting expression data for multiple cells"""
        # Wait for metadata to be loaded
        small_slaf.wait_for_metadata()

        # Get expression for first two cells
        cell_ids = small_slaf.obs["cell_id"][:2].to_list()
        result = small_slaf.get_cell_expression(cell_ids)

        # Check that we got a polars DataFrame
        assert isinstance(result, pl.DataFrame)

        # Check that it has the expected columns
        assert "cell_id" in result.columns
        assert "gene_id" in result.columns
        assert "value" in result.columns

        # Check that all rows have the expected cell IDs
        if len(result) > 0:
            assert all(cid in cell_ids for cid in result["cell_id"].to_list())

    def test_get_gene_expression_multiple(self, small_slaf):
        """Test getting expression data for multiple genes"""
        # Wait for metadata to be loaded
        small_slaf.wait_for_metadata()

        # Get expression for first two genes
        gene_ids = small_slaf.var["gene_id"][:2].to_list()
        result = small_slaf.get_gene_expression(gene_ids)

        # Check that we got a polars DataFrame
        assert isinstance(result, pl.DataFrame)

        # Check that it has the expected columns
        assert "cell_id" in result.columns
        assert "gene_id" in result.columns
        assert "value" in result.columns

        # Check that all rows have the expected gene IDs
        if len(result) > 0:
            assert all(gid in gene_ids for gid in result["gene_id"].to_list())

    def test_query_method(self, small_slaf):
        """Test the SQL query method"""
        # Test a simple query
        result = small_slaf.query("SELECT COUNT(*) as total_cells FROM cells")

        # Check that we got a polars DataFrame
        assert isinstance(result, pl.DataFrame)

        # Check that it has the expected columns
        assert "total_cells" in result.columns

        # Check that we got a result
        assert len(result) == 1
        assert result["total_cells"][0] == 10

        # Test a more complex query
        result2 = small_slaf.query(
            "SELECT cell_type, COUNT(*) as count FROM cells GROUP BY cell_type"
        )

        # Check that we got a polars DataFrame
        assert isinstance(result2, pl.DataFrame)

        # Check that it has the expected columns
        assert "cell_type" in result2.columns
        assert "count" in result2.columns

        # Check that we got results for each cell type
        assert len(result2) >= 1

    def test_query_with_expression_data(self, small_slaf):
        """Test SQL queries that involve expression data"""
        # Test a query that joins cells and expression
        result = small_slaf.query(
            """
            SELECT c.cell_type, AVG(e.value) as avg_expression
            FROM cells c
            JOIN expression e ON c.cell_integer_id = e.cell_integer_id
            GROUP BY c.cell_type
            """
        )

        # Check that we got a polars DataFrame
        assert isinstance(result, pl.DataFrame)

        # Check that it has the expected columns
        assert "cell_type" in result.columns
        assert "avg_expression" in result.columns

        # Check that we got results
        assert len(result) >= 1

    def test_error_handling(self, small_slaf):
        """Test error handling for invalid queries"""
        # Test invalid column name
        with pytest.raises(ValueError):
            small_slaf.filter_cells(invalid_column="value")

        # Test invalid SQL
        with pytest.raises((ValueError, RuntimeError, Exception)):
            small_slaf.query("SELECT * FROM invalid_table")

    def test_boolean_filtering(self, small_slaf):
        """Test filtering with boolean values"""
        # Filter by highly variable genes
        result = small_slaf.filter_genes(highly_variable=True)

        # Check that we got a polars DataFrame
        assert isinstance(result, pl.DataFrame)

        # Check that all genes are highly variable
        if len(result) > 0:
            assert all(result["highly_variable"])

    def test_numeric_filtering(self, small_slaf):
        """Test filtering with numeric values"""
        # Filter cells with specific total counts
        result = small_slaf.filter_cells(total_counts=5)

        # Check that we got a polars DataFrame
        assert isinstance(result, pl.DataFrame)

        # Check that all cells have the specified total counts
        if len(result) > 0:
            assert all(result["total_counts"] == 5)

    def test_dataset_properties(self, small_slaf):
        """Test basic dataset properties"""
        # Test shape
        assert small_slaf.shape == (10, 5)

        # Test obs and var are polars DataFrames
        assert isinstance(small_slaf.obs, pl.DataFrame)
        assert isinstance(small_slaf.var, pl.DataFrame)

        # Test that obs and var have the expected number of rows
        assert len(small_slaf.obs) == 10
        assert len(small_slaf.var) == 5

        # Test that obs and var have the expected columns
        assert "cell_type" in small_slaf.obs.columns
        assert "total_counts" in small_slaf.obs.columns
        assert "gene_type" in small_slaf.var.columns
        assert "highly_variable" in small_slaf.var.columns

    def test_lance_datasets_loaded(self, small_slaf):
        """Test that Lance datasets are properly loaded"""
        # Test that Lance datasets are available
        assert hasattr(small_slaf, "expression")
        assert hasattr(small_slaf, "cells")
        assert hasattr(small_slaf, "genes")

    def test_get_submatrix(self, small_slaf):
        """Test getting a small submatrix of expression data"""

        # Test a small 5x2 submatrix using slice selectors
        result = small_slaf.get_submatrix(
            cell_selector=slice(0, 5), gene_selector=slice(0, 2)
        )

        # Check that we got a polars DataFrame
        assert isinstance(result, pl.DataFrame)

        # Check that it has the expected columns
        assert "cell_id" in result.columns
        assert "gene_id" in result.columns
        assert "value" in result.columns

        # Check that we got some data (may be empty if no expression in that range)
        print(f"Submatrix result shape: {result.shape}")
        print(f"Submatrix columns: {result.columns}")
        if len(result) > 0:
            print(f"Sample data:\n{result.head()}")

        # The result should be a reasonable size (could be 0 if no expression in range)
        assert len(result) >= 0

    def test_row_index_mapper_integration(self, small_slaf):
        """Test RowIndexMapper integration with SLAFArray."""
        # Wait for metadata to be loaded
        small_slaf.wait_for_metadata()

        # Check that row_mapper is initialized
        assert hasattr(small_slaf, "row_mapper")
        assert small_slaf.row_mapper is not None

        # Check that cell start index is available
        assert small_slaf._cell_start_index is not None
        assert (
            len(small_slaf._cell_start_index) == small_slaf.shape[0] + 1
        )  # +1 for prepended 0

        # Test getting row ranges for a single cell
        row_indices = small_slaf.row_mapper.get_cell_row_ranges_by_selector(0)
        assert isinstance(row_indices, list)
        assert len(row_indices) >= 0

        # Test with a slice
        row_indices = small_slaf.row_mapper.get_cell_row_ranges_by_selector(slice(0, 2))
        assert isinstance(row_indices, list)
        assert len(row_indices) >= 0

    def test_get_submatrix_with_different_selectors(self, small_slaf):
        """Test get_submatrix with various selector types."""
        # Wait for metadata to be loaded
        small_slaf.wait_for_metadata()

        # Test with integer selectors
        result = small_slaf.get_submatrix(cell_selector=0, gene_selector=0)
        assert isinstance(result, pl.DataFrame)

        # Test with list selectors
        result = small_slaf.get_submatrix(cell_selector=[0, 1], gene_selector=[0, 1])
        assert isinstance(result, pl.DataFrame)

        # Test with None selectors (all cells/genes)
        result = small_slaf.get_submatrix(cell_selector=None, gene_selector=slice(0, 2))
        assert isinstance(result, pl.DataFrame)

    def test_expression_methods_consistency(self, small_slaf):
        """Test that expression methods return consistent results."""
        # Wait for metadata to be loaded
        small_slaf.wait_for_metadata()

        if len(small_slaf.obs) > 0:
            # Get cell expression using the new optimized method
            cell_id = small_slaf.obs["cell_id"][0]
            cell_result = small_slaf.get_cell_expression(cell_id)

            # Get the same data using submatrix (should be consistent)
            submatrix_result = small_slaf.get_submatrix(
                cell_selector=0, gene_selector=None
            )

            # Both should be DataFrames with same columns
            assert isinstance(cell_result, pl.DataFrame)
            assert isinstance(submatrix_result, pl.DataFrame)
            assert set(cell_result.columns) == set(submatrix_result.columns)

            # If both have data, cell IDs should match
            if len(cell_result) > 0 and len(submatrix_result) > 0:
                assert all(cell_result["cell_id"] == cell_id)
                assert all(submatrix_result["cell_id"] == cell_id)

    def test_info_method_backward_compatibility(self, tmp_path):
        """Test that info method works with both new and old format versions"""
        import json

        import numpy as np

        # Create a small test dataset
        import scanpy as sc
        from scipy import sparse

        from slaf.data import SLAFConverter

        n_cells, n_genes = 10, 5
        X = sparse.csr_matrix(np.random.randint(0, 10, (n_cells, n_genes)))

        obs = (
            pl.DataFrame(
                {
                    "cell_type": np.random.choice(["A", "B"], n_cells),
                    "total_counts": X.sum(axis=1).A1,
                }
            )
            .with_row_index("cell_id", offset=0)
            .with_columns(
                pl.col("cell_id").map_elements(
                    lambda x: f"cell_{x}", return_dtype=pl.Utf8
                )
            )
        )

        var = (
            pl.DataFrame(
                {
                    "gene_type": np.random.choice(
                        ["protein_coding", "lncRNA"], n_genes
                    ),
                    "highly_variable": np.random.choice([True, False], n_genes),
                }
            )
            .with_row_index("gene_id", offset=0)
            .with_columns(
                pl.col("gene_id").map_elements(
                    lambda x: f"gene_{x}", return_dtype=pl.Utf8
                )
            )
        )

        # Convert to pandas for AnnData compatibility
        obs_pd = obs.to_pandas().set_index("cell_id")
        var_pd = var.to_pandas().set_index("gene_id")

        adata = sc.AnnData(X=X, obs=obs_pd, var=var_pd)

        # Test new format (0.2) - should use pre-computed metadata
        h5ad_path = tmp_path / "test.h5ad"
        adata.write(h5ad_path)

        output_path_new = tmp_path / "test_new.slaf"
        converter = SLAFConverter()
        converter.convert(str(h5ad_path), str(output_path_new))

        slaf_array_new = SLAFArray(str(output_path_new))

        # Capture info output for new format
        import io
        import sys

        captured_output = io.StringIO()
        sys.stdout = captured_output
        slaf_array_new.info()
        sys.stdout = sys.__stdout__
        new_format_output = captured_output.getvalue()

        # Verify new format output contains metadata
        assert "Expression records:" in new_format_output
        assert "Sparsity:" in new_format_output
        assert "Density:" in new_format_output
        assert "Expression statistics:" in new_format_output

        # Test old format (0.1) - should fall back to querying
        # Create old format by modifying config
        with open(output_path_new / "config.json") as f:
            config_old = json.load(f)

        # Downgrade to old format
        config_old["format_version"] = "0.1"
        if "metadata" in config_old:
            del config_old["metadata"]

        output_path_old = tmp_path / "test_old.slaf"
        output_path_old.mkdir(exist_ok=True)

        # Copy Lance files
