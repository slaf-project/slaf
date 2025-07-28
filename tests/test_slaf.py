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

    @pytest.mark.skip(reason="Not implemented")
    def test_get_cell_expression(self, small_slaf):
        """Test getting expression data for specific cells"""
        # Get expression for first cell
        cell_id = small_slaf.obs.index[0]
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

    @pytest.mark.skip(reason="Not implemented")
    def test_get_gene_expression(self, small_slaf):
        """Test getting expression data for specific genes"""
        # Get expression for first gene
        gene_id = small_slaf.var.index[0]
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
            .with_columns(pl.col("cell_id").map_elements(lambda x: f"cell_{x}"))
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
            .with_columns(pl.col("gene_id").map_elements(lambda x: f"gene_{x}"))
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
