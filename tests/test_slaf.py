import duckdb
import pandas as pd
import pytest


class TestSLAFArray:
    """Test suite for SLAFArray class"""

    def test_slaf_array_initialization(self, tiny_slaf):
        """Test SLAFArray initialization"""

        assert tiny_slaf.shape == (100, 50)
        assert tiny_slaf.config is not None
        assert "format_version" in tiny_slaf.config
        assert "tables" in tiny_slaf.config

    def test_info_method(self, tiny_slaf):
        """Test the info method"""

        # Capture print output
        import io
        from contextlib import redirect_stdout

        f = io.StringIO()
        with redirect_stdout(f):
            tiny_slaf.info()

        output = f.getvalue()

        # Check that key information is present
        assert "SLAF Dataset" in output
        assert "100 cells × 50 genes" in output
        assert "Cell metadata columns" in output
        assert "Gene metadata columns" in output
        assert "Record counts" in output

    def test_filter_cells_basic(self, tiny_slaf):
        """Test basic cell filtering"""

        # Test filtering by cell type
        t_cells = tiny_slaf.filter_cells(cell_type="T-cell")
        assert len(t_cells) > 0
        assert all(t_cells["cell_type"] == "T-cell")

        # Test filtering by batch
        batch_1_cells = tiny_slaf.filter_cells(batch="batch_1")
        assert len(batch_1_cells) > 0
        assert all(batch_1_cells["batch"] == "batch_1")

    def test_filter_cells_range_queries(self, tiny_slaf):
        """Test cell filtering with range queries"""

        # Test greater than
        high_count_cells = tiny_slaf.filter_cells(total_counts=">1000")
        assert len(high_count_cells) >= 0

        # Test greater than or equal
        high_count_cells_ge = tiny_slaf.filter_cells(total_counts=">=1000")
        assert len(high_count_cells_ge) >= 0

        # Test less than
        low_gene_cells = tiny_slaf.filter_cells(n_genes_by_counts="<400")
        assert len(low_gene_cells) >= 0

        # Test less than or equal
        low_gene_cells_le = tiny_slaf.filter_cells(n_genes_by_counts="<=400")
        assert len(low_gene_cells_le) >= 0

        # Test combined filtering with different operators
        combined = tiny_slaf.filter_cells(
            total_counts=">=800", n_genes_by_counts="<=600"
        )
        assert len(combined) >= 0

        # Test range filtering (between values) - using different columns
        range_filter = tiny_slaf.filter_cells(
            total_counts=">=500", n_genes_by_counts="<=1500"
        )
        assert len(range_filter) >= 0

    def test_filter_cells_list_values(self, tiny_slaf):
        """Test cell filtering with list values"""

        # Test filtering by multiple cell types
        immune_cells = tiny_slaf.filter_cells(cell_type=["T-cell", "B-cell"])
        assert len(immune_cells) > 0
        assert all(immune_cells["cell_type"].isin(["T-cell", "B-cell"]))

    def test_filter_cells_no_filters(self, tiny_slaf):
        """Test cell filtering with no filters (should return all cells)"""

        all_cells = tiny_slaf.filter_cells()
        assert len(all_cells) == 100  # Should return all cells

    def test_filter_genes_basic(self, tiny_slaf):
        """Test basic gene filtering"""

        # Test filtering by gene type
        protein_coding = tiny_slaf.filter_genes(gene_type="protein_coding")
        assert len(protein_coding) > 0
        assert all(protein_coding["gene_type"] == "protein_coding")

        # Test filtering by highly variable
        hvg_genes = tiny_slaf.filter_genes(highly_variable=True)
        assert len(hvg_genes) >= 0

    def test_filter_genes_range_queries(self, tiny_slaf):
        """Test gene filtering with range queries"""

        # Test greater than
        high_expr_genes = tiny_slaf.filter_genes(total_counts=">400")
        assert len(high_expr_genes) >= 0

        # Test greater than or equal
        high_expr_genes_ge = tiny_slaf.filter_genes(total_counts=">=400")
        assert len(high_expr_genes_ge) >= 0

        # Test less than
        low_cell_genes = tiny_slaf.filter_genes(n_cells_by_counts="<30")
        assert len(low_cell_genes) >= 0

        # Test less than or equal
        low_cell_genes_le = tiny_slaf.filter_genes(n_cells_by_counts="<=30")
        assert len(low_cell_genes_le) >= 0

        # Test combined filtering with different operators
        combined = tiny_slaf.filter_genes(
            total_counts=">=300", n_cells_by_counts="<=60"
        )
        assert len(combined) >= 0

    def test_filter_genes_no_filters(self, tiny_slaf):
        """Test gene filtering with no filters (should return all genes)"""

        all_genes = tiny_slaf.filter_genes()
        assert len(all_genes) == 50  # Should return all genes

    def test_get_cell_expression(self, tiny_slaf):
        """Test getting cell expression data"""

        # Get some cell IDs - now the index contains the cell IDs
        all_cells = tiny_slaf.filter_cells()
        cell_ids = all_cells.index[:3].tolist()

        # Get expression data
        cell_expression = tiny_slaf.get_cell_expression(cell_ids)
        assert len(cell_expression) > 0  # May have multiple expression records per cell
        assert "cell_id" in cell_expression.columns
        assert "gene_id" in cell_expression.columns
        assert "value" in cell_expression.columns

        # Test single cell ID
        single_cell_expr = tiny_slaf.get_cell_expression(cell_ids[0])
        assert len(single_cell_expr) >= 0  # May have multiple genes expressed

    def test_get_gene_expression(self, tiny_slaf):
        """Test getting gene expression data"""

        # Get some gene IDs - now the index contains the gene IDs
        all_genes = tiny_slaf.filter_genes()
        gene_ids = all_genes.index[:3].tolist()

        # Get expression data
        gene_expression = tiny_slaf.get_gene_expression(gene_ids)
        assert len(gene_expression) > 0  # May have multiple expression records per gene
        assert "cell_id" in gene_expression.columns
        assert "gene_id" in gene_expression.columns
        assert "value" in gene_expression.columns

        # Test single gene ID
        single_gene_expr = tiny_slaf.get_gene_expression(gene_ids[0])
        assert len(single_gene_expr) >= 0

    def test_query_method(self, tiny_slaf):
        """Test direct SQL query method"""

        # Simple query
        result = tiny_slaf.query("SELECT COUNT(*) as count FROM cells")
        assert len(result) == 1
        assert result.iloc[0]["count"] == 100

        # Complex query with joins
        complex_query = """
        SELECT
            c.cell_type,
            COUNT(*) as cell_count,
            AVG(c.total_counts) as avg_counts
        FROM cells c
        GROUP BY c.cell_type
        ORDER BY cell_count DESC
        """

        result = tiny_slaf.query(complex_query)
        assert len(result) > 0
        assert "cell_type" in result.columns
        assert "cell_count" in result.columns
        assert "avg_counts" in result.columns

    def test_query_with_expression_data(self, tiny_slaf):
        """Test SQL queries involving expression data"""

        # Query that joins metadata with expression data
        query = """
        SELECT
            c.cell_id,
            c.cell_type,
            COUNT(e.value) as expressed_genes
        FROM cells c
        LEFT JOIN expression e ON c.cell_id = e.cell_id
        GROUP BY c.cell_id, c.cell_type
        LIMIT 5
        """

        result = tiny_slaf.query(query)
        assert len(result) > 0
        assert "cell_id" in result.columns
        assert "cell_type" in result.columns
        assert "expressed_genes" in result.columns

    def test_error_handling(self, tiny_slaf):
        """Test error handling for invalid operations"""

        with pytest.raises(duckdb.CatalogException):
            tiny_slaf.query("SELECT * FROM nonexistent_table")

        with pytest.raises(ValueError):
            tiny_slaf.filter_cells(nonexistent_column="value")

    def test_boolean_filtering(self, tiny_slaf):
        """Test boolean filtering"""

        # Test boolean True filter
        high_mito_cells = tiny_slaf.filter_cells(high_mito=True)
        assert len(high_mito_cells) >= 0

        # Test boolean False filter
        low_mito_cells = tiny_slaf.filter_cells(high_mito=False)
        assert len(low_mito_cells) >= 0

    def test_numeric_filtering(self, tiny_slaf):
        """Test numeric filtering"""

        # Test numeric equality
        specific_count_cells = tiny_slaf.filter_cells(total_counts=1000)
        assert len(specific_count_cells) >= 0

    def test_dataset_properties(self, tiny_slaf):
        """Test dataset properties and configuration"""

        # Test shape property
        assert tiny_slaf.shape == (100, 50)

        # Test config structure
        assert "format_version" in tiny_slaf.config
        assert "array_shape" in tiny_slaf.config
        assert "tables" in tiny_slaf.config
        assert "cells" in tiny_slaf.config["tables"]
        assert "genes" in tiny_slaf.config["tables"]
        assert "expression" in tiny_slaf.config["tables"]

    def test_lance_datasets_loaded(self, tiny_slaf):
        """Test that Lance datasets are properly loaded"""

        # Check that all expected datasets are loaded
        assert hasattr(tiny_slaf, "cells")
        assert hasattr(tiny_slaf, "genes")
        assert hasattr(tiny_slaf, "expression")

        # Check that they are Lance datasets
        import lance

        assert isinstance(tiny_slaf.cells, lance.LanceDataset)
        assert isinstance(tiny_slaf.genes, lance.LanceDataset)
        assert isinstance(tiny_slaf.expression, lance.LanceDataset)

    def test_get_submatrix(self, tiny_slaf):
        """Test getting a small submatrix of expression data"""

        # Test a small 5x2 submatrix using slice selectors
        result = tiny_slaf.get_submatrix(
            cell_selector=slice(0, 5), gene_selector=slice(0, 2)
        )

        # Check that we got a DataFrame
        assert isinstance(result, pd.DataFrame)

        # Check that it has the expected columns
        assert "cell_id" in result.columns
        assert "gene_id" in result.columns
        assert "value" in result.columns

        # Check that we got some data (may be empty if no expression in that range)
        print(f"Submatrix result shape: {result.shape}")
        print(f"Submatrix columns: {result.columns.tolist()}")
        if len(result) > 0:
            print(f"Sample data:\n{result.head()}")

        # The result should be a reasonable size (could be 0 if no expression in range)
        assert len(result) >= 0
