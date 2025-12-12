"""
Unit tests for LazyExpressionMatrix with layers support.

Tests LazyExpressionMatrix when used with layers table:
- Compute layer matrix
- Slice layer matrix
- SQL query on layers table (wide format)
- Layer respects cell/gene selectors
"""

import pytest
import scipy.sparse

from slaf.integrations.anndata import LazyAnnData, LazyExpressionMatrix


class TestLazyExpressionMatrixLayers:
    """Test LazyExpressionMatrix with layers table"""

    def test_layer_matrix_compute(self, slaf_with_layers):
        """Test computing a layer matrix"""
        adata = LazyAnnData(slaf_with_layers)
        spliced = adata.layers["spliced"]

        matrix = spliced.compute()

        # Verify it's a sparse matrix
        assert isinstance(matrix, scipy.sparse.csr_matrix)
        assert matrix.shape == (3, 2)

        # Verify values (from fixture: spliced = [1.5, 2.5, 3.5, 4.5, 5.5, 6.5])
        # Cell 0, Gene 0: 1.5
        # Cell 0, Gene 1: 2.5
        # Cell 1, Gene 0: 3.5
        # Cell 1, Gene 1: 4.5
        # Cell 2, Gene 0: 5.5
        # Cell 2, Gene 1: 6.5
        assert matrix[0, 0] == 1.5
        assert matrix[0, 1] == 2.5
        assert matrix[1, 0] == 3.5
        assert matrix[1, 1] == 4.5
        assert matrix[2, 0] == 5.5
        assert matrix[2, 1] == 6.5

    def test_layer_matrix_slicing(self, slaf_with_layers):
        """Test slicing a layer matrix"""
        adata = LazyAnnData(slaf_with_layers)
        spliced = adata.layers["spliced"]

        # Slice to first 2 cells, first gene
        spliced_subset = spliced[:2, :1]

        assert spliced_subset.shape == (2, 1)

        # Compute and verify values
        matrix = spliced_subset.compute()
        assert isinstance(matrix, scipy.sparse.csr_matrix)
        assert matrix.shape == (2, 1)
        assert matrix[0, 0] == 1.5
        assert matrix[1, 0] == 3.5

    def test_layer_matrix_query(self, slaf_with_layers):
        """Test SQL query on layers table (wide format)"""
        # Build a query using the layer matrix's internal query building
        from slaf.core.query_optimizer import QueryOptimizer

        sql_query = QueryOptimizer.build_submatrix_query(
            cell_selector=None,
            gene_selector=None,
            cell_count=3,
            gene_count=2,
            table_name="layers",
            layer_name="spliced",
        )

        # Verify query format (wide format: select layer column directly)
        assert "SELECT" in sql_query
        assert "FROM layers" in sql_query
        assert "spliced as value" in sql_query
        assert "cell_integer_id" in sql_query
        assert "gene_integer_id" in sql_query

    def test_layer_matrix_subsetting(self, slaf_with_layers):
        """Test that layer respects cell/gene selectors from parent LazyAnnData"""
        adata = LazyAnnData(slaf_with_layers)

        # Subset to first 2 cells, first gene
        adata_subset = adata[:2, :1]

        # Access layer on subset
        spliced_subset = adata_subset.layers["spliced"]

        # Verify shape matches subset
        assert spliced_subset.shape == (2, 1)
        assert spliced_subset.shape == adata_subset.X.shape

        # Compute and verify values
        matrix = spliced_subset.compute()
        assert matrix.shape == (2, 1)
        assert matrix[0, 0] == 1.5  # Cell 0, Gene 0
        assert matrix[1, 0] == 3.5  # Cell 1, Gene 0

    def test_layer_matrix_direct_initialization(self, slaf_with_layers):
        """Test creating LazyExpressionMatrix directly for a layer"""
        layer_matrix = LazyExpressionMatrix(
            slaf_with_layers, table_name="layers", layer_name="spliced"
        )

        assert layer_matrix.table_name == "layers"
        assert layer_matrix.layer_name == "spliced"
        assert layer_matrix.shape == slaf_with_layers.shape

    def test_layer_matrix_initialization_requires_layer_name(self, slaf_with_layers):
        """Test that layer_name is required when table_name='layers'"""
        with pytest.raises(ValueError, match="layer_name must be provided"):
            LazyExpressionMatrix(slaf_with_layers, table_name="layers", layer_name=None)

    def test_layer_matrix_vs_expression_matrix(self, slaf_with_layers):
        """Test that layer matrix and expression matrix have same shape but different values"""
        adata = LazyAnnData(slaf_with_layers)

        # Get expression matrix
        X = adata.X
        X_matrix = X.compute()

        # Get layer matrix
        spliced = adata.layers["spliced"]
        spliced_matrix = spliced.compute()

        # Shapes should match
        assert X_matrix.shape == spliced_matrix.shape

        # Values should be different (spliced has different values in fixture)
        # Expression: [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
        # Spliced: [1.5, 2.5, 3.5, 4.5, 5.5, 6.5]
        assert X_matrix[0, 0] == 1.0
        assert spliced_matrix[0, 0] == 1.5
        assert X_matrix[0, 0] != spliced_matrix[0, 0]

    def test_multiple_layers_different_values(self, slaf_with_layers):
        """Test that different layers have different values"""
        adata = LazyAnnData(slaf_with_layers)

        spliced = adata.layers["spliced"].compute()
        unspliced = adata.layers["unspliced"].compute()

        # Shapes should match
        assert spliced.shape == unspliced.shape

        # Values should be different
        # Spliced: [1.5, 2.5, 3.5, 4.5, 5.5, 6.5]
        # Unspliced: [0.5, 1.5, 2.5, 3.5, 4.5, 5.5]
        assert spliced[0, 0] == 1.5
        assert unspliced[0, 0] == 0.5
        assert spliced[0, 0] != unspliced[0, 0]

    def test_layer_matrix_chained_slicing(self, slaf_with_layers):
        """Test chained slicing on layer matrix"""
        adata = LazyAnnData(slaf_with_layers)
        spliced = adata.layers["spliced"]

        # Chain slices
        spliced_subset = spliced[:2, :1]

        # Should work without errors
        assert spliced_subset.shape == (2, 1)

        # Can compute
        matrix = spliced_subset.compute()
        assert matrix.shape == (2, 1)

    def test_layer_matrix_shape_consistency(self, slaf_with_layers):
        """Test that layer matrix shape is always consistent with X"""
        adata = LazyAnnData(slaf_with_layers)

        for layer_name in adata.layers.keys():
            layer_matrix = adata.layers[layer_name]
            assert layer_matrix.shape == adata.X.shape

    def test_layer_matrix_parent_adata_reference(self, slaf_with_layers):
        """Test that layer matrix has parent_adata reference"""
        adata = LazyAnnData(slaf_with_layers)
        spliced = adata.layers["spliced"]

        assert spliced.parent_adata is adata


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
