"""
Unit tests for SLAFArray layers support.

Tests SLAFArray functionality with layers:
- Loading dataset with layers
- Loading old dataset without layers (backward compat)
- SQL query including layers table (wide format)
- get_submatrix() from layers table
"""

import pytest


class TestSLAFArrayLayers:
    """Test SLAFArray layers support"""

    def test_slaf_array_with_layers(self, slaf_with_layers):
        """Test loading dataset with layers"""
        assert slaf_with_layers.layers is not None
        assert slaf_with_layers.shape == (3, 2)

        # Verify layers table is accessible
        layers_dataset = slaf_with_layers.layers
        assert layers_dataset is not None

        # Verify schema has layer columns
        schema = layers_dataset.schema
        column_names = [field.name for field in schema]
        assert "cell_integer_id" in column_names
        assert "gene_integer_id" in column_names
        assert "spliced" in column_names
        assert "unspliced" in column_names

    def test_slaf_array_without_layers(self, slaf_without_layers):
        """Test loading old dataset without layers (backward compatibility)"""
        assert slaf_without_layers.layers is None
        assert slaf_without_layers.shape == (3, 2)

        # Should still work for expression queries
        result = slaf_without_layers.query("SELECT COUNT(*) as count FROM expression")
        assert len(result) > 0

    def test_query_layers_table(self, slaf_with_layers):
        """Test SQL query including layers table (wide format)"""
        # Query to get spliced values
        result = slaf_with_layers.query(
            """
            SELECT cell_integer_id, gene_integer_id, spliced as value
            FROM layers
            WHERE cell_integer_id = 0 AND gene_integer_id = 0
            """
        )

        assert len(result) == 1
        assert result["value"][0] == 1.5

    def test_query_layers_table_wide_format(self, slaf_with_layers):
        """Test that layers table uses wide format (one column per layer)"""
        # Query multiple layers at once
        result = slaf_with_layers.query(
            """
            SELECT cell_integer_id, gene_integer_id, spliced, unspliced
            FROM layers
            WHERE cell_integer_id = 0 AND gene_integer_id = 0
            """
        )

        assert len(result) == 1
        assert "spliced" in result.columns
        assert "unspliced" in result.columns
        assert result["spliced"][0] == 1.5
        assert result["unspliced"][0] == 0.5

    def test_query_without_layers_table(self, slaf_without_layers):
        """Test that query fails gracefully when layers table doesn't exist"""
        # Query should work for expression table
        result = slaf_without_layers.query("SELECT COUNT(*) as count FROM expression")
        assert len(result) > 0

        # Query to layers table should fail (table not registered)
        import polars.exceptions

        with pytest.raises(
            polars.exceptions.SQLInterfaceError, match="relation 'layers' was not found"
        ):
            slaf_without_layers.query("SELECT * FROM layers")

    def test_get_submatrix_layers(self, slaf_with_layers):
        """Test get_submatrix() from layers table"""
        # Get submatrix from layers table
        result = slaf_with_layers.get_submatrix(
            cell_selector=slice(0, 2),
            gene_selector=slice(0, 1),
            table_name="layers",
            layer_name="spliced",
        )

        # Should return DataFrame with cell_id, gene_id, value (joined with metadata)
        assert "cell_id" in result.columns
        assert "gene_id" in result.columns
        assert "value" in result.columns

        # Verify values
        values = result["value"].to_list()
        assert 1.5 in values  # Cell 0, Gene 0
        assert 3.5 in values  # Cell 1, Gene 0

    def test_get_submatrix_layers_requires_layer_name(self, slaf_with_layers):
        """Test that get_submatrix requires layer_name for layers table"""
        with pytest.raises(ValueError, match="layer_name must be provided"):
            slaf_with_layers.get_submatrix(table_name="layers", layer_name=None)

    def test_get_submatrix_layers_table_not_available(self, slaf_without_layers):
        """Test that get_submatrix fails when layers table is not available"""
        with pytest.raises(ValueError, match="Layers table not available"):
            slaf_without_layers.get_submatrix(table_name="layers", layer_name="spliced")

    def test_config_layers_metadata(self, slaf_with_layers):
        """Test that config.json has layers metadata"""
        config = slaf_with_layers.config

        assert "layers" in config
        layers_config = config["layers"]

        assert "available" in layers_config
        assert "immutable" in layers_config
        assert "mutable" in layers_config

        assert set(layers_config["available"]) == {"spliced", "unspliced"}
        assert set(layers_config["immutable"]) == {"spliced", "unspliced"}
        assert layers_config["mutable"] == []

    def test_config_no_layers_metadata(self, slaf_without_layers):
        """Test that config.json doesn't have layers metadata when no layers exist"""
        config = slaf_without_layers.config

        # Should not have layers key
        assert "layers" not in config

        # Should not have layers in tables
        assert "layers" not in config.get("tables", {})

    def test_layers_table_registered_in_query(self, slaf_with_layers):
        """Test that layers table is registered in query() method"""
        # Query that joins layers with expression
        result = slaf_with_layers.query(
            """
            SELECT e.cell_integer_id, e.gene_integer_id, e.value as expr, l.spliced
            FROM expression e
            JOIN layers l ON e.cell_integer_id = l.cell_integer_id
                         AND e.gene_integer_id = l.gene_integer_id
            WHERE e.cell_integer_id = 0 AND e.gene_integer_id = 0
            """
        )

        assert len(result) == 1
        assert "expr" in result.columns
        assert "spliced" in result.columns
        assert result["expr"][0] == 1.0
        assert result["spliced"][0] == 1.5


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
