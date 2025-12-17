"""
End-to-end tests for layers infrastructure.

Tests basic layers infrastructure:
- Config.json schema with format_version 0.4
- Layers table loading in SLAFArray
- Layers table registration in query() method
- Backward compatibility (datasets without layers)
"""

import json
import tempfile

import lance
import pyarrow as pa
import pytest

from slaf.core.slaf import SLAFArray


@pytest.fixture
def temp_slaf_dir():
    """Create a temporary SLAF directory for testing"""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir


@pytest.fixture
def slaf_dataset_without_layers(temp_slaf_dir):
    """Create a minimal SLAF dataset without layers (backward compatibility test)"""
    # Create basic tables
    expression_data = pa.table(
        {
            "cell_integer_id": [0, 0, 1, 1],
            "gene_integer_id": [0, 1, 0, 1],
            "value": [1.0, 2.0, 3.0, 4.0],
        }
    )
    cells_data = pa.table(
        {
            "cell_integer_id": [0, 1],
            "cell_id": ["cell_0", "cell_1"],
        }
    )
    genes_data = pa.table(
        {
            "gene_integer_id": [0, 1],
            "gene_id": ["gene_0", "gene_1"],
        }
    )

    # Write Lance datasets
    lance.write_dataset(expression_data, f"{temp_slaf_dir}/expression.lance")
    lance.write_dataset(cells_data, f"{temp_slaf_dir}/cells.lance")
    lance.write_dataset(genes_data, f"{temp_slaf_dir}/genes.lance")

    # Create config.json (format 0.4 but no layers)
    config = {
        "format_version": "0.4",
        "array_shape": [2, 2],
        "n_cells": 2,
        "n_genes": 2,
        "tables": {
            "expression": "expression.lance",
            "cells": "cells.lance",
            "genes": "genes.lance",
        },
        "optimizations": {
            "use_integer_keys": True,
            "optimize_storage": True,
        },
        "metadata": {
            "expression_count": 4,
            "sparsity": 0.0,
            "density": 1.0,
            "total_possible_elements": 4,
        },
    }

    with open(f"{temp_slaf_dir}/config.json", "w") as f:
        json.dump(config, f, indent=2)

    return temp_slaf_dir


@pytest.fixture
def slaf_dataset_with_layers(temp_slaf_dir):
    """Create a minimal SLAF dataset with layers (format 0.4)"""
    # Create basic tables
    expression_data = pa.table(
        {
            "cell_integer_id": [0, 0, 1, 1],
            "gene_integer_id": [0, 1, 0, 1],
            "value": [1.0, 2.0, 3.0, 4.0],
        }
    )
    cells_data = pa.table(
        {
            "cell_integer_id": [0, 1],
            "cell_id": ["cell_0", "cell_1"],
        }
    )
    genes_data = pa.table(
        {
            "gene_integer_id": [0, 1],
            "gene_id": ["gene_0", "gene_1"],
        }
    )

    # Create layers table (wide format: one column per layer)
    layers_data = pa.table(
        {
            "cell_integer_id": [0, 0, 1, 1],
            "gene_integer_id": [0, 1, 0, 1],
            "spliced": [1.5, 2.5, 3.5, 4.5],
            "unspliced": [0.5, 1.5, 2.5, 3.5],
        }
    )

    # Write Lance datasets
    lance.write_dataset(expression_data, f"{temp_slaf_dir}/expression.lance")
    lance.write_dataset(cells_data, f"{temp_slaf_dir}/cells.lance")
    lance.write_dataset(genes_data, f"{temp_slaf_dir}/genes.lance")
    lance.write_dataset(layers_data, f"{temp_slaf_dir}/layers.lance")

    # Create config.json with layers
    config = {
        "format_version": "0.4",
        "array_shape": [2, 2],
        "n_cells": 2,
        "n_genes": 2,
        "tables": {
            "expression": "expression.lance",
            "cells": "cells.lance",
            "genes": "genes.lance",
            "layers": "layers.lance",
        },
        "layers": {
            "available": ["spliced", "unspliced"],
            "immutable": ["spliced", "unspliced"],
            "mutable": [],
        },
        "optimizations": {
            "use_integer_keys": True,
            "optimize_storage": True,
        },
        "metadata": {
            "expression_count": 4,
            "sparsity": 0.0,
            "density": 1.0,
            "total_possible_elements": 4,
        },
    }

    with open(f"{temp_slaf_dir}/config.json", "w") as f:
        json.dump(config, f, indent=2)

    return temp_slaf_dir


def test_load_dataset_without_layers(slaf_dataset_without_layers):
    """Test that datasets without layers load correctly (backward compatibility)"""
    # Disable async metadata loading to avoid errors with temporary datasets
    slaf = SLAFArray(slaf_dataset_without_layers, load_metadata=False)

    # Verify basic properties
    assert slaf.shape == (2, 2)
    assert slaf.config["format_version"] == "0.4"

    # Verify layers table is None
    assert slaf.layers is None

    # Verify query still works
    result = slaf.query("SELECT COUNT(*) as count FROM cells")
    assert result["count"][0] == 2


def test_load_dataset_with_layers(slaf_dataset_with_layers):
    """Test that datasets with layers load correctly"""
    # Disable async metadata loading to avoid errors with temporary datasets
    slaf = SLAFArray(slaf_dataset_with_layers, load_metadata=False)

    # Verify basic properties
    assert slaf.shape == (2, 2)
    assert slaf.config["format_version"] == "0.4"

    # Verify layers table is loaded
    assert slaf.layers is not None

    # Verify layers metadata in config
    assert "layers" in slaf.config
    assert "available" in slaf.config["layers"]
    assert set(slaf.config["layers"]["available"]) == {"spliced", "unspliced"}
    assert set(slaf.config["layers"]["immutable"]) == {"spliced", "unspliced"}
    assert slaf.config["layers"]["mutable"] == []


def test_query_with_layers_table(slaf_dataset_with_layers):
    """Test that layers table is registered in query() method"""
    # Disable async metadata loading to avoid errors with temporary datasets
    slaf = SLAFArray(slaf_dataset_with_layers, load_metadata=False)

    # Query layers table
    result = slaf.query(
        """
        SELECT cell_integer_id, gene_integer_id, spliced
        FROM layers
        WHERE spliced IS NOT NULL
        ORDER BY cell_integer_id, gene_integer_id
        """
    )

    # Verify results
    assert len(result) == 4
    assert "spliced" in result.columns
    assert result["spliced"][0] == 1.5
    assert result["spliced"][1] == 2.5


def test_query_without_layers_table(slaf_dataset_without_layers):
    """Test that query fails gracefully when layers table doesn't exist"""
    # Disable async metadata loading to avoid errors with temporary datasets
    slaf = SLAFArray(slaf_dataset_without_layers, load_metadata=False)

    # Query should fail if trying to use layers table
    import polars.exceptions

    with pytest.raises(
        polars.exceptions.SQLInterfaceError, match="relation 'layers' was not found"
    ):
        slaf.query("SELECT * FROM layers")


def test_config_format_version_0_4(slaf_dataset_with_layers):
    """Test that config.json has format_version 0.4"""
    # Disable async metadata loading to avoid errors with temporary datasets
    slaf = SLAFArray(slaf_dataset_with_layers, load_metadata=False)

    assert slaf.config["format_version"] == "0.4"


def test_layers_table_schema(slaf_dataset_with_layers):
    """Test that layers table has correct wide format schema"""
    # Disable async metadata loading to avoid errors with temporary datasets
    slaf = SLAFArray(slaf_dataset_with_layers, load_metadata=False)

    # Get schema from layers dataset
    schema = slaf.layers.schema

    # Verify columns exist
    column_names = [field.name for field in schema]
    assert "cell_integer_id" in column_names
    assert "gene_integer_id" in column_names
    assert "spliced" in column_names
    assert "unspliced" in column_names

    # Verify it's wide format (one column per layer, not layer_name column)
    assert "layer_name" not in column_names


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
