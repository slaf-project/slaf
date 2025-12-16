"""
End-to-end tests for layer access functionality.

Tests layer access functionality:
- LazyLayersView dictionary-like interface
- Accessing layers as LazyExpressionMatrix
- Listing layers (keys, __contains__, __len__)
- Layer access after subsetting
- Backward compatibility (datasets without layers)
"""

import json
import tempfile

import lance
import pyarrow as pa
import pytest

from slaf.core.slaf import SLAFArray
from slaf.integrations.anndata import LazyAnnData


@pytest.fixture
def temp_slaf_dir():
    """Create a temporary SLAF directory for testing"""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir


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


def test_layers_keys(slaf_dataset_with_layers):
    """Test listing layer names"""
    slaf = SLAFArray(slaf_dataset_with_layers, load_metadata=False)
    adata = LazyAnnData(slaf)

    # Test keys() method
    layer_keys = list(adata.layers.keys())
    assert set(layer_keys) == {"spliced", "unspliced"}


def test_layers_contains(slaf_dataset_with_layers):
    """Test checking if layer exists"""
    slaf = SLAFArray(slaf_dataset_with_layers, load_metadata=False)
    adata = LazyAnnData(slaf)

    # Test __contains__
    assert "spliced" in adata.layers
    assert "unspliced" in adata.layers
    assert "nonexistent" not in adata.layers


def test_layers_len(slaf_dataset_with_layers):
    """Test getting number of layers"""
    slaf = SLAFArray(slaf_dataset_with_layers, load_metadata=False)
    adata = LazyAnnData(slaf)

    # Test __len__
    assert len(adata.layers) == 2


def test_layers_getitem(slaf_dataset_with_layers):
    """Test accessing a layer"""
    slaf = SLAFArray(slaf_dataset_with_layers, load_metadata=False)
    adata = LazyAnnData(slaf)

    # Access a layer
    spliced = adata.layers["spliced"]
    assert spliced.shape == adata.X.shape
    assert spliced.shape == (2, 2)

    # Verify it's a LazyExpressionMatrix
    from slaf.integrations.anndata import LazyExpressionMatrix

    assert isinstance(spliced, LazyExpressionMatrix)


def test_layers_getitem_nonexistent(slaf_dataset_with_layers):
    """Test accessing a non-existent layer raises KeyError"""
    slaf = SLAFArray(slaf_dataset_with_layers, load_metadata=False)
    adata = LazyAnnData(slaf)

    with pytest.raises(KeyError, match="Layer 'nonexistent' not found"):
        _ = adata.layers["nonexistent"]


def test_layers_iter(slaf_dataset_with_layers):
    """Test iterating over layers"""
    slaf = SLAFArray(slaf_dataset_with_layers, load_metadata=False)
    adata = LazyAnnData(slaf)

    # Test iteration
    layer_names = list(adata.layers)
    assert set(layer_names) == {"spliced", "unspliced"}


def test_layers_empty_dataset(slaf_dataset_without_layers):
    """Test layers on dataset without layers (backward compatibility)"""
    slaf = SLAFArray(slaf_dataset_without_layers, load_metadata=False)
    adata = LazyAnnData(slaf)

    # Should return empty list
    assert len(adata.layers) == 0
    assert list(adata.layers.keys()) == []
    assert "spliced" not in adata.layers


def test_layers_after_subsetting(slaf_dataset_with_layers):
    """Test layer access after subsetting LazyAnnData"""
    slaf = SLAFArray(slaf_dataset_with_layers, load_metadata=False)
    adata = LazyAnnData(slaf)

    # Subset to first cell and first gene
    # Note: This triggers metadata loading which has a known type issue
    adata_subset = adata[:1, :1]

    # Access layer on subset
    spliced_subset = adata_subset.layers["spliced"]
    assert spliced_subset.shape == (1, 1)
    assert spliced_subset.shape == adata_subset.X.shape


def test_layers_compute(slaf_dataset_with_layers):
    """Test computing a layer matrix"""
    slaf = SLAFArray(slaf_dataset_with_layers, load_metadata=False)
    adata = LazyAnnData(slaf)

    # Access and compute layer
    spliced = adata.layers["spliced"]
    matrix = spliced.compute()

    # Verify it's a sparse matrix
    import scipy.sparse

    assert isinstance(matrix, scipy.sparse.csr_matrix)
    assert matrix.shape == (2, 2)


def test_layers_slicing(slaf_dataset_with_layers):
    """Test slicing a layer matrix"""
    slaf = SLAFArray(slaf_dataset_with_layers, load_metadata=False)
    adata = LazyAnnData(slaf)

    # Access layer and slice it
    spliced = adata.layers["spliced"]
    spliced_subset = spliced[:1, :1]

    # Verify shape
    assert spliced_subset.shape == (1, 1)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
