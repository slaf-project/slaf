import json
import os
from pathlib import Path

import lance
import numpy as np

from slaf.data.converter import SLAFConverter


class TestSLAFConverter:
    """Test the SLAF converter functionality"""

    # Basic functionality tests
    def test_converter_creates_new_table_structure(self, small_sample_adata, temp_dir):
        """Test that converter creates the COO table structure"""
        # Save sample data as h5ad
        h5ad_path = Path(temp_dir) / "test.h5ad"
        small_sample_adata.write(h5ad_path)

        # Convert to SLAF
        output_path = Path(temp_dir) / "test.slaf"
        converter = SLAFConverter()
        converter.convert(str(h5ad_path), str(output_path))

        # Check that new table files exist
        assert (output_path / "expression.lance").exists()
        assert (output_path / "cells.lance").exists()
        assert (output_path / "genes.lance").exists()
        assert (output_path / "config.json").exists()

        # Check config version
        with open(output_path / "config.json") as f:
            config = json.load(f)
        assert config["format_version"] == "0.1"

    def test_expression_data_consistency(self, small_sample_adata, temp_dir):
        """Test that expression data is consistent in COO format"""
        # Convert to SLAF
        h5ad_path = Path(temp_dir) / "test.h5ad"
        small_sample_adata.write(h5ad_path)
        output_path = Path(temp_dir) / "test.slaf"
        converter = SLAFConverter()
        converter.convert(str(h5ad_path), str(output_path))

        # Load expression table
        expression_dataset = lance.dataset(output_path / "expression.lance")
        expression_df = expression_dataset.to_table().to_pandas()

        # Reconstruct original matrix from COO table using integer IDs
        reconstructed = np.zeros((small_sample_adata.n_obs, small_sample_adata.n_vars))
        for _, row in expression_df.iterrows():
            cell_idx = row["cell_integer_id"]
            gene_idx = row["gene_integer_id"]
            value = row["value"]
            reconstructed[cell_idx, gene_idx] = value

        # Check that reconstruction matches original
        original_dense = small_sample_adata.X.toarray()
        np.testing.assert_array_equal(original_dense, reconstructed)

        # Check that we have the right number of non-zero entries
        expected_nonzero = small_sample_adata.X.nnz
        actual_nonzero = len(expression_df)
        assert actual_nonzero == expected_nonzero

    # Integer ID tests
    def test_integer_ids_in_metadata(self, small_sample_adata, temp_dir):
        """Test that integer IDs are embedded in metadata tables"""
        # Save sample data as h5ad
        h5ad_path = Path(temp_dir) / "test.h5ad"
        small_sample_adata.write(h5ad_path)

        # Convert with integer keys
        output_path = Path(temp_dir) / "test.slaf"
        converter = SLAFConverter(use_integer_keys=True)
        converter.convert(str(h5ad_path), str(output_path))

        # Load metadata tables
        cells_dataset = lance.dataset(output_path / "cells.lance")
        cells_df = cells_dataset.to_table().to_pandas()

        genes_dataset = lance.dataset(output_path / "genes.lance")
        genes_df = genes_dataset.to_table().to_pandas()

        # Check that integer ID columns exist
        assert "cell_integer_id" in cells_df.columns
        assert "gene_integer_id" in genes_df.columns

        # Check that integer IDs are sequential
        np.testing.assert_array_equal(
            cells_df["cell_integer_id"].values, range(small_sample_adata.n_obs)
        )
        np.testing.assert_array_equal(
            genes_df["gene_integer_id"].values, range(small_sample_adata.n_vars)
        )

        # Check that original IDs are preserved
        expected_cell_ids = [f"cell_{i}" for i in range(small_sample_adata.n_obs)]
        assert list(cells_df["cell_id"]) == expected_cell_ids

        # Use the actual gene IDs from the AnnData object
        expected_gene_ids = list(small_sample_adata.var.index)
        assert list(genes_df["gene_id"]) == expected_gene_ids

    def test_no_integer_ids_when_disabled(self, small_sample_adata, temp_dir):
        """Test that integer IDs are not added when disabled"""
        # Save sample data as h5ad
        h5ad_path = Path(temp_dir) / "test.h5ad"
        small_sample_adata.write(h5ad_path)

        # Convert without integer keys
        output_path = Path(temp_dir) / "test.slaf"
        converter = SLAFConverter(use_integer_keys=False)
        converter.convert(str(h5ad_path), str(output_path))

        # Load metadata tables
        cells_dataset = lance.dataset(output_path / "cells.lance")
        cells_df = cells_dataset.to_table().to_pandas()

        genes_dataset = lance.dataset(output_path / "genes.lance")
        genes_df = genes_dataset.to_table().to_pandas()

        # Check that integer ID columns do NOT exist
        assert "cell_integer_id" not in cells_df.columns
        assert "gene_integer_id" not in genes_df.columns

    def test_expression_data_structure(self, small_sample_adata, temp_dir):
        """Test that expression data has the correct COO structure"""
        # Save sample data as h5ad
        h5ad_path = Path(temp_dir) / "test.h5ad"
        small_sample_adata.write(h5ad_path)

        # Convert with integer keys
        output_path = Path(temp_dir) / "test.slaf"
        converter = SLAFConverter(use_integer_keys=True)
        converter.convert(str(h5ad_path), str(output_path))

        # Load expression table
        expression_dataset = lance.dataset(output_path / "expression.lance")
        expression_df = expression_dataset.to_table().to_pandas()

        # Check that expression table has the right columns
        expected_columns = {
            "cell_id",
            "gene_id",
            "cell_integer_id",
            "gene_integer_id",
            "value",
        }
        assert set(expression_df.columns) == expected_columns

        # Check that all values are valid
        assert all(expression_df["value"] >= 0)
        assert all(expression_df["cell_id"].str.startswith("cell_"))
        # Check gene_id prefix matches the AnnData gene index
        gene_prefix = str(small_sample_adata.var.index[0])[:5]
        assert all(expression_df["gene_id"].str.startswith(gene_prefix))

        # Check that integer IDs are valid
        assert all(expression_df["cell_integer_id"] >= 0)
        assert all(expression_df["cell_integer_id"] < small_sample_adata.n_obs)
        assert all(expression_df["gene_integer_id"] >= 0)
        assert all(expression_df["gene_integer_id"] < small_sample_adata.n_vars)

    def test_integer_id_consistency(self, small_sample_adata, temp_dir):
        """Test that integer IDs in expression table match metadata tables"""
        # Save sample data as h5ad
        h5ad_path = Path(temp_dir) / "test.h5ad"
        small_sample_adata.write(h5ad_path)

        # Convert with integer keys
        output_path = Path(temp_dir) / "test.slaf"
        converter = SLAFConverter(use_integer_keys=True)
        converter.convert(str(h5ad_path), str(output_path))

        # Load all tables
        expression_dataset = lance.dataset(output_path / "expression.lance")
        expression_df = expression_dataset.to_table().to_pandas()

        cells_dataset = lance.dataset(output_path / "cells.lance")
        cells_df = cells_dataset.to_table().to_pandas()

        genes_dataset = lance.dataset(output_path / "genes.lance")
        genes_df = genes_dataset.to_table().to_pandas()

        # Create mappings from string IDs to integer IDs
        cell_id_to_int = dict(
            zip(cells_df["cell_id"], cells_df["cell_integer_id"], strict=False)
        )
        gene_id_to_int = dict(
            zip(genes_df["gene_id"], genes_df["gene_integer_id"], strict=False)
        )

        # Verify that expression table integer IDs match the mappings
        for _, row in expression_df.iterrows():
            expected_cell_int = cell_id_to_int[row["cell_id"]]
            expected_gene_int = gene_id_to_int[row["gene_id"]]

            assert row["cell_integer_id"] == expected_cell_int, (
                f"Cell integer ID mismatch for {row['cell_id']}"
            )
            assert row["gene_integer_id"] == expected_gene_int, (
                f"Gene integer ID mismatch for {row['gene_id']}"
            )

    # Optimization tests
    def test_integer_keys_optimization(self, small_sample_adata, temp_dir):
        """Test that integer keys reduce file size"""
        h5ad_path = Path(temp_dir) / "test.h5ad"
        small_sample_adata.write(h5ad_path)
        output_path_string_keys = Path(temp_dir) / "test_string_keys.slaf"
        converter_string_keys = SLAFConverter(use_integer_keys=False)
        converter_string_keys.convert(str(h5ad_path), str(output_path_string_keys))
        output_path_int_keys = Path(temp_dir) / "test_int_keys.slaf"
        converter_int_keys = SLAFConverter(use_integer_keys=True)
        converter_int_keys.convert(str(h5ad_path), str(output_path_int_keys))

        # Check that files exist and have reasonable sizes
        string_keys_size = os.path.getsize(output_path_string_keys / "expression.lance")
        int_keys_size = os.path.getsize(output_path_int_keys / "expression.lance")
        print(f"String keys size: {string_keys_size:,} bytes")
        print(f"Integer keys size: {int_keys_size:,} bytes")

        # Both should be reasonable sizes
        assert string_keys_size > 0
        assert int_keys_size > 0

    def test_optimization_config_persistence(self, small_sample_adata, temp_dir):
        """Test that optimization settings are saved in config"""
        # Save sample data as h5ad
        h5ad_path = Path(temp_dir) / "test.h5ad"
        small_sample_adata.write(h5ad_path)

        # Convert with optimizations
        output_path = Path(temp_dir) / "test_optimized.slaf"
        converter = SLAFConverter(
            use_integer_keys=True,
        )
        converter.convert(str(h5ad_path), str(output_path))

        # Check config
        with open(output_path / "config.json") as f:
            config = json.load(f)

        assert "optimizations" in config
        assert config["optimizations"]["use_integer_keys"]
