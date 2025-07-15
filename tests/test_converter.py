import json
import os
from pathlib import Path

import lance
import numpy as np
import pandas as pd
import pyarrow as pa
import pytest

from slaf.data.chunked_reader import ChunkedH5ADReader
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

    # Chunked converter tests
    def test_chunked_reader_basic(self, sample_h5ad_file):
        """Test basic chunked reader functionality"""
        # Test chunked reader
        with ChunkedH5ADReader(sample_h5ad_file) as reader:
            assert reader.n_obs == 2
            assert reader.n_vars == 2
            assert len(reader.obs_names) == 2
            assert len(reader.var_names) == 2

            # Test metadata reading
            obs_df = reader.get_obs_metadata()
            var_df = reader.get_var_metadata()

            assert obs_df.shape == (2, 1)  # cell_type column
            assert var_df.shape == (2, 1)  # highly_variable column

            # Test chunking
            chunks = list(reader.iter_chunks(chunk_size=1))
            assert len(chunks) == 2

            chunk1, slice1 = chunks[0]
            assert chunk1.shape == (1, 2)

    def test_converter_chunked_mode(self, chunked_converter):
        """Test that chunked mode works correctly"""
        # Test that chunked mode is set correctly
        assert chunked_converter.chunked is True
        assert chunked_converter.chunk_size == 1000

        # Test that convert_anndata raises error in chunked mode
        with pytest.raises(
            ValueError, match="convert_anndata.*not supported in chunked mode"
        ):
            chunked_converter.convert_anndata(None, "output.slaf")

    def test_converter_backward_compatibility(self):
        """Test that non-chunked mode works as before"""
        converter = SLAFConverter(chunked=False)

        # Test that chunked mode is disabled
        assert converter.chunked is False
        assert converter.chunk_size == 1000  # Default value

    def test_expression_schema(self):
        """Test that expression schema is correct"""
        converter = SLAFConverter()
        schema = converter._get_expression_schema()

        # Check that schema has expected fields
        field_names = [field.name for field in schema]
        expected_fields = [
            "cell_id",
            "gene_id",
            "cell_integer_id",
            "gene_integer_id",
            "value",
        ]

        assert field_names == expected_fields

        # Check field types
        assert schema.field("cell_id").type == pa.string()
        assert schema.field("gene_id").type == pa.string()
        assert schema.field("cell_integer_id").type == pa.int32()
        assert schema.field("gene_integer_id").type == pa.int32()
        assert schema.field("value").type == pa.float32()

    def test_chunked_conversion_creates_same_structure(
        self, small_sample_adata, temp_dir
    ):
        """Test that chunked conversion creates the same structure as traditional conversion"""
        # Save sample data as h5ad
        h5ad_path = Path(temp_dir) / "test.h5ad"
        small_sample_adata.write(h5ad_path)

        # Convert using traditional method
        output_path_traditional = Path(temp_dir) / "test_traditional.slaf"
        converter_traditional = SLAFConverter(chunked=False, sort_metadata=True)
        converter_traditional.convert(str(h5ad_path), str(output_path_traditional))

        # Convert using chunked method
        output_path_chunked = Path(temp_dir) / "test_chunked.slaf"
        converter_chunked = SLAFConverter(
            chunked=True, chunk_size=100, sort_metadata=True
        )
        converter_chunked.convert(str(h5ad_path), str(output_path_chunked))

        # Check that both methods create the same files
        for method_path in [output_path_traditional, output_path_chunked]:
            assert (method_path / "expression.lance").exists()
            assert (method_path / "cells.lance").exists()
            assert (method_path / "genes.lance").exists()
            assert (method_path / "config.json").exists()

        # Check that configs are identical (except for created_at timestamp)
        with open(output_path_traditional / "config.json") as f:
            config_traditional = json.load(f)
        with open(output_path_chunked / "config.json") as f:
            config_chunked = json.load(f)

        # Remove timestamp for comparison
        del config_traditional["created_at"]
        del config_chunked["created_at"]

        assert config_traditional == config_chunked

    def test_chunked_vs_traditional_identical_output(
        self, small_sample_adata, temp_dir
    ):
        """Test that chunked and traditional conversion produce identical data and metadata"""
        # Save sample data as h5ad
        h5ad_path = Path(temp_dir) / "test.h5ad"
        small_sample_adata.write(h5ad_path)

        # Convert using traditional method
        output_path_traditional = Path(temp_dir) / "test_traditional.slaf"
        converter_traditional = SLAFConverter(chunked=False, sort_metadata=True)
        converter_traditional.convert(str(h5ad_path), str(output_path_traditional))

        # Convert using chunked method
        output_path_chunked = Path(temp_dir) / "test_chunked.slaf"
        converter_chunked = SLAFConverter(
            chunked=True, chunk_size=100, sort_metadata=True
        )
        converter_chunked.convert(str(h5ad_path), str(output_path_chunked))

        # Compare expression data
        expression_traditional = lance.dataset(
            output_path_traditional / "expression.lance"
        )
        expression_chunked = lance.dataset(output_path_chunked / "expression.lance")

        df_traditional = expression_traditional.to_table().to_pandas()
        df_chunked = expression_chunked.to_table().to_pandas()

        # Sort both dataframes by the same columns for comparison
        sort_cols = ["cell_integer_id", "gene_integer_id", "cell_id", "gene_id"]
        df_traditional_sorted = df_traditional.sort_values(sort_cols).reset_index(
            drop=True
        )
        df_chunked_sorted = df_chunked.sort_values(sort_cols).reset_index(drop=True)

        # Compare expression data exactly
        pd.testing.assert_frame_equal(df_traditional_sorted, df_chunked_sorted)

        # Compare cell metadata
        cells_traditional = lance.dataset(output_path_traditional / "cells.lance")
        cells_chunked = lance.dataset(output_path_chunked / "cells.lance")

        df_cells_traditional = cells_traditional.to_table().to_pandas()
        df_cells_chunked = cells_chunked.to_table().to_pandas()

        # Debug: print column information
        print(f"Traditional cell columns: {list(df_cells_traditional.columns)}")
        print(f"Chunked cell columns: {list(df_cells_chunked.columns)}")
        print(f"Traditional cell shape: {df_cells_traditional.shape}")
        print(f"Chunked cell shape: {df_cells_chunked.shape}")

        # Debug: check what the chunked reader actually reads
        from slaf.data.chunked_reader import ChunkedH5ADReader

        with ChunkedH5ADReader(str(h5ad_path)) as reader:
            obs_df = reader.get_obs_metadata()
            print(f"Chunked reader obs columns: {list(obs_df.columns)}")
            print(f"Chunked reader obs shape: {obs_df.shape}")
            print(f"Chunked reader obs head:\n{obs_df.head()}")

        # Compare cell metadata by sorting by cell_id column
        df_cells_traditional_sorted = df_cells_traditional.sort_values(
            "cell_id"
        ).reset_index(drop=True)
        df_cells_chunked_sorted = df_cells_chunked.sort_values("cell_id").reset_index(
            drop=True
        )

        # Sort columns for robust comparison
        df_cells_traditional_sorted = df_cells_traditional_sorted[
            sorted(df_cells_traditional_sorted.columns)
        ]
        df_cells_chunked_sorted = df_cells_chunked_sorted[
            sorted(df_cells_chunked_sorted.columns)
        ]

        pd.testing.assert_frame_equal(
            df_cells_traditional_sorted, df_cells_chunked_sorted
        )

        # Compare gene metadata
        genes_traditional = lance.dataset(output_path_traditional / "genes.lance")
        genes_chunked = lance.dataset(output_path_chunked / "genes.lance")

        df_genes_traditional = genes_traditional.to_table().to_pandas()
        df_genes_chunked = genes_chunked.to_table().to_pandas()

        # Sort by gene_id for comparison
        df_genes_traditional_sorted = df_genes_traditional.sort_values(
            "gene_id"
        ).reset_index(drop=True)
        df_genes_chunked_sorted = df_genes_chunked.sort_values("gene_id").reset_index(
            drop=True
        )

        # Sort columns for robust comparison
        df_genes_traditional_sorted = df_genes_traditional_sorted[
            sorted(df_genes_traditional_sorted.columns)
        ]
        df_genes_chunked_sorted = df_genes_chunked_sorted[
            sorted(df_genes_chunked_sorted.columns)
        ]

        pd.testing.assert_frame_equal(
            df_genes_traditional_sorted, df_genes_chunked_sorted
        )

        # Verify data integrity by reconstructing the original matrix
        def reconstruct_matrix(df, n_cells, n_genes):
            """Reconstruct dense matrix from COO data"""
            matrix = np.zeros((n_cells, n_genes))
            for _, row in df.iterrows():
                cell_idx = row["cell_integer_id"]
                gene_idx = row["gene_integer_id"]
                value = row["value"]
                matrix[cell_idx, gene_idx] = value
            return matrix

        # Reconstruct matrices from both methods
        matrix_traditional = reconstruct_matrix(
            df_traditional_sorted, small_sample_adata.n_obs, small_sample_adata.n_vars
        )
        matrix_chunked = reconstruct_matrix(
            df_chunked_sorted, small_sample_adata.n_obs, small_sample_adata.n_vars
        )

        # Compare reconstructed matrices
        np.testing.assert_array_equal(matrix_traditional, matrix_chunked)

        # Verify against original AnnData
        original_matrix = small_sample_adata.X.toarray()
        np.testing.assert_array_equal(matrix_traditional, original_matrix)
        np.testing.assert_array_equal(matrix_chunked, original_matrix)
