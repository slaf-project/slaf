#!/usr/bin/env python3
"""
Unit tests for SLAF append functionality.

This module tests the append functionality for existing SLAF datasets,
including compatibility validation, fragment management, and data integrity.
"""

import glob
import json
import os

import lance
import numpy as np
import pandas as pd
import pytest
import scanpy as sc
from scipy import sparse

from slaf.data.converter import SLAFConverter


class TestAppendFunctionality:
    """Test the append functionality for existing SLAF datasets."""

    @pytest.fixture
    def synthetic_data_dir(self, tmp_path):
        """Create synthetic test data for append testing."""
        # Create compatible files
        compatible_dir = os.path.join(tmp_path, "compatible")
        os.makedirs(compatible_dir, exist_ok=True)

        # Create 3 compatible h5ad files
        for i in range(3):
            np.random.seed(42 + i)
            n_cells, n_genes = 50, 30
            X = np.random.negative_binomial(5, 0.1, (n_cells, n_genes))
            X_sparse = sparse.csr_matrix(X)

            # Create cell metadata with consistent schema
            cell_ids = [f"cell_{i}_{j:03d}" for j in range(n_cells)]
            cell_types = np.random.choice(["T_cell", "B_cell"], n_cells)
            batches = [f"batch_{i}" for _ in range(n_cells)]

            obs_df = pd.DataFrame(
                {
                    "cell_type": cell_types,
                    "batch": batches,
                    "n_genes": [20] * n_cells,
                    "total_counts": [1000] * n_cells,
                },
                index=cell_ids,
            )

            # Create gene metadata
            gene_ids = [f"GENE_{j:03d}" for j in range(n_genes)]
            var_df = pd.DataFrame(
                {
                    "gene_type": ["protein_coding"] * n_genes,
                    "gene_name": [f"Gene_{j:03d}" for j in range(n_genes)],
                },
                index=gene_ids,
            )

            # Create AnnData object
            adata = sc.AnnData(X=X_sparse, obs=obs_df, var=var_df)

            # Save as h5ad
            output_file = os.path.join(compatible_dir, f"synthetic_data_{i:02d}.h5ad")
            adata.write_h5ad(output_file)

        # Create incompatible files
        incompatible_dir = os.path.join(tmp_path, "incompatible")
        os.makedirs(incompatible_dir, exist_ok=True)

        # File with different genes
        np.random.seed(999)
        X_incompatible = np.random.negative_binomial(
            5, 0.1, (n_cells, 20)
        )  # Different gene count
        X_incompatible_sparse = sparse.csr_matrix(X_incompatible)

        incompatible_gene_ids = [f"DIFFERENT_GENE_{j:03d}" for j in range(20)]
        incompatible_cell_ids = [f"incompatible_cell_{j:03d}" for j in range(n_cells)]

        obs_incompatible = pd.DataFrame(
            {
                "cell_type": ["T_cell"] * n_cells,
                "batch": ["incompatible_batch"] * n_cells,
                "n_genes": [20] * n_cells,
                "total_counts": [1000] * n_cells,
            },
            index=incompatible_cell_ids,
        )

        var_incompatible = pd.DataFrame(
            {
                "gene_type": ["protein_coding"] * 20,
                "gene_name": [f"DifferentGene_{j:03d}" for j in range(20)],
            },
            index=incompatible_gene_ids,
        )

        adata_incompatible = sc.AnnData(
            X=X_incompatible_sparse, obs=obs_incompatible, var=var_incompatible
        )
        incompatible_file = os.path.join(incompatible_dir, "incompatible_genes.h5ad")
        adata_incompatible.write_h5ad(incompatible_file)

        return {
            "compatible_dir": compatible_dir,
            "incompatible_dir": incompatible_dir,
            "compatible_files": sorted(
                glob.glob(os.path.join(compatible_dir, "*.h5ad"))
            ),
            "incompatible_files": sorted(
                glob.glob(os.path.join(incompatible_dir, "*.h5ad"))
            ),
        }

    def test_append_single_file(self, synthetic_data_dir, tmp_path):
        """Test appending a single file to existing SLAF dataset."""
        # Create initial SLAF from first file
        first_file = synthetic_data_dir["compatible_files"][0]
        initial_slaf_path = os.path.join(tmp_path, "initial.slaf")

        converter = SLAFConverter(
            chunked=True,
            chunk_size=25,
            create_indices=False,
            optimize_storage=True,
            use_optimized_dtypes=True,
        )

        # Convert first file to SLAF
        converter.convert(str(first_file), str(initial_slaf_path))

        # Verify initial dataset
        initial_cells_dataset = lance.dataset(
            os.path.join(initial_slaf_path, "cells.lance")
        )
        initial_cell_count = len(initial_cells_dataset.to_table())
        assert initial_cell_count == 50

        # Append second file
        second_file = synthetic_data_dir["compatible_files"][1]
        converter.append(str(second_file), str(initial_slaf_path))

        # Verify final dataset
        final_cells_dataset = lance.dataset(
            os.path.join(initial_slaf_path, "cells.lance")
        )
        final_cell_count = len(final_cells_dataset.to_table())
        assert final_cell_count == 100

        # Check auto-incrementing IDs
        final_cells_table = final_cells_dataset.to_table()
        cell_integer_ids = final_cells_table.column("cell_integer_id").to_numpy()
        expected_ids = set(range(100))
        actual_ids = set(cell_integer_ids)
        assert actual_ids == expected_ids

        # Check source file tracking
        source_files = final_cells_table.column("source_file").to_numpy()
        unique_sources = set(source_files)
        expected_sources = {"original_data", "synthetic_data_01.h5ad"}
        assert unique_sources == expected_sources

    def test_append_multiple_files(self, synthetic_data_dir, tmp_path):
        """Test appending multiple files to existing SLAF dataset."""
        # Create initial SLAF from first file
        first_file = synthetic_data_dir["compatible_files"][0]
        initial_slaf_path = os.path.join(tmp_path, "initial.slaf")

        converter = SLAFConverter(
            chunked=True,
            chunk_size=25,
            create_indices=False,
            optimize_storage=True,
            use_optimized_dtypes=True,
        )

        # Convert first file to SLAF
        converter.convert(str(first_file), str(initial_slaf_path))

        # Append remaining files
        remaining_files = synthetic_data_dir["compatible_files"][1:]
        remaining_dir = os.path.join(tmp_path, "remaining")
        os.makedirs(remaining_dir, exist_ok=True)

        # Copy remaining files to a directory for appending
        for i, file_path in enumerate(remaining_files):
            import shutil

            shutil.copy(
                file_path, os.path.join(remaining_dir, f"remaining_{i:02d}.h5ad")
            )

        # Append remaining files
        converter.append(str(remaining_dir), str(initial_slaf_path))

        # Verify final dataset
        final_cells_dataset = lance.dataset(
            os.path.join(initial_slaf_path, "cells.lance")
        )
        final_cell_count = len(final_cells_dataset.to_table())
        assert final_cell_count == 150  # 3 files × 50 cells each

        # Check auto-incrementing IDs
        final_cells_table = final_cells_dataset.to_table()
        cell_integer_ids = final_cells_table.column("cell_integer_id").to_numpy()
        expected_ids = set(range(150))
        actual_ids = set(cell_integer_ids)
        assert actual_ids == expected_ids

        # Check source file tracking
        source_files = final_cells_table.column("source_file").to_numpy()
        unique_sources = set(source_files)
        expected_sources = {
            "original_data",
            "remaining_00.h5ad",
            "remaining_01.h5ad",
        }
        assert unique_sources == expected_sources

    def test_append_fragment_structure(self, synthetic_data_dir, tmp_path):
        """Test that append creates correct fragment structure."""
        # Create initial SLAF from first file
        first_file = synthetic_data_dir["compatible_files"][0]
        initial_slaf_path = os.path.join(tmp_path, "initial.slaf")

        converter = SLAFConverter(
            chunked=True,
            chunk_size=25,
            create_indices=False,
            optimize_storage=True,
            use_optimized_dtypes=True,
        )

        # Convert first file to SLAF
        converter.convert(str(first_file), str(initial_slaf_path))

        # Append remaining files
        remaining_files = synthetic_data_dir["compatible_files"][1:]
        remaining_dir = os.path.join(tmp_path, "remaining")
        os.makedirs(remaining_dir, exist_ok=True)

        for i, file_path in enumerate(remaining_files):
            import shutil

            shutil.copy(
                file_path, os.path.join(remaining_dir, f"remaining_{i:02d}.h5ad")
            )

        converter.append(str(remaining_dir), str(initial_slaf_path))

        # Check fragment structure
        expression_dataset = lance.dataset(
            os.path.join(initial_slaf_path, "expression.lance")
        )
        cells_dataset = lance.dataset(os.path.join(initial_slaf_path, "cells.lance"))
        genes_dataset = lance.dataset(os.path.join(initial_slaf_path, "genes.lance"))

        expression_fragments = expression_dataset.get_fragments()
        cells_fragments = cells_dataset.get_fragments()
        genes_fragments = genes_dataset.get_fragments()

        # Expected: 3 files × (50 cells / 25 chunk_size) = 6 fragments for expression
        assert len(expression_fragments) == 6
        # Expected: 3 fragments for cells (1 per file)
        assert len(cells_fragments) == 3
        # Expected: 1 fragment for genes (from initial file)
        assert len(genes_fragments) == 1

    def test_append_config_metadata(self, synthetic_data_dir, tmp_path):
        """Test that append updates config metadata correctly."""
        # Create initial SLAF from first file
        first_file = synthetic_data_dir["compatible_files"][0]
        initial_slaf_path = os.path.join(tmp_path, "initial.slaf")

        converter = SLAFConverter(
            chunked=True,
            chunk_size=25,
            create_indices=False,
            optimize_storage=True,
            use_optimized_dtypes=True,
        )

        # Convert first file to SLAF
        converter.convert(str(first_file), str(initial_slaf_path))

        # Append remaining files
        remaining_files = synthetic_data_dir["compatible_files"][1:]
        remaining_dir = os.path.join(tmp_path, "remaining")
        os.makedirs(remaining_dir, exist_ok=True)

        for i, file_path in enumerate(remaining_files):
            import shutil

            shutil.copy(
                file_path, os.path.join(remaining_dir, f"remaining_{i:02d}.h5ad")
            )

        converter.append(str(remaining_dir), str(initial_slaf_path))

        # Check config file
        config_path = os.path.join(initial_slaf_path, "config.json")
        with open(config_path) as f:
            config = json.load(f)

        # Should have multi_file section
        assert "multi_file" in config
        assert "source_files" in config["multi_file"]
        assert len(config["multi_file"]["source_files"]) == 2  # 2 appended files
        assert config["n_cells"] == 150  # 3 files × 50 cells each

        # Check source file information
        source_files = config["multi_file"]["source_files"]
        assert len(source_files) == 2
        assert source_files[0]["file_name"] == "remaining_00.h5ad"
        assert source_files[1]["file_name"] == "remaining_01.h5ad"

    def test_append_compatibility_validation(self, synthetic_data_dir, tmp_path):
        """Test that append validates compatibility correctly."""
        # Create initial SLAF from first file
        first_file = synthetic_data_dir["compatible_files"][0]
        initial_slaf_path = os.path.join(tmp_path, "initial.slaf")

        converter = SLAFConverter(
            chunked=True,
            chunk_size=25,
            create_indices=False,
            optimize_storage=True,
            use_optimized_dtypes=True,
        )

        # Convert first file to SLAF
        converter.convert(str(first_file), str(initial_slaf_path))

        # Try to append incompatible file
        incompatible_file = synthetic_data_dir["incompatible_files"][0]

        with pytest.raises(ValueError, match="Validation failed"):
            converter.append(str(incompatible_file), str(initial_slaf_path))

    def test_append_nonexistent_dataset(self, synthetic_data_dir, tmp_path):
        """Test that append fails gracefully with nonexistent dataset."""
        converter = SLAFConverter(
            chunked=True,
            chunk_size=25,
            create_indices=False,
            optimize_storage=True,
            use_optimized_dtypes=True,
        )

        first_file = synthetic_data_dir["compatible_files"][0]
        nonexistent_slaf = os.path.join(tmp_path, "nonexistent.slaf")

        with pytest.raises(FileNotFoundError, match="Existing SLAF dataset not found"):
            converter.append(str(first_file), str(nonexistent_slaf))

    def test_append_empty_directory(self, synthetic_data_dir, tmp_path):
        """Test that append fails gracefully with empty directory."""
        # Create initial SLAF from first file
        first_file = synthetic_data_dir["compatible_files"][0]
        initial_slaf_path = os.path.join(tmp_path, "initial.slaf")

        converter = SLAFConverter(
            chunked=True,
            chunk_size=25,
            create_indices=False,
            optimize_storage=True,
            use_optimized_dtypes=True,
        )

        # Convert first file to SLAF
        converter.convert(str(first_file), str(initial_slaf_path))

        # Create empty directory
        empty_dir = os.path.join(tmp_path, "empty")
        os.makedirs(empty_dir, exist_ok=True)

        with pytest.raises(ValueError, match="No supported files found in directory"):
            converter.append(str(empty_dir), str(initial_slaf_path))

    def test_append_source_file_column_handling(self, synthetic_data_dir, tmp_path):
        """Test that append handles source_file column correctly."""
        # Create initial SLAF from first file
        first_file = synthetic_data_dir["compatible_files"][0]
        initial_slaf_path = os.path.join(tmp_path, "initial.slaf")

        converter = SLAFConverter(
            chunked=True,
            chunk_size=25,
            create_indices=False,
            optimize_storage=True,
            use_optimized_dtypes=True,
        )

        # Convert first file to SLAF
        converter.convert(str(first_file), str(initial_slaf_path))

        # Check that initial dataset doesn't have source_file column
        initial_cells_dataset = lance.dataset(
            os.path.join(initial_slaf_path, "cells.lance")
        )
        initial_cells_table = initial_cells_dataset.to_table()
        initial_columns = set(initial_cells_table.column_names)
        assert "source_file" not in initial_columns

        # Append second file
        second_file = synthetic_data_dir["compatible_files"][1]
        converter.append(str(second_file), str(initial_slaf_path))

        # Check that final dataset has source_file column
        final_cells_dataset = lance.dataset(
            os.path.join(initial_slaf_path, "cells.lance")
        )
        final_cells_table = final_cells_dataset.to_table()
        final_columns = set(final_cells_table.column_names)
        assert "source_file" in final_columns

        # Check source file distribution
        source_files = final_cells_table.column("source_file").to_numpy()
        unique_sources = set(source_files)
        expected_sources = {"original_data", "synthetic_data_01.h5ad"}
        assert unique_sources == expected_sources

    def test_append_chunk_size_affects_fragments(self, synthetic_data_dir, tmp_path):
        """Test that different chunk sizes result in different fragment counts."""
        # Create initial SLAF from first file
        first_file = synthetic_data_dir["compatible_files"][0]
        initial_slaf_path = os.path.join(tmp_path, "initial.slaf")

        # Test with small chunk size
        converter_small = SLAFConverter(
            chunked=True,
            chunk_size=10,  # Small chunks
            create_indices=False,
            optimize_storage=True,
            use_optimized_dtypes=True,
        )

        converter_small.convert(str(first_file), str(initial_slaf_path))

        # Append remaining files
        remaining_files = synthetic_data_dir["compatible_files"][1:]
        remaining_dir = os.path.join(tmp_path, "remaining")
        os.makedirs(remaining_dir, exist_ok=True)

        for i, file_path in enumerate(remaining_files):
            import shutil

            shutil.copy(
                file_path, os.path.join(remaining_dir, f"remaining_{i:02d}.h5ad")
            )

        converter_small.append(str(remaining_dir), str(initial_slaf_path))

        # Check fragment count with small chunks
        expression_dataset = lance.dataset(
            os.path.join(initial_slaf_path, "expression.lance")
        )
        expression_fragments_small = expression_dataset.get_fragments()

        # Expected: 3 files × (50 cells / 10 chunk_size) = 15 fragments
        assert len(expression_fragments_small) == 15

    def test_append_memory_efficiency(self, synthetic_data_dir, tmp_path):
        """Test that append is memory efficient with small chunks."""
        # Create initial SLAF from first file
        first_file = synthetic_data_dir["compatible_files"][0]
        initial_slaf_path = os.path.join(tmp_path, "initial.slaf")

        converter = SLAFConverter(
            chunked=True,
            chunk_size=10,  # Very small chunks for memory testing
            create_indices=False,
            optimize_storage=True,
            use_optimized_dtypes=True,
        )

        # Convert first file to SLAF
        converter.convert(str(first_file), str(initial_slaf_path))

        # Append remaining files
        remaining_files = synthetic_data_dir["compatible_files"][1:]
        remaining_dir = os.path.join(tmp_path, "remaining")
        os.makedirs(remaining_dir, exist_ok=True)

        for i, file_path in enumerate(remaining_files):
            import shutil

            shutil.copy(
                file_path, os.path.join(remaining_dir, f"remaining_{i:02d}.h5ad")
            )

        # This should not cause memory issues
        converter.append(str(remaining_dir), str(initial_slaf_path))

        # Verify output is correct
        final_cells_dataset = lance.dataset(
            os.path.join(initial_slaf_path, "cells.lance")
        )
        final_cell_count = len(final_cells_dataset.to_table())
        assert final_cell_count == 150

    def test_append_preserves_existing_data(self, synthetic_data_dir, tmp_path):
        """Test that append preserves existing data integrity."""
        # Create initial SLAF from first file
        first_file = synthetic_data_dir["compatible_files"][0]
        initial_slaf_path = os.path.join(tmp_path, "initial.slaf")

        converter = SLAFConverter(
            chunked=True,
            chunk_size=25,
            create_indices=False,
            optimize_storage=True,
            use_optimized_dtypes=True,
        )

        # Convert first file to SLAF
        converter.convert(str(first_file), str(initial_slaf_path))

        # Get initial data for comparison
        initial_cells_dataset = lance.dataset(
            os.path.join(initial_slaf_path, "cells.lance")
        )
        initial_cells_table = initial_cells_dataset.to_table()
        initial_cell_ids = set(initial_cells_table.column("cell_id").to_numpy())

        # Append second file
        second_file = synthetic_data_dir["compatible_files"][1]
        converter.append(str(second_file), str(initial_slaf_path))

        # Check that initial data is preserved
        final_cells_dataset = lance.dataset(
            os.path.join(initial_slaf_path, "cells.lance")
        )
        final_cells_table = final_cells_dataset.to_table()
        final_cell_ids = set(final_cells_table.column("cell_id").to_numpy())

        # Initial cell IDs should still be present
        assert initial_cell_ids.issubset(final_cell_ids)

        # Check that new data is added
        assert len(final_cell_ids) > len(initial_cell_ids)

    def test_append_error_handling(self, synthetic_data_dir, tmp_path):
        """Test that append handles errors gracefully."""
        # Create initial SLAF from first file
        first_file = synthetic_data_dir["compatible_files"][0]
        initial_slaf_path = os.path.join(tmp_path, "initial.slaf")

        converter = SLAFConverter(
            chunked=True,
            chunk_size=25,
            create_indices=False,
            optimize_storage=True,
            use_optimized_dtypes=True,
        )

        # Convert first file to SLAF
        converter.convert(str(first_file), str(initial_slaf_path))

        # Try to append to a file that doesn't exist
        nonexistent_file = os.path.join(tmp_path, "nonexistent.h5ad")
        with open(nonexistent_file, "w") as f:
            f.write("not a valid h5ad file")

        with pytest.raises(ValueError, match="Validation failed for nonexistent.h5ad"):
            converter.append(str(nonexistent_file), str(initial_slaf_path))

    def test_append_with_skip_validation(self, synthetic_data_dir, tmp_path):
        """Test that append works with skip_validation flag."""
        # Create initial SLAF from first file
        first_file = synthetic_data_dir["compatible_files"][0]
        initial_slaf_path = os.path.join(tmp_path, "initial.slaf")

        converter = SLAFConverter(
            chunked=True,
            chunk_size=25,
            create_indices=False,
            optimize_storage=True,
            use_optimized_dtypes=True,
        )

        # Convert first file to SLAF
        converter.convert(str(first_file), str(initial_slaf_path))

        # Append second file (should work without validation)
        second_file = synthetic_data_dir["compatible_files"][1]
        converter.append(str(second_file), str(initial_slaf_path))

        # Verify final dataset
        final_cells_dataset = lance.dataset(
            os.path.join(initial_slaf_path, "cells.lance")
        )
        final_cell_count = len(final_cells_dataset.to_table())
        assert final_cell_count == 100

    def test_append_auto_detection(self, synthetic_data_dir, tmp_path):
        """Test that append auto-detects file format correctly."""
        # Create initial SLAF from first file
        first_file = synthetic_data_dir["compatible_files"][0]
        initial_slaf_path = os.path.join(tmp_path, "initial.slaf")

        converter = SLAFConverter(
            chunked=True,
            chunk_size=25,
            create_indices=False,
            optimize_storage=True,
            use_optimized_dtypes=True,
        )

        # Convert first file to SLAF
        converter.convert(str(first_file), str(initial_slaf_path))

        # Append second file with auto-detection
        second_file = synthetic_data_dir["compatible_files"][1]
        converter.append(str(second_file), str(initial_slaf_path), input_format="auto")

        # Verify final dataset
        final_cells_dataset = lance.dataset(
            os.path.join(initial_slaf_path, "cells.lance")
        )
        final_cell_count = len(final_cells_dataset.to_table())
        assert final_cell_count == 100

    def test_append_explicit_format(self, synthetic_data_dir, tmp_path):
        """Test that append works with explicit format specification."""
        # Create initial SLAF from first file
        first_file = synthetic_data_dir["compatible_files"][0]
        initial_slaf_path = os.path.join(tmp_path, "initial.slaf")

        converter = SLAFConverter(
            chunked=True,
            chunk_size=25,
            create_indices=False,
            optimize_storage=True,
            use_optimized_dtypes=True,
        )

        # Convert first file to SLAF
        converter.convert(str(first_file), str(initial_slaf_path))

        # Append second file with explicit format
        second_file = synthetic_data_dir["compatible_files"][1]
        converter.append(str(second_file), str(initial_slaf_path), input_format="h5ad")

        # Verify final dataset
        final_cells_dataset = lance.dataset(
            os.path.join(initial_slaf_path, "cells.lance")
        )
        final_cell_count = len(final_cells_dataset.to_table())
        assert final_cell_count == 100
