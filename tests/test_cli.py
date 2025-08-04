"""Tests for SLAF CLI functionality."""

import subprocess
from pathlib import Path
from unittest.mock import Mock, patch

import pytest
import typer
from typer.testing import CliRunner

from slaf.cli import app


class TestCLI:
    """Test CLI functionality."""

    @pytest.fixture
    def runner(self):
        """Create CLI runner for testing."""
        return CliRunner()

    def test_version_command(self, runner):
        """Test version command."""
        result = runner.invoke(app, ["version"])
        assert result.exit_code == 0
        assert "SLAF version" in result.stdout

    @patch("slaf.cli.check_dependencies")
    @patch("slaf.cli.Path.exists")
    @patch("slaf.cli.subprocess.run")
    def test_docs_build(self, mock_run, mock_exists, mock_check_deps, runner):
        """Test docs build command."""
        mock_exists.return_value = True
        mock_run.return_value = Mock(returncode=0)

        result = runner.invoke(app, ["docs", "--build"])
        assert result.exit_code == 0
        assert "Building documentation" in result.stdout
        mock_run.assert_called_once()

    @patch("slaf.cli.check_dependencies")
    @patch("slaf.cli.Path.exists")
    def test_docs_no_mkdocs_yml(self, mock_exists, mock_check_deps, runner):
        """Test docs command when mkdocs.yml is missing."""
        mock_exists.return_value = False

        result = runner.invoke(app, ["docs", "--build"])
        assert result.exit_code == 1
        assert "mkdocs.yml not found" in result.stdout

    @patch("slaf.cli.check_dependencies")
    @patch("slaf.cli.Path.exists")
    @patch("slaf.cli.subprocess.run")
    def test_docs_serve(self, mock_run, mock_exists, mock_check_deps, runner):
        """Test docs serve command."""
        mock_exists.return_value = True
        mock_run.side_effect = KeyboardInterrupt()

        result = runner.invoke(app, ["docs", "--serve"])
        assert result.exit_code == 0
        assert "Starting documentation server" in result.stdout

    @patch("slaf.cli.check_dependencies")
    @patch("slaf.cli.Path.exists")
    def test_examples_list(self, mock_exists, mock_check_deps, runner):
        """Test examples list command."""
        mock_exists.return_value = True

        # Mock examples directory with some files
        with patch("slaf.cli.Path") as mock_path:
            mock_examples_dir = Mock()
            mock_examples_dir.glob.return_value = [
                Path("examples/01-getting-started.py"),
                Path("examples/02-lazy-processing.py"),
            ]
            mock_path.return_value = mock_examples_dir

            result = runner.invoke(app, ["examples", "--list"])
            assert result.exit_code == 0
            assert "Available examples" in result.stdout

    @patch("slaf.cli.check_dependencies")
    @patch("slaf.cli.Path.exists")
    def test_examples_no_directory(self, mock_exists, mock_check_deps, runner):
        """Test examples command when directory doesn't exist."""
        mock_exists.return_value = False

        result = runner.invoke(app, ["examples", "--list"])
        assert result.exit_code == 1
        assert "Examples directory not found" in result.stdout

    @patch("slaf.cli.check_dependencies")
    @patch("slaf.cli.subprocess.run")
    def test_convert_command(self, mock_run, mock_check_deps, runner):
        """Test convert command."""

        # Patch Path.exists to simulate input file exists, output dir does not
        def exists_side_effect(self):
            return str(self).endswith("input.h5ad")

        with patch("slaf.cli.Path.exists", new=exists_side_effect):
            with patch("slaf.data.SLAFConverter", create=True) as mock_converter:
                mock_converter_instance = Mock()
                mock_converter.return_value = mock_converter_instance
                result = runner.invoke(app, ["convert", "input.h5ad", "output_dir"])
                assert result.exit_code == 0
                assert "Converting" in result.stdout

    @patch("slaf.cli.check_dependencies")
    def test_convert_input_not_found(self, mock_check_deps, runner):
        """Test convert command when input file doesn't exist."""
        with patch("slaf.cli.Path.exists", return_value=False):
            result = runner.invoke(app, ["convert", "nonexistent.h5ad", "output_dir"])
            assert result.exit_code == 1
            assert "Input file not found" in result.stdout

    @patch("slaf.cli.check_dependencies")
    def test_info_command(self, mock_check_deps, runner):
        """Test info command."""
        with patch("slaf.cli.Path") as mock_path:
            mock_path_instance = Mock()
            mock_path_instance.exists.return_value = True
            mock_path.return_value = mock_path_instance
            with patch("slaf.SLAFArray", create=True) as mock_slaf:
                mock_dataset = Mock()
                mock_slaf.return_value = mock_dataset
                result = runner.invoke(app, ["info", "test_dataset"])
                assert result.exit_code == 0

    @patch("slaf.cli.check_dependencies")
    def test_info_dataset_not_found(self, mock_check_deps, runner):
        """Test info command when dataset doesn't exist."""
        with patch("slaf.cli.Path.exists", return_value=False):
            result = runner.invoke(app, ["info", "nonexistent_dataset"])
            assert result.exit_code == 1
            assert "Dataset not found" in result.stdout

    @patch("slaf.cli.check_dependencies")
    def test_query_command(self, mock_check_deps, runner):
        """Test query command."""
        with patch("slaf.cli.Path") as mock_path:
            mock_path_instance = Mock()
            mock_path_instance.exists.return_value = True
            mock_path.return_value = mock_path_instance
            with patch("slaf.SLAFArray", create=True) as mock_slaf:
                mock_dataset = Mock()
                mock_dataset.query.return_value = Mock()
                mock_slaf.return_value = mock_dataset
                result = runner.invoke(
                    app, ["query", "test_dataset", "SELECT * FROM expression LIMIT 5"]
                )
                assert result.exit_code == 0

    @patch("slaf.cli.check_dependencies")
    @patch("slaf.cli.Path.exists")
    def test_query_dataset_not_found(self, mock_exists, mock_check_deps, runner):
        """Test query command when dataset doesn't exist."""
        mock_exists.return_value = False

        result = runner.invoke(
            app, ["query", "nonexistent_dataset", "SELECT * FROM expression"]
        )
        assert result.exit_code == 1
        assert "Dataset not found" in result.stdout

    @patch("slaf.cli.check_dependencies")
    @patch("slaf.cli.Path.exists")
    @patch("slaf.cli.subprocess.run")
    def test_release_build(self, mock_run, mock_exists, mock_check_deps, runner):
        """Test release build command."""
        mock_exists.return_value = True
        mock_run.return_value = Mock(returncode=0)

        result = runner.invoke(app, ["release", "build"])
        assert result.exit_code == 0
        assert "Building package" in result.stdout

    @patch("slaf.cli.check_dependencies")
    @patch("slaf.cli.Path.exists")
    @patch("slaf.cli.subprocess.run")
    def test_release_test(self, mock_run, mock_exists, mock_check_deps, runner):
        """Test release test command."""
        mock_exists.return_value = True
        mock_run.return_value = Mock(returncode=0)

        result = runner.invoke(app, ["release", "test"])
        assert result.exit_code == 0
        assert "Running tests" in result.stdout

    @patch("slaf.cli.check_dependencies")
    @patch("slaf.cli.Path.exists")
    @patch("slaf.cli.subprocess.run")
    def test_release_check(self, mock_run, mock_exists, mock_check_deps, runner):
        """Test release check command."""
        mock_exists.return_value = True
        mock_run.return_value = Mock(returncode=0)

        result = runner.invoke(app, ["release", "check"])
        assert result.exit_code == 0
        assert "Checking package" in result.stdout

    @patch("slaf.cli.check_dependencies")
    @patch("slaf.cli.Path.exists")
    @patch("slaf.cli.subprocess.run")
    def test_release_publish(self, mock_run, mock_exists, mock_check_deps, runner):
        """Test release publish command."""
        mock_exists.return_value = True
        mock_run.return_value = Mock(returncode=0)

        # Mock the functions that publish calls
        with patch("slaf.cli.update_version"):
            with patch("slaf.cli.run_tests"):
                with patch("slaf.cli.build_package"):
                    with patch("slaf.cli.check_package"):
                        with patch("slaf.cli.create_tag"):
                            result = runner.invoke(
                                app, ["release", "publish", "--version", "0.2.0"]
                            )
                            assert result.exit_code == 0
                            assert "Publishing version" in result.stdout

    @patch("slaf.cli.check_dependencies")
    @patch("slaf.cli.Path.exists")
    @patch("slaf.cli.subprocess.run")
    def test_release_prepare(self, mock_run, mock_exists, mock_check_deps, runner):
        """Test release prepare command."""
        mock_exists.return_value = True
        mock_run.return_value = Mock(returncode=0)

        # Mock get_current_version and update_version
        with patch("slaf.cli.get_current_version") as mock_get_version:
            with patch("slaf.cli.update_version"):
                with patch("slaf.cli.generate_changelog"):
                    mock_get_version.return_value = "0.1.0"

                    result = runner.invoke(app, ["release", "prepare"])
                    assert result.exit_code == 0
                    assert "Preparing release" in result.stdout

    @patch("slaf.cli.check_dependencies")
    @patch("slaf.cli.Path.exists")
    @patch("slaf.cli.subprocess.run")
    def test_release_prepare_with_version(
        self, mock_run, mock_exists, mock_check_deps, runner
    ):
        """Test release prepare command with specific version."""
        mock_exists.return_value = True
        mock_run.return_value = Mock(returncode=0)

        # Mock update_version and generate_changelog
        with patch("slaf.cli.update_version"):
            with patch("slaf.cli.generate_changelog"):
                result = runner.invoke(
                    app, ["release", "prepare", "--version", "0.2.0"]
                )
                assert result.exit_code == 0
                assert "Preparing release" in result.stdout

    @patch("slaf.cli.check_dependencies")
    @patch("slaf.cli.Path.exists")
    @patch("slaf.cli.subprocess.run")
    def test_release_patch(self, mock_run, mock_exists, mock_check_deps, runner):
        """Test release patch command."""
        mock_exists.return_value = True
        mock_run.return_value = Mock(returncode=0)

        # Mock get_current_version and update_version
        with patch("slaf.cli.get_current_version") as mock_get_version:
            with patch("slaf.cli.update_version"):
                with patch("slaf.cli.generate_changelog"):
                    mock_get_version.return_value = "0.1.0"

                    result = runner.invoke(
                        app, ["release", "prepare", "--type", "patch"]
                    )
                    assert result.exit_code == 0
                    assert "Preparing release" in result.stdout

    @patch("slaf.cli.check_dependencies")
    @patch("slaf.cli.Path.exists")
    @patch("slaf.cli.subprocess.run")
    def test_release_minor(self, mock_run, mock_exists, mock_check_deps, runner):
        """Test release minor command."""
        mock_exists.return_value = True
        mock_run.return_value = Mock(returncode=0)

        # Mock get_current_version and update_version
        with patch("slaf.cli.get_current_version") as mock_get_version:
            with patch("slaf.cli.update_version"):
                with patch("slaf.cli.generate_changelog"):
                    mock_get_version.return_value = "0.1.0"

                    result = runner.invoke(
                        app, ["release", "prepare", "--type", "minor"]
                    )
                    assert result.exit_code == 0
                    assert "Preparing release" in result.stdout

    @patch("slaf.cli.check_dependencies")
    @patch("slaf.cli.Path.exists")
    @patch("slaf.cli.subprocess.run")
    def test_release_major(self, mock_run, mock_exists, mock_check_deps, runner):
        """Test release major command."""
        mock_exists.return_value = True
        mock_run.return_value = Mock(returncode=0)

        # Mock get_current_version and update_version
        with patch("slaf.cli.get_current_version") as mock_get_version:
            with patch("slaf.cli.update_version"):
                with patch("slaf.cli.generate_changelog"):
                    mock_get_version.return_value = "0.1.0"

                    result = runner.invoke(
                        app, ["release", "prepare", "--type", "major"]
                    )
                    assert result.exit_code == 0
                    assert "Preparing release" in result.stdout

    def test_invalid_release_action(self, runner):
        """Test release command with invalid action."""
        result = runner.invoke(app, ["release", "invalid_action"])
        assert result.exit_code == 1
        assert "Unknown action" in result.stdout

    @patch("slaf.cli.importlib.util.find_spec")
    def test_check_dependencies_missing(self, mock_find_spec):
        """Test check_dependencies with missing packages."""
        mock_find_spec.return_value = None

        with pytest.raises(typer.Exit):
            from slaf.cli import check_dependencies

            check_dependencies()

    @patch("slaf.cli.importlib.util.find_spec")
    def test_check_dependencies_all_present(self, mock_find_spec):
        """Test check_dependencies with all packages present."""
        mock_find_spec.return_value = Mock()

        from slaf.cli import check_dependencies

        # Should not raise an exception
        check_dependencies()

    @patch("slaf.cli.subprocess.run")
    def test_run_command_success(self, mock_run):
        """Test run_command with successful execution."""
        mock_run.return_value = Mock(returncode=0)

        from slaf.cli import run_command

        result = run_command("echo test")
        assert result.returncode == 0

    @patch("slaf.cli.subprocess.run")
    def test_run_command_failure(self, mock_run):
        """Test run_command with failed execution."""
        mock_run.side_effect = subprocess.CalledProcessError(1, "echo test")

        from slaf.cli import run_command

        with pytest.raises(subprocess.CalledProcessError):
            run_command("echo test")

    def test_calculate_new_version(self):
        """Test calculate_new_version function."""
        from slaf.cli import calculate_new_version

        # Test patch
        assert calculate_new_version("1.2.3", "patch") == "1.2.4"
        assert calculate_new_version("1.2.3", "minor") == "1.3.0"
        assert calculate_new_version("1.2.3", "major") == "2.0.0"

        # Test edge cases
        assert calculate_new_version("0.1.0", "patch") == "0.1.1"
        assert calculate_new_version("0.1.0", "minor") == "0.2.0"
        assert calculate_new_version("0.1.0", "major") == "1.0.0"

    # Multi-format conversion tests
    @patch("slaf.cli.check_dependencies")
    def test_convert_auto_detection_h5ad(self, mock_check_deps, runner):
        """Test convert command with auto-detection for h5ad files."""

        def exists_side_effect(self):
            return str(self).endswith("input.h5ad")

        with patch("slaf.cli.Path.exists", new=exists_side_effect):
            with patch("slaf.data.SLAFConverter", create=True) as mock_converter:
                mock_converter_instance = Mock()
                mock_converter.return_value = mock_converter_instance

                result = runner.invoke(app, ["convert", "input.h5ad", "output_dir"])

                assert result.exit_code == 0
                assert "Converting" in result.stdout
                # Verify auto-detection was used (no explicit format)
                mock_converter_instance.convert.assert_called_once_with(
                    "input.h5ad", "output_dir"
                )

    @patch("slaf.cli.check_dependencies")
    def test_convert_explicit_format_h5ad(self, mock_check_deps, runner):
        """Test convert command with explicit h5ad format specification."""

        def exists_side_effect(self):
            return str(self).endswith("input.h5ad")

        with patch("slaf.cli.Path.exists", new=exists_side_effect):
            with patch("slaf.data.SLAFConverter", create=True) as mock_converter:
                mock_converter_instance = Mock()
                mock_converter.return_value = mock_converter_instance

                result = runner.invoke(
                    app, ["convert", "input.h5ad", "output_dir", "--format", "h5ad"]
                )

                assert result.exit_code == 0
                assert "Converting" in result.stdout
                # Verify explicit format was used
                mock_converter_instance.convert.assert_called_once_with(
                    "input.h5ad", "output_dir", input_format="h5ad"
                )

    @patch("slaf.cli.check_dependencies")
    def test_convert_explicit_format_10x_mtx(self, mock_check_deps, runner):
        """Test convert command with explicit 10x_mtx format specification."""

        def exists_side_effect(self):
            return str(self).endswith("mtx_dir")

        with patch("slaf.cli.Path.exists", new=exists_side_effect):
            with patch("slaf.data.SLAFConverter", create=True) as mock_converter:
                mock_converter_instance = Mock()
                mock_converter.return_value = mock_converter_instance

                result = runner.invoke(
                    app, ["convert", "mtx_dir", "output_dir", "--format", "10x_mtx"]
                )

                assert result.exit_code == 0
                assert "Converting" in result.stdout
                # Verify explicit format was used
                mock_converter_instance.convert.assert_called_once_with(
                    "mtx_dir", "output_dir", input_format="10x_mtx"
                )

    @patch("slaf.cli.check_dependencies")
    def test_convert_explicit_format_10x_h5(self, mock_check_deps, runner):
        """Test convert command with explicit 10x_h5 format specification."""

        def exists_side_effect(self):
            return str(self).endswith("data.h5")

        with patch("slaf.cli.Path.exists", new=exists_side_effect):
            with patch("slaf.data.SLAFConverter", create=True) as mock_converter:
                mock_converter_instance = Mock()
                mock_converter.return_value = mock_converter_instance

                result = runner.invoke(
                    app, ["convert", "data.h5", "output_dir", "--format", "10x_h5"]
                )

                assert result.exit_code == 0
                assert "Converting" in result.stdout
                # Verify explicit format was used
                mock_converter_instance.convert.assert_called_once_with(
                    "data.h5", "output_dir", input_format="10x_h5"
                )

    @patch("slaf.cli.check_dependencies")
    def test_convert_format_mapping_anndata(self, mock_check_deps, runner):
        """Test convert command with legacy 'anndata' format mapping."""

        def exists_side_effect(self):
            return str(self).endswith("input.h5ad")

        with patch("slaf.cli.Path.exists", new=exists_side_effect):
            with patch("slaf.data.SLAFConverter", create=True) as mock_converter:
                mock_converter_instance = Mock()
                mock_converter.return_value = mock_converter_instance

                result = runner.invoke(
                    app, ["convert", "input.h5ad", "output_dir", "--format", "anndata"]
                )

                assert result.exit_code == 0
                assert "Converting" in result.stdout
                # Verify format mapping was applied
                mock_converter_instance.convert.assert_called_once_with(
                    "input.h5ad", "output_dir", input_format="h5ad"
                )

    @patch("slaf.cli.check_dependencies")
    def test_convert_format_mapping_hdf5(self, mock_check_deps, runner):
        """Test convert command with legacy 'hdf5' format mapping."""

        def exists_side_effect(self):
            return str(self).endswith("data.h5")

        with patch("slaf.cli.Path.exists", new=exists_side_effect):
            with patch("slaf.data.SLAFConverter", create=True) as mock_converter:
                mock_converter_instance = Mock()
                mock_converter.return_value = mock_converter_instance

                result = runner.invoke(
                    app, ["convert", "data.h5", "output_dir", "--format", "hdf5"]
                )

                assert result.exit_code == 0
                assert "Converting" in result.stdout
                # Verify format mapping was applied
                mock_converter_instance.convert.assert_called_once_with(
                    "data.h5", "output_dir", input_format="10x_h5"
                )

    @patch("slaf.cli.check_dependencies")
    def test_convert_chunked_processing(self, mock_check_deps, runner):
        """Test convert command with chunked processing."""

        def exists_side_effect(self):
            return str(self).endswith("input.h5ad")

        with patch("slaf.cli.Path.exists", new=exists_side_effect):
            with patch("slaf.data.SLAFConverter", create=True) as mock_converter:
                mock_converter_instance = Mock()
                mock_converter.return_value = mock_converter_instance

                result = runner.invoke(
                    app,
                    [
                        "convert",
                        "input.h5ad",
                        "output_dir",
                        "--chunked",
                        "--chunk-size",
                        "5000",
                    ],
                )

                assert result.exit_code == 0
                assert "Converting" in result.stdout
                assert "Using chunked processing" in result.stdout
                # Verify chunked processing was used
                mock_converter.assert_called_once_with(
                    chunked=True,
                    chunk_size=5000,
                    create_indices=False,
                    optimize_storage=True,
                    use_optimized_dtypes=True,
                    enable_v2_manifest=True,
                    compact_after_write=True,
                )

    @patch("slaf.cli.check_dependencies")
    def test_convert_verbose_output(self, mock_check_deps, runner):
        """Test convert command with verbose output."""

        def exists_side_effect(self):
            return str(self).endswith("input.h5ad")

        with patch("slaf.cli.Path.exists", new=exists_side_effect):
            with patch("slaf.data.SLAFConverter", create=True) as mock_converter:
                mock_converter_instance = Mock()
                mock_converter.return_value = mock_converter_instance

                # Mock SLAFArray for verbose output
                with patch("slaf.SLAFArray", create=True) as mock_slaf:
                    mock_dataset = Mock()
                    mock_dataset.shape = (1000, 20000)
                    mock_dataset.obs.columns = ["cell_type", "batch"]
                    mock_dataset.var.columns = ["gene_symbol", "highly_variable"]
                    mock_slaf.return_value = mock_dataset

                    result = runner.invoke(
                        app, ["convert", "input.h5ad", "output_dir", "--verbose"]
                    )

                    assert result.exit_code == 0
                    assert "Converting" in result.stdout
                    assert "Dataset info:" in result.stdout
                    assert "1,000 cells" in result.stdout
                    assert "20,000 genes" in result.stdout

    @patch("slaf.cli.check_dependencies")
    def test_convert_output_directory_exists(self, mock_check_deps, runner):
        """Test convert command when output directory already exists."""

        def exists_side_effect(self):
            return str(self).endswith("input.h5ad") or str(self).endswith("output_dir")

        with patch("slaf.cli.Path.exists", new=exists_side_effect):
            result = runner.invoke(app, ["convert", "input.h5ad", "output_dir"])
            assert result.exit_code == 1
            assert "Output directory already exists" in result.stdout

    @patch("slaf.cli.check_dependencies")
    def test_convert_conversion_error(self, mock_check_deps, runner):
        """Test convert command when conversion fails."""

        def exists_side_effect(self):
            return str(self).endswith("input.h5ad")

        with patch("slaf.cli.Path.exists", new=exists_side_effect):
            with patch("slaf.data.SLAFConverter", create=True) as mock_converter:
                mock_converter_instance = Mock()
                mock_converter_instance.convert.side_effect = ValueError(
                    "Conversion failed"
                )
                mock_converter.return_value = mock_converter_instance

                result = runner.invoke(app, ["convert", "input.h5ad", "output_dir"])
                assert result.exit_code == 1
                assert "Conversion failed" in result.stdout

    @patch("slaf.cli.check_dependencies")
    def test_convert_help(self, mock_check_deps, runner):
        """Test convert command help text."""
        result = runner.invoke(app, ["convert", "--help"])
        assert result.exit_code == 0
        assert "Input file or directory path" in result.stdout
        assert "Output SLAF directory path" in result.stdout
        assert "Input format" in result.stdout
        assert "h5ad," in result.stdout
        assert "10x_mtx," in result.stdout
        assert "10x_h5" in result.stdout
        assert "chunked" in result.stdout
        assert "verbose" in result.stdout

        # Benchmark command tests

    @pytest.mark.skip(
        reason="Benchmark tests disabled until benchmark module is refactored"
    )
    def test_benchmark_help(self, runner):
        """Test benchmark command help."""
        result = runner.invoke(app, ["benchmark", "--help"])
        assert result.exit_code == 0
        assert "run" in result.stdout
        assert "summary" in result.stdout
        assert "docs" in result.stdout
        assert "all" in result.stdout

    @pytest.mark.skip(
        reason="Benchmark tests disabled until benchmark module is refactored"
    )
    def test_benchmark_invalid_action(self, runner):
        """Test benchmark command with invalid action."""
        result = runner.invoke(app, ["benchmark", "invalid_action"])
        assert result.exit_code == 1
        assert "Unknown action" in result.stdout

    @pytest.mark.skip(
        reason="Benchmark tests disabled until benchmark module is refactored"
    )
    @patch("slaf.cli.check_dependencies")
    def test_benchmark_run_action(self, mock_check_deps, runner):
        """Test benchmark run action."""
        with patch("slaf.cli.Path") as mock_path:
            mock_path_instance = Mock()
            mock_path_instance.exists.return_value = True
            mock_path_instance.glob.return_value = [Path("data/pbmc3k_processed.h5ad")]
            mock_path.return_value = mock_path_instance

            # Mock benchmark functions
            with patch(
                "slaf.cli.run_benchmark_suite", create=True
            ) as mock_run_benchmark:
                mock_run_benchmark.return_value = {"test": "results"}

                result = runner.invoke(
                    app, ["benchmark", "run", "--datasets", "pbmc3k"]
                )
                assert result.exit_code == 0
                assert "Running SLAF benchmarks" in result.stdout

    @pytest.mark.skip(
        reason="Benchmark tests disabled until benchmark module is refactored"
    )
    @patch("slaf.cli.check_dependencies")
    def test_benchmark_summary_action(self, mock_check_deps, runner):
        """Test benchmark summary action."""
        with patch("slaf.cli.Path.exists", return_value=True):
            with patch(
                "slaf.cli.generate_benchmark_summary", create=True
            ) as mock_summary:
                result = runner.invoke(app, ["benchmark", "summary"])
                assert result.exit_code == 0
                assert "Generating benchmark summary" in result.stdout
                mock_summary.assert_called_once()

    @pytest.mark.skip(
        reason="Benchmark tests disabled until benchmark module is refactored"
    )
    @patch("slaf.cli.check_dependencies")
    def test_benchmark_docs_action(self, mock_check_deps, runner):
        """Test benchmark docs action."""
        with patch("slaf.cli.Path.exists", return_value=True):
            with patch("slaf.cli.update_performance_docs", create=True) as mock_docs:
                result = runner.invoke(app, ["benchmark", "docs"])
                assert result.exit_code == 0
                assert "Updating performance documentation" in result.stdout
                mock_docs.assert_called_once()

    @pytest.mark.skip(
        reason="Benchmark tests disabled until benchmark module is refactored"
    )
    @patch("slaf.cli.check_dependencies")
    def test_benchmark_all_action(self, mock_check_deps, runner):
        """Test benchmark all action."""
        with patch("slaf.cli.Path") as mock_path:
            mock_path_instance = Mock()
            mock_path_instance.exists.return_value = True
            mock_path_instance.glob.return_value = [Path("data/pbmc3k_processed.h5ad")]
            mock_path.return_value = mock_path_instance

            # Mock all benchmark functions
            with patch("slaf.cli.run_benchmark_suite", create=True) as mock_run:
                with patch("slaf.cli.generate_benchmark_summary", create=True):
                    with patch("slaf.cli.update_performance_docs", create=True):
                        mock_run.return_value = {"test": "results"}

                        result = runner.invoke(
                            app, ["benchmark", "all", "--datasets", "pbmc3k"]
                        )
                        assert result.exit_code == 0
                        assert "Running complete benchmark workflow" in result.stdout

    @pytest.mark.skip(
        reason="Benchmark tests disabled until benchmark module is refactored"
    )
    @patch("slaf.cli.check_dependencies")
    def test_benchmark_missing_results_file(self, mock_check_deps, runner):
        """Test benchmark summary with missing results file."""
        with patch("slaf.cli.Path.exists", return_value=False):
            result = runner.invoke(app, ["benchmark", "summary"])
            assert result.exit_code == 1
            assert "Results file not found" in result.stdout

    @pytest.mark.skip(
        reason="Benchmark tests disabled until benchmark module is refactored"
    )
    @patch("slaf.cli.check_dependencies")
    def test_benchmark_missing_summary_file(self, mock_check_deps, runner):
        """Test benchmark docs with missing summary file."""
        with patch("slaf.cli.Path.exists", side_effect=[True, False]):
            result = runner.invoke(app, ["benchmark", "docs"])
            assert result.exit_code == 1
            assert "Summary file not found" in result.stdout

    @pytest.mark.skip(
        reason="Benchmark tests disabled until benchmark module is refactored"
    )
    @patch("slaf.cli.check_dependencies")
    def test_benchmark_missing_docs_file(self, mock_check_deps, runner):
        """Test benchmark docs with missing docs file."""
        with patch("slaf.cli.Path.exists", side_effect=[True, True, False]):
            result = runner.invoke(app, ["benchmark", "docs"])
            assert result.exit_code == 1
            assert "Docs file not found" in result.stdout

    @patch("slaf.cli.check_dependencies")
    def test_convert_with_slaf_import_error(self, mock_check_deps, runner):
        """Test convert command when SLAF import fails."""

        def exists_side_effect(self):
            return str(self).endswith("input.h5ad")

        with patch("slaf.cli.Path.exists", new=exists_side_effect):
            with patch(
                "slaf.data.SLAFConverter", side_effect=ImportError("SLAF not found")
            ):
                result = runner.invoke(app, ["convert", "input.h5ad", "output_dir"])
                assert result.exit_code == 1
                assert "Conversion failed: SLAF not found" in result.stdout

    @patch("slaf.cli.check_dependencies")
    def test_info_with_slaf_import_error(self, mock_check_deps, runner):
        """Test info command when SLAF import fails."""
        with patch("slaf.cli.Path.exists", return_value=True):
            with patch("slaf.SLAFArray", side_effect=ImportError("SLAF not found")):
                result = runner.invoke(app, ["info", "test_dataset"])
                assert result.exit_code == 1
                assert "Failed to load dataset: SLAF not found" in result.stdout

    @patch("slaf.cli.check_dependencies")
    def test_query_with_slaf_import_error(self, mock_check_deps, runner):
        """Test query command when SLAF import fails."""
        with patch("slaf.cli.Path.exists", return_value=True):
            with patch("slaf.SLAFArray", side_effect=ImportError("SLAF not found")):
                result = runner.invoke(
                    app, ["query", "test_dataset", "SELECT * FROM expression"]
                )
                assert result.exit_code == 1
                assert "Query failed: SLAF not found" in result.stdout

    @patch("slaf.cli.check_dependencies")
    def test_query_with_output_file(self, mock_check_deps, runner):
        """Test query command with output file."""
        with patch("slaf.cli.Path") as mock_path:
            mock_path_instance = Mock()
            mock_path_instance.exists.return_value = True
            mock_path.return_value = mock_path_instance
            with patch("slaf.SLAFArray", create=True) as mock_slaf:
                mock_dataset = Mock()
                mock_result = Mock()
                mock_result.to_csv = Mock()
                mock_dataset.query.return_value = mock_result
                mock_slaf.return_value = mock_dataset

                result = runner.invoke(
                    app,
                    [
                        "query",
                        "test_dataset",
                        "SELECT * FROM expression",
                        "--output",
                        "results.csv",
                    ],
                )
                assert result.exit_code == 0
                assert "Results saved to results.csv" in result.stdout
                mock_result.to_csv.assert_called_once_with("results.csv", index=False)

    @patch("slaf.cli.check_dependencies")
    def test_query_with_polars_dataframe(self, mock_check_deps, runner):
        """Test query command with Polars DataFrame (simulating the new behavior)."""
        with patch("slaf.cli.Path") as mock_path:
            mock_path_instance = Mock()
            mock_path_instance.exists.return_value = True
            mock_path.return_value = mock_path_instance
            with patch("slaf.SLAFArray", create=True) as mock_slaf:
                mock_dataset = Mock()
                # Simulate a Polars DataFrame (no to_csv method, has write_csv)
                mock_result = Mock()
                # Remove to_csv method to simulate Polars DataFrame
                del mock_result.to_csv
                mock_result.write_csv = Mock()
                mock_dataset.query.return_value = mock_result
                mock_slaf.return_value = mock_dataset

                result = runner.invoke(
                    app,
                    [
                        "query",
                        "test_dataset",
                        "SELECT * FROM expression",
                        "--output",
                        "results.csv",
                    ],
                )
                assert result.exit_code == 0
                assert "Results saved to results.csv" in result.stdout
                mock_result.write_csv.assert_called_once_with("results.csv")

    @patch("slaf.cli.check_dependencies")
    def test_query_with_pandas_dataframe(self, mock_check_deps, runner):
        """Test query command with Pandas DataFrame (backward compatibility)."""
        with patch("slaf.cli.Path") as mock_path:
            mock_path_instance = Mock()
            mock_path_instance.exists.return_value = True
            mock_path.return_value = mock_path_instance
            with patch("slaf.SLAFArray", create=True) as mock_slaf:
                mock_dataset = Mock()
                # Simulate a Pandas DataFrame (has to_string method)
                mock_result = Mock()
                mock_result.to_string = Mock(return_value="pandas output")
                mock_result.to_csv = Mock()
                mock_dataset.query.return_value = mock_result
                mock_slaf.return_value = mock_dataset

                result = runner.invoke(
                    app,
                    [
                        "query",
                        "test_dataset",
                        "SELECT * FROM expression",
                    ],
                )
                assert result.exit_code == 0
                assert "Query results:" in result.stdout
                mock_result.to_string.assert_called_once_with(index=False)

    @patch("slaf.cli.check_dependencies")
    def test_query_display_polars_dataframe(self, mock_check_deps, runner):
        """Test query command display with Polars DataFrame."""
        with patch("slaf.cli.Path") as mock_path:
            mock_path_instance = Mock()
            mock_path_instance.exists.return_value = True
            mock_path.return_value = mock_path_instance
            with patch("slaf.SLAFArray", create=True) as mock_slaf:
                mock_dataset = Mock()
                # Simulate a Polars DataFrame
                mock_result = Mock()
                # Remove to_string method to simulate Polars DataFrame
                del mock_result.to_string
                mock_result.__str__ = Mock(return_value="polars output")
                mock_dataset.query.return_value = mock_result
                mock_slaf.return_value = mock_dataset

                result = runner.invoke(
                    app,
                    [
                        "query",
                        "test_dataset",
                        "SELECT * FROM expression",
                    ],
                )
                assert result.exit_code == 0
                assert "Query results:" in result.stdout
                assert "polars output" in result.stdout

    @patch("slaf.cli.check_dependencies")
    def test_query_with_limit(self, mock_check_deps, runner):
        """Test query command with custom limit."""
        with patch("slaf.cli.Path") as mock_path:
            mock_path_instance = Mock()
            mock_path_instance.exists.return_value = True
            mock_path.return_value = mock_path_instance
            with patch("slaf.SLAFArray", create=True) as mock_slaf:
                mock_dataset = Mock()
                mock_dataset.query.return_value = Mock()
                mock_slaf.return_value = mock_dataset

                result = runner.invoke(
                    app,
                    [
                        "query",
                        "test_dataset",
                        "SELECT * FROM expression",
                        "--limit",
                        "5",
                    ],
                )
                assert result.exit_code == 0
                # Verify the query was called with LIMIT 5
                mock_dataset.query.assert_called_once()
                call_args = mock_dataset.query.call_args[0][0]
                assert "LIMIT 5" in call_args

    @patch("slaf.cli.check_dependencies")
    def test_query_failure(self, mock_check_deps, runner):
        """Test query command when query fails."""
        with patch("slaf.cli.Path") as mock_path:
            mock_path_instance = Mock()
            mock_path_instance.exists.return_value = True
            mock_path.return_value = mock_path_instance
            with patch("slaf.SLAFArray", create=True) as mock_slaf:
                mock_dataset = Mock()
                mock_dataset.query.side_effect = ValueError("Invalid SQL")
                mock_slaf.return_value = mock_dataset

                result = runner.invoke(app, ["query", "test_dataset", "INVALID SQL"])
                assert result.exit_code == 1
                assert "Query failed" in result.stdout

    @patch("slaf.cli.check_dependencies")
    def test_info_failure(self, mock_check_deps, runner):
        """Test info command when dataset loading fails."""
        with patch("slaf.cli.Path") as mock_path:
            mock_path_instance = Mock()
            mock_path_instance.exists.return_value = True
            mock_path.return_value = mock_path_instance
            with patch("slaf.SLAFArray", create=True) as mock_slaf:
                mock_slaf.side_effect = ValueError("Invalid dataset")

                result = runner.invoke(app, ["info", "test_dataset"])
                assert result.exit_code == 1
                assert "Failed to load dataset" in result.stdout

    def test_get_current_version_success(self):
        """Test get_current_version with valid pyproject.toml."""
        with patch("slaf.cli.Path.exists", return_value=True):
            with patch("slaf.cli.Path.read_text", return_value='version = "1.2.3"'):
                from slaf.cli import get_current_version

                version = get_current_version()
                assert version == "1.2.3"

    def test_get_current_version_file_not_found(self):
        """Test get_current_version when pyproject.toml doesn't exist."""
        with patch("slaf.cli.Path.exists", return_value=False):
            from slaf.cli import get_current_version

            with pytest.raises(FileNotFoundError):
                get_current_version()

    def test_get_current_version_no_version(self):
        """Test get_current_version when version not found in pyproject.toml."""
        with patch("slaf.cli.Path.exists", return_value=True):
            with patch("slaf.cli.Path.read_text", return_value="no version here"):
                from slaf.cli import get_current_version

                with pytest.raises(ValueError):
                    get_current_version()

    def test_calculate_new_version_patch(self):
        """Test calculate_new_version with patch release."""
        from slaf.cli import calculate_new_version

        assert calculate_new_version("1.2.3", "patch") == "1.2.4"

    def test_calculate_new_version_minor(self):
        """Test calculate_new_version with minor release."""
        from slaf.cli import calculate_new_version

        assert calculate_new_version("1.2.3", "minor") == "1.3.0"

    def test_calculate_new_version_major(self):
        """Test calculate_new_version with major release."""
        from slaf.cli import calculate_new_version

        assert calculate_new_version("1.2.3", "major") == "2.0.0"

    def test_calculate_new_version_invalid(self):
        """Test calculate_new_version with invalid release type."""
        from slaf.cli import calculate_new_version

        with pytest.raises(ValueError):
            calculate_new_version("1.2.3", "invalid")

    def test_calculate_new_version_invalid_format(self):
        """Test calculate_new_version with invalid version format."""
        from slaf.cli import calculate_new_version

        with pytest.raises(ValueError):
            calculate_new_version("1.2", "patch")

    @patch("slaf.cli.Path")
    @patch("slaf.cli.run_command")
    def test_update_version(self, mock_run_command, mock_path):
        """Test update_version function."""
        mock_path_instance = Mock()
        mock_path_instance.read_text.return_value = '[project]\nversion = "1.2.3"\n'
        mock_path.return_value = mock_path_instance

        from slaf.cli import update_version

        update_version("1.2.4")

        # Verify the file was written with new version
        mock_path_instance.write_text.assert_called_once()
        written_content = mock_path_instance.write_text.call_args[0][0]
        assert 'version = "1.2.4"' in written_content

    @patch("slaf.cli.run_command")
    def test_create_tag_success(self, mock_run_command):
        """Test create_tag function."""
        mock_run_command.side_effect = [
            Mock(returncode=0, stdout=""),  # git tag -l
            Mock(returncode=0),  # git tag
            Mock(returncode=0),  # git push
        ]

        from slaf.cli import create_tag

        create_tag("1.2.3")

        assert mock_run_command.call_count == 3

    @patch("slaf.cli.run_command")
    def test_create_tag_already_exists(self, mock_run_command):
        """Test create_tag when tag already exists."""
        mock_run_command.side_effect = [
            Mock(returncode=0, stdout="v1.2.3"),  # git tag -l
        ]

        from slaf.cli import create_tag

        create_tag("1.2.3")

        # Should only call git tag -l, not create or push
        assert mock_run_command.call_count == 1

    @patch("slaf.cli.run_command")
    def test_generate_changelog(self, mock_run_command):
        """Test generate_changelog function."""
        mock_run_command.side_effect = [
            Mock(returncode=0, stdout="v1.2.0"),  # git tag
            Mock(returncode=0, stdout="commit1\ncommit2"),  # git log
        ]

        with patch("slaf.cli.Path") as mock_path:
            mock_path_instance = Mock()
            mock_path_instance.exists.return_value = True
            mock_path_instance.read_text.return_value = "# Changelog\n\n"
            mock_path.return_value = mock_path_instance

            from slaf.cli import generate_changelog

            generate_changelog("1.2.3")

            # Verify changelog was written
            mock_path_instance.write_text.assert_called_once()
            written_content = mock_path_instance.write_text.call_args[0][0]
            assert "## [1.2.3]" in written_content

    @patch("slaf.cli.check_dependencies")
    def test_convert_with_verbose_and_chunked(self, mock_check_deps, runner):
        """Test convert command with both verbose and chunked options."""

        def exists_side_effect(self):
            return str(self).endswith("input.h5ad")

        with patch("slaf.cli.Path.exists", new=exists_side_effect):
            with patch("slaf.data.SLAFConverter", create=True) as mock_converter:
                mock_converter_instance = Mock()
                mock_converter.return_value = mock_converter_instance

                # Mock SLAFArray for verbose output
                with patch("slaf.SLAFArray", create=True) as mock_slaf:
                    mock_dataset = Mock()
                    mock_dataset.shape = (1000, 20000)
                    mock_dataset.obs.columns = ["cell_type", "batch"]
                    mock_dataset.var.columns = ["gene_symbol", "highly_variable"]
                    mock_slaf.return_value = mock_dataset

                    result = runner.invoke(
                        app,
                        [
                            "convert",
                            "input.h5ad",
                            "output_dir",
                            "--chunked",
                            "--chunk-size",
                            "5000",
                            "--verbose",
                        ],
                    )

                    assert result.exit_code == 0
                    assert "Converting" in result.stdout
                    assert "Using chunked processing" in result.stdout
                    assert "Dataset info:" in result.stdout
                    # Verify chunked processing was used
                    mock_converter.assert_called_once_with(
                        chunked=True,
                        chunk_size=5000,
                        create_indices=False,
                        optimize_storage=True,
                        use_optimized_dtypes=True,
                        enable_v2_manifest=True,
                        compact_after_write=True,
                    )

    @patch("slaf.cli.check_dependencies")
    def test_convert_with_custom_chunk_size(self, mock_check_deps, runner):
        """Test convert command with custom chunk size."""

        def exists_side_effect(self):
            return str(self).endswith("input.h5ad")

        with patch("slaf.cli.Path.exists", new=exists_side_effect):
            with patch("slaf.data.SLAFConverter", create=True) as mock_converter:
                mock_converter_instance = Mock()
                mock_converter.return_value = mock_converter_instance

                result = runner.invoke(
                    app,
                    [
                        "convert",
                        "input.h5ad",
                        "output_dir",
                        "--chunked",
                        "--chunk-size",
                        "2000",
                    ],
                )

                assert result.exit_code == 0
                assert (
                    "Using chunked processing (chunk size: 2,000 cells)"
                    in result.stdout
                )
                # Verify custom chunk size was used
                mock_converter.assert_called_once_with(
                    chunked=True,
                    chunk_size=2000,
                    create_indices=False,
                    optimize_storage=True,
                    use_optimized_dtypes=True,
                    enable_v2_manifest=True,
                    compact_after_write=True,
                )

    @patch("slaf.cli.check_dependencies")
    def test_convert_with_format_and_chunked(self, mock_check_deps, runner):
        """Test convert command with explicit format and chunked processing."""

        def exists_side_effect(self):
            return str(self).endswith("input.h5ad")

        with patch("slaf.cli.Path.exists", new=exists_side_effect):
            with patch("slaf.data.SLAFConverter", create=True) as mock_converter:
                mock_converter_instance = Mock()
                mock_converter.return_value = mock_converter_instance

                result = runner.invoke(
                    app,
                    [
                        "convert",
                        "input.h5ad",
                        "output_dir",
                        "--format",
                        "h5ad",
                        "--chunked",
                    ],
                )

                assert result.exit_code == 0
                assert "Converting" in result.stdout
                # Verify both format and chunked processing were used
                mock_converter_instance.convert.assert_called_once_with(
                    "input.h5ad", "output_dir", input_format="h5ad"
                )
                mock_converter.assert_called_once_with(
                    chunked=True,
                    chunk_size=25000,
                    create_indices=False,
                    optimize_storage=True,
                    use_optimized_dtypes=True,
                    enable_v2_manifest=True,
                    compact_after_write=True,
                )

    def test_convert_help_text_contains_chunked_info(self, runner):
        """Test that convert help text contains chunked processing information."""
        result = runner.invoke(app, ["convert", "--help"])
        assert result.exit_code == 0
        # Check that help text contains basic CLI information
        assert "chunked" in result.stdout
        assert "verbose" in result.stdout
