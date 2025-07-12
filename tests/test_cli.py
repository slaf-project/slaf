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

        # Benchmark command tests

    def test_benchmark_help(self, runner):
        """Test benchmark command help."""
        result = runner.invoke(app, ["benchmark", "--help"])
        assert result.exit_code == 0
        assert "run" in result.stdout
        assert "summary" in result.stdout
        assert "docs" in result.stdout
        assert "all" in result.stdout

    def test_benchmark_invalid_action(self, runner):
        """Test benchmark command with invalid action."""
        result = runner.invoke(app, ["benchmark", "invalid_action"])
        assert result.exit_code == 1
        assert "Unknown action" in result.stdout
