#!/usr/bin/env python3
# type: ignore
"""SLAF Command Line Interface."""

import importlib.util
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

import typer

from slaf.core.slaf import display_ascii_art

app = typer.Typer(
    name="slaf",
    help="SLAF (Sparse Lazy Array Format) - High-performance single-cell data storage and analysis",
    add_completion=False,
)


def check_dependencies() -> None:
    """Check if required dependencies are installed."""
    required_packages = {
        "mkdocs": "mkdocs",
        "marimo": "marimo",
        "typer": "typer",
    }

    missing = []
    for package, import_name in required_packages.items():
        if importlib.util.find_spec(import_name) is None:
            missing.append(package)

    if missing:
        typer.echo(f"❌ Missing required packages: {', '.join(missing)}")
        typer.echo("Install with: uv add " + " ".join(missing))
        raise typer.Exit(1) from None


@app.command()
def version():
    """Show SLAF version."""
    display_ascii_art()
    from loguru import logger

    logger.info("")
    try:
        import slaf

        typer.echo(f"SLAF version: {getattr(slaf, '__version__', 'unknown')}")
    except ImportError:
        typer.echo("SLAF not installed or not in PYTHONPATH")


@app.command()
def docs(
    build: bool = typer.Option(False, "--build", "-b", help="Build documentation"),
    serve: bool = typer.Option(
        False, "--serve", "-s", help="Serve documentation locally"
    ),
):
    """Manage SLAF documentation."""
    check_dependencies()

    if not Path("mkdocs.yml").exists():
        typer.echo("❌ mkdocs.yml not found. Are you in the project root?")
        raise typer.Exit(1) from None

    if build:
        typer.echo("🏗️ Building documentation...")
        try:
            subprocess.run([sys.executable, "-m", "mkdocs", "build"], check=True)
            typer.echo("✅ Documentation built successfully!")
        except subprocess.CalledProcessError as e:
            typer.echo(f"❌ Failed to build documentation: {e}")
            raise typer.Exit(1) from e

    if serve:
        typer.echo("🌐 Starting documentation server...")
        typer.echo("📖 Open http://127.0.0.1:8000 in your browser")
        try:
            subprocess.run([sys.executable, "-m", "mkdocs", "serve"])
        except KeyboardInterrupt:
            typer.echo("\n👋 Server stopped")
        except subprocess.CalledProcessError as e:
            typer.echo(f"❌ Failed to start server: {e}")
            raise typer.Exit(1) from e

    if not any([build, serve]):
        # Default to serve if no options specified
        typer.echo("🌐 Starting documentation server...")
        typer.echo("📖 Open http://127.0.0.1:8000 in your browser")
        try:
            subprocess.run([sys.executable, "-m", "mkdocs", "serve"])
        except KeyboardInterrupt:
            typer.echo("\n👋 Server stopped")
        except subprocess.CalledProcessError as e:
            typer.echo(f"❌ Failed to start server: {e}")
            raise typer.Exit(1) from e


@app.command()
def examples(
    export: bool = typer.Option(
        False, "--export", "-e", help="Export examples to HTML"
    ),
    list_examples: bool = typer.Option(
        False, "--list", "-l", help="List available examples"
    ),
    example: str | None = typer.Argument(None, help="Specific example to export"),
):
    """Manage SLAF examples."""
    check_dependencies()

    examples_dir = Path("examples")
    if not examples_dir.exists():
        typer.echo("❌ Examples directory not found")
        raise typer.Exit(1) from None

    if list_examples:
        typer.echo("📚 Available examples:")
        for py_file in examples_dir.glob("*.py"):
            if not py_file.name.startswith("__"):
                typer.echo(f"  - {py_file.stem}")
        return

    if export:
        docs_examples_dir = Path("docs/examples")
        docs_examples_dir.mkdir(exist_ok=True)

        if example:
            # Export specific example
            example_file = examples_dir / f"{example}.py"
            if not example_file.exists():
                typer.echo(f"❌ Example '{example}' not found")
                raise typer.Exit(1) from None

            output_file = docs_examples_dir / f"{example}.html"
            typer.echo(f"🔧 Exporting {example_file}...")
            try:
                subprocess.run(
                    [
                        sys.executable,
                        "-m",
                        "marimo",
                        "export",
                        "html",
                        str(example_file),
                        "-o",
                        str(output_file),
                    ],
                    check=True,
                )
                typer.echo(f"✅ Exported to {output_file}")
            except subprocess.CalledProcessError as e:
                typer.echo(f"❌ Failed to export: {e}")
                raise typer.Exit(1) from e
        else:
            # Export all examples
            typer.echo("🔧 Exporting all examples...")
            exported_files = []

            for py_file in examples_dir.glob("*.py"):
                if not py_file.name.startswith("__"):
                    output_file = docs_examples_dir / f"{py_file.stem}.html"
                    typer.echo(f"  Exporting {py_file.name}...")
                    try:
                        subprocess.run(
                            [
                                sys.executable,
                                "-m",
                                "marimo",
                                "export",
                                "html",
                                str(py_file),
                                "-o",
                                str(output_file),
                            ],
                            check=True,
                        )
                        exported_files.append(output_file)
                    except subprocess.CalledProcessError as e:
                        typer.echo(f"❌ Failed to export {py_file.name}: {e}")
                        raise typer.Exit(1) from e

            typer.echo(f"✅ Exported {len(exported_files)} examples:")
            for file in exported_files:
                typer.echo(f"  - {file}")


@app.command()
def convert(
    input_path: str = typer.Argument(..., help="Input file or directory path"),
    output_path: str = typer.Argument(..., help="Output SLAF directory path"),
    format: str | None = typer.Option(
        None,
        "--format",
        "-f",
        help="Input format: h5ad, 10x_mtx, 10x_h5 (auto-detected if not specified)",
    ),
    chunked: bool = typer.Option(
        True,  # Changed from False to True - make chunked the default
        "--chunked/--no-chunked",  # Added --no-chunked option for users who want non-chunked
        "-c",
        help="Use chunked processing for memory efficiency (default: True, supports all formats)",
    ),
    chunk_size: int = typer.Option(
        25000, "--chunk-size", help="Number of cells per chunk (default: 25000)"
    ),
    create_indices: bool = typer.Option(
        False,
        "--create-indices",
        help="Create indices for query performance (recommended for large datasets)",
    ),
    optimize_storage: bool = typer.Option(
        True,
        "--optimize-storage/--no-optimize-storage",
        help="Only store integer IDs in expression table to reduce storage size by 50-80% (default: True)",
    ),
    use_optimized_dtypes: bool = typer.Option(
        True,
        "--optimized-dtypes/--no-optimized-dtypes",
        help="Use uint16/uint32 data types for better compression (default: True)",
    ),
    enable_v2_manifest: bool = typer.Option(
        True,
        "--v2-manifest/--no-v2-manifest",
        help="Enable v2 manifest paths for better query performance (default: True)",
    ),
    compact_after_write: bool = typer.Option(
        True,
        "--compact/--no-compact",
        help="Compact dataset after writing for optimal storage (default: True)",
    ),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Verbose output"),
):
    """Convert single-cell datasets to SLAF format with optimized storage."""
    try:
        from slaf.data import SLAFConverter
    except ImportError as e:
        typer.echo("❌ SLAF not installed or not in PYTHONPATH")
        raise typer.Exit(1) from e

    input_file = Path(input_path)
    if not input_file.exists():
        typer.echo(f"❌ Input file not found: {input_path}")
        raise typer.Exit(1) from None

    output_dir = Path(output_path)
    if output_dir.exists():
        typer.echo(f"❌ Output directory already exists: {output_path}")
        raise typer.Exit(1) from None

    typer.echo(f"🔄 Converting {input_path} to SLAF format...")

    if chunked:
        typer.echo(f"📦 Using chunked processing (chunk size: {chunk_size:,} cells)")
    else:
        typer.echo("📦 Using non-chunked processing (may use more memory)")

    try:
        converter = SLAFConverter(
            chunked=chunked,
            chunk_size=chunk_size,
            create_indices=create_indices,  # Use the CLI parameter instead of hardcoding False
            optimize_storage=optimize_storage,
            use_optimized_dtypes=use_optimized_dtypes,
            enable_v2_manifest=enable_v2_manifest,
            compact_after_write=compact_after_write,
        )

        # Use the new converter with auto-detection
        if format:
            # Convert CLI format names to converter format names
            format_mapping = {
                "anndata": "h5ad",
                "hdf5": "10x_h5",
                "10x_mtx": "10x_mtx",
                "10x_h5": "10x_h5",
                "h5ad": "h5ad",
            }
            input_format = format_mapping.get(format, format)
            converter.convert(input_path, output_path, input_format=input_format)
        else:
            # Use auto-detection
            converter.convert(input_path, output_path)

        typer.echo(f"✅ Successfully converted to {output_path}")

        if verbose:
            # Show some info about the converted dataset
            from slaf import SLAFArray

            slaf_array = SLAFArray(output_path)
            typer.echo("📊 Dataset info:")
            typer.echo(
                f"  Shape: {slaf_array.shape[0]:,} cells × {slaf_array.shape[1]:,} genes"
            )
            typer.echo(f"  Cell metadata columns: {len(slaf_array.obs.columns)}")
            typer.echo(f"  Gene metadata columns: {len(slaf_array.var.columns)}")

    except Exception as e:
        typer.echo(f"❌ Conversion failed: {e}")
        raise typer.Exit(1) from e


def get_current_version() -> str:
    """Get current version from pyproject.toml."""
    pyproject_path = Path("pyproject.toml")
    if not pyproject_path.exists():
        raise FileNotFoundError("pyproject.toml not found")

    content = pyproject_path.read_text()
    match = re.search(r'version = "([^"]+)"', content)
    if not match:
        raise ValueError("Could not find version in pyproject.toml")

    return match.group(1)


def update_version(new_version: str) -> None:
    """Update version in pyproject.toml and uv.lock."""
    pyproject_path = Path("pyproject.toml")
    content = pyproject_path.read_text()

    # Replace only the project version line, not other version-like strings
    # Look for the specific line in the [project] section
    lines = content.split("\n")
    in_project_section = False
    updated_lines = []

    for line in lines:
        if line.strip() == "[project]":
            in_project_section = True
        elif line.strip().startswith("[") and line.strip().endswith("]"):
            in_project_section = False

        if in_project_section and line.strip().startswith("version = "):
            updated_lines.append(f'version = "{new_version}"')
        else:
            updated_lines.append(line)

    new_content = "\n".join(updated_lines)
    pyproject_path.write_text(new_content)
    typer.echo(f"Updated version to {new_version}")

    # Update uv.lock to reflect the new version
    try:
        typer.echo("Updating uv.lock...")
        run_command("uv lock --no-upgrade")
        typer.echo("Updated uv.lock")
    except Exception as e:
        typer.echo(f"Warning: Failed to update uv.lock: {e}")
        typer.echo("You may need to run 'uv lock' manually")


def calculate_new_version(current_version: str, release_type: str) -> str:
    """Calculate new version based on release type."""
    parts = current_version.split(".")
    if len(parts) < 3:
        raise ValueError(f"Invalid version format: {current_version}")

    major, minor, patch = int(parts[0]), int(parts[1]), int(parts[2])

    if release_type == "major":
        return f"{major + 1}.0.0"
    elif release_type == "minor":
        return f"{major}.{minor + 1}.0"
    elif release_type == "patch":
        return f"{major}.{minor}.{patch + 1}"
    else:
        raise ValueError(f"Invalid release type: {release_type}")


def run_command(cmd: str, check: bool = True) -> subprocess.CompletedProcess:
    """Run a shell command and return the result."""
    typer.echo(f"Running: {cmd}")
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if check and result.returncode != 0:
        typer.echo(f"Error: {result.stderr}")
        raise typer.Exit(1)
    return result


def run_tests() -> None:
    """Run the test suite."""
    typer.echo("Running tests...")
    run_command("pytest tests/ --cov=slaf --cov-report=term-missing")


def build_package() -> None:
    """Build the package."""
    typer.echo("Building package...")
    run_command("python -m build")


def check_package() -> None:
    """Check the built package."""
    typer.echo("Checking package...")
    run_command("twine check dist/*")


def create_tag(version: str) -> None:
    """Create and push a git tag."""
    tag_name = f"v{version}"
    typer.echo(f"Creating tag: {tag_name}")

    # Check if tag already exists
    result = run_command(f"git tag -l {tag_name}", check=False)
    if tag_name in result.stdout:
        typer.echo(f"Warning: Tag {tag_name} already exists")
        return

    run_command(f"git tag {tag_name}")
    run_command(f"git push origin {tag_name}")


def generate_changelog(version: str) -> None:
    """Generate a basic changelog entry."""
    typer.echo("Generating changelog...")

    # Get current date
    from datetime import datetime

    current_date = datetime.now().strftime("%Y-%m-%d")

    # Get commits since last tag
    result = run_command("git tag --sort=-version:refname | head -1", check=False)
    last_tag = result.stdout.strip()

    changelog_entry = f"""
## [{version}] - {current_date}

### Added
"""

    if last_tag:
        # Get commits since last tag
        commits = run_command(f"git log --oneline --no-merges {last_tag}..HEAD")
        for line in commits.stdout.strip().split("\n"):
            if line.strip():
                changelog_entry += f"- {line.strip()}\n"
    else:
        # Get all commits
        commits = run_command("git log --oneline --no-merges")
        for line in commits.stdout.strip().split("\n"):
            if line.strip():
                changelog_entry += f"- {line.strip()}\n"

    # Append to CHANGELOG.md
    changelog_path = Path("CHANGELOG.md")
    if changelog_path.exists():
        content = changelog_path.read_text()
        new_content = changelog_entry + "\n" + content
    else:
        new_content = f"# Changelog\n\n{changelog_entry}"

    changelog_path.write_text(new_content)
    typer.echo("Changelog updated")


@app.command()
def release(
    action: str = typer.Argument(
        ...,
        help="Action to perform: prepare, publish, test, build, or check",
        case_sensitive=False,
    ),
    version: str | None = typer.Option(
        None,
        "--version",
        "-v",
        help="New version (e.g., 0.1.1) (for prepare/publish actions)",
    ),
    release_type: str = typer.Option(
        "patch",
        "--type",
        "-t",
        help="Release type (for prepare action)",
        case_sensitive=False,
    ),
    no_tag: bool = typer.Option(
        False, "--no-tag", help="Don't create git tag (for publish action)"
    ),
):
    r"""
    Manage SLAF releases and publishing.

    Actions:
      prepare - Prepare a new release (update version, generate changelog)
      publish - Publish a release (build, test, tag, prepare for PyPI)
      test    - Run the test suite
      build   - Build the package
      check   - Build and check the package

    Examples:
      slaf release prepare --type minor
      slaf release publish --version 0.2.0
      slaf release test
      slaf release build
      slaf release check
    """

    if action.lower() == "prepare":
        if not version:
            try:
                current_version = get_current_version()
                new_version = calculate_new_version(current_version, release_type)
            except (FileNotFoundError, ValueError) as e:
                typer.echo(f"❌ Error: {e}")
                raise typer.Exit(1) from e
        else:
            new_version = version

        typer.echo(f"Preparing release {new_version}")
        try:
            update_version(new_version)
            generate_changelog(new_version)
            typer.echo(f"✅ Release {new_version} prepared. Review changes and commit.")
        except Exception as e:
            typer.echo(f"❌ Error preparing release: {e}")
            raise typer.Exit(1) from e

    elif action.lower() == "publish":
        if not version:
            typer.echo("❌ Error: --version required for publish action")
            raise typer.Exit(1) from None

        typer.echo(f"Publishing version {version}")
        try:
            update_version(version)
            run_tests()
            build_package()
            check_package()

            if not no_tag:
                create_tag(version)

            typer.echo(f"✅ Release {version} completed!")
            typer.echo(
                "🚀 The GitHub Actions workflow will automatically publish to PyPI"
            )
        except Exception as e:
            typer.echo(f"❌ Error publishing release: {e}")
            raise typer.Exit(1) from e

    elif action.lower() == "test":
        try:
            run_tests()
            typer.echo("✅ Tests completed successfully!")
        except Exception as e:
            typer.echo(f"❌ Error running tests: {e}")
            raise typer.Exit(1) from e

    elif action.lower() == "build":
        try:
            build_package()
            typer.echo("✅ Package built successfully!")
        except Exception as e:
            typer.echo(f"❌ Error building package: {e}")
            raise typer.Exit(1) from e

    elif action.lower() == "check":
        try:
            build_package()
            check_package()
            typer.echo("✅ Package check completed successfully!")
        except Exception as e:
            typer.echo(f"❌ Error checking package: {e}")
            raise typer.Exit(1) from e

    else:
        typer.echo(f"❌ Unknown action: {action}")
        typer.echo("Available actions: prepare, publish, test, build, check")
        raise typer.Exit(1) from None


@app.command()
def info(
    dataset_path: str = typer.Argument(..., help="Path to SLAF dataset"),
):
    """Show information about a SLAF dataset."""
    try:
        from slaf import SLAFArray
    except ImportError as e:
        typer.echo("❌ SLAF not installed or not in PYTHONPATH")
        raise typer.Exit(1) from e

    dataset_dir = Path(dataset_path)
    if not dataset_dir.exists():
        typer.echo(f"❌ Dataset not found: {dataset_path}")
        raise typer.Exit(1) from None

    try:
        slaf_array = SLAFArray(dataset_path)
        slaf_array.info()
    except Exception as e:
        typer.echo(f"❌ Failed to load dataset: {e}")
        raise typer.Exit(1) from e


@app.command()
def query(
    dataset_path: str = typer.Argument(..., help="Path to SLAF dataset"),
    sql: str = typer.Argument(..., help="SQL query to execute"),
    output: str | None = typer.Option(
        None, "--output", "-o", help="Output file path (CSV)"
    ),
    limit: int = typer.Option(10, "--limit", "-l", help="Limit number of results"),
):
    """Execute SQL query on SLAF dataset."""
    try:
        from slaf import SLAFArray
    except ImportError as e:
        typer.echo("❌ SLAF not installed or not in PYTHONPATH")
        raise typer.Exit(1) from e

    dataset_dir = Path(dataset_path)
    if not dataset_dir.exists():
        typer.echo(f"❌ Dataset not found: {dataset_path}")
        raise typer.Exit(1) from None

    try:
        slaf_array = SLAFArray(dataset_path)

        # Add LIMIT if not present
        if "LIMIT" not in sql.upper():
            sql += f" LIMIT {limit}"

        typer.echo(f"🔍 Executing query: {sql}")
        results = slaf_array.query(sql)

        if output:
            # Handle both pandas and polars DataFrames
            if hasattr(results, "to_csv"):
                results.to_csv(output, index=False)
            else:
                # Polars DataFrame
                results.write_csv(output)
            typer.echo(f"✅ Results saved to {output}")
        else:
            typer.echo("📊 Query results:")
            # Handle both pandas and polars DataFrames
            if hasattr(results, "to_string"):
                # Pandas DataFrame
                typer.echo(results.to_string(index=False))
            else:
                # Polars DataFrame
                typer.echo(str(results))

    except Exception as e:
        typer.echo(f"❌ Query failed: {e}")
        raise typer.Exit(1) from e


@app.command()
def benchmark(  # type: ignore[misc, assignment, attr-defined]
    action: str = typer.Argument(
        ...,
        help="Action to perform: run, summary, docs, or all",
        case_sensitive=False,
    ),
    datasets: list[str] = typer.Option(
        ["pbmc3k"],
        "--datasets",
        "-d",
        help="Dataset names to benchmark (for run/all actions)",
    ),
    data_dir: str = typer.Option(
        None, "--data-dir", help="Directory containing datasets (for run/all actions)"
    ),
    types: list[str] = typer.Option(
        None,
        "--types",
        "-t",
        help="Specific benchmark types to run (for run/all actions)",
    ),
    auto_convert: bool = typer.Option(
        False,
        "--auto-convert",
        help="Auto-convert h5ad to SLAF if needed (for run/all actions)",
    ),
    verbose: bool = typer.Option(
        False, "--verbose", "-v", help="Verbose output (for run/all actions)"
    ),
    results_file: str = typer.Option(
        "benchmarks/comprehensive_benchmark_results.json",
        "--results",
        help="Results file path (for summary action)",
    ),
    summary_file: str = typer.Option(
        "benchmarks/benchmark_summary.json",
        "--summary",
        help="Summary file path (for docs action)",
    ),
    docs_file: str = typer.Option(
        "docs/benchmarks/bioinformatics_benchmarks.md",
        "--docs",
        help="Performance docs file path (for docs action)",
    ),
):
    """
    Manage SLAF benchmarks and performance testing.

    Actions:
      run     - Run benchmarks on specified datasets
      summary - Generate documentation summary from results
      docs    - Update performance.md with benchmark data
      all     - Run complete workflow (benchmarks + summary + docs)

    Examples:
      slaf benchmark run --datasets pbmc3k --auto-convert
      slaf benchmark summary --results comprehensive_benchmark_results.json
      slaf benchmark docs --summary benchmark_summary.json
      slaf benchmark all --datasets pbmc3k --auto-convert
    """

    # Import benchmark functions
    try:
        from benchmarks import (
            generate_benchmark_summary,
            run_benchmark_suite,
            update_performance_docs,
        )
    except ImportError as e:
        typer.echo(f"❌ Failed to import benchmark modules: {e}")
        typer.echo("Make sure you're in the project root directory")
        raise typer.Exit(1) from e

    # Set default data directory
    if not data_dir:
        data_dir = str(Path(__file__).parent.parent.parent / "slaf-datasets")

    if action.lower() == "run":
        typer.echo("🚀 Running SLAF benchmarks...")

        data_path = Path(data_dir)
        all_dataset_results = {}

        for dataset_name in datasets:
            typer.echo(f"\n🎯 Benchmarking dataset: {dataset_name}")
            typer.echo("=" * 60)

            # Check if dataset is already a SLAF file
            if dataset_name.endswith(".slaf"):
                slaf_path = Path(dataset_name)
                if not slaf_path.exists():
                    typer.echo(f"❌ SLAF file not found: {dataset_name}")
                    continue

                # For SLAF files, we need to find the corresponding h5ad file
                # or create a dummy path for comparison
                h5ad_path = slaf_path.with_suffix(".h5ad")
                if not h5ad_path.exists():
                    typer.echo(
                        f"⚠️  No corresponding h5ad file found for {dataset_name}"
                    )
                    typer.echo("   SLAF-only benchmarking may have limited comparisons")
                    h5ad_path = None
            else:
                # Find dataset files
                h5ad_pattern = f"{dataset_name}*.h5ad"
                h5ad_files = list(data_path.glob(h5ad_pattern))

                if not h5ad_files:
                    typer.echo(f"❌ No h5ad file found for {dataset_name}")
                    typer.echo(f"   Looking for: {data_path / h5ad_pattern}")
                    continue

                h5ad_path = h5ad_files[0]
                slaf_path = data_path / f"{dataset_name}.slaf"

            # Run benchmark suite for this dataset
            if h5ad_path is None:
                typer.echo("⚠️  Skipping benchmarks that require h5ad comparison")
                continue

            dataset_results = run_benchmark_suite(
                h5ad_path=str(h5ad_path),
                slaf_path=str(slaf_path),
                benchmark_types=types,
                verbose=verbose,
                auto_convert=auto_convert,
            )

            if dataset_results:
                all_dataset_results[dataset_name] = dataset_results

        # Save results
        if all_dataset_results:
            import json

            import numpy as np

            # Prepare results for JSON serialization
            json_results: dict[str, Any] = {}
            for dataset, results in all_dataset_results.items():
                json_results[dataset] = {}
                for benchmark_type, type_results in results.items():  # type: ignore
                    # type_results can be either list[dict] or dict depending on benchmark type
                    if benchmark_type == "data_vs_tokenization_timing":
                        # Handle timing breakdown results (dictionary format)
                        json_results[dataset][benchmark_type] = {}
                        for k, v in type_results.items():
                            if isinstance(v, dict):
                                clean_dict = {}
                                for dict_k, dict_v in v.items():
                                    if isinstance(dict_v, np.integer | np.floating):
                                        clean_dict[dict_k] = float(dict_v)
                                    elif hasattr(dict_v, "to_dict"):
                                        clean_dict[dict_k] = dict_v.to_dict()
                                    elif hasattr(dict_v, "tolist"):
                                        clean_dict[dict_k] = dict_v.tolist()
                                    else:
                                        clean_dict[dict_k] = dict_v
                                json_results[dataset][benchmark_type][k] = clean_dict
                            else:
                                json_results[dataset][benchmark_type][k] = v
                    else:
                        # Handle standard list-based results
                        json_results[dataset][benchmark_type] = []
                        for result in type_results:
                            clean_result = {}
                            for k, v in result.items():
                                if isinstance(v, np.integer | np.floating):
                                    clean_result[k] = float(v)
                                elif hasattr(v, "to_dict"):  # Handle pandas objects
                                    clean_result[k] = v.to_dict()
                                elif hasattr(v, "tolist"):  # Handle numpy arrays
                                    clean_result[k] = v.tolist()
                                else:
                                    clean_result[k] = v
                            json_results[dataset][benchmark_type].append(clean_result)

            # Save comprehensive results
            with open(results_file, "w") as f:
                json.dump(json_results, f, indent=2)

            typer.echo(f"\n💾 Results saved to: {results_file}")

    elif action.lower() == "summary":
        typer.echo("📊 Generating benchmark summary...")

        if not Path(results_file).exists():
            typer.echo(f"❌ Results file not found: {results_file}")
            raise typer.Exit(1) from None

        generate_benchmark_summary(results_file, summary_file)

    elif action.lower() == "docs":
        typer.echo("📝 Updating performance documentation...")

        if not Path(summary_file).exists():
            typer.echo(f"❌ Summary file not found: {summary_file}")
            raise typer.Exit(1) from None

        if not Path(docs_file).exists():
            typer.echo(f"❌ Docs file not found: {docs_file}")
            raise typer.Exit(1) from None

        update_performance_docs(summary_file, docs_file)

    elif action.lower() == "all":
        typer.echo("🚀 Running complete benchmark workflow...")

        # Step 1: Run benchmarks
        typer.echo("\n📊 Step 1: Running benchmarks...")
        data_path = Path(data_dir)
        all_dataset_results = {}

        for dataset_name in datasets:
            typer.echo(f"\n🎯 Benchmarking dataset: {dataset_name}")
            typer.echo("=" * 60)

            # Check if dataset is already a SLAF file
            if dataset_name.endswith(".slaf"):
                slaf_path = Path(dataset_name)
                if not slaf_path.exists():
                    typer.echo(f"❌ SLAF file not found: {dataset_name}")
                    continue

                # For SLAF files, we need to find the corresponding h5ad file
                # or create a dummy path for comparison
                h5ad_path = slaf_path.with_suffix(".h5ad")
                if not h5ad_path.exists():
                    typer.echo(
                        f"⚠️  No corresponding h5ad file found for {dataset_name}"
                    )
                    typer.echo("   SLAF-only benchmarking may have limited comparisons")
                    h5ad_path = None
            else:
                # Find dataset files
                h5ad_pattern = f"{dataset_name}*.h5ad"
                h5ad_files = list(data_path.glob(h5ad_pattern))

                if not h5ad_files:
                    typer.echo(f"❌ No h5ad file found for {dataset_name}")
                    typer.echo(f"   Looking for: {data_path / h5ad_pattern}")
                    continue

                h5ad_path = h5ad_files[0]
                slaf_path = data_path / f"{dataset_name}.slaf"

            # Run benchmark suite for this dataset
            if h5ad_path is None:
                typer.echo("⚠️  Skipping benchmarks that require h5ad comparison")
                continue

            dataset_results = run_benchmark_suite(
                h5ad_path=str(h5ad_path),
                slaf_path=str(slaf_path),
                benchmark_types=types,
                verbose=verbose,
                auto_convert=auto_convert,
            )

            if dataset_results:
                all_dataset_results[dataset_name] = dataset_results

        # Save comprehensive results
        if all_dataset_results:
            import json

            import numpy as np

            # Prepare results for JSON serialization
            json_results = {}
            for dataset, results in all_dataset_results.items():
                json_results[dataset] = {}
                for benchmark_type, type_results in results.items():  # type: ignore
                    if benchmark_type == "data_vs_tokenization_timing":
                        # Handle timing breakdown results (dictionary format)
                        json_results[dataset][benchmark_type] = {}
                        for k, v in type_results.items():
                            if isinstance(v, dict):
                                clean_dict = {}
                                for dict_k, dict_v in v.items():
                                    if isinstance(dict_v, np.integer | np.floating):
                                        clean_dict[dict_k] = float(dict_v)
                                    elif hasattr(dict_v, "to_dict"):
                                        clean_dict[dict_k] = dict_v.to_dict()
                                    elif hasattr(dict_v, "tolist"):
                                        clean_dict[dict_k] = dict_v.tolist()
                                    else:
                                        clean_dict[dict_k] = dict_v
                                json_results[dataset][benchmark_type][k] = clean_dict
                            else:
                                json_results[dataset][benchmark_type][k] = v
                    else:
                        # Handle standard list-based results
                        json_results[dataset][benchmark_type] = []
                        for result in type_results:
                            clean_result = {}
                            for k, v in result.items():
                                if isinstance(v, np.integer | np.floating):
                                    clean_result[k] = float(v)
                                elif hasattr(v, "to_dict"):  # Handle pandas objects
                                    clean_result[k] = v.to_dict()
                                elif hasattr(v, "tolist"):  # Handle numpy arrays
                                    clean_result[k] = v.tolist()
                                else:
                                    clean_result[k] = v
                            json_results[dataset][benchmark_type].append(clean_result)

            # Save comprehensive results
            with open(results_file, "w") as f:
                json.dump(json_results, f, indent=2)

            typer.echo(f"\n💾 Results saved to: {results_file}")

            # Step 2: Generate summary
            typer.echo("\n📊 Step 2: Generating summary...")
            generate_benchmark_summary(results_file, summary_file)

            # Step 3: Update docs
            typer.echo("\n📊 Step 3: Updating documentation...")
            update_performance_docs(summary_file, docs_file)

            typer.echo("\n✅ Complete workflow finished!")
        else:
            typer.echo("\n❌ No benchmarks completed successfully")

    else:
        typer.echo(f"❌ Unknown action: {action}")
        typer.echo("Available actions: run, summary, docs, all")
        raise typer.Exit(1) from None


if __name__ == "__main__":
    app()
