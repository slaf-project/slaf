#!/usr/bin/env python3
"""SLAF Command Line Interface."""

import importlib.util
import re
import subprocess
import sys
from pathlib import Path

import typer

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
    deploy: bool = typer.Option(False, "--deploy", "-d", help="Deploy to GitHub Pages"),
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

    if deploy:
        typer.echo("🚀 Deploying to GitHub Pages...")
        try:
            subprocess.run([sys.executable, "-m", "mkdocs", "gh-deploy"], check=True)
            typer.echo("✅ Documentation deployed successfully!")
        except subprocess.CalledProcessError as e:
            typer.echo(f"❌ Failed to deploy: {e}")
            raise typer.Exit(1) from e

    if not any([build, serve, deploy]):
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
    input_path: str = typer.Argument(..., help="Input file path (AnnData, HDF5, etc.)"),
    output_path: str = typer.Argument(..., help="Output SLAF directory path"),
    format: str | None = typer.Option(
        None, "--format", "-f", help="Input format (auto-detected if not specified)"
    ),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Verbose output"),
):
    """Convert datasets to SLAF format."""
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

    try:
        converter = SLAFConverter()

        # Auto-detect format if not specified
        if not format:
            if input_path.endswith(".h5ad") or input_path.endswith(".h5"):
                format = "anndata"
            elif input_path.endswith(".h5"):
                format = "hdf5"
            else:
                format = "anndata"  # Default

        if format == "anndata":
            converter.convert_anndata(input_path, output_path)
        else:
            typer.echo(f"❌ Unsupported format: {format}")
            raise typer.Exit(1) from None

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
    """Update version in pyproject.toml."""
    pyproject_path = Path("pyproject.toml")
    content = pyproject_path.read_text()

    # Replace version
    new_content = re.sub(r'version = "[^"]+"', f'version = "{new_version}"', content)

    pyproject_path.write_text(new_content)
    typer.echo(f"Updated version to {new_version}")


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

    # Get commits since last tag
    result = run_command("git tag --sort=-version:refname | head -1", check=False)
    last_tag = result.stdout.strip()

    changelog_entry = f"""
## [{version}] - $(date +%Y-%m-%d)

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
        help="Action to perform",
        case_sensitive=False,
    ),
    version: str | None = typer.Option(
        None, "--version", "-v", help="New version (e.g., 0.1.1)"
    ),
    release_type: str = typer.Option(
        "patch", "--type", "-t", help="Release type", case_sensitive=False
    ),
    no_tag: bool = typer.Option(False, "--no-tag", help="Don't create git tag"),
):
    """Manage SLAF releases and publishing."""

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
            results.to_csv(output, index=False)
            typer.echo(f"✅ Results saved to {output}")
        else:
            typer.echo("📊 Query results:")
            typer.echo(results.to_string(index=False))

    except Exception as e:
        typer.echo(f"❌ Query failed: {e}")
        raise typer.Exit(1) from e


if __name__ == "__main__":
    app()
