#!/usr/bin/env python3
"""SLAF Command Line Interface."""

import typer
from pathlib import Path
import subprocess
import sys
from typing import Optional, List
import importlib.util

app = typer.Typer(
    name="slaf",
    help="SLAF (Sparse Lance Array Format) - High-performance single-cell data storage and analysis",
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
        raise typer.Exit(1)


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
        raise typer.Exit(1)

    if build:
        typer.echo("🏗️ Building documentation...")
        try:
            subprocess.run([sys.executable, "-m", "mkdocs", "build"], check=True)
            typer.echo("✅ Documentation built successfully!")
        except subprocess.CalledProcessError as e:
            typer.echo(f"❌ Failed to build documentation: {e}")
            raise typer.Exit(1)

    if serve:
        typer.echo("🌐 Starting documentation server...")
        typer.echo("📖 Open http://127.0.0.1:8000 in your browser")
        try:
            subprocess.run([sys.executable, "-m", "mkdocs", "serve"])
        except KeyboardInterrupt:
            typer.echo("\n👋 Server stopped")
        except subprocess.CalledProcessError as e:
            typer.echo(f"❌ Failed to start server: {e}")
            raise typer.Exit(1)

    if deploy:
        typer.echo("🚀 Deploying to GitHub Pages...")
        try:
            subprocess.run([sys.executable, "-m", "mkdocs", "gh-deploy"], check=True)
            typer.echo("✅ Documentation deployed successfully!")
        except subprocess.CalledProcessError as e:
            typer.echo(f"❌ Failed to deploy: {e}")
            raise typer.Exit(1)

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
            raise typer.Exit(1)


@app.command()
def examples(
    export: bool = typer.Option(
        False, "--export", "-e", help="Export examples to HTML"
    ),
    list_examples: bool = typer.Option(
        False, "--list", "-l", help="List available examples"
    ),
    example: Optional[str] = typer.Argument(None, help="Specific example to export"),
):
    """Manage SLAF examples."""
    check_dependencies()

    examples_dir = Path("examples")
    if not examples_dir.exists():
        typer.echo("❌ Examples directory not found")
        raise typer.Exit(1)

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
                raise typer.Exit(1)

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
                raise typer.Exit(1)
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
                        raise typer.Exit(1)

            typer.echo(f"✅ Exported {len(exported_files)} examples:")
            for file in exported_files:
                typer.echo(f"  - {file}")


@app.command()
def convert(
    input_path: str = typer.Argument(..., help="Input file path (AnnData, HDF5, etc.)"),
    output_path: str = typer.Argument(..., help="Output SLAF directory path"),
    format: Optional[str] = typer.Option(
        None, "--format", "-f", help="Input format (auto-detected if not specified)"
    ),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Verbose output"),
):
    """Convert datasets to SLAF format."""
    try:
        from slaf.data import SLAFConverter
    except ImportError:
        typer.echo("❌ SLAF not installed or not in PYTHONPATH")
        raise typer.Exit(1)

    input_file = Path(input_path)
    if not input_file.exists():
        typer.echo(f"❌ Input file not found: {input_path}")
        raise typer.Exit(1)

    output_dir = Path(output_path)
    if output_dir.exists():
        typer.echo(f"❌ Output directory already exists: {output_path}")
        raise typer.Exit(1)

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
            raise typer.Exit(1)

        typer.echo(f"✅ Successfully converted to {output_path}")

        if verbose:
            # Show some info about the converted dataset
            from slaf import SLAFArray

            slaf_array = SLAFArray(output_path)
            typer.echo(f"📊 Dataset info:")
            typer.echo(
                f"  Shape: {slaf_array.shape[0]:,} cells × {slaf_array.shape[1]:,} genes"
            )
            typer.echo(f"  Cell metadata columns: {len(slaf_array.obs.columns)}")
            typer.echo(f"  Gene metadata columns: {len(slaf_array.var.columns)}")

    except Exception as e:
        typer.echo(f"❌ Conversion failed: {e}")
        raise typer.Exit(1)


@app.command()
def info(
    dataset_path: str = typer.Argument(..., help="Path to SLAF dataset"),
):
    """Show information about a SLAF dataset."""
    try:
        from slaf import SLAFArray
    except ImportError:
        typer.echo("❌ SLAF not installed or not in PYTHONPATH")
        raise typer.Exit(1)

    dataset_dir = Path(dataset_path)
    if not dataset_dir.exists():
        typer.echo(f"❌ Dataset not found: {dataset_path}")
        raise typer.Exit(1)

    try:
        slaf_array = SLAFArray(dataset_path)
        slaf_array.info()
    except Exception as e:
        typer.echo(f"❌ Failed to load dataset: {e}")
        raise typer.Exit(1)


@app.command()
def query(
    dataset_path: str = typer.Argument(..., help="Path to SLAF dataset"),
    sql: str = typer.Argument(..., help="SQL query to execute"),
    output: Optional[str] = typer.Option(
        None, "--output", "-o", help="Output file path (CSV)"
    ),
    limit: int = typer.Option(10, "--limit", "-l", help="Limit number of results"),
):
    """Execute SQL query on SLAF dataset."""
    try:
        from slaf import SLAFArray
    except ImportError:
        typer.echo("❌ SLAF not installed or not in PYTHONPATH")
        raise typer.Exit(1)

    dataset_dir = Path(dataset_path)
    if not dataset_dir.exists():
        typer.echo(f"❌ Dataset not found: {dataset_path}")
        raise typer.Exit(1)

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
        raise typer.Exit(1)


if __name__ == "__main__":
    app()
