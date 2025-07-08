# CI/CD Pipeline

This project uses GitHub Actions for continuous integration and deployment. The pipeline ensures code quality, security, and reliability through automated testing and checks.

## Pipeline Overview

The CI/CD pipeline consists of several jobs that run in parallel and sequentially:

### Core Quality Checks

1. **Lint and Format Check** (`lint`)

   - Checks code formatting with `ruff format`
   - Runs linting with `ruff check`
   - Ensures consistent code style

2. **Type Check** (`type-check`)

   - Runs `mypy` for static type checking
   - Validates type annotations across the codebase

3. **Test Suite** (`test`)

   - Runs pytest with coverage reporting
   - Generates coverage reports (XML, HTML, terminal)
   - Uploads coverage to Codecov
   - Requires minimum 80% code coverage

4. **Dependency Check** (`dependencies`)

   - Checks for dependency conflicts with `pipdeptree`
   - Identifies outdated dependencies
   - Ensures clean dependency tree

5. **Security Check** (`security`)
   - Runs `bandit` for security vulnerability scanning
   - Runs `safety` for known security vulnerabilities in dependencies
   - Uploads security reports as artifacts

### Build and Integration

6. **Build Package** (`build`)

   - Builds the package using `python -m build`
   - Uploads build artifacts
   - Only runs after lint and test pass

7. **Integration Tests** (`integration-test`)

   - Runs additional integration tests
   - Excludes slow tests with `-m "not slow"`
   - Runs after core tests pass

8. **Benchmark Tests** (`benchmark`)
   - Runs performance benchmarks
   - Only runs on pushes to main branch
   - Uploads benchmark results as artifacts

### Quality Gate

9. **Quality Gate** (`quality-gate`)
   - Final validation step
   - Ensures all required jobs passed
   - Provides clear pass/fail status

## Running Locally

### Pre-commit Hooks

Install and run pre-commit hooks locally:

```bash
pip install pre-commit
pre-commit install
```

This will run the same checks locally before each commit:

- Code formatting with `ruff`
- Linting with `ruff`
- Basic file checks

### Manual Testing

Run the same checks locally that the CI pipeline runs:

```bash
# Install development dependencies
pip install -e ".[dev,test]"

# Run linting
ruff check .
ruff format --check .

# Run type checking
mypy slaf/ --ignore-missing-imports

# Run tests with coverage
pytest tests/ --cov=slaf --cov-report=html --cov-report=term-missing

# Run security checks
bandit -r slaf/
safety check

# Build package
python -m build
```

## Coverage Requirements

The pipeline requires a minimum of 80% code coverage. To check coverage locally:

```bash
pytest tests/ --cov=slaf --cov-report=html --cov-report=term-missing --cov-fail-under=80
```

## Artifacts

The pipeline generates several artifacts:

- **Coverage Reports**: HTML and XML coverage reports
- **Security Reports**: Bandit and Safety scan results
- **Build Artifacts**: Package distribution files
- **Benchmark Results**: Performance benchmark data

## Configuration

### Pytest Configuration

The project uses `pytest.ini` for test configuration:

- Test discovery in `tests/` directory
- Markers for different test types (slow, integration, unit, benchmark)
- Warning filters for cleaner output
- Coverage and reporting settings

### Ruff Configuration

Code formatting and linting is configured in `pyproject.toml`:

- Line length: 88 characters
- Target Python version: 3.12
- Selected lint rules for comprehensive checks

## Troubleshooting

### Common Issues

1. **Coverage below threshold**: Add more tests or improve existing test coverage
2. **Type check failures**: Add proper type annotations or ignore specific lines
3. **Linting errors**: Run `ruff check --fix` to auto-fix issues
4. **Security warnings**: Review and address security findings

### Local Development

For faster development cycles:

```bash
# Run only fast tests
pytest tests/ -m "not slow"

# Run specific test file
pytest tests/test_slaf.py

# Run with verbose output
pytest tests/ -v -s
```

## Branch Protection

Consider setting up branch protection rules for the main branch:

- Require status checks to pass before merging
- Require up-to-date branches before merging
- Restrict direct pushes to main branch
- Require pull request reviews
