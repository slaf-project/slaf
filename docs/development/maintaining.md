# For Maintainers

This guide covers release management, CI/CD, and maintenance tasks for SLAF maintainers.

## CI/CD Pipeline

The project uses GitHub Actions for automated testing and deployment:

### Automated Checks

- **Lint and Format**: Code formatting and style checks
- **Type Check**: Static type checking with mypy
- **Test Suite**: Full test suite with coverage reporting
- **Security**: Vulnerability scanning with bandit and safety
- **Build**: Package building and validation
- **Integration Tests**: Additional integration tests

### Deployment

- **Documentation**: Automatically deployed to GitHub Pages on main branch
- **Coverage**: Uploaded to Codecov with 70% minimum requirement
- **Security Reports**: Generated as artifacts

## Release Management

### PyPI Setup (One-time)

1. **Create PyPI Account**:

   - Go to [PyPI](https://pypi.org/account/register/)
   - Create an account and verify email

2. **Create API Token**:

   - Go to [PyPI API Tokens](https://pypi.org/manage/account/token/)
   - Create token with "Entire account" scope
   - Copy the token

3. **Add to GitHub Secrets**:
   - Go to your GitHub repo → `Settings` → `Secrets and variables` → `Actions`
   - Add secret: `PYPI_API_TOKEN` with your PyPI token

### Release Process

The release process is automated through GitHub Actions:

1. **Prepare Release**:

   - Go to `Actions` → `Prepare Release`
   - Choose release type: `patch`, `minor`, or `major`
   - The workflow will:
     - Calculate the next version number
     - Update `pyproject.toml` and `uv.lock`
     - Generate a changelog with recent commits
     - Create and push a release branch
     - Provide direct links to create a PR

2. **Create Pull Request**:

   - Click the provided link in the workflow output
   - Or go to: `https://github.com/slaf-project/slaf/compare/main...release-{VERSION}`
   - Use the default title and description below

3. **Create Release Tag**:

   ```bash
   git pull origin main
   git tag v0.2.1  # Use the version from the PR
   git push origin v0.2.1
   ```

4. **Automatic Publishing**:
   - The release workflow automatically:
     - Builds the package
     - Runs tests
     - Publishes to PyPI
     - Creates a GitHub release

### CLI Release Commands

For manual release management:

```bash
# Prepare release (updates version and changelog)
slaf release prepare --type patch

# Run tests
slaf release test

# Build package
slaf release build

# Check package
slaf release check
```

### Default PR Title and Message

When creating a release PR, use these defaults:

**Title:**

```
Release 0.1.1
```

**Description:**

```markdown
## Release 0.1.1

This PR prepares the release for version 0.1.1.

### Changes:

- Updated version in pyproject.toml
- Updated uv.lock
- Generated changelog

### Next steps:

1. Review the changes
2. Merge this PR
3. Create a tag: `git tag v0.1.1`
4. Push the tag: `git push origin v0.1.1`

The release workflow will automatically publish to PyPI when the tag is pushed.
```

## Package Configuration

### Version Management

The package version is managed in `pyproject.toml`:

```toml
[project]
name = "slaf"
version = "0.2.0"  # Updated automatically during release
```

### Dependencies

Dependencies are organized in optional groups:

```toml
[project.optional-dependencies]
dev = ["pytest>=8.0.0", "ruff==0.12.2", "mypy>=1.8.0", ...]
docs = ["mkdocs>=1.5.0", "mkdocs-material>=9.5.0", ...]
test = ["pytest>=8.0.0", "pytest-cov>=6.2.0", "coverage>=7.9.1"]
```

## Documentation Management

### Local Development

```bash
# Serve docs locally
slaf docs --serve

# Build docs for testing
slaf docs --build
```

### Examples Management

```bash
# List available examples
slaf examples --list

# Export all examples to HTML
slaf examples --export

# Export specific example
slaf examples --export getting-started
```

## Troubleshooting

### Common Issues

1. **Release fails**: Check PyPI API token in GitHub secrets
2. **Tests fail**: Review coverage reports and add tests
3. **Build fails**: Check package configuration in pyproject.toml
4. **Documentation fails**: Verify mkdocs configuration
