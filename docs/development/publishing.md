# Publishing to PyPI

This guide covers the automated PyPI publishing setup for the SLAF project.

## 🚀 Quick Start

### 1. Set up PyPI API Tokens

You need to add API tokens to your GitHub repository secrets:

#### For Production PyPI:

1. Go to [PyPI](https://pypi.org/manage/account/token/)
2. Create a new API token with "Entire account" scope
3. Copy the token
4. Go to your GitHub repo → `Settings` → `Secrets and variables` → `Actions`
5. Add secret: `PYPI_API_TOKEN` with your PyPI token

#### For Test PyPI (Optional):

1. Go to [Test PyPI](https://test.pypi.org/manage/account/token/)
2. Create a new API token
3. Add secret: `TEST_PYPI_API_TOKEN` with your Test PyPI token

### 2. Release Process

#### Automated Release (Recommended)

1. **Prepare Release**:

   - Go to `Actions` → `Prepare Release`
   - Click "Run workflow"
   - Choose release type: `patch`, `minor`, or `major`
   - Check "prerelease" if it's an alpha/beta/rc release
   - Click "Run workflow"

2. **Review and Merge**:

   - Review the generated PR
   - Merge the PR to main branch

3. **Create and Push Tag**:

   ```bash
   git pull origin main
   git tag v0.1.1  # Use the version from the PR
   git push origin v0.1.1
   ```

4. **Automatic Publishing**:
   - The release workflow will automatically:
     - Build the package
     - Run tests
     - Publish to PyPI (or Test PyPI for prereleases)
     - Create a GitHub release

#### Manual Release

```bash
# 1. Prepare release (updates version and changelog)
slaf release prepare --type patch

# 2. Run tests
slaf release test

# 3. Build and check package
slaf release build
slaf release check

# 4. Publish (creates tag and triggers GitHub Actions)
slaf release publish --version 0.1.1
```

Or use the CLI for individual steps:

```bash
# Update version and generate changelog
slaf release prepare --version 0.1.1

# Run tests
slaf release test

# Build package
slaf release build

# Check package
slaf release check

# Create tag manually
git tag v0.1.1
git push origin v0.1.1
```

## 📦 Package Configuration

### Version Management

The package version is managed in `pyproject.toml`:

```toml
[project]
name = "slaf"
version = "0.1.0"  # This gets updated automatically
```

### Package Metadata

Key metadata in `pyproject.toml`:

```toml
[project]
name = "slaf"
version = "0.1.0"
description = "Sparse Lazy Array Format - MVP for single-cell data"
authors = [{ name = "Pavan Ramkumar", email = "pavan.ramkumar@gmail.com" }]
readme = "README.md"
license = { text = "Apache-2.0" }
requires-python = ">=3.12,<3.13"
```

### Dependencies

Dependencies are organized in optional groups:

```toml
[project.optional-dependencies]
dev = ["pytest>=8.0.0", ...]
docs = ["mkdocs>=1.5.0", ...]
test = ["pytest>=8.0.0", "pytest-cov>=6.2.0", "coverage>=7.9.1"]
```

## 🔄 Release Workflows

### 1. Prepare Release Workflow

**Trigger**: Manual workflow dispatch
**Purpose**: Prepare release with version bump and changelog

**Features**:

- Automatic version bumping (patch/minor/major)
- Prerelease support (alpha/beta/rc)
- Changelog generation from git commits
- Pull request creation for review

**Usage**:

1. Go to `Actions` → `Prepare Release`
2. Choose release type
3. Review generated PR
4. Merge PR
5. Create and push tag

### 2. Release to PyPI Workflow

**Trigger**: Push to version tags (`v*`)
**Purpose**: Build and publish to PyPI

**Features**:

- Automatic version extraction from tag
- Test execution before release
- Package building and validation
- PyPI/Test PyPI publishing
- GitHub release creation

### 3. CLI Release Commands

The SLAF CLI includes built-in release management commands:

#### `slaf release prepare`

Prepare a new release by updating version and generating changelog.

```bash
# Auto-calculate next patch version
slaf release prepare --type patch

# Auto-calculate next minor version
slaf release prepare --type minor

# Auto-calculate next major version
slaf release prepare --type major

# Specify exact version
slaf release prepare --version 0.1.1
```

#### `slaf release test`

Run the test suite with coverage.

```bash
slaf release test
```

#### `slaf release build`

Build the package distribution files.

```bash
slaf release build
```

#### `slaf release check`

Build and validate the package.

```bash
slaf release check
```

#### `slaf release publish`

Complete release process including tests, build, and tag creation.

```bash
slaf release publish --version 0.1.1
```

## 🏷️ Versioning Strategy

### Semantic Versioning

- **Major** (`1.0.0`): Breaking changes
- **Minor** (`0.2.0`): New features, backward compatible
- **Patch** (`0.1.1`): Bug fixes, backward compatible

### Prereleases

- **Alpha**: `0.1.0-alpha.20231201`
- **Beta**: `0.1.0-beta.1`
- **Release Candidate**: `0.1.0-rc.1`

### Tag Format

- Production: `v0.1.0`
- Prerelease: `v0.1.0-alpha.20231201`

## 🔧 Configuration

### GitHub Secrets Required

| Secret                | Description         | Required |
| --------------------- | ------------------- | -------- |
| `PYPI_API_TOKEN`      | PyPI API token      | ✅       |
| `TEST_PYPI_API_TOKEN` | Test PyPI API token | Optional |

### Workflow Permissions

The workflows require these permissions:

- `contents: read/write` - For repository access
- `pull-requests: write` - For PR creation
- `id-token: write` - For PyPI authentication

## 📋 Release Checklist

### Before Release

- [ ] All tests pass
- [ ] Documentation is up to date
- [ ] Changelog is updated
- [ ] Version is correct in `pyproject.toml`
- [ ] PyPI tokens are configured

### During Release

- [ ] Run "Prepare Release" workflow
- [ ] Review generated PR
- [ ] Merge PR to main
- [ ] Create and push version tag
- [ ] Monitor release workflow

### After Release

- [ ] Verify package on PyPI
- [ ] Test installation: `pip install slaf`
- [ ] Update documentation if needed
- [ ] Announce release

## 🐛 Troubleshooting

### Common Issues

1. **Authentication Failed**

   - Check PyPI API tokens in GitHub secrets
   - Ensure tokens have correct permissions

2. **Version Conflicts**

   - Ensure version in `pyproject.toml` matches tag
   - Check for existing releases with same version

3. **Build Failures**

   - Check `pyproject.toml` syntax
   - Verify all dependencies are available

4. **Test Failures**
   - Fix failing tests before release
   - Ensure coverage requirements are met

### Manual Recovery

If automated release fails:

```bash
# 1. Fix issues
# 2. Update version manually
# 3. Build and test locally
python -m build
twine check dist/*

# 4. Upload manually (if needed)
twine upload dist/*
```

## 📊 Release Monitoring

### PyPI Statistics

Monitor your package on PyPI:

- [PyPI Package Page](https://pypi.org/project/slaf/)
- Download statistics
- User feedback

### GitHub Releases

Track releases on GitHub:

- Release notes
- Download statistics
- User feedback

## 🔗 Useful Links

- [PyPI](https://pypi.org/) - Python Package Index
- [Test PyPI](https://test.pypi.org/) - Test Package Index
- [PyPI API Tokens](https://pypi.org/manage/account/token/)
- [Python Packaging User Guide](https://packaging.python.org/)
- [Twine Documentation](https://twine.readthedocs.io/)
