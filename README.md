# SKADA-Bench: Benchmarking Unsupervised Domain Adaptation Methods with Realistic Validation

Welcome to the official implementation of [SKADA-Bench: Benchmarking Unsupervised Domain Adaptation Methods with Realistic Validation](https://arxiv.org/pdf/2407.11676).

## Website Documentation Update Guide

This guide explains how to update the documentation website with new versions.

### Prerequisites
- Access to both `main` and `website` branches
- Python environment with documentation building dependencies installed
- Write permissions to the repository

### Step-by-Step Instructions

#### 1. Create New Version Branch
1. Checkout the `main` branch
   ```bash
   git checkout main
   git pull origin main
   ```
2. Create and switch to a new version branch
   ```bash
   git checkout -b version-X.Y.Z
   ```

#### 2. Update Version Information
1. Locate and open `version.py`
2. Update the version number to match your new version
3. Commit the changes
   ```bash
   git add version.py
   git commit -m "Update version to X.Y.Z"
   ```

#### 3. Generate Documentation
1. Run the documentation build script:
   ```bash
   python make_docs.py
   ```
2. Verify the documentation was generated successfully in `docs/build/`

#### 4. Update Website Branch
1. Copy the generated documentation files:
   - `docs/build/html`
   - `docs/build/doctrees`

2. Switch to the website branch:
   ```bash
   git checkout website
   git pull origin website
   ```

3. Create new version directory:
   ```bash
   mkdir -p build/vX.Y.Z
   ```

4. Add documentation files:
   - Copy `html` folder to `build/vX.Y.Z/html`
   - Copy `doctrees` folder to `build/vX.Y.Z/doctrees`

#### 5. Update Navigation Files
1. In the `docs` folder:
   - Add new version to `index.html`
   - Format: `<a href="../build/vX.Y.Z/html/index.html">Version X.Y.Z</a>`

2. Update `all_versions.html`:
   - Add new version entry
   - Keep versions in descending order
   - Use consistent formatting with existing entries

#### 6. Finalize Changes
1. Commit all changes:
   ```bash
   git add .
   git commit -m "Add documentation for version X.Y.Z"
   ```

2. Push changes to remote:
   ```bash
   git push origin website
   ```

### Verification
- Visit the website to ensure:
  - New version is accessible
  - Links work correctly
  - Navigation between versions functions properly

### Notes
- Maintain consistent formatting across version documentation
- Keep version numbers in semantic versioning format (X.Y.Z)
- Backup important files before major changes
- Test all links and navigation before finalizing

For any issues or questions, please open an issue in the repository.