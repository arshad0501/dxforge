# Contributing to DxForge

First off, thank you for considering contributing to DxForge! It's people like you that make open source such a great community.

## Where do I go from here?

If you've noticed a bug or have a feature request, please make one! It's generally best if you get confirmation of your bug or approval for your feature request before starting work.

### Step 1: Fork
Fork the project on GitHub and clone your fork locally.

```bash
git clone https://github.com/your-username/dxforge.git
cd dxforge
```

### Step 2: Branch
Create a new branch for your feature or bug fix.

```bash
git checkout -b your-feature-name
```

### Step 3: Setup Environment
It is recommended to use a virtual environment.

```bash
python -m venv .venv
source .venv/bin/activate  # On Windows use: .venv\Scripts\activate
pip install -r requirements-full.txt
```

### Step 4: Run Tests
Before making your changes, make sure all tests pass.

```bash
pytest -q
python scripts/run_benchmark.py
```

### Step 5: Make Changes and Commit
Make your changes, and write tests for them. 

```bash
git add .
git commit -m "feat: Add your descriptive commit message here"
```

### Step 6: Push and Create a Pull Request
Push your branch to your fork, and create a pull request to the main repository.

```bash
git push origin your-feature-name
```

## Coding Standards
* Follow standard Python PEP 8 conventions.
* Add type hints to all new functions.
* Ensure unit tests cover new functionality.
* Do not introduce AI logic where deterministic protocol checks are required.

Thank you for contributing!
