crvUSD Risk Modeling
=======================================

[![Checked with mypy](https://www.mypy-lang.org/static/mypy_badge.svg)](https://mypy-lang.org/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Linting: Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/charliermarsh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)

Overview
---
Curve risk modeling simulations.

# Setup

We currently rely on development versions of `curvesim` and `crvusdsim` for some of our modeling. Both packages are under active development, so we point to specific commits in each repo as our dependencies. These are in `requirements.txt`. Do the following:

```
python3 -m venv venv
```

```
python3 -m pip install -r requirements.txt
```

```
source venv/bin/activate
```

This will install the working versions of `curvesim` and `crvusdsim`. Eventually we will replace these dependencies with stable releases.