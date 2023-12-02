crvusd Risk Modeling
=======================================

[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![CI](https://github.com/xenophonlabs/crvUSDrisk/actions/workflows/CI.yml/badge.svg)](https://github.com/xenophonlabs/crvUSDrisk/actions/workflows/CI.yml/badge.svg)


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