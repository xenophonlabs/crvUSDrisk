crvusd Risk Modeling
=======================================

![Run Black](https://github.com/xenophonlabs/crvusdrisk/workflows/Run%20Black/badge.svg)
![Run Pylint](https://github.com/xenophonlabs/crvusdrisk/workflows/Run%20Pylint/badge.svg)
![Run Mypy](https://github.com/xenophonlabs/crvusdrisk/workflows/Run%20Mypy/badge.svg)


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