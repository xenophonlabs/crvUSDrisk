"""
Provides the configuration for shocks to scenarios.

We shock four possible parameters:
- GBM drift (mu) for collateral assets
- GBM volatility (sigma) for collateral assets
- Total debt in the system
- Debt : crvUSD liquidity ratio
"""

# Tags
NEUTRAL = "neutral"
ADVERSE = "adverse"
SEVERE = "severe"

# Types
LIQUIDITY = "liquidity"
DEBT = "debt"
MU = "mu"
VOL = "vol"

### ============ Mu ============ ###

SHOCK_MU_NEUTRAL = {
    "target": {"WBTC": 0, "tBTC": 0, "WETH": 0, "sfrxETH": 0, "wstETH": 0},
    "type": MU,
    "tag": NEUTRAL,
    "description": "Assume no drift.",
}

SHOCK_MU_ADVERSE = {
    "target": {
        "WBTC": -19.10171,
        "tBTC": -19.10171,
        "WETH": -24.68442,
        "sfrxETH": -24.68442,
        "wstETH": -24.68442,
    },
    "type": MU,
    "tag": SEVERE,
    "description": "p05 intraday mu observed since 2020.",
}

SHOCK_MU_SEVERE = {
    "target": {
        "WBTC": -37.011385,
        "tBTC": -37.011385,
        "WETH": -47.030258,
        "sfrxETH": -47.030258,
        "wstETH": -47.030258,
    },
    "type": MU,
    "tag": SEVERE,
    "description": "p01 intraday mu observed since 2020.",
}

### =========== Debt =========== ###

SHOCK_DEBT_NEUTRAL = {
    "target": {
        "wsteth": 0.33,
        "weth": 0.075,
        "wbtc": 0.2,
        "sfrxeth": 0.2,
    },
    "type": DEBT,
    "tag": NEUTRAL,
    "description": "Average debt observed over Q4 2023 as pct of debt ceiling (total: 115M).",
}

SHOCK_DEBT_ADVERSE = {
    "target": {
        "wsteth": 0.5,
        "weth": 0.125,
        "wbtc": 0.275,
        "sfrxeth": 0.3,
    },
    "type": DEBT,
    "tag": ADVERSE,
    "description": "p99 debt oberserved over Q4 2023 as pct of debt ceiling (total: 170M).",
}

SHOCK_DEBT_SEVERE = {
    "target": {
        "wsteth": 0.99,
        "weth": 0.99,
        "wbtc": 0.99,
        "sfrxeth": 0.99,
    },
    "type": DEBT,
    "tag": SEVERE,
    "description": "99% of the current debt ceiling (total: 594M)",
}


### ============ Vol ============ ###

# Doesn't need neutral scenario, contained in price config file.

SHOCK_VOL_ADVERSE = {
    "target": {
        "WBTC": 1.211855,
        "tBTC": 1.211855,
        "WETH": 1.507219,
        "sfrxETH": 1.507219,
        "wstETH": 1.507219,
    },
    "type": VOL,
    "tag": ADVERSE,
    "description": "p95 intraday vol observed since 2020.",
}

SHOCK_VOL_SEVERE = {
    "target": {
        "WBTC": 1.877968,
        "tBTC": 1.877968,
        "WETH": 2.445831,
        "sfrxETH": 2.445831,
        "wstETH": 2.445831,
    },
    "type": VOL,
    "tag": SEVERE,
    "description": "p99 intraday vol observed since 2020.",
}

### ========== Liquidity ========== ###

SHOCK_LIQUIDITY_NEUTRAL = {
    "target": 2.362817,
    "type": LIQUIDITY,
    "tag": NEUTRAL,
    "description": "Average debt:liquidity ratio for crvUSD over Q4 2023.",
}

SHOCK_LIQUIDITY_ADVERSE = {
    "target": 3.018794,
    "type": LIQUIDITY,
    "tag": ADVERSE,
    "description": "p99 debt:liquidity ratio for crvUSD over Q4 2023.",
}

SHOCK_LIQUIDITY_SEVERE = {
    "target": 5.0,
    "type": LIQUIDITY,
    "tag": SEVERE,
    "description": "2x worse liquidity than the neutral scenario.",
}
