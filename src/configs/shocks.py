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
        "wsteth": 50_000_000,
        "weth": 15_000_000,
        "wbtc": 40_000_000,
        "sfrxeth": 10_000_000,
    },
    "type": DEBT,
    "tag": NEUTRAL,
    "description": "Average debt observed over Q4 2023 (total: 115M).",
}

SHOCK_DEBT_ADVERSE = {
    "target": {
        "wsteth": 75_000_000,
        "weth": 25_000_000,
        "wbtc": 55_000_000,
        "sfrxeth": 15_000_000,
    },
    "type": DEBT,
    "tag": ADVERSE,
    "description": "p99 debt oberserved over Q4 2023 (total: 170M).",
}

SHOCK_DEBT_SEVERE = {
    "target": {
        "wsteth": 500_000_000,
        "weth": 150_000_000,
        "wbtc": 400_000_000,
        "sfrxeth": 100_000_000,
    },
    "type": DEBT,
    "tag": SEVERE,
    "description": "10x the average debt observed over Q4 2023 (total: 1.15B).",
    # TODO need to increase debt ceiling for this one
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
