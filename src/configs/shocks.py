"""
Provides the configuration for shocks to scenarios.

We shock four possible parameters:
- GBM drift (mu) for collateral assets
- GBM volatility (sigma) for collateral assets
- Total debt in the system
- Debt : crvUSD liquidity ratio
"""
from .tokens import WETH, WSTETH, SFRXETH, WBTC, TBTC, USDC

# Tags
NEUTRAL = "neutral"
ADVERSE = "adverse"
SEVERE = "severe"
VERY_SEVERE = "very severe"

# Types
LIQUIDITY = "liquidity"
DEBT = "debt"
MU = "mu"
VOL = "vol"
JUMP = "jump"

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
        "WBTC": 1.877968,
        "tBTC": 1.877968,
        "WETH": 2.445831,
        "sfrxETH": 2.445831,
        "wstETH": 2.445831,
    },
    "type": VOL,
    "tag": ADVERSE,
    "description": "p99 intraday vol observed since 2020.",
}

SHOCK_VOL_SEVERE = {
    "target": {
        "WBTC": 2.816952,
        "tBTC": 2.816952,
        "WETH": 3.6687465,
        "sfrxETH": 3.6687465,
        "wstETH": 3.6687465,
    },
    "type": VOL,
    "tag": SEVERE,
    "description": "50% worse than the adverse scenario. This is massive volatility.",
}

### ========== Liquidity ========== ###

SHOCK_LIQUIDITY_NEUTRAL = {
    "target": 2.362817,
    "type": LIQUIDITY,
    "tag": NEUTRAL,
    "description": "Average debt:liquidity ratio for crvUSD over Q4 2023.",
}

SHOCK_LIQUIDITY_ADVERSE = {
    "target": 3.5,
    "type": LIQUIDITY,
    "tag": ADVERSE,
    "description": "p99.9 debt:liquidity ratio for crvUSD over Q4 2023.",
}

SHOCK_LIQUIDITY_SEVERE = {
    "target": 5.0,
    "type": LIQUIDITY,
    "tag": SEVERE,
    "description": "2x worse liquidity than the neutral scenario.",
}

SHOCK_LIQUIDITY_VERY_SEVERE = {
    "target": 10.0,
    "type": LIQUIDITY,
    "tag": VERY_SEVERE,
    "description": "4x worse liquidity than the neutral scenario.",
}

### ============ Jumps ============ ###

SHOCK_FLASH_CRASH_ADVERSE = {
    "target": {
        "type": "flash_crash",
        "coins_j": [WETH, WSTETH, SFRXETH, WBTC, TBTC],
        "mu_j": [-0.022666, -0.022666, -0.022666, -0.018658, -0.018658],
        "cov_j": [
            [0.00017597, 0.00017597, 0.00017597, 0.00014909, 0.00014909],
            [0.00017597, 0.00017597, 0.00017597, 0.00014909, 0.00014909],
            [0.00017597, 0.00017597, 0.00017597, 0.00014909, 0.00014909],
            [0.00014909, 0.00014909, 0.00014909, 0.00012631, 0.00012631],
            [0.00014909, 0.00014909, 0.00014909, 0.00012631, 0.00012631],
        ],
    },
    "type": JUMP,
    "tag": ADVERSE,
    "description": "Enforce a 5x std flash crash on all collateral assets.",
}


SHOCK_FLASH_CRASH_SEVERE = {
    "target": {
        "type": "flash_crash",
        "coins_j": [WETH, WSTETH, SFRXETH, WBTC, TBTC],
        "mu_j": [-0.074193, -0.074193, -0.074193, -0.061255, -0.061255],
        "cov_j": [
            [0.00095642, 0.00095642, 0.00095642, 0.00083179, 0.00083179],
            [0.00095642, 0.00095642, 0.00095642, 0.00083179, 0.00083179],
            [0.00095642, 0.00095642, 0.00095642, 0.00083179, 0.00083179],
            [0.00083179, 0.00083179, 0.00083179, 0.00072339, 0.00072339],
            [0.00083179, 0.00083179, 0.00083179, 0.00072339, 0.00072339],
        ],
    },
    "type": JUMP,
    "tag": SEVERE,
    "description": "Enforce a 20x std flash crash on all collateral assets.",
}

SHOCK_DEPEG_ADVERSE = {
    "target": {
        "type": "depeg",
        "coins_j": [USDC],
        "size": -0.2,
    },
    "type": JUMP,
    "tag": ADVERSE,
    "description": "Enforce a 20% depeg on USDC.",
}

SHOCK_DEPEG_SEVERE = {
    "target": {
        "type": "depeg",
        "coins_j": [USDC],
        "size": -0.99,
    },
    "type": JUMP,
    "tag": SEVERE,
    "description": "Enforce a 99% depeg on USDC.",
}
