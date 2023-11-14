"""
Master config with basic constants.
"""

COINGECKO_URL = "https://api.coingecko.com/api/v3/"

# Stables
USDC = "0xA0b86991c6218b36c1d19D4a2e9Eb0cE3606eB48"
USDT = "0xdac17f958d2ee523a2206206994597c13d831ec7"
USDP = "0x8e870d67f660d95d5be530380d0ec0bd388289e1"
TUSD = "0x0000000000085d4780B73119b644AE5ecd22b376"

# Collateral
WETH = "0xc02aaa39b223fe8d0a0e5c4f27ead9083c756cc2"
WSTETH = "0x7f39c581f595b53c5cb19bd0b3f8da6c935e2ca0"
SFRXETH = "0xac3e018457b222d93114458476f3e3416abbe38f"
WBTC = "0x2260fac5e5542a773aa44fbcfedf7c193bc2c599"
TBTC = "0x18084fba666a33d37592fa2633fd49a74dd93a88"

STABLES = [USDC, USDT, USDP, TUSD]
COLLATERAL = [WETH, WBTC]  # , SFRXETH, TBTC, WSTETH]
COINS = STABLES + COLLATERAL

# Roughly: min trade should be around $1, max trade is around $100M
ALL = {
    "USDC": {
        "address": USDC,
        "decimals": 6,
        "min_trade_size": 1,
        "max_trade_size": 1e8,
    },
    "USDT": {
        "address": USDT,
        "decimals": 6,
        "min_trade_size": 1,
        "max_trade_size": 1e8,
    },
    "USDP": {
        "address": USDP,
        "decimals": 18,
        "min_trade_size": 1,
        "max_trade_size": 1e8,
    },
    "TUSD": {
        "address": TUSD,
        "decimals": 18,
        "min_trade_size": 1,
        "max_trade_size": 1e8,
    },
    "WETH": {
        "address": WETH,
        "decimals": 18,
        "min_trade_size": 1e-3,
        "max_trade_size": 50000,
    },
    "wstETH": {
        "address": WSTETH,
        "decimals": 18,
        "min_trade_size": 1e-3,
        "max_trade_size": 50000,
    },
    "sfrxETH": {
        "address": SFRXETH,
        "decimals": 18,
        "min_trade_size": 1e-3,
        "max_trade_size": 50000,
    },
    "WBTC": {
        "address": WBTC,
        "decimals": 18,
        "min_trade_size": 1e-5,
        "max_trade_size": 3000,
    },
    "tBTC": {
        "address": TBTC,
        "decimals": 18,
        "min_trade_size": 1e-5,
        "max_trade_size": 3000,
    },
}

COINGECKO_IDS = {
    USDC: "usd-coin",
    USDT: "tether",
    USDP: "paxos-standard",
    TUSD: "true-usd",
    WETH: "weth",
    WBTC: "wrapped-bitcoin",
}

STABLE_CG_IDS = [COINGECKO_IDS[coin] for coin in STABLES]
