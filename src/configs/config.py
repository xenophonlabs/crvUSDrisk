"""
Master config with basic constants.
"""

COINGECKO_URL = "https://api.coingecko.com/api/v3/"

USDC = "0xA0b86991c6218b36c1d19D4a2e9Eb0cE3606eB48"
USDT = "0xdac17f958d2ee523a2206206994597c13d831ec7"
USDP = "0x8e870d67f660d95d5be530380d0ec0bd388289e1"
TUSD = "0x0000000000085d4780B73119b644AE5ecd22b376"
WETH = "0xc02aaa39b223fe8d0a0e5c4f27ead9083c756cc2"
WBTC = "0x2260fac5e5542a773aa44fbcfedf7c193bc2c599"

STABLES = [USDC, USDT, USDP, TUSD]
COLLATERAL = [WETH, WBTC]
COINS = STABLES + COLLATERAL

COINGECKO_IDS = {
    USDC: "usd-coin",
    USDT: "tether",
    USDP: "paxos-standard",
    TUSD: "true-usd",
    WETH: "weth",
    WBTC: "wrapped-bitcoin",
}

STABLE_CG_IDS = [COINGECKO_IDS[coin] for coin in STABLES]
