"""
Master config with basic constants.
"""

import os
from dotenv import load_dotenv

load_dotenv()

username = os.getenv("PG_USERNAME")
password = os.getenv("PG_PASSWORD")
database = os.getenv("PG_DATABASE")

URI = (
    f"postgresql://{username}:{password}@localhost/{database}"  # defaults to port 5432
)

COINGECKO_URL = "https://api.coingecko.com/api/v3/"

# Stables
USDC = "0xa0b86991c6218b36c1d19d4a2e9eb0ce3606eb48"
USDT = "0xdac17f958d2ee523a2206206994597c13d831ec7"
USDP = "0x8e870d67f660d95d5be530380d0ec0bd388289e1"
TUSD = "0x0000000000085d4780b73119b644ae5ecd22b376"

# Collateral
WETH = "0xc02aaa39b223fe8d0a0e5c4f27ead9083c756cc2"
WSTETH = "0x7f39c581f595b53c5cb19bd0b3f8da6c935e2ca0"
SFRXETH = "0xac3e018457b222d93114458476f3e3416abbe38f"
WBTC = "0x2260fac5e5542a773aa44fbcfedf7c193bc2c599"
TBTC = "0x18084fba666a33d37592fa2633fd49a74dd93a88"

STABLES = [USDC, USDT, USDP, TUSD]
COLLATERAL = [WETH, WBTC]  # , SFRXETH, TBTC, WSTETH]
COINS = STABLES + COLLATERAL

# Roughly: min trade should be around $1000, max trade is around $100M
# We set the floor at $1000 to prevent the router from looking for profitable
# arbitrages on small trades in random protocols, which screws up the impact calc.
# TODO make these dynamic?
ALL = {
    "USDC": {
        "address": USDC,
        "decimals": 6,
        "min_trade_size": 1e3,
        "max_trade_size": 1e8,
    },
    "USDT": {
        "address": USDT,
        "decimals": 6,
        "min_trade_size": 1e3,
        "max_trade_size": 1e8,
    },
    "USDP": {
        "address": USDP,
        "decimals": 18,
        "min_trade_size": 1e3,
        "max_trade_size": 1e8,
    },
    "TUSD": {
        "address": TUSD,
        "decimals": 18,
        "min_trade_size": 1e3,
        "max_trade_size": 1e8,
    },
    "WETH": {
        "address": WETH,
        "decimals": 18,
        "min_trade_size": 0.5,
        "max_trade_size": 50000,
    },
    "wstETH": {
        "address": WSTETH,
        "decimals": 18,
        "min_trade_size": 0.5,
        "max_trade_size": 50000,
    },
    "sfrxETH": {
        "address": SFRXETH,
        "decimals": 18,
        "min_trade_size": 0.5,
        "max_trade_size": 50000,
    },
    "WBTC": {
        "address": WBTC,
        "decimals": 18,
        "min_trade_size": 0.03,
        "max_trade_size": 3000,
    },
    "tBTC": {
        "address": TBTC,
        "decimals": 18,
        "min_trade_size": 0.03,
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

protocols = [
    # {
    #     "id": "UNISWAP_V1",
    #     "title": "Uniswap V1",
    #     "img": "https://cdn.1inch.io/liquidity-sources-logo/uniswap.png",
    #     "img_color": "https://cdn.1inch.io/liquidity-sources-logo/uniswap_color.png",
    # },
    {
        "id": "UNISWAP_V2",
        "title": "Uniswap V2",
        "img": "https://cdn.1inch.io/liquidity-sources-logo/uniswap.png",
        "img_color": "https://cdn.1inch.io/liquidity-sources-logo/uniswap_color.png",
    },
    {
        "id": "SUSHI",
        "title": "SushiSwap",
        "img": "https://cdn.1inch.io/liquidity-sources-logo/sushiswap.png",
        "img_color": "https://cdn.1inch.io/liquidity-sources-logo/sushiswap_color.png",
    },
    # {
    #     "id": "MOONISWAP",
    #     "title": "Mooniswap",
    #     "img": "https://cdn.1inch.io/liquidity-sources-logo/mooniswap.png",
    #     "img_color": "https://cdn.1inch.io/liquidity-sources-logo/mooniswap_color.png",
    # },
    {
        "id": "BALANCER",
        "title": "Balancer",
        "img": "https://cdn.1inch.io/liquidity-sources-logo/balancer.png",
        "img_color": "https://cdn.1inch.io/liquidity-sources-logo/balancer_color.png",
    },
    {
        "id": "COMPOUND",
        "title": "Compound",
        "img": "https://cdn.1inch.io/liquidity-sources-logo/compound.png",
        "img_color": "https://cdn.1inch.io/liquidity-sources-logo/compound_color.png",
    },
    {
        "id": "CURVE",
        "title": "Curve",
        "img": "https://cdn.1inch.io/liquidity-sources-logo/curve.png",
        "img_color": "https://cdn.1inch.io/liquidity-sources-logo/curve_color.png",
    },
    {
        "id": "CURVE_V2_SPELL_2_ASSET",
        "title": "Curve Spell",
        "img": "https://cdn.1inch.io/liquidity-sources-logo/curve.png",
        "img_color": "https://cdn.1inch.io/liquidity-sources-logo/curve_color.png",
    },
    {
        "id": "CURVE_V2_SGT_2_ASSET",
        "title": "Curve SGT",
        "img": "https://cdn.1inch.io/liquidity-sources-logo/curve.png",
        "img_color": "https://cdn.1inch.io/liquidity-sources-logo/curve_color.png",
    },
    {
        "id": "CURVE_V2_THRESHOLDNETWORK_2_ASSET",
        "title": "Curve Threshold",
        "img": "https://cdn.1inch.io/liquidity-sources-logo/curve.png",
        "img_color": "https://cdn.1inch.io/liquidity-sources-logo/curve_color.png",
    },
    {
        "id": "CHAI",
        "title": "Chai",
        "img": "https://cdn.1inch.io/liquidity-sources-logo/chai.png",
        "img_color": "https://cdn.1inch.io/liquidity-sources-logo/chai_color.png",
    },
    {
        "id": "OASIS",
        "title": "Oasis",
        "img": "https://cdn.1inch.io/liquidity-sources-logo/oasis.png",
        "img_color": "https://cdn.1inch.io/liquidity-sources-logo/oasis_color.png",
    },
    {
        "id": "KYBER",
        "title": "Kyber",
        "img": "https://cdn.1inch.io/liquidity-sources-logo/kyber.png",
        "img_color": "https://cdn.1inch.io/liquidity-sources-logo/kyber_color.png",
    },
    {
        "id": "AAVE",
        "title": "Aave",
        "img": "https://cdn.1inch.io/liquidity-sources-logo/aave.png",
        "img_color": "https://cdn.1inch.io/liquidity-sources-logo/aave_color.png",
    },
    {
        "id": "IEARN",
        "title": "yearn",
        "img": "https://cdn.1inch.io/liquidity-sources-logo/yearn.png",
        "img_color": "https://cdn.1inch.io/liquidity-sources-logo/yearn_color.png",
    },
    {
        "id": "BANCOR",
        "title": "Bancor",
        "img": "https://cdn.1inch.io/liquidity-sources-logo/bancor.png",
        "img_color": "https://cdn.1inch.io/liquidity-sources-logo/bancor_color.png",
    },
    {
        "id": "SWERVE",
        "title": "Swerve",
        "img": "https://cdn.1inch.io/liquidity-sources-logo/swerve.png",
        "img_color": "https://cdn.1inch.io/liquidity-sources-logo/swerve_color.png",
    },
    {
        "id": "BLACKHOLESWAP",
        "title": "BlackholeSwap",
        "img": "https://cdn.1inch.io/liquidity-sources-logo/blackholeswap.png",
        "img_color": "https://cdn.1inch.io/liquidity-sources-logo/blackholeswap_color.png",
    },
    {
        "id": "DODO",
        "title": "DODO",
        "img": "https://cdn.1inch.io/liquidity-sources-logo/dodo.png",
        "img_color": "https://cdn.1inch.io/liquidity-sources-logo/dodo_color.png",
    },
    {
        "id": "DODO_V2",
        "title": "DODO v2",
        "img": "https://cdn.1inch.io/liquidity-sources-logo/dodo.png",
        "img_color": "https://cdn.1inch.io/liquidity-sources-logo/dodo_color.png",
    },
    {
        "id": "VALUELIQUID",
        "title": "Value Liquid",
        "img": "https://cdn.1inch.io/liquidity-sources-logo/valueliquid.png",
        "img_color": "https://cdn.1inch.io/liquidity-sources-logo/valueliquid_color.png",
    },
    {
        "id": "SHELL",
        "title": "Shell",
        "img": "https://cdn.1inch.io/liquidity-sources-logo/shell.png",
        "img_color": "https://cdn.1inch.io/liquidity-sources-logo/shell_color.png",
    },
    {
        "id": "DEFISWAP",
        "title": "DeFi Swap",
        "img": "https://cdn.1inch.io/liquidity-sources-logo/defiswap.png",
        "img_color": "https://cdn.1inch.io/liquidity-sources-logo/defiswap_color.png",
    },
    {
        "id": "SAKESWAP",
        "title": "Sake Swap",
        "img": "https://cdn.1inch.io/liquidity-sources-logo/sakeswap.png",
        "img_color": "https://cdn.1inch.io/liquidity-sources-logo/sakeswap_color.png",
    },
    {
        "id": "LUASWAP",
        "title": "Lua Swap",
        "img": "https://cdn.1inch.io/liquidity-sources-logo/luaswap.png",
        "img_color": "https://cdn.1inch.io/liquidity-sources-logo/luaswap_color.png",
    },
    {
        "id": "MINISWAP",
        "title": "Mini Swap",
        "img": "https://cdn.1inch.io/liquidity-sources-logo/miniswap.png",
        "img_color": "https://cdn.1inch.io/liquidity-sources-logo/miniswap_color.png",
    },
    {
        "id": "MSTABLE",
        "title": "MStable",
        "img": "https://cdn.1inch.io/liquidity-sources-logo/mstable.png",
        "img_color": "https://cdn.1inch.io/liquidity-sources-logo/mstable_color.png",
    },
    {
        "id": "PMM2",
        "title": "PMM2",
        "img": "https://cdn.1inch.io/liquidity-sources-logo/pmm.png",
        "img_color": "https://cdn.1inch.io/liquidity-sources-logo/pmm_color.png",
    },
    {
        "id": "SYNTHETIX",
        "title": "Synthetix",
        "img": "https://cdn.1inch.io/liquidity-sources-logo/synthetix.png",
        "img_color": "https://cdn.1inch.io/liquidity-sources-logo/synthetix_color.png",
    },
    {
        "id": "AAVE_V2",
        "title": "Aave V2",
        "img": "https://cdn.1inch.io/liquidity-sources-logo/aave.png",
        "img_color": "https://cdn.1inch.io/liquidity-sources-logo/aave_color.png",
    },
    {
        "id": "ST_ETH",
        "title": "LiDo",
        "img": "https://cdn.1inch.io/liquidity-sources-logo/steth.png",
        "img_color": "https://cdn.1inch.io/liquidity-sources-logo/steth_color.png",
    },
    {
        "id": "ONE_INCH_LP",
        "title": "1INCH LP v1.0",
        "img": "https://cdn.1inch.io/liquidity-sources-logo/1inch.png",
        "img_color": "https://cdn.1inch.io/liquidity-sources-logo/1inch_color.png",
    },
    {
        "id": "ONE_INCH_LP_1_1",
        "title": "1INCH LP v1.1",
        "img": "https://cdn.1inch.io/liquidity-sources-logo/1inch.png",
        "img_color": "https://cdn.1inch.io/liquidity-sources-logo/1inch_color.png",
    },
    {
        "id": "LINKSWAP",
        "title": "LINKSWAP",
        "img": "https://cdn.1inch.io/liquidity-sources-logo/linkswap.png",
        "img_color": "https://cdn.1inch.io/liquidity-sources-logo/linkswap_color.png",
    },
    {
        "id": "S_FINANCE",
        "title": "sFinance",
        "img": "https://cdn.1inch.io/liquidity-sources-logo/sfinance.png",
        "img_color": "https://cdn.1inch.io/liquidity-sources-logo/sfinance_color.png",
    },
    {
        "id": "PSM",
        "title": "PSM USDC",
        "img": "https://cdn.1inch.io/liquidity-sources-logo/maker.png",
        "img_color": "https://cdn.1inch.io/liquidity-sources-logo/maker_color.png",
    },
    {
        "id": "POWERINDEX",
        "title": "POWERINDEX",
        "img": "https://cdn.1inch.io/liquidity-sources-logo/powerindex.png",
        "img_color": "https://cdn.1inch.io/liquidity-sources-logo/powerindex_color.png",
    },
    {
        "id": "XSIGMA",
        "title": "xSigma",
        "img": "https://cdn.1inch.io/liquidity-sources-logo/xsigma.png",
        "img_color": "https://cdn.1inch.io/liquidity-sources-logo/xsigma_color.png",
    },
    {
        "id": "SMOOTHY_FINANCE",
        "title": "Smoothy Finance",
        "img": "https://cdn.1inch.io/liquidity-sources-logo/smoothy.png",
        "img_color": "https://cdn.1inch.io/liquidity-sources-logo/smoothy_color.png",
    },
    {
        "id": "SADDLE",
        "title": "Saddle Finance",
        "img": "https://cdn.1inch.io/liquidity-sources-logo/saddle.png",
        "img_color": "https://cdn.1inch.io/liquidity-sources-logo/saddle_color.png",
    },
    {
        "id": "KYBER_DMM",
        "title": "Kyber DMM",
        "img": "https://cdn.1inch.io/liquidity-sources-logo/kyber.png",
        "img_color": "https://cdn.1inch.io/liquidity-sources-logo/kyber_color.png",
    },
    {
        "id": "BALANCER_V2",
        "title": "Balancer V2",
        "img": "https://cdn.1inch.io/liquidity-sources-logo/balancer.png",
        "img_color": "https://cdn.1inch.io/liquidity-sources-logo/balancer_color.png",
    },
    {
        "id": "UNISWAP_V3",
        "title": "Uniswap V3",
        "img": "https://cdn.1inch.io/liquidity-sources-logo/uniswap.png",
        "img_color": "https://cdn.1inch.io/liquidity-sources-logo/uniswap_color.png",
    },
    {
        "id": "SETH_WRAPPER",
        "title": "sETH Wrapper",
        "img": "https://cdn.1inch.io/liquidity-sources-logo/synthetix.png",
        "img_color": "https://cdn.1inch.io/liquidity-sources-logo/synthetix_color.png",
    },
    {
        "id": "CURVE_V2",
        "title": "Curve V2",
        "img": "https://cdn.1inch.io/liquidity-sources-logo/curve.png",
        "img_color": "https://cdn.1inch.io/liquidity-sources-logo/curve_color.png",
    },
    {
        "id": "CURVE_V2_EURS_2_ASSET",
        "title": "Curve V2 EURS",
        "img": "https://cdn.1inch.io/liquidity-sources-logo/curve.png",
        "img_color": "https://cdn.1inch.io/liquidity-sources-logo/curve_color.png",
    },
    {
        "id": "CURVE_V2_ETH_CRV",
        "title": "Curve V2 ETH CRV",
        "img": "https://cdn.1inch.io/liquidity-sources-logo/curve.png",
        "img_color": "https://cdn.1inch.io/liquidity-sources-logo/curve_color.png",
    },
    {
        "id": "CURVE_V2_ETH_CVX",
        "title": "Curve V2 ETH CVX",
        "img": "https://cdn.1inch.io/liquidity-sources-logo/curve.png",
        "img_color": "https://cdn.1inch.io/liquidity-sources-logo/curve_color.png",
    },
    {
        "id": "CONVERGENCE_X",
        "title": "Convergence X",
        "img": "https://cdn.1inch.io/liquidity-sources-logo/convergence.png",
        "img_color": "https://cdn.1inch.io/liquidity-sources-logo/convergence_color.png",
    },
    {
        "id": "ONE_INCH_LIMIT_ORDER",
        "title": "1inch Limit Order Protocol",
        "img": "https://cdn.1inch.io/liquidity-sources-logo/1inch.png",
        "img_color": "https://cdn.1inch.io/liquidity-sources-logo/1inch_color.png",
    },
    {
        "id": "ONE_INCH_LIMIT_ORDER_V2",
        "title": "1inch Limit Order Protocol V2",
        "img": "https://cdn.1inch.io/liquidity-sources-logo/1inch.png",
        "img_color": "https://cdn.1inch.io/liquidity-sources-logo/1inch_color.png",
    },
    {
        "id": "ONE_INCH_LIMIT_ORDER_V3",
        "title": "1inch Limit Order Protocol V3",
        "img": "https://cdn.1inch.io/liquidity-sources-logo/1inch.png",
        "img_color": "https://cdn.1inch.io/liquidity-sources-logo/1inch_color.png",
    },
    {
        "id": "DFX_FINANCE",
        "title": "DFX Finance",
        "img": "https://cdn.1inch.io/liquidity-sources-logo/dfx.png",
        "img_color": "https://cdn.1inch.io/liquidity-sources-logo/dfx_color.png",
    },
    {
        "id": "FIXED_FEE_SWAP",
        "title": "Fixed Fee Swap",
        "img": "https://cdn.1inch.io/liquidity-sources-logo/1inch.png",
        "img_color": "https://cdn.1inch.io/liquidity-sources-logo/1inch_color.png",
    },
    {
        "id": "DXSWAP",
        "title": "Swapr",
        "img": "https://cdn.1inch.io/liquidity-sources-logo/swapr.png",
        "img_color": "https://cdn.1inch.io/liquidity-sources-logo/swapr_color.png",
    },
    {
        "id": "SHIBASWAP",
        "title": "ShibaSwap",
        "img": "https://cdn.1inch.io/liquidity-sources-logo/shiba.png",
        "img_color": "https://cdn.1inch.io/liquidity-sources-logo/shiba_color.png",
    },
    {
        "id": "UNIFI",
        "title": "Unifi",
        "img": "https://cdn.1inch.io/liquidity-sources-logo/unifi.png",
        "img_color": "https://cdn.1inch.io/liquidity-sources-logo/unifi_color.png",
    },
    {
        "id": "PSM_PAX",
        "title": "PSM USDP",
        "img": "https://cdn.1inch.io/liquidity-sources-logo/maker.png",
        "img_color": "https://cdn.1inch.io/liquidity-sources-logo/maker_color.png",
    },
    {
        "id": "WSTETH",
        "title": "wstETH",
        "img": "https://cdn.1inch.io/liquidity-sources-logo/steth.png",
        "img_color": "https://cdn.1inch.io/liquidity-sources-logo/steth_color.png",
    },
    {
        "id": "DEFI_PLAZA",
        "title": "DeFi Plaza",
        "img": "https://cdn.1inch.io/liquidity-sources-logo/defiplaza.png",
        "img_color": "https://cdn.1inch.io/liquidity-sources-logo/defiplaza_color.png",
    },
    {
        "id": "FIXED_FEE_SWAP_V3",
        "title": "Fixed Rate Swap",
        "img": "https://cdn.1inch.io/liquidity-sources-logo/1inch.png",
        "img_color": "https://cdn.1inch.io/liquidity-sources-logo/1inch_color.png",
    },
    {
        "id": "SYNTHETIX_WRAPPER",
        "title": "Wrapped Synthetix",
        "img": "https://cdn.1inch.io/liquidity-sources-logo/synthetix.png",
        "img_color": "https://cdn.1inch.io/liquidity-sources-logo/synthetix_color.png",
    },
    {
        "id": "SYNAPSE",
        "title": "Synapse",
        "img": "https://cdn.1inch.io/liquidity-sources-logo/synapse.png",
        "img_color": "https://cdn.1inch.io/liquidity-sources-logo/synapse_color.png",
    },
    {
        "id": "CURVE_V2_YFI_2_ASSET",
        "title": "Curve Yfi",
        "img": "https://cdn.1inch.io/liquidity-sources-logo/curve.png",
        "img_color": "https://cdn.1inch.io/liquidity-sources-logo/curve_color.png",
    },
    {
        "id": "CURVE_V2_ETH_PAL",
        "title": "Curve V2 ETH Pal",
        "img": "https://cdn.1inch.io/liquidity-sources-logo/curve.png",
        "img_color": "https://cdn.1inch.io/liquidity-sources-logo/curve_color.png",
    },
    {
        "id": "POOLTOGETHER",
        "title": "Pooltogether",
        "img": "https://cdn.1inch.io/liquidity-sources-logo/pooltogether.png",
        "img_color": "https://cdn.1inch.io/liquidity-sources-logo/pooltogether_color.png",
    },
    {
        "id": "ETH_BANCOR_V3",
        "title": "Bancor V3",
        "img": "https://cdn.1inch.io/liquidity-sources-logo/bancor.png",
        "img_color": "https://cdn.1inch.io/liquidity-sources-logo/bancor_color.png",
    },
    {
        "id": "ELASTICSWAP",
        "title": "ElasticSwap",
        "img": "https://cdn.1inch.io/liquidity-sources-logo/elastic_swap.png",
        "img_color": "https://cdn.1inch.io/liquidity-sources-logo/elastic_swap_color.png",
    },
    {
        "id": "BALANCER_V2_WRAPPER",
        "title": "Balancer V2 Aave Wrapper",
        "img": "https://cdn.1inch.io/liquidity-sources-logo/balancer.png",
        "img_color": "https://cdn.1inch.io/liquidity-sources-logo/balancer_color.png",
    },
    {
        "id": "FRAXSWAP",
        "title": "FraxSwap",
        "img": "https://cdn.1inch.io/liquidity-sources-logo/frax_swap.png",
        "img_color": "https://cdn.1inch.io/liquidity-sources-logo/frax_swap_color.png",
    },
    {
        "id": "RADIOSHACK",
        "title": "RadioShack",
        "img": "https://cdn.1inch.io/liquidity-sources-logo/radioshack.png",
        "img_color": "https://cdn.1inch.io/liquidity-sources-logo/radioshack_color.png",
    },
    {
        "id": "KYBERSWAP_ELASTIC",
        "title": "KyberSwap Elastic",
        "img": "https://cdn.1inch.io/liquidity-sources-logo/kyber.png",
        "img_color": "https://cdn.1inch.io/liquidity-sources-logo/kyber_color.png",
    },
    {
        "id": "CURVE_V2_TWO_CRYPTO",
        "title": "Curve V2 2Crypto",
        "img": "https://cdn.1inch.io/liquidity-sources-logo/curve.png",
        "img_color": "https://cdn.1inch.io/liquidity-sources-logo/curve_color.png",
    },
    {
        "id": "STABLE_PLAZA",
        "title": "Stable Plaza",
        "img": "https://cdn.1inch.io/liquidity-sources-logo/defiplaza.png",
        "img_color": "https://cdn.1inch.io/liquidity-sources-logo/defiplaza_color.png",
    },
    {
        "id": "ZEROX_LIMIT_ORDER",
        "title": "0x Limit Order",
        "img": "https://cdn.1inch.io/liquidity-sources-logo/0x.png",
        "img_color": "https://cdn.1inch.io/liquidity-sources-logo/0x_color.png",
    },
    {
        "id": "CURVE_3CRV",
        "title": "Curve 3CRV",
        "img": "https://cdn.1inch.io/liquidity-sources-logo/curve.png",
        "img_color": "https://cdn.1inch.io/liquidity-sources-logo/curve_color.png",
    },
    {
        "id": "KYBER_DMM_STATIC",
        "title": "Kyber DMM Static",
        "img": "https://cdn.1inch.io/liquidity-sources-logo/kyber.png",
        "img_color": "https://cdn.1inch.io/liquidity-sources-logo/kyber_color.png",
    },
    {
        "id": "ANGLE",
        "title": "Angle",
        "img": "https://cdn.1inch.io/liquidity-sources-logo/angle.png",
        "img_color": "https://cdn.1inch.io/liquidity-sources-logo/angle_color.png",
    },
    {
        "id": "ROCKET_POOL",
        "title": "Rocket Pool",
        "img": "https://cdn.1inch.io/liquidity-sources-logo/rocketpool.png",
        "img_color": "https://cdn.1inch.io/liquidity-sources-logo/rocketpool_color.png",
    },
    {
        "id": "ETHEREUM_ELK",
        "title": "ELK",
        "img": "https://cdn.1inch.io/liquidity-sources-logo/elk.png",
        "img_color": "https://cdn.1inch.io/liquidity-sources-logo/elk_color.png",
    },
    {
        "id": "ETHEREUM_PANCAKESWAP_V2",
        "title": "Pancake Swap",
        "img": "https://cdn.1inch.io/liquidity-sources-logo/pancakeswap.png",
        "img_color": "https://cdn.1inch.io/liquidity-sources-logo/pancakeswap_color.png",
    },
    {
        "id": "SYNTHETIX_ATOMIC_SIP288",
        "title": "Synthetix Atomic SIP288",
        "img": "https://cdn.1inch.io/liquidity-sources-logo/synthetix.png",
        "img_color": "https://cdn.1inch.io/liquidity-sources-logo/synthetix_color.png",
    },
    {
        "id": "PSM_GUSD",
        "title": "PSM GUSD",
        "img": "https://cdn.1inch.io/liquidity-sources-logo/maker.png",
        "img_color": "https://cdn.1inch.io/liquidity-sources-logo/maker_color.png",
    },
    {
        "id": "INTEGRAL",
        "title": "Integral",
        "img": "https://cdn.1inch.io/liquidity-sources-logo/integral.png",
        "img_color": "https://cdn.1inch.io/liquidity-sources-logo/integral_color.png",
    },
    {
        "id": "MAINNET_SOLIDLY",
        "title": "Solidly",
        "img": "https://cdn.1inch.io/liquidity-sources-logo/solidly.png",
        "img_color": "https://cdn.1inch.io/liquidity-sources-logo/solidly_color.png",
    },
    {
        "id": "NOMISWAP_STABLE",
        "title": "Nomiswap Stable",
        "img": "https://cdn.1inch.io/liquidity-sources-logo/nomiswap.png",
        "img_color": "https://cdn.1inch.io/liquidity-sources-logo/nomiswap_color.png",
    },
    {
        "id": "CURVE_V2_TWOCRYPTO_META",
        "title": "Curve V2 2Crypto Meta",
        "img": "https://cdn.1inch.io/liquidity-sources-logo/curve.png",
        "img_color": "https://cdn.1inch.io/liquidity-sources-logo/curve_color.png",
    },
    {
        "id": "MAVERICK_V1",
        "title": "Maverick V1",
        "img": "https://cdn.1inch.io/liquidity-sources-logo/maverick.png",
        "img_color": "https://cdn.1inch.io/liquidity-sources-logo/maverick_color.png",
    },
    {
        "id": "VERSE",
        "title": "Verse",
        "img": "https://cdn.1inch.io/liquidity-sources-logo/verse.png",
        "img_color": "https://cdn.1inch.io/liquidity-sources-logo/verse_color.png",
    },
    {
        "id": "DFX_FINANCE_V2",
        "title": "DFX Finance V2",
        "img": "https://cdn.1inch.io/liquidity-sources-logo/dfx.png",
        "img_color": "https://cdn.1inch.io/liquidity-sources-logo/dfx_color.png",
    },
    {
        "id": "ZK_BOB",
        "title": "BobSwap",
        "img": "https://cdn.1inch.io/liquidity-sources-logo/zkbob.png",
        "img_color": "https://cdn.1inch.io/liquidity-sources-logo/zkbob_color.png",
    },
    {
        "id": "PANCAKESWAP_V3",
        "title": "Pancake Swap V3",
        "img": "https://cdn.1inch.io/liquidity-sources-logo/pancakeswap.png",
        "img_color": "https://cdn.1inch.io/liquidity-sources-logo/pancakeswap_color.png",
    },
    {
        "id": "NOMISWAPEPCS",
        "title": "Nomiswap-epcs",
        "img": "https://cdn.1inch.io/liquidity-sources-logo/nomiswap.png",
        "img_color": "https://cdn.1inch.io/liquidity-sources-logo/nomiswap_color.png",
    },
    {
        "id": "XFAI",
        "title": "Xfai",
        "img": "https://cdn.1inch.io/liquidity-sources-logo/xfai.png",
        "img_color": "https://cdn.1inch.io/liquidity-sources-logo/xfai_color.png",
    },
    {
        "id": "CURVE_V2_LLAMMA",
        "title": "Curve Llama",
        "img": "https://cdn.1inch.io/liquidity-sources-logo/curve.png",
        "img_color": "https://cdn.1inch.io/liquidity-sources-logo/curve_color.png",
    },
    {
        "id": "CURVE_V2_TRICRYPTO_NG",
        "title": "Curve 3Crypto",
        "img": "https://cdn.1inch.io/liquidity-sources-logo/curve.png",
        "img_color": "https://cdn.1inch.io/liquidity-sources-logo/curve_color.png",
    },
    {
        "id": "PMM8_2",
        "title": "PMM8_2",
        "img": "https://cdn.1inch.io/liquidity-sources-logo/pmm.png",
        "img_color": "https://cdn.1inch.io/liquidity-sources-logo/pmm_color.png",
    },
    {
        "id": "SUSHISWAP_V3",
        "title": "SushiSwap V3",
        "img": "https://cdn.1inch.io/liquidity-sources-logo/sushiswap.png",
        "img_color": "https://cdn.1inch.io/liquidity-sources-logo/sushiswap_color.png",
    },
    {
        "id": "SFRX_ETH",
        "title": "sFrxEth",
        "img": "https://cdn.1inch.io/liquidity-sources-logo/frax_swap.png",
        "img_color": "https://cdn.1inch.io/liquidity-sources-logo/frax_swap_color.png",
    },
    {
        "id": "SDAI",
        "title": "sDAI",
        "img": "https://cdn.1inch.io/liquidity-sources-logo/maker.png",
        "img_color": "https://cdn.1inch.io/liquidity-sources-logo/maker_color.png",
    },
    {
        "id": "ETHEREUM_WOMBATSWAP",
        "title": "Wombat",
        "img": "https://cdn.1inch.io/liquidity-sources-logo/wombat.png",
        "img_color": "https://cdn.1inch.io/liquidity-sources-logo/wombat_color.png",
    },
    {
        "id": "CARBON",
        "title": "Carbon",
        "img": "https://cdn.1inch.io/liquidity-sources-logo/carbon.png",
        "img_color": "https://cdn.1inch.io/liquidity-sources-logo/carbon_color.png",
    },
    {
        "id": "COMPOUND_V3",
        "title": "Compound V3",
        "img": "https://cdn.1inch.io/liquidity-sources-logo/compound.png",
        "img_color": "https://cdn.1inch.io/liquidity-sources-logo/compound_color.png",
    },
    {
        "id": "DODO_V3",
        "title": "DODO v3",
        "img": "https://cdn.1inch.io/liquidity-sources-logo/dodo.png",
        "img_color": "https://cdn.1inch.io/liquidity-sources-logo/dodo_color.png",
    },
    {
        "id": "SMARDEX",
        "title": "Smardex",
        "img": "https://cdn.1inch.io/liquidity-sources-logo/smardex.png",
        "img_color": "https://cdn.1inch.io/liquidity-sources-logo/smardex_color.png",
    },
    {
        "id": "TRADERJOE_V2_1",
        "title": "TraderJoe V2.1",
        "img": "https://cdn.1inch.io/liquidity-sources-logo/traderjoe.png",
        "img_color": "https://cdn.1inch.io/liquidity-sources-logo/traderjoe_color.png",
    },
    {
        "id": "SOLIDLY_V3",
        "title": "Solidly v3",
        "img": "https://cdn.1inch.io/liquidity-sources-logo/solidly.png",
        "img_color": "https://cdn.1inch.io/liquidity-sources-logo/solidly_color.png",
    },
    {
        "id": "RAFT_PSM",
        "title": "Raft PSM",
        "img": "https://cdn.1inch.io/liquidity-sources-logo/raftpsm.png",
        "img_color": "https://cdn.1inch.io/liquidity-sources-logo/raftpsm_color.png",
    },
]
