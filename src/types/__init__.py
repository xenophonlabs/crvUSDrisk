from dataclasses import dataclass
import pandas as pd
import json
from curvesim.pool import SimCurvePool


@dataclass
class QuoteResponse:
    src: str
    dst: str
    in_amount: int
    out_amount: int
    gas: int
    timestamp: int
    in_decimals: int
    out_decimals: int
    price: float
    protocols: list

    def __init__(self, res: dict, in_amount: int, timestamp: int):
        self.src = res["fromToken"]["address"]
        self.dst = res["toToken"]["address"]
        self.in_amount = int(in_amount)
        self.out_amount = int(res["toAmount"])
        self.gas = int(res["gas"])
        self.timestamp = timestamp
        self.in_decimals = res["fromToken"]["decimals"]
        self.out_decimals = res["toToken"]["decimals"]
        self.price = (self.out_amount / 10**self.out_decimals) / (
            self.in_amount / 10**self.in_decimals
        )
        self.protocols = res["protocols"]
        # Cost of buying 1 unit of dst token using src token

    def to_df(self) -> pd.DataFrame:
        """
        Note
        ----
        Dumps protocols field into a JSON string. Is there
        a better approach?
        """
        return pd.DataFrame(
            [
                {
                    "src": self.src,
                    "dst": self.dst,
                    "in_amount": self.in_amount,
                    "out_amount": self.out_amount,
                    "gas": self.gas,
                    "price": self.price,
                    "protocols": json.dumps(self.protocols),
                    "timestamp": self.timestamp,
                }
            ]
        )


@dataclass
class Trade:
    """
    Trade class stores an arbitrage trade like:

    1. Sell `size` of stablecoin1 and buy crvUSD from `pool1`
    2. Sell crvUSD and buy stablecoin2 from `pool2`

    where the market price of stablecoin1/stablecoin2 is `p`.
    """

    size: int
    profit: float
    pool1: SimCurvePool  # Pool to buy crvUSD from
    pool2: SimCurvePool  # Pool to sell crvUSD to
    p: float  # Market price of stablecoins (e.g. USDC/USDT)

    def unpack(self):
        return self.size, self.pool1, self.pool2, self.p
