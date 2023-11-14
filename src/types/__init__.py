from dataclasses import dataclass
import pandas as pd


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
        self.src = res["fromToken"]["symbol"]
        self.dst = res["toToken"]["symbol"]
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
        return pd.DataFrame(
            [
                {
                    "src": self.src,
                    "dst": self.dst,
                    "in_amount": self.in_amount,
                    "out_amount": self.out_amount,
                    "gas": self.gas,
                    "price": self.price,
                    "timestamp": self.timestamp,
                }
            ]
        )
