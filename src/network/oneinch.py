"""
Provides the `OneInchQuotes` class to 
fetch quotes from the 1inch API, and the `QuoteResponse` class
to store the response data.
"""
import json
import logging
from datetime import datetime
from dataclasses import dataclass
from itertools import permutations
from typing import List, Dict, Any
import requests as req
import pandas as pd
import numpy as np
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
)
from src.data_transfer_objects import TokenDTO

MAX_RETRIES = 3


def is_rate_limit_error(e):
    """Check if exception is a rate limit exception"""
    return isinstance(e, req.exceptions.HTTPError) and e.response.status_code == 429


# pylint: disable=too-many-instance-attributes
@dataclass
class QuoteResponse:
    """Store 1inch quote response."""

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


class OneInchQuotes:
    """
    Get quotes from 1inch for specified token pairs and construct price impact
    curves.

    Partly inspired by
    https://github.com/RichardAtCT/1inch_wrapper/blob/master/oneinch_py/main.py.

    NOTE this uses the legacy 1inch API! Could update with new Fusion API.
    """

    version = "v5.2"
    base_url = "https://api.1inch.dev/swap"

    chains = {
        "ethereum": "1",
        "binance": "56",
        "polygon": "137",
        "optimism": "10",
        "arbitrum": "42161",
        "gnosis": "100",
        "avalanche": "43114",
        "fantom": "250",
        "klaytn": "8217",
        "aurora": "1313161554",
        "zksync": "324",
    }

    def __init__(
        self,
        api_key: str,
        config: Dict[str, TokenDTO],
        chain: str = "ethereum",
        calls: int = 20,
    ):
        """
        Note
        ----
        The config file should be a dictionary of TokenDTO objects, where
        the key is the token address.
        """
        self.chain_id = self.chains[chain]
        self.api_key = api_key
        self.calls = calls  # default number of calls to construct curve
        self.config = config

    @property
    def quote_url(self) -> str:
        """URL endpoint for 1inch quotes."""
        return f"{self.base_url}/{self.version}/{self.chain_id}/quote"

    @property
    def protocols_url(self) -> str:
        """URL endpoint for 1inch protocols."""
        return f"{self.base_url}/{self.version}/{self.chain_id}/liquidity-sources"

    @property
    def header(self) -> dict:
        """Header for 1inch API requests."""
        return {"Authorization": f"Bearer {self.api_key}", "accept": "application/json"}

    def protocols(self) -> dict:
        """GET 1inch protocols."""
        res = req.get(self.protocols_url, headers=self.header, timeout=5)
        return res.json()

    @retry(
        stop=stop_after_attempt(MAX_RETRIES),
        wait=wait_exponential(multiplier=1, min=1, max=5),
        retry=retry_if_exception_type(req.HTTPError),
    )
    def quote(self, in_token: str, out_token: str, in_amount: int) -> QuoteResponse:
        """GET a quote from 1inch API. Retry if rate limit error."""
        params: Dict[str, Any] = {
            "src": in_token,
            "dst": out_token,
            "amount": str(in_amount),
            "includeGas": True,
            "includeTokensInfo": True,
            "includeProtocols": True,
        }
        res = req.get(self.quote_url, params=params, headers=self.header, timeout=5)
        res.raise_for_status()  # retry if rate limit error
        ts = int(datetime.now().timestamp())
        return QuoteResponse(res.json(), in_amount, ts)

    def quotes_for_pair(
        self, pair: tuple, calls: int | None = None
    ) -> List[QuoteResponse]:
        """
        GET quotes for a pair of tokens using the specified amount
        range in the config. The number of calls is specified by
        the `calls` parameter. If `calls` is None, use the default
        number of calls. Calls are spaced logarithmically between
        the min and max trade sizes.
        """
        calls = calls if calls else self.calls  # default to self.calls
        in_token, out_token = pair
        in_amounts = np.geomspace(
            self.config[in_token].min_trade_size,
            self.config[in_token].max_trade_size,
            calls,
        )
        # add some noise to get a more complete distribution
        noise = 1 + np.random.uniform(-0.5, 0.5, calls)
        in_amounts *= noise
        in_amounts *= 10 ** self.config[in_token].decimals
        responses = []
        for in_amount in in_amounts:
            res = self.quote(
                self.config[in_token].address,
                self.config[out_token].address,
                int(in_amount),
            )
            responses.append(res)
        return responses

    def all_quotes(
        self, tokens: List[str], calls: int | None = None
    ) -> List[QuoteResponse]:
        """GET the quotes for all pairs of the input tokens."""
        pairs = list(permutations(tokens, 2))
        n = len(pairs)
        responses = []
        for i, pair in enumerate(pairs):
            logging.info("Fetching: %s... %d/%d", pair, i + 1, n)
            responses.extend(self.quotes_for_pair(pair, calls=calls))
        return responses

    def to_df(self, responses: List[QuoteResponse], fn=None) -> pd.DataFrame:
        """Dump quote responses into a pd.DataFrame"""
        df = pd.concat([res.to_df() for res in responses])
        if fn:
            df.to_csv(fn, index=False)
        return df
