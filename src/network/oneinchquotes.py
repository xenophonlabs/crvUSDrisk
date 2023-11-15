import pandas as pd
import numpy as np
import requests as req
from typing import List
from datetime import datetime
from itertools import permutations
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception

from src.types import QuoteResponse

MAX_RETRIES = 3


def is_rate_limit_error(e):
    """Check if exception is a rate limit exception"""
    return isinstance(e, req.exceptions.HTTPError) and e.response.status_code == 429


class OneInchQuotes:
    """
    Get quotes from 1inch for specified token pairs and construct slippage
    curves.

    Partly inspired by
    https://github.com/RichardAtCT/1inch_wrapper/blob/master/oneinch_py/main.py.

    NOTE this uses the legacy 1inch API! Could update with new Fusion API.
    TODO add logging
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
        self, api_key: str, config: dict, chain: str = "ethereum", calls: int = 10
    ):
        """
        Note
        ----
        The config file should be a dictionary with the following structure:
        config[token] = {
            "address": str,
            "decimals": int,
            "min_trade_size": float, # this is the minimum trade size to quote for
            "max_trade_size": float, # this is the maximum trade size to quote for
        }
        They token key can be the address itself, or a natural language name like USDC.
        """
        self.chain_id = self.chains[chain]
        self.api_key = api_key
        self.calls = calls  # default number of calls to construct cure
        self.config = config

    @property
    def quote_url(self) -> str:
        return f"{self.base_url}/{self.version}/{self.chain_id}/quote"

    @property
    def protocols_url(self) -> str:
        return f"{self.base_url}/{self.version}/{self.chain_id}/liquidity-sources"

    @property
    def header(self) -> dict:
        return {"Authorization": f"Bearer {self.api_key}", "accept": "application/json"}

    def protocols(self) -> dict:
        res = req.get(self.protocols_url, headers=self.header)
        return res.json()

    @retry(
        stop=stop_after_attempt(MAX_RETRIES),
        wait=wait_exponential(multiplier=1, min=1, max=5),
        retry=retry_if_exception(is_rate_limit_error),
    )
    def quote(self, in_token: str, out_token: str, in_amount: int) -> QuoteResponse:
        """Get a quote from 1inch API. Retry if rate limit error."""
        params = {
            "src": in_token,
            "dst": out_token,
            "amount": str(in_amount),
            "includeGas": True,
            "includeTokensInfo": True,
            "includeProtocols": True,
        }
        res = req.get(self.quote_url, params=params, headers=self.header)
        ts = int(datetime.now().timestamp())
        if res.status_code == 200:
            return QuoteResponse(res.json(), in_amount, ts)
        else:
            raise res.raise_for_status()  # retry if rate limit error

    def quotes_for_pair(self, pair: tuple, calls: int = None) -> List[QuoteResponse]:
        """
        Get quotes for a pair of tokens using the specified amount
        range in the config.
        """
        calls = calls if calls else self.calls  # default to self.calls
        in_token, out_token = pair
        in_amounts = (
            np.geomspace(
                self.config[in_token]["min_trade_size"],
                self.config[in_token]["max_trade_size"],
                calls,
            )
            * 10 ** self.config[in_token]["decimals"]
        )
        in_amounts = [int(i) for i in in_amounts]
        responses = []
        for in_amount in in_amounts:
            res = self.quote(
                self.config[in_token]["address"],
                self.config[out_token]["address"],
                in_amount,
            )
            responses.append(res)
        return responses

    def all_quotes(
        self, tokens: List[str], calls: int = None
    ) -> List[List[QuoteResponse]]:
        """Get the quotes for all pairs of the input tokens."""
        pairs = list(permutations(tokens, 2))
        n = len(pairs)
        responses = []
        for i, pair in enumerate(pairs):
            print(f"Fetching: {pair}... {i+1}/{n}")
            responses.append(self.quotes_for_pair(pair, calls=calls))
        return responses

    def to_df(
        self, responses: List[QuoteResponse], fn=None
    ) -> pd.DataFrame:
        """Dump quote responses into a pd.DataFrame"""
        flat_responses = [item for row in responses for item in row]
        df = pd.concat([res.to_df() for res in flat_responses])
        if fn:
            df.to_csv(fn, index=False)
        return df
