import numpy as np
from datetime import datetime
import requests as req
import pandas as pd
from ...exceptions import coingeckoRateLimitException
from ...configs.config import COINGECKO_URL, STABLE_CG_IDS
from curvesim.network.coingecko import coin_ids_from_addresses_sync

### ========== Get Historical Prices ========== ###


def get_price(coin_id: str, start: int, end: int) -> list:
    """
    Get price data from coingecko API.

    Parameters
    ----------
    coin_id : str
        Coin id from coingecko API.
    start : int
        Unix timestamp in milliseconds.
    end : int
        Unix timestamp in milliseconds.

    Returns
    -------
    list
        List of price data.

    Note
    ----
    The coingecko API returns data in the following
    granularity:
        1 day from current time = 5-minutely data
        1 day from anytime (except from current time) = hourly data
        2-90 days from current time or anytime = hourly data
        above 90 days from current time or anytime = daily data (00:00 UTC)
    """
    url = COINGECKO_URL + f"coins/{coin_id}/market_chart/range"
    p = {"vs_currency": "usd", "from": start, "to": end}
    r = req.get(url, params=p)
    if r.status_code == 200:
        return r.json()["prices"]
    elif r.status_code == 429:
        raise coingeckoRateLimitException("Coingecko API Rate Limit Exceeded.")
    else:
        raise RuntimeError(f"Request failed with status code {r.status_code}.")


def get_prices_df(coins: str, start: int, end: int, freq: str = "1d") -> pd.DataFrame:
    """
    Get price data from coingecko API and convert
    into a formatted DataFrame.

    Parameters
    ----------
    coin_id : str
        Coin id from coingecko API or Ethereum address.
    start : int
        Unix timestamp in milliseconds.
    end : int
        Unix timestamp in milliseconds.
    freq : Optional[str]
        Frequency of price data. Default is daily.

    Returns
    -------
    df : pd.DataFrame
        DataFrame of price data.
    """
    dfs = []
    for coin in coins:
        if "0x" in coin:
            # Convert Ethereum address to Coingecko coin id
            coin = coin_ids_from_addresses_sync([coin], "mainnet")[0]
        prices = get_price(coin, start, end)
        cdf = pd.DataFrame(prices, columns=["timestamp", coin])
        cdf.index = pd.to_datetime(cdf["timestamp"], unit="ms")
        cdf.index.name = "datetime"
        cdf.drop(["timestamp"], axis=1, inplace=True)
        cdf = cdf.resample(freq).mean()
        dfs.append(cdf)
    df = pd.concat(dfs, axis=1)
    df["timestamp"] = df.index
    df["timestamp"] = df["timestamp"].apply(lambda x: int(datetime.timestamp(x)))
    df = df.ffill()
    return df


### ========== Analyze Historical Prices ========== ###

# E.g., determine mu, sigma for each asset
# E.g., determine correlation between assets


def factor(freq):
    if freq == "1d":
        return 365
    elif freq == "1h":
        return 365 * 24
    else:
        raise ValueError(f"Invalid frequency: {freq}")


def process_prices(df, freq="1d"):
    """
    Given a DataFrame of prices, compute:
        1. mu, sigma for each asset
        2. covariance matrix between assets
        3. Jumps - TODO

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame of prices.
    freq : Optional[str]
        Frequency of price data. Default is daily.

    Returns
    -------
    results : dict
        Dictionary of results.

    Note
    ----
    Currently we set drift of stablecoins to zero.
    This is a hack to prevent the stablecoins from
    drifting away from their pegs. Ideally, we could
    use a more sophisticated model for stablecoins,
    like Ornstein-Uhlenbeck.
    """
    df = df.copy()  # Avoid modifying original DataFrame
    mus, sigmas, cov = {}, {}, {}

    annual_factor = factor(freq)

    # Assume that each col is a different asset, with the exception of timestamp
    cols = [col for col in df.columns if col != "timestamp"]
    for col in cols:
        df[f"{col}_returns"] = df[col].pct_change()
        df[f"{col}_log_returns"] = np.log1p(df[f"{col}_returns"])

        if col in STABLE_CG_IDS:
            mus[col] = 0.0
        else:
            mus[col] = df[f"{col}_log_returns"].mean() * annual_factor
        sigmas[col] = df[f"{col}_log_returns"].std() * np.sqrt(annual_factor)

    df.dropna(inplace=True)

    # Calculate the covariance matrix of the log returns
    log_return_cols = [f"{col}_log_returns" for col in cols]
    cov = df[log_return_cols].cov() * annual_factor

    results = {
        "mus": mus,
        "sigmas": sigmas,
        "cov": cov,
    }
    return results


### ========== Generate Simulated Prices ========== ###


def gen_cor_gbm(n, T, dt, S0s, mus, sigmas, cov):
    """
    Generate a matrix of correlated GBMs using
    Cholesky decomposition.

    Parameters
    ----------
    n : int
        Number of assets.
    T : int
        Time horizon, in years
    dt : float
        Length of each step, in years.
    S0s : dict
        Dictionary of initial prices for each asset.
    mus : dict
        Dictionary of mu values for each asset,
        computed on log returns.
    sigmas : dict
        Dictionary of sigma values for each asset,
        computed on log returns.
    cov : List[List[float]] or pd.DataFrame
        Covariance matrix, computed on log returns.

    Returns
    -------
    S : pd.DataFrame
        Correlated GBMs.

    Note
    ----
    mus, sigmas, and cov are all calculated assuming
    annual returns, so T and dt must also be in years.
    """
    N = int(T / dt)  # Number of steps

    assets = sorted(S0s.keys())  # Sort keys to ensure consistent ordering
    S0s = np.array([S0s[asset] for asset in assets])
    mus = np.array([mus[asset] for asset in assets])
    sigmas = np.array([sigmas[asset] for asset in assets])

    # Generate uncorrelated Brownian motions
    dW = np.sqrt(dt) * np.random.randn(N, n)

    # Apply Cholesky decomposition to get correlated Brownian motions
    L = np.linalg.cholesky(cov)
    dW_correlated = dW.dot(L.T)

    # Initialize the price matrix and simulate the paths
    S = np.zeros((N, n))
    S[0] = S0s

    for t in range(1, N):
        S[t] = S[t - 1] * np.exp(
            (mus - 0.5 * sigmas**2) * dt + sigmas * dW_correlated[t - 1]
        )

    S = pd.DataFrame(S, columns=assets)

    return S


### ========== Outdated ========== ###


def gen_cor_matrix(n_assets, sparse_cor):
    cor_matrix = np.identity(n_assets)
    for i in range(n_assets):
        for j in range(n_assets):
            pair = sorted([i, j])
            if i == j:
                cor_matrix[i][j] = 1.0
            else:
                cor_matrix[i][j] = sparse_cor[str(pair[0])][str(pair[1])]
    return cor_matrix


def gen_cor_jump_gbm2(assets, cor_matrix, T, dt):
    cor_matrix = gen_cor_matrix(len(assets), cor_matrix)

    n_steps = int(T / dt)
    n_assets = len(assets)

    # Generate uncorrelated Brownian motions
    dW = np.sqrt(dt) * np.random.randn(n_steps, n_assets)
    # Apply Cholesky decomposition to get correlated Brownian motions
    L = np.linalg.cholesky(cor_matrix)
    # get the dot product of the weiner processes and the transpose of the cholesky matrix
    dW_correlated = dW.dot(L.T)

    # Initialize asset prices
    for index, asset in enumerate(assets):
        # jump_data = "size","prob","rec_perc","rec_speed","limit","count"
        asset["jump_data"] = sorted(asset["jump_data"], key=lambda x: x["annual_prob"])
        S = np.zeros(n_steps)
        S[0] = asset["S0"]
        asset["S"] = S
        asset["recovery_period"] = 0
        asset["jump_to_recover"] = 0

    # Iterate over each time step
    for t in range(1, n_steps):
        for index, asset in enumerate(assets):
            rand_num = np.random.rand()

            if asset["recovery_period"] > 0:
                asset["recovery_period"] -= 1
                asset["S"][t] = (
                    asset["S"][t - 1] + (asset["jump_to_recover"])
                ) * np.exp(
                    (asset["mu"] - 0.5 * asset["sigma"] ** 2) * dt
                    + asset["sigma"] * dW_correlated[t][index]
                )
            else:
                # jump diffusion based on poisson process
                for jump in asset["jump_data"]:
                    lag = jump["lag_days"] * 24
                    if (
                        rand_num < (jump["annual_prob"] / (365 * 24))
                        and jump["count"] < asset["jump_limit"]
                        and t > lag
                    ):
                        asset["S"][t] = asset["S"][t - 1] * (1 + jump["size"])
                        asset["recovery_period"] = jump["rec_speed_days"] * 24
                        asset["jump_to_recover"] = (
                            -1 * jump["rec_perc"] * jump["size"] * asset["S"][t - 1]
                        ) / (asset["recovery_period"])
                        jump["count"] = 1 + jump["count"]
                        break
                    else:
                        asset["S"][t] = asset["S"][t - 1] * np.exp(
                            (asset["mu"] - 0.5 * asset["sigma"] ** 2) * dt
                            + asset["sigma"] * dW_correlated[t][index]
                        )
    return assets


def gen_jump_gbm2(assets, T, dt):
    n_steps = int(T / dt)

    # Initialize asset prices
    for index, asset in enumerate(assets):
        # jump_data = "size","prob","rec_perc","rec_speed","limit","count"
        asset["jump_data"] = sorted(asset["jump_data"], key=lambda x: x["annual_prob"])
        S = np.zeros(n_steps)
        S[0] = asset["S0"]
        asset["S"] = S
        asset["recovery_period"] = 0
        asset["jump_to_recover"] = 0

    # Iterate over each time step
    for t in range(1, n_steps):
        for index, asset in enumerate(assets):
            rand_num = np.random.rand()
            W = np.random.normal(loc=0, scale=np.sqrt(dt))
            if asset["recovery_period"] > 0:
                asset["recovery_period"] -= 1
                asset["S"][t] = (
                    asset["S"][t - 1] + (asset["jump_to_recover"])
                ) * np.exp(
                    (asset["mu"] - 0.5 * asset["sigma"] ** 2) * dt + asset["sigma"] * W
                )
            else:
                # jump diffusion based on poisson process
                for jump in asset["jump_data"]:
                    lag = jump["lag_days"] * 24
                    if (
                        rand_num < (jump["annual_prob"] / (365 * 24))
                        and jump["count"] < asset["jump_limit"]
                        and t > lag
                    ):
                        asset["S"][t] = asset["S"][t - 1] * (1 + jump["size"])
                        asset["recovery_period"] = jump["rec_speed_days"] * 24
                        asset["jump_to_recover"] = (
                            -1 * jump["rec_perc"] * jump["size"] * asset["S"][t - 1]
                        ) / (asset["recovery_period"])
                        jump["count"] = 1 + jump["count"]
                        break
                    else:
                        asset["S"][t] = asset["S"][t - 1] * np.exp(
                            (asset["mu"] - 0.5 * asset["sigma"] ** 2) * dt
                            + asset["sigma"] * W
                        )
    return assets
