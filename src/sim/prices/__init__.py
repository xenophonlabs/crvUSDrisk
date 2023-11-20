import numpy as np
import scipy.optimize as so
import logging
import json
import pandas as pd
import matplotlib.pyplot as plt
from typing import List
from datetime import datetime
from dataclasses import dataclass
from ...configs.config import STABLE_CG_IDS
from ...plotting import plot_prices
from ...network.coingecko import get_prices_df, address_from_coin_id, get_current_prices

# TODO structure this code/directory better jfc


@dataclass
class PriceSample:
    timestamp: int
    prices: dict


class PricePaths:
    """
    Convenient class for storing params required
    to generate prices.
    """

    def __init__(self, fn: str, N: int):
        """
        Generate price paths from config file.

        Parameters
        ----------
        fn : str
            Path to price config file.
        N : int
            Number of timesteps.

        TODO integrate with curvesim PriceSampler?
        """

        with open(fn, "r") as f:
            logging.info(f"Reading price config from {fn}.")
            config = json.load(f)

        self.N = N
        self.params = config["params"]
        self.cov = pd.DataFrame.from_dict(config["cov"])
        self.freq = config["freq"]
        self.gran = gran(self.freq)  # in seconds
        self.coin_ids = list(self.params.keys())
        self.coins = [address_from_coin_id(coin_id) for coin_id in self.coin_ids]
        self.S0s = get_current_prices(self.coin_ids)
        self.annual_factor = factor(self.freq)
        self.dt = 1 / self.annual_factor
        self.T = self.N * self.dt
        self.S = gen_cor_prices(
            self.coin_ids,
            self.T,
            self.dt,
            self.S0s,
            self.cov,
            self.params,
            timestamps=True,
            gran=self.gran,
        )

    def __iter__(self):
        """
        Yields
        ------
        :class: `PriceSample`
        """
        for ts, prices in self.S.iterrows():
            yield PriceSample(ts, prices.to_dict())


def gen_price_config(
    fn: str,
    coins: List[str],
    start: int,
    end: int,
    freq: str = "1h",
    plot: bool = False,
) -> dict:
    """
    Generate a config file for prices for each input coin.
    The file will contain the generative parameters for the coin
    based on the historical prices between `start` and `end`.

    Parameters for
        Stablecoins: OU process with `theta`, `mu`, and `sigma`.
        Other: GBM process with `mu`, and `sigma`.

    Parameters
    ----------
    fn : str
        Filename to save config to.
    coins : List[str]
        List of coin addresses
    start : int
        Unix timestamp in milliseconds.
    end : int
        Unix timestamp in milliseconds.
    freq : Optional[str]
        Frequency of price data. Default is hourly.
    plot : Optional[bool]
        Whether to plot simulated prices.

    Returns
    -------
    config : dict
        Dictionary of parameters for each coin.

    Note
    ----
    TODO Add configs for jumps.
    """
    assert fn, "Must provide filename to write to."
    assert freq in ["1h", "1d"], f"Frequency must be 1h or 1d, not {freq}."

    logging.info("Fetching price data.")
    df = get_prices_df(coins, start, end, freq)
    assert df.shape[0] > 0, "Price data is empty."

    logging.info("Processing parameters.")
    params, cov = process_prices(df, freq)
    logging.info(f"Params\n{pd.DataFrame.from_dict(params)}")
    logging.info(f"Cov\n{cov}")
    config = {
        "params": params,
        "cov": cov.to_dict(),
        "freq": freq,
        "start": start,
        "end": end,
    }

    with open(fn, "w") as f:
        logging.info(f"Writing price config to {fn}.")
        json.dump(config, f)

    if plot:
        start = datetime.fromtimestamp(start).strftime("%Y-%m-%d")
        end = datetime.fromtimestamp(end).strftime("%Y-%m-%d")
        coin_ids = df.drop(["timestamp"], axis=1).columns.tolist()
        logging.info(
            f"Plotting empirical and simulated prices from {start} to {end}.\n"
        )
        annual_factor = factor(freq)
        T = df.shape[0] / annual_factor
        dt = 1 / annual_factor
        S0s = df[coin_ids].iloc[0].to_dict()  # Get first row of prices
        S = gen_cor_prices(coin_ids, T, dt, S0s, cov, params)
        _ = plot_prices(df[coin_ids], df2=S)
        plt.show()

    return config


### ========== Analyze Historical Prices ========== ###


def factor(freq: str) -> int:
    """Annualizing factor."""
    if freq == "1d":
        return 365
    elif freq == "1h":
        return 365 * 24
    else:
        raise ValueError(f"Invalid frequency: {freq}")


def gran(freq: str) -> int:
    """Granularity in seconds"""
    if freq == "1d":
        return 60 * 60 * 24
    elif freq == "1h":
        return 60 * 60
    else:
        raise ValueError(f"Invalid frequency: {freq}")


def process_prices(df: pd.DataFrame, freq: str = "1d") -> tuple:
    """
    Given a DataFrame of prices, compute:
        1. Parameters for the generative process
        2. covariance matrix between coins
        3. Jumps - TODO

    The parameters for the gen process are either
    mu, sigma for a simple GBM (collateral tokens)
    or theta, mu, sigma for an OU mean-reverting
    process (stablecoins).

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame of prices.
    freq : Optional[str]
        Frequency of price data. Default is daily.

    Returns
    -------
    params : dict
        Dictionary of params.
    cov : pd.DataFrame
        Covariance matrix.

    Note
    ----
    For stablecoins: params = {theta, mu, sigma, type=OU}
    Else: params = {mu, sigma, type=GBM}
    """
    df = df.copy()  # Avoid modifying original DataFrame

    params = {}

    annual_factor = factor(freq)

    dt = 1 / annual_factor  # Time step

    # Assume that each col is a different asset, with the exception of timestamp
    cols = [col for col in df.columns if col != "timestamp"]
    for col in cols:
        df[f"{col}_returns"] = df[col].pct_change()
        df[f"{col}_log_returns"] = np.log1p(df[f"{col}_returns"])

        if col in STABLE_CG_IDS:
            # Estimate an OU Process
            theta, mu, sigma = estimate_ou_parameters_MLE(df[col], dt)
            sigma *= np.sqrt(annual_factor)  # annualize
            params[col] = {"theta": theta, "mu": mu, "sigma": sigma, "type": "OU"}
        else:
            # Simple GBM estimation
            mu = df[f"{col}_log_returns"].mean() * annual_factor
            sigma = df[f"{col}_log_returns"].std() * np.sqrt(annual_factor)
            params[col] = {"mu": mu, "sigma": sigma, "type": "GBM"}

    df.dropna(inplace=True)

    # Calculate the covariance matrix of the log returns
    log_return_cols = [f"{col}_log_returns" for col in cols]
    cov = df[log_return_cols].cov() * annual_factor
    cov.columns = cols
    cov.index = cols

    return params, cov


def log_likelihood(params, X, dt):
    """
    Log-likelihood for OU process based
    on: http://www.investmentscience.com/Content/howtoArticles/MLE_for_OR_mean_reverting.pdf

    Parameters
    ----------
    params : tuple
        (theta, mu, sigma)
    X : np.array
        Time series of observations
    dt : float
        Time step (annualized)

    Returns
    -------
    float
        Negative Log-likelihood
    """
    theta, mu, sigma = params
    n = len(X)

    sigma_tilde_squared = (sigma**2) / (2 * theta) * (1 - np.exp(-2 * theta * dt))

    summation_term = 0
    for i in range(1, n):
        summation_term += (
            X.iloc[i]
            - X.iloc[i - 1] * np.exp(-theta * dt)
            - mu * (1 - np.exp(-theta * dt))
        ) ** 2

    summation_term = -summation_term / (2 * n * sigma_tilde_squared)

    # Corrected the log likelihood calculation
    log_likelihood = (
        (-np.log(2 * np.pi) / 2)
        - (np.log(np.sqrt(sigma_tilde_squared)))
        + summation_term
    )

    return -log_likelihood  # Negative for minimization


def estimate_ou_parameters_MLE(X: np.array, dt: float) -> tuple:
    """
    Estimate the parameters of an OU process using MLE.

    Parameters
    ----------
    X : np.array
        Time series of observations
    dt : float
        Time step (annualized)

    Returns
    -------
    tuple
        (theta, mu, sigma)

    Note
    ----
    TODO results are very sensitive to the initial
    guess for theta. This might indicate that the surface
    isn't actually convex, or is too flat. Still not sure
    what to do about that, but these initial guesses seem
    to give pretty reasonable prices.
    """
    bounds = ((1, None), (1e-6, None), (1e-6, None))  # theta > 0, mu ∈ ℝ, sigma > 0
    initial_guess = (1e4, np.mean(X), np.std(X))

    # Minimize the negative log likelihood
    result = so.minimize(
        log_likelihood,
        initial_guess,
        args=(X, dt),
        bounds=bounds,
        method="L-BFGS-B",
        tol=1e-6,
    )

    if result.success:
        return result.x
    else:
        raise RuntimeError("Maximum likelihood estimation failed to converge")


### ========== Generate Simulated Prices ========== ###


def gen_dW(dt, shape):
    if isinstance(shape, tuple):
        N, n = shape
        return np.sqrt(dt) * np.random.randn(N, n)
    return np.sqrt(dt) * np.random.randn(shape)


def gen_ou(theta, mu, sigma, dt, S0, N, dW=None):
    """
    Generate an Ornstein-Uhlenbeck process.
    Optionally correlated with other tokens.

    Parameters
    ----------
    theta : float
        Rate of mean reversion.
    mu : float
        Long-run mean of the process.
    sigma : float
        Volatility of the process.
    dt : float
        Time step (annualized).
    S0 : float
        Initial value of the process.
    N : int
        Number of steps to simulate.
    dW : Optional[np.array]
        Correlated Wiener process.

    Returns
    -------
    np.array
        Simulated process.
    """
    dW = dW if dW is not None else gen_dW(dt, N)
    S = np.zeros(N)
    S[0] = S0
    # Pre-calculate constants
    exp_minus_theta_dt = np.exp(-theta * dt)
    sqrt_variance_dt = np.sqrt((1 - np.exp(-2 * theta * dt)) / (2 * theta))
    for t in range(1, N):
        S[t] = (
            S[t - 1] * exp_minus_theta_dt
            + mu * (1 - exp_minus_theta_dt)
            + sigma * sqrt_variance_dt * dW[t - 1]
        )
    return S


def gen_gbm(mu, sigma, dt, S0, N, dW=None):
    """
    Generate a simple GBM process.
    Optionally correlated with other tokens.

    Parameters
    ----------
    mu : float
        Drift term.
    sigma : float
        Volatility term.
    dt : float
        Time step (annualized).
    S0 : float
        Initial value of the process.
    N : int
        Number of steps to simulate.
    dW : Optional[np.array]
        Correlated Wiener process.

    Returns
    -------
    np.array
        Simulated process.
    """
    dW = dW if dW is not None else gen_dW(dt, N)
    S = np.zeros(N)
    S[0] = S0
    for t in range(1, N):
        S[t] = S[t - 1] * np.exp((mu - 0.5 * sigma**2) * dt + sigma * dW[t - 1])
    return S


def gen_cor_prices(
    coins: List[str],
    T: float,
    dt: float,
    S0s: dict,
    cov: pd.DataFrame,
    params: dict,
    timestamps: bool = False,
    gran: int = None,
):
    """
    Generate a matrix of correlated GBMs using
    Cholesky decomposition.

    Parameters
    ----------
    coins : List[str]
        Asset ids/names.
    T : float
        Time horizon, in years
    dt : float
        Length of each step, in years.
    S0s : dict
        Dictionary of initial prices for each asset.
    cov : pd.DataFrame
        Covariance matrix, computed on log returns.
    params : dict
        Dictionary of parameters for each asset.
    timestamps : bool
        Add timestamps to df index.
    gran : int
        Timestamp granularity

    Returns
    -------
    pd.DataFrame
        Correlated prices.

    Note
    ----
    mus, sigmas, and cov are all calculated assuming
    annual returns, so T and dt must also be in years.
    """
    N = int(T / dt)  # Number of steps
    n = len(coins)

    # Generate uncorrelated Brownian motions
    dW = gen_dW(dt, (N, n))

    # Apply Cholesky decomposition to get correlated Brownian motions
    cov = cov[coins]  # Ensure correct ordering
    L = np.linalg.cholesky(cov)
    dW_correlated = dW.dot(L.T)

    # Initialize the price matrix and simulate the paths
    S = np.zeros((N, n))

    for i, asset in enumerate(coins):
        if asset in STABLE_CG_IDS:
            theta, mu, sigma = (
                params[asset]["theta"],
                params[asset]["mu"],
                params[asset]["sigma"],
            )
            S[:, i] = gen_ou(theta, mu, sigma, dt, S0s[asset], N)
        else:
            mu, sigma = params[asset]["mu"], params[asset]["sigma"]
            S[:, i] = gen_gbm(mu, sigma, dt, S0s[asset], N, dW_correlated[:, i])

    coin_ids = [address_from_coin_id(c) for c in coins]
    df = pd.DataFrame(S, columns=coin_ids)

    if timestamps:
        assert gran
        now = int(datetime.now().timestamp())
        df.index = list(range(now, now + N * gran, gran))

    return df


### ========== Outdated ========== ###


# def gen_cor_matrix(n_coins, sparse_cor):
#     cor_matrix = np.identity(n_coins)
#     for i in range(n_coins):
#         for j in range(n_coins):
#             pair = sorted([i, j])
#             if i == j:
#                 cor_matrix[i][j] = 1.0
#             else:
#                 cor_matrix[i][j] = sparse_cor[str(pair[0])][str(pair[1])]
#     return cor_matrix


# def gen_cor_jump_gbm2(coins, cor_matrix, T, dt):
#     cor_matrix = gen_cor_matrix(len(coins), cor_matrix)

#     n_steps = int(T / dt)
#     n_coins = len(coins)

#     # Generate uncorrelated Brownian motions
#     dW = np.sqrt(dt) * np.random.randn(n_steps, n_coins)
#     # Apply Cholesky decomposition to get correlated Brownian motions
#     L = np.linalg.cholesky(cor_matrix)
#     # get the dot product of the weiner processes and the transpose of the cholesky matrix
#     dW_correlated = dW.dot(L.T)

#     # Initialize asset prices
#     for index, asset in enumerate(coins):
#         # jump_data = "size","prob","rec_perc","rec_speed","limit","count"
#         asset["jump_data"] = sorted(asset["jump_data"], key=lambda x: x["annual_prob"])
#         S = np.zeros(n_steps)
#         S[0] = asset["S0"]
#         asset["S"] = S
#         asset["recovery_period"] = 0
#         asset["jump_to_recover"] = 0

#     # Iterate over each time step
#     for t in range(1, n_steps):
#         for index, asset in enumerate(coins):
#             rand_num = np.random.rand()

#             if asset["recovery_period"] > 0:
#                 asset["recovery_period"] -= 1
#                 asset["S"][t] = (
#                     asset["S"][t - 1] + (asset["jump_to_recover"])
#                 ) * np.exp(
#                     (asset["mu"] - 0.5 * asset["sigma"] ** 2) * dt
#                     + asset["sigma"] * dW_correlated[t][index]
#                 )
#             else:
#                 # jump diffusion based on poisson process
#                 for jump in asset["jump_data"]:
#                     lag = jump["lag_days"] * 24
#                     if (
#                         rand_num < (jump["annual_prob"] / (365 * 24))
#                         and jump["count"] < asset["jump_limit"]
#                         and t > lag
#                     ):
#                         asset["S"][t] = asset["S"][t - 1] * (1 + jump["size"])
#                         asset["recovery_period"] = jump["rec_speed_days"] * 24
#                         asset["jump_to_recover"] = (
#                             -1 * jump["rec_perc"] * jump["size"] * asset["S"][t - 1]
#                         ) / (asset["recovery_period"])
#                         jump["count"] = 1 + jump["count"]
#                         break
#                     else:
#                         asset["S"][t] = asset["S"][t - 1] * np.exp(
#                             (asset["mu"] - 0.5 * asset["sigma"] ** 2) * dt
#                             + asset["sigma"] * dW_correlated[t][index]
#                         )
#     return coins


# def gen_jump_gbm2(coins, T, dt):
#     n_steps = int(T / dt)

#     # Initialize asset prices
#     for index, asset in enumerate(coins):
#         # jump_data = "size","prob","rec_perc","rec_speed","limit","count"
#         asset["jump_data"] = sorted(asset["jump_data"], key=lambda x: x["annual_prob"])
#         S = np.zeros(n_steps)
#         S[0] = asset["S0"]
#         asset["S"] = S
#         asset["recovery_period"] = 0
#         asset["jump_to_recover"] = 0

#     # Iterate over each time step
#     for t in range(1, n_steps):
#         for index, asset in enumerate(coins):
#             rand_num = np.random.rand()
#             W = np.random.normal(loc=0, scale=np.sqrt(dt))
#             if asset["recovery_period"] > 0:
#                 asset["recovery_period"] -= 1
#                 asset["S"][t] = (
#                     asset["S"][t - 1] + (asset["jump_to_recover"])
#                 ) * np.exp(
#                     (asset["mu"] - 0.5 * asset["sigma"] ** 2) * dt + asset["sigma"] * W
#                 )
#             else:
#                 # jump diffusion based on poisson process
#                 for jump in asset["jump_data"]:
#                     lag = jump["lag_days"] * 24
#                     if (
#                         rand_num < (jump["annual_prob"] / (365 * 24))
#                         and jump["count"] < asset["jump_limit"]
#                         and t > lag
#                     ):
#                         asset["S"][t] = asset["S"][t - 1] * (1 + jump["size"])
#                         asset["recovery_period"] = jump["rec_speed_days"] * 24
#                         asset["jump_to_recover"] = (
#                             -1 * jump["rec_perc"] * jump["size"] * asset["S"][t - 1]
#                         ) / (asset["recovery_period"])
#                         jump["count"] = 1 + jump["count"]
#                         break
#                     else:
#                         asset["S"][t] = asset["S"][t - 1] * np.exp(
#                             (asset["mu"] - 0.5 * asset["sigma"] ** 2) * dt
#                             + asset["sigma"] * W
#                         )
#     return coins
