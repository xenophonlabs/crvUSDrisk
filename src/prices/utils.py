"""
Provides utilities for generating stochastic 
prices based on historical data. This includes:
1. Generating correlated prices using Cholesky decomposition.
2. Estimating generative parameters for GBMs and OU processes.
    a. We use a Log-Likelihood MLE for estimating OU parameters.
"""
import json
from typing import List, Tuple, Union
from datetime import datetime
import numpy as np
import pandas as pd
import scipy.optimize as so
from ..configs import STABLE_CG_IDS
from ..plotting import plot_prices
from ..network.coingecko import get_prices_df, address_from_coin_id
from ..logging import get_logger


logger = get_logger(__name__)


# pylint: disable=too-many-arguments, too-many-locals
def gen_price_config(
    fn: str,
    coins: List[str],
    start: int,
    end: int,
    freq: str,
    plot: bool = False,
    plot_fn: str | None = None,
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
    freq : str
        Frequency of price data.
    plot : bool
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

    logger.info("Fetching price data.")
    df = get_prices_df(coins, start, end, freq)
    assert df.shape[0] > 0, "Price data is empty."

    logger.info("Processing parameters.")
    params, cov = process_prices(df, freq)
    logger.info("Params\n%s", pd.DataFrame.from_dict(params))
    logger.info("Cov\n%s", cov)
    config = {
        "params": params,
        "cov": cov.to_dict(),
        "freq": freq,
        "start": start,
        "end": end,
    }

    with open(fn, "w", encoding="utf-8") as f:
        logger.info("Writing price config to %s.", fn)
        json.dump(config, f)

    if plot:
        coin_ids = df.drop(["timestamp"], axis=1).columns.tolist()
        logger.info(
            "Plotting empirical and simulated prices from %s to %s.",
            datetime.fromtimestamp(start).strftime("%Y-%m-%d"),
            datetime.fromtimestamp(end).strftime("%Y-%m-%d"),
        )
        annual_factor = get_factor(freq)
        T = df.shape[0] / annual_factor
        dt = 1 / annual_factor
        S0s = df[coin_ids].iloc[0].to_dict()  # Get first row of prices
        prices = gen_cor_prices(
            coin_ids,
            T,
            dt,
            S0s,
            cov,
            params,
            timestamps=True,
            gran=get_gran(freq),
            now=df.index[0],
        )
        _ = plot_prices(df[coin_ids], df2=prices, fn=plot_fn)

    return config


### ========== Analyze Historical Prices ========== ###


def get_factor(freq: str) -> int:
    """Annualizing factor."""
    if freq == "1min":
        return 365 * 24 * 60
    if freq == "5min":
        return 365 * 24 * 12
    if freq == "1h":
        return 365 * 24
    if freq == "1d":
        return 365
    raise ValueError(f"Invalid frequency: {freq}")


def get_gran(freq: str) -> int:
    """Granularity in seconds"""
    if freq == "1min":
        return 60
    if freq == "5min":
        return 60 * 5
    if freq == "1h":
        return 60 * 60
    if freq == "1d":
        return 60 * 60 * 24
    raise ValueError(f"Invalid frequency: {freq}")


def process_prices(df: pd.DataFrame, freq: str = "1d") -> Tuple[dict, pd.DataFrame]:
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
    freq : str
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

    annual_factor = get_factor(freq)

    dt = 1 / annual_factor  # Time step

    # Assume that each col is a different asset, with the exception of timestamp
    cols = [col for col in df.columns if col != "timestamp"]
    for col in cols:
        df[f"{col}_returns"] = df[col].pct_change()
        df[f"{col}_log_returns"] = np.log1p(df[f"{col}_returns"])

        if col in STABLE_CG_IDS:
            # Estimate an OU Process
            theta, mu, sigma = estimate_ou_parameters_mle(df[col], dt)
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
    cov.columns = cols  # type: ignore
    cov.index = cols  # type: ignore

    return params, cov


def log_likelihood(params: tuple, X: pd.Series, dt: float) -> float:
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
    ll = (
        (-np.log(2 * np.pi) / 2)
        - (np.log(np.sqrt(sigma_tilde_squared)))
        + summation_term
    )

    return -ll  # Negative for minimization


def estimate_ou_parameters_mle(X: pd.Series, dt: float) -> tuple:
    """
    Estimate the parameters of an OU process using MLE.

    Parameters
    ----------
    X : List[float]
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
    raise RuntimeError("Maximum likelihood estimation failed to converge")


### ========== Generate Simulated Prices ========== ###


def gen_dW(dt: float, shape: Union[tuple, int]) -> np.ndarray:
    """
    Generate a Wiener process of shape `shape`.
    """
    if isinstance(shape, tuple):
        N, n = shape
        return np.sqrt(dt) * np.random.randn(N, n)
    return np.sqrt(dt) * np.random.randn(shape)


def gen_ou(
    theta: float,
    mu: float,
    sigma: float,
    dt: float,
    S0: float,
    N: int,
    dW: np.ndarray | None = None,
) -> np.ndarray:
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
    dW : np.array | None
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


def gen_gbm(
    mu: float, sigma: float, dt: float, S0: float, N: int, dW: np.ndarray | None = None
) -> np.ndarray:
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
    dW : np.array | None
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
    gran: int | None = None,
    now: pd.Timestamp | None = None,
) -> pd.DataFrame:
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
    gran : int | None
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

    # TODO convert all CG id references to ERC20 addresses
    # and delete the STABLE_CG_IDS dict
    for i, coin in enumerate(coins):
        if coin in STABLE_CG_IDS:
            theta, mu, sigma = (
                params[coin]["theta"],
                params[coin]["mu"],
                params[coin]["sigma"],
            )
            S[:, i] = gen_ou(theta, mu, sigma, dt, S0s[coin], N)
        else:
            mu, sigma = params[coin]["mu"], params[coin]["sigma"]
            S[:, i] = gen_gbm(mu, sigma, dt, S0s[coin], N, dW_correlated[:, i])

    # TODO add jumps
    coin_ids = [address_from_coin_id(c) for c in coins]
    df = pd.DataFrame(S, columns=coin_ids)

    if timestamps:
        assert gran
        now = now or datetime.now()
        ts = int(now.timestamp())
        df.index = pd.Index(
            pd.to_datetime(list(range(ts, ts + N * gran, gran)), unit="s")
        )

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
