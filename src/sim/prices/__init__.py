import numpy as np
import scipy.optimize as so
import requests as req
import pandas as pd
from datetime import datetime
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
        1. Parameters for the generative process
        2. covariance matrix between assets
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


def estimate_ou_parameters_MLE(X, dt):
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


def gen_cor_prices(assets, T, dt, S0s, cov, params):
    """
    Generate a matrix of correlated GBMs using
    Cholesky decomposition.

    Parameters
    ----------
    assets : List[str]
        Asset ids/names.
    T : int
        Time horizon, in years
    dt : float
        Length of each step, in years.
    S0s : dict
        Dictionary of initial prices for each asset.
    cov : pd.DataFrame
        Covariance matrix, computed on log returns.
    params : dict
        Dictionary of parameters for each asset.

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
    n = len(assets)

    # Generate uncorrelated Brownian motions
    dW = gen_dW(dt, (N, n))

    # Apply Cholesky decomposition to get correlated Brownian motions
    cov = cov[assets]  # Ensure correct ordering
    L = np.linalg.cholesky(cov)
    dW_correlated = dW.dot(L.T)

    # Initialize the price matrix and simulate the paths
    S = np.zeros((N, n))

    for i, asset in enumerate(assets):
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

    return pd.DataFrame(S, columns=assets)


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
