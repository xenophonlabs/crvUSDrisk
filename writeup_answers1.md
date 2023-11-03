# Model Overview

## Price Path Generation
<p>
1. A polished description of the current price generation approach we are taking of correlated GBMs, ~2 or so paragraphs, and a pretty visualization. This should indicate that we are fitting the necessary parameters using historical data.

We are currently modeling cryptoasset prices as path-dependent stochastic processes. This is a well established methodology in fincial analysis. In particular we model two main types of stochastic processes. 
</p>

### GBM
The first is for singular assets and is called geometric brownian motion (GBM for short), which presupposes a notion of a drift over time and a lognormally distributed volatility at each time step. This allows us to simulate random paths in assets that follow relatively predictable patterns. The two main equations for a GBM are as follows:

1. $dS_t = \mu S_t \, dt + \sigma S_t \, dW_t$

2. $dW_t = \eta \sqrt{dt}$

The equation $dS_t = \mu S_t \, dt + \sigma S_t \, dW_t$ describes the dynamics of an asset price that follows a Geometric Brownian Motion (GBM). The components of this equation are:

- $S_t$: The price of the asset at time $t$.
- $dS_t$: The infinitesimal change in the asset price at time $t$.
- $\mu$: The drift coefficient, representing the expected return of the asset per unit of time. This reflects the average directional movement of the asset price.
- $\sigma$: The volatility coefficient, representing the standard deviation of the asset's returns. This captures the risk or uncertainty associated with the asset price movements.
- $dt$: An infinitesimal increment of time.
- $dW_t$: The increment of a Wiener process (or standard Brownian motion) at time $t$, introducing randomness into the model. This represents the unpredictable component of the asset price changes.

In this model, the term $\mu S_t \, dt$ accounts for the expected growth rate of the asset, while the term $\sigma S_t \, dW_t$ models the random fluctuations around this trend, reflecting the inherent unpredictability of asset prices in financial markets. The volatility term $\sigma S_t$ suggests that the asset's price volatility is proportionally related to its current level, consistent with the observed behavior of many financial assets.

GBM is a key model in financial mathematics for simulating asset price behaviors, capturing both the stochastic nature of short-term price fluctuations and the long-term trend. It is also foundational in the development of financial derivatives pricing models, such as the Black-Scholes formula.
</p>

### Cholesky Decomoposition
The second is called a Cholesky Decomposition and we are using it to model stochastic price paths in multiple assets at once, where those assets may have cross correlations to each other. With regards to cryptoassets, this is often a useful layer of realism for simulation. The Cholesky Decomposition is a matrix factorization technique that decomposes a symmetric positive-definite matrix into the product of a lower triangular matrix and its conjugate transpose. It is expressed as:

$$ A = LL^T $$

where:

- $A$ is the original symmetric positive-definite matrix.
- $L$ is a lower triangular matrix with real and positive diagonal entries.
- $L^T$ is the transpose of $L$.

The decomposition is used for various numerical calculations, such as solving systems of linear equations, inverting matrices, and Monte Carlo simulations. It is particularly useful because it is more efficient and numerically stable than other methods, like the LU decomposition, when dealing with positive-definite matrices.

The process of Cholesky Decomposition involves finding the elements of $L$ using the following formulas:

For the diagonal elements of $L$:
$$ l_{jj} = \sqrt{a_{jj} - \sum_{k=1}^{j-1} l_{jk}^2} $$

For the off-diagonal elements:
$$ l_{ij} = \frac{1}{l_{jj}} \left( a_{ij} - \sum_{k=1}^{j-1} l_{ik} l_{jk} \right) \text{ for } i > j $$

The decomposition is unique; given a particular matrix $A$, there is only one lower triangular matrix $L$ with positive diagonal entries that satisfies the equation $A = LL^T$.


## Jumps/Depegs 
<p>
2. A polished description of the current approach to enforce jumps/depegs, ~2 of so paragraphs, and a pretty visualization (ideally showing a depeg, perhaps compared to the empirical USDC depeg if you think its appropriate).
</p>

## Liquidity, Slippage, and Price Impact
<p>
3. A polished description of the liquidity/slippage analysis you did, and how we are incorporating it into our model. Specifically, would be great to show the regression results on the Univ3 data and a corresponding visualization. This should describe why this approach is reasonable, and discuss some potential limitations/enhancements, ~5 or so paragraphs.
</p>

## Borrower Distributions
<p>
4. A tentative plan for modeling the "replenishing" distirbutions you mentioned for borrowers.
</p>