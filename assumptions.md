Current assumptions and plans for eliminating [some of] them. The vast majority of these assumptions revolve around liquidity on external liquidity venues like Uniswap, and internal liquidity venues like TriCrypto-ng and Stableswap pools.

# Weak Assumptions

These are probably not a big deal for now.

1.1. External liquidity venues (e.g. Uniswap) instantly equilibrate against external price trajectories. That is, external price trajectories are an independent input to the system, and Uniswap price will match this trajectory.
1.2. Liquidations are limited to flash swaps against DeFi CFMMs, like Uni v3.

# Strong Assumptions

These are a big deal and we want to eliminate them

2.1. Oracle price is an EMA of external market price. This is obviously not true: oracle price is an EMA of TriCrypto-ng and crvUSD StableSwap pools.
2.2. Liquidators can buy/sell crvUSD with no slippage. Eventually we will want to model slippage from purchasing crvUSD from StableSwap pools.
2.3. Liquidators have unlimited inventory for the token they put into a liquidation. For a soft liquidation, this means they have (e.g.) unlimited WETH or unlimited crvUSD.
2.4. External liquidity (e.g. Uniswap) has static liquidity. We assume a simple CFMM where $x \cdot y = k$, where some initial $x,y,k$ are known and $k$ remains constant. This means that for some $\Delta y$ we can calculate slippage as:

$$s(\Delta y; x, y, k) = \frac{k}{x} - y.$$

We rebalance $x, y$ at every time step to ensure the CFMM tracks the input price trajectory. Of course, we are implicitly also assuming a no-arbitrage condition on external liquidity venues.
2.5. There are no gas costs. Eventually we will want to plug in gas costs that are either a function of price volatility or are an input to stress test (e.g., "What happens if gas 10x?").
2.6. Liquidations are always full liquidations. This honestly doesn't seem like a heinous assumption looking at the data; all liquidations for wstETH as of Oct 17th 2023 have been full liquidations (even the `liquidate_extended` method calls are called with `frac=1`).
2.7. Liquidations do not affect the liquidity on external liquidity venues, and the liquidity on these external liquidity venues rebalance instantaneously at every timestep. Both of these are not true, and we will probably want to incorporate some system like: liquidations deplete liquidity on external liquidity venues which mean revert with a half life of $T$, or something like that. This creates a mechanism whereby liquidators will liquidate a fraction of positions as the liquidity slowly mean-reverts.
2.8. Borrowers will only borrow/withdraw/repay to maintain a pre-set distribution of collateral. For example, borrowers might change their positions to keep a normal distribution to the left of the current price. We might later change this to model these events as point processes (e.g., Poisson point processes). 

# Improvements

3.1. Liquidations are bounded by slippage on purchasing the input asset (i.e. flash swaps). When putting crvUSD into LLAMMA, this means they flash borrow a stablecoin and then swap it in a StableSwap pool for crvUSD. This means that they are bounded both by USDC liquidity on (e.g.) Uni v3 WETH<>USDC, AND by crvUSD<>USDC liquidity on StableSwap.
3.2. Liquidations are bounded by the slippage on crvUSD Peg Keeper markets. This eliminates assumption (4). This is basically the same as (3.1.)
3.3. External liquidity is a function of price volatility, that equilibrates against external prices on a regular cadence. This is a weaker assumption than (2.4.).
3.4. Incorporate flat gas costs.
3.5. Make Oracle price a function of TriCrypto-ng and StableSwap reserves. This means liquidations against these pools affect oracle prices.
3.6. We can easily incorporate some optimization function for liquidators to determine the best fraction of a user's position to liquidate and eliminate assumption 2.6. First, we must understand why partial liquidations are not being performed. Perhaps, liquidity simply has not been constrained enough for partial liquidations to be profitable (i.e., under infinite liquidity profits are linear with size).

# Future Assumptions

4.1. We might want to assume that the only trades against PK pools will be: PK updates (mint/burn), liquidations (both buy/sell).
4.2. The quote token for a liquidator will be USDC (currently is crvUSD). This means liquidators at the end of a liquidation want to sell all collateral/crvUSD into USDC which could be assumed to be \$1.