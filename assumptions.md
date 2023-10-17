Current assumptions and plans for eliminating [some of] them.

# Weak Assumptions

These are probably not a big deal for now.

1.1. External liquidity venues (e.g. Uniswap) instantly equilibrate against external price trajectories. That is, external price trajectories are an independent input to the system, and Uniswap price will match this trajectory.
1.2. Liquidations are limited to flash swaps against DeFi CFMMs, like Uni v3.
1.3. Liquidations do not move the price or liquidity of external liquidity venues.

# Strong Assumptions

These are a big deal and we want to eliminate them

2.1. Oracle price is an EMA of external market price. This is obviously not true: oracle price is an EMA of TriCrypto-ng and crvUSD StableSwap pools.
2.2. Liquidators can buy/sell crvUSD with no slippage. Eventually we will want to model slippage from purchasing crvUSD from StableSwap pools.
2.3. Liquidators have unlimited inventory for the token they put into a liquidation. For a soft liquidation, this means they have (e.g.) unlimited WETH or unlimited crvUSD.
2.4. External liquidity (e.g. Uniswap) has static liquidity. We assume a simple CFMM where $x \cdot y = k$, where some initial $x,y,k$ are known and $k$ remains constant. This means that for some $\Delta y$ we can calculate slippage 

$$s(\Delta y; x, y, k) = \frac{k}{x} - y.$$

We rebalance $x, y$ at every time step to ensure the CFMM tracks the input price trajectory. Of course, we are implicitly also assuming a no-arbitrage condition on external liquidity venues.
2.5. There are no gas costs. Eventually we will want to plug in gas costs that are either a function of price volatility or are an input to stress test (e.g., "What happens if gas 10x?").

# Improvements

3.1. Liquidations are bounded by slippage on purchasing the input asset (i.e. flash swaps). When putting crvUSD into LLAMMA, this means they flash borrow a stablecoin and then swap it in a StableSwap pool for crvUSD. This means that they are bounded both by USDC liquidity on (e.g.) Uni v3 WETH<>USDC, AND by crvUSD<>USDC liquidity on StableSwap.
3.2. Liquidations are bounded by the slippage on crvUSD Peg Keeper markets. This eliminates assumption (4). This is basically the same as (3.1.)
3.3. External liquidity is a function of price volatility, that equilibrates against external prices on a regular cadence. This is a weaker assumption than (2.4.).
3.4. Incorporate flat gas costs.
3.5. Make Oracle price a function of TriCrypto-ng and StableSwap reserves. This means liquidations against these pools affect oracle prices.

# Future Assumptions

4.1. We might want to assume that the only trades against PK pools will be: PK updates (mint/burn), liquidations (both buy/sell).
4.2. The quote token for a liquidator will be USDC (currently is crvUSD). This means liquidators at the end of a liquidation want to sell all collateral/crvUSD into USDC which could be assumed to be \$1.