# crvUSD Risk Assumptions and Limitations

This risk model makes some assumptions about agent behavior, price movements, and market liquidity. These assumptions simplify the system to make it more computationally tractable and interpretable, without (ideally) detracting from the usefulness of the model's results. These assumptions do not come without risks: if the assumptions deviate meaningfully from reality, then so will the results. 

The model also comes with some technical limitations. Limitations differ from assumptions in that they are not simplifications of the system, but rather are incorrect representations of certain components.

## Assumptions

### Prices

<ol>
<li><b>Collateral GBMs.</b> Collateral prices behave as Geometric Brownian Motion processes.</li>
<li><b>Stablecoin OUs.</b> Stablecoin prices behave as Ornstein-Uhlenbeck processes.</li>
<li><b>Exogenous Prices.</b> Prices are exogenous to the system. Although trades incur slippage at any given timestep, they do not permanently affect the price trajectory of the simulation.</li>
<li><b>Endogenous crvUSD Prices.</b> crvUSD liquidity and price discovery is entirely contained in the simulated LLAMMA, StableSwap, and TriCrypto pools. The simulated Oracles and Stable Aggregator point to these pools, and any simulated trades will affect these prices.</li>
</ol>

### Liquidity

<ol>
<li><b>Isotonic External Liquidity.</b> Trades against external markets (i.e. selling collateral on Uniswap) are modeled using an <a href="https://scikit-learn.org/stable/modules/generated/sklearn.isotonic.IsotonicRegression.html">Isotonic Regressor</a> using historical 1Inch swap quotes from <a href="https://github.com/xenophonlabs/oneinch-quotes">this</a> API.</li>
<li><b>DEX Liquidity.</b> All external liquidity is contained in DEXes indexed by 1Inch.</li>
<li><b>Endogenous crvUSD Liquidity.</b> All crvUSD liquidity in endogenous to the system and contained in the simulated StableSwap, Tricrypto, and LLAMMA pools.</li>
</ol>

### Agents

<ol>
<li><b>Atomicity.</b> All trades are atomic (i.e. one timestep).</li>
<li><b>Cyclicality.</b> All trades are cyclic, meaning they are of the form: A → B → C → A for assets A, B, C.</li>
<li> <b>Passive borrowers.</b> Borrower agents are loaded in from the latest subgraph data for each LLAMMA. At the initialization step, borrowers may optionally borrow more or repay their debt. This randomizes the initial distribution of collateral and debt in the system, making the simulation results more comprehensive. Throughout the course of the simulation, borrowers do not borrow more or repay debt. This assumption becomes less reasonable on longer simulation time horizons (e.g. 1 day+).</li>
<li> <b>Passive liquidity providers.</b> Liquidity is initialized in two ways: Curve liquidity is initialized from the latest subgraph data, whereas External liquidity is extracted from historical 1inch quotes from <a href="https://github.com/xenophonlabs/oneinch-quotes">this</a> API. At the initialization step, LP agents may optionally add or remove liquidity (both on Curve pools and External Markets). This randomizes the initial liquidity for each token, making the simulation results more comprehensive. Different scenarios, such as the <b>Flash Crash</b> scenario, may enforce a liquidity crunch by having LPs withdraw a large part of the system's liquidity. Over the course of the simulation, LPs do not add or remove liquidity. This assumption becomes less reasonable on longer simulation time horizons (e.g. 1 day+).</li>
<li> <b>Greedy Arbitrageurs.</b> The purpose of the Arbitrageur is to equilibrate crvUSD pool prices against external market prices, while accounting for price impact and fees. Arbitrageurs will always execute trades that exceed their profit tolerance, which defaults to 1 USD. In this model, arbitrageurs will arbitrage all crvUSD pools to equilibrate prices at a profit. Arbitrages involving LLAMMAs are called "soft liquidations", and may incur borrower losses. All arbitrages are cyclic and are performed in the following order:
<ul>
    <li>LLAMMA → StableSwap → External Market</li>
    <li>StableSwap → LLAMMA → External Market</li>
    <li>LLAMMA → LLAMMA → External Market</li>
    <li>StableSwap → StableSwap → External Market</li>
</ul>
Every timestep, the arbitrageur will consider all cycles matching the above structure, optimize the allocation to those cycles, and <b>greedily</b> execute the most profitable cycle. They will repeat this process until no profitable arbitrages are left.
<b>Example:</b> USDC → [StableSwap crvUSD/USDC] → crvUSD → [LLAMMA crvUSD/ETH] → ETH → [External Market ETH/USDC] → USDC.</li>
<li><b>Risk Averse Liquidators.</b>The purpose of the Liquidator is to close underwater positions at a profit before they become "bad debt". The Liquidator will liquidate any position as long as a liquidation will exceed their profit tolerance. Similar to arbitrages, liquidations are cyclic and follow the form: StableSwap → LLAMMA → External Market, meaning the liquidator first sources crvUSD liquidity from the simulated StableSwap pools (that is, liquidators do not hold crvUSD inventory). Liquidators will only source crvUSD from the USDC and USDT StableSwap pools.</li>
</ol>

*Notice that by assuming arbitrages and liquidations are cyclic, we are implicitly modeling liquidations as flash swaps originating from External Markets (largely Uniswap v3 pools).*

## Limitations

We are working on fixing the below limitations.

<ol>
<li><b>Staked oracles.</b> Staked collateral pools (e.g. stETH/ETH) are not modelled explicitly. It is assumed that these pools are centered at the simulated price (e.g. stETH/ETH simulated price) and do not pose a meaningful risk vector. Notice that the wstETH and sfrxETH LLAMMAs use these staked pools in their oracles.</li>
<li><b>Frozen oracles.</b> Oracle prices do not change within a cyclic arbitrage, meaning that the first trade will not meaningfully move oracle prices before the second trade is executed. This drastically simplifies and accelerates computation, although is not completely accurate.</li>
<li><b>Missing crvUSD Liquidity.</b> The model provides a conservative estimate for crvUSD liquidity, it does not exhaustively simulate all pools. It currently includes: all LLAMMAs, all Pegged StableSwap pools, TricryptoLLAMA.</li>
<li><b>Gas.</b> The model does not currently model gas costs.</li>
</ol>
