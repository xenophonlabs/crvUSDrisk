import numpy as np
# from ..modules.llamma import LLAMMA as lm
# from ..modules.controller import Controller as cntrlr
# from ..modules.oracle import Oracle as orcl
# from ..agents.liquidator import Liquidator as lqdtr
# from ..modules.pegkeeperv1 import PegKeeperV1 as pk

# import plotly.express as px
# import numpy as np
# import pandas as pd


# def gen_gbm(S0, mu, sigma, dt, T):
#     W = np.random.normal(loc=0, scale=np.sqrt(dt), size=int(T / dt))
#     S = S0 * np.exp(np.cumsum((mu - 0.5 * sigma**2) * dt + sigma * W))
#     return S


# # Graphing
# def graph(df, y1: str, y2: int = False):
#     if y2 != False:
#         fig = px.line(df, x=df.index, y=y1, labels={"X": "Timestep", "Y": y1})
#         fig.add_trace(
#             go.Scatter(x=df.index, y=df[y2], mode="lines", name=y2, yaxis="y2")
#         )
#         fig.update_layout(yaxis2=dict(overlaying="y", side="right"))
#         fig.show()
#     else:
#         fig = px.line(df, x=df.index, y=y1, labels={"X": "Timestep", "Y": y1})
#         fig.show()


# def calc_p_impact(x, y, original_swap_x, fee):
#     """
#     @notice calculate slippage when selling x or y to the open market.
#     Assuming that we trade against Uniswap is a conservative assumption
#     """
#     # TODO: Also need to incorporate original_swap_y in case we are selling collateral
#     # TODO: Need to incorporate concentrated liquidity (e.g., no more liquidity beyond ]a, b[)
#     # TODO: Need to incorproate the StableSwap invariant for crvUSD pool liquidity
#     # x = 2e6
#     # y = 1e3
#     k = x * y
#     original_price = x / y
#     # original_swap_x = 10e3
#     # fee=0.00
#     swap_x = original_swap_x * (1 - fee)
#     new_x = x + swap_x
#     new_y = k / new_x
#     swap_y = y - new_y
#     trade_price = swap_x / swap_y
#     new_price = new_x / new_y

#     return (trade_price - original_price) / original_price


# def sim(
#     T,  # number of time periods, eg 1 year
#     dt,  # resolution of time steps, eg 1/365 for daily
#     collat_base_price,  # for the gbm
#     collat_mu,  # expected drift for the collat
#     collat_sigma,  # expected volatility for collat
#     spot_stable_sigma,  # expected volatility for the stable
#     external_stable_liquidity,  #
#     external_swap_fee,
# ):
#     # NOTE: For now assume Gas is 0? But DO create the Gas variable, and set it to 0.
#     # NOTE: Eventually optimize as just DataFrame operations.

#     dfs = []  # Store data from run

#     # Generate collateral distribution <- This means Borrowers will create loans
#     llamma = lm.LLAMMA(
#         A=100, base_price=1800, oracle=orcl.Oracle(), fee=0.01, admin_fee=0.01
#     )
#     controller = cntrlr.Controller()
#     controller.create()  # NOTE: Might want to just query subgraph?

#     # Gen from GBM. The price is collateral/USD price
#     spot_collateral_prices = gen_gbm(
#         S0=collat_base_price, mu=collat_mu, sigma=collat_sigma, dt=dt, T=T
#     )

#     # TODO: Eventually, this price will be a function of Collateral/USDC and Collateral/USDT -> crvUSD/USD (from PKs) -> turn into Collateral/USD
#     # ETH/USDC and crvUSD/USDC (Tricrypto) -> ETH/crvUSD (PK pool)
#     # ETH/USDT and crvUSD/USDT (Tricrypto) -> ETH/crvUSD (PK pool)
#     # LWA -> p = ETH/crvUSD
#     # p_s = Aggregator price crvUSD/USD (all PK pools)
#     # p = p * p_s -> ETH/USD <- this is the oracle price
#     # for now, just generate ETH/USD from GBM?
#     # Ultimately, will need to generate 6 price paths (+ crvUSD price path?)
#     spot_stable_prices = gen_gbm(S0=1, mu=0, sigma=spot_stable_mu, dt=dt, T=T)

#     # This loops through timesteps
#     for t in range(int(T / dt)):
#         # Get price
#         p_spot = spot_collateral_prices[t]
#         p_ema = pd.Series(p_spot).ewm(span=5, adjust=False).mean()

#         # First: update Peg Keepers. For now: pass <- this involves arbitrage and the update() function
#         # This mints/burns crvUSD
#         # TODO: implement peg keeper
#         pegkeeper = pk.PegKeeper()
#         pegkeeper.update()

#         # Update oracle price <- This updates position healths
#         # TODO: implement oracle
#         oracle = orcl.Oracle()
#         p_oracle = oracle.price()

#         # Create external slippage Curve for ETH and crvUSD?
#         # define liquidity by $ amount in pool
#         collat_liquidity = external_stable_liquidity / p_spot
#         swap_amount = 10e3
#         price_impact = calc_p_impact(
#             x=external_stable_liquidity,
#             y=collat_liquidity,
#             original_swap_x=swap_amount,
#             fee=external_swap_fee,
#         )

#         # Liquidators liquidate positions or arbitrage LLAMMA <- This updates LLAMMA/Controller
#         # NOTE: Liquidators do whatever is most profitable < check hard liquidations first, then arbs (soft liquidations)
#         # NOTE: This is where slippage/liquidity is important
#         liquidator = lqdtr.Liquidator()
#         liquidator.liquidate(controller=controller, user=0)
#         liquidator.arbitrage()

#         # Borrowers update positions or create new loans <- This updates LLAMMA/Controller
#         # TODO: How will borrowers update positions?
#         # Try to have distribution be fixed (e.g. Normally around current price)
#         # VC: I think instead of simlating indidivudal loans here it might make more sense to simulate the whole distribution of loans at once, but we can discuss

#         # Compute gains and losses from liquidations and swaps
#         liquidation_pnl = np.zeros(int(T / dt))
#         admin_fees = np.zeros(int(T / dt))
#         arb_losses = np.zeros(int(T / dt))
#         total_gains_and_losses = pd.DataFrame(liquidation_pnl + admin_fees + arb_losses)

#     # Update metrics in dfs <- e.g., calculate loss/bad debt
#     return total_gains_and_losses
## END ARCHIVE


from pricegenerator import PriceGenerator
from slippage import Slippage

def main():
    ## price generation
    price_generator = PriceGenerator()
    
    # Multi Stablecoin Paths
    # T = 1
    # dt = 1/(365*24)
    # n_assets = 8
    # mu = np.zeros(n_assets),
    # sigma = np.full(n_assets, 0.05)
    # S0 = np.full(n_assets,1)  # Initial price for each asset
    # # jump list of ordered pairs (jump_size, cumulative probability)
    # jump_list = [(0.02, 0.02/24),(0.05,0.01/24)]    
    # jump_direction = [1,-1]
    # recovery_perc = 1
    # recovery_speed=9*24
    # sparse_cor = {
    # 0: {1: 0.95, 2: 0.95, 3: 0.95, 4: 0.95, 5: 0.95, 6: 0.95, 7: 0.95},
    # 1: {2: 0.9025, 3: 0.9025, 4: 0.9025, 5: 0.9025, 6: 0.9025, 7: 0.9025},
    # 2: {3: 0.9025, 4: 0.9025, 5: 0.9025, 6: 0.9025, 7: 0.9025},
    # 3: {4: 0.9025, 5: 0.9025, 6: 0.9025, 7: 0.9025},
    # 4: {5: 0.9025, 6: 0.9025, 7: 0.9025},
    # 5: {6: 0.9025, 7: 0.9025},
    # 6: {7: 0.9025}}
    # cor_matrix=price_generator.gen_cor_matrix(n_assets,sparse_cor)
    # generated_prices = price_generator.gen_cor_jump_gbm(n_assets,T,dt,mu,sigma,S0,cor_matrix,jump_list,jump_direction,recovery_perc,recovery_speed)
    # S = generated_prices
    # plot GBM paths
    # price_generator.plot_gbms(S,n_assets)

    # Single Collateral Path
    # T = 1
    # dt = 1/(365*24)
    # n_assets = 1
    # mu = 0.0
    # sigma = 0.05
    # S0 = 1500  # Initial price for each asset
    # # jump list of ordered pairs (jump_size, cumulative probability)
    # jump_list = [(0.02, 0.02/24),(0.05,0.01/24)]    
    # jump_direction = [-1]
    # recovery_perc = .9
    # recovery_speed=3*24
    # generated_prices = price_generator.gen_single_jump_gbm(S0, mu, sigma, dt, T,jump_list,jump_direction,recovery_perc,recovery_speed)
    # S = generated_prices
    # plot GBM paths
    # price_generator.plot_gbms(S,n_assets)

    # Slippage    
    slippage_engine = Slippage()
    slippage_engine.plot_lin_collat_slippage(low=-9,high=6,x_type="log")

if __name__ == "__main__":
    main()
