# Necessary imports for data manipulation, stochastic processes, and file operations
import numpy as np
from pricegenerator import PriceGenerator  # Assumed to contain financial models for price generation
import json

def main():
    # Instantiate a PriceGenerator object to simulate asset price paths
    price_generator = PriceGenerator()
    
    # Define the time horizon (T) for the price path simulation in years (here 1 year)
    # and the time step (dt) in hours, assuming 24-hour trading and 365 trading days per year
    T = 1
    dt = 1/(365*24)
    
    # Load simulation configuration parameters from a JSON file
    with open("./configs/config_1.json","r") as infile:
        config = json.load(infile)

    # Extract key parameters from the configuration file
    title = config["title"]  # Title for the simulation, possibly used in plots or reporting
    assets = config["assets"]  # List of asset identifiers to simulate
    type = config["type"]  # Type of simulation to perform

    # Check the type of simulation and generate the price paths accordingly
    if type[0] == "multi_corr":
        # If the type is 'multi_corr', a correlated multi-asset simulation is conducted
        sparse_cor = config["sparse_cor"]
        # Generate correlated GBM price paths with jumps for multiple assets
        assets = price_generator.gen_cor_jump_gbm2(assets, sparse_cor, T, dt)
    else:
        # For other types, assume no correlation is needed in the price path generation
        sparse_cor = None
        # Generate GBM price paths with jumps for assets without considering correlations
        assets = price_generator.gen_jump_gbm2(assets, T, dt)
    
    # Plot the generated GBM price paths for visualization and analysis
    price_generator.plot_gbms(T, dt, assets, title=title)

# This conditional checks if the script is executed as the main program and not imported as a module
if __name__ == "__main__":
    main()




################ Archived Code ################
    # Single Collateral Path
    # T = 1
    # dt = 1/(365*24)
    # n_assets = 1
    # mu = 8.569081760129549e-05
    # sigma = 0.022516670770215422
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
    # slippage_engine = Slippage()
   
    #@TODO: change logspace index from power to number of tokens
    # Linear Slippage
    # slippage_engine.plot_lin_collat_slippage(low=-9,high=6,x_type="log")
    
    # Multivariate Slippage
    # slippage_engine.plot_multi_var_collat_slippage(low_tokens=1e-9,high_tokens=1e5,low_vol=0,high_vol=.4,x0_type="lin",x1_type="lin")
    # slippage_engine.collateral_auction(tokens_in=1e5,price=2000,price_path=np.random.normal(1000,1500,10))

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