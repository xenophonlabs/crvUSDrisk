from metricsprocessor import MetricsProcessor
from scenario import Scenario


def generate(config: str):
    """
    Generate the necessary inputs, modules, and agents for a simulation.

    Parameters
    ----------
    config : str
        The filepath for the stress test scenario config file.

    Returns
    -------
    scenario : Scenario
        An object storing the necessary

    Note
    ----
    TODO
        1. Currently defaulting to params from
        ./configs/prices/1h_1694885962_1700073562.json <- 60 days of 1h data from
        Coingecko API. Should ideally allow the scenario to specify logic for
        choosing a parameter file. For example: specify that we want parameters
        learned from daily data from Coingecko for >1y of data.
        2. The initial price for simulation is the current price reported from
        Coingecko API. Include a way for user to specify a different date for
        the initial price?
    """
    scenario = Scenario(config)

    # Generate inputs
    pricepaths = scenario.generate_pricepaths()

    # Generate modules
    markets = scenario.generate_markets()

    return scenario, pricepaths, markets


def simulate(config: str):
    """
    Simulate a stress test scenario.

    Parameters
    ----------
    config : str
        The filepath for the stress test scenario config file.

    Returns
    -------
    metrics : MetricsProcessor
        A metrics processor object, containing the
        necessary metrics for analyzing the simulation.
    """
    inputs, modules, agents = generate(config)
    metrics = MetricsProcessor()
    return metrics


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


# def main():
#     ## Price Generation
#     price_generator = PriceGenerator()

#     # Multi Stablecoin Paths
#     # T = 1
#     # dt = 1/(365*24)
#     # n_assets = 8
#     # mu = np.full(n_assets,-7.048389410022259e-07),
#     # sigma = np.full(n_assets, 0.0030879241521733092)
#     # S0 = np.full(n_assets,1)  # Initial price for each asset
#     # # jump list of ordered pairs (jump_size, cumulative probability)
#     # jump_list = [(0.02, 0.02/24),(0.05,0.01/24)]
#     # jump_direction = [1,-1]
#     # recovery_perc = 1
#     # recovery_speed=9*24
#     # sparse_cor = {
#     # 0: {1: 0.95, 2: 0.95, 3: 0.95, 4: 0.95, 5: 0.95, 6: 0.95, 7: 0.95},
#     # 1: {2: 0.9025, 3: 0.9025, 4: 0.9025, 5: 0.9025, 6: 0.9025, 7: 0.9025},
#     # 2: {3: 0.9025, 4: 0.9025, 5: 0.9025, 6: 0.9025, 7: 0.9025},
#     # 3: {4: 0.9025, 5: 0.9025, 6: 0.9025, 7: 0.9025},
#     # 4: {5: 0.9025, 6: 0.9025, 7: 0.9025},
#     # 5: {6: 0.9025, 7: 0.9025},
#     # 6: {7: 0.9025}}
#     # cor_matrix=price_generator.gen_cor_matrix(n_assets,sparse_cor)
#     # generated_prices = price_generator.gen_cor_jump_gbm(n_assets,T,dt,mu,sigma,S0,cor_matrix,jump_list,jump_direction,recovery_perc,recovery_speed)
#     # S = generated_prices
#     # plot GBM paths
#     # price_generator.plot_gbms(S,n_assets)

#     # Single Collateral Path
#     # T = 1
#     # dt = 1/(365*24)
#     # n_assets = 1
#     # mu = 8.569081760129549e-05
#     # sigma = 0.022516670770215422
#     # S0 = 1500  # Initial price for each asset
#     # # jump list of ordered pairs (jump_size, cumulative probability)
#     # jump_list = [(0.02, 0.02/24),(0.05,0.01/24)]
#     # jump_direction = [-1]
#     # recovery_perc = .9
#     # recovery_speed=3*24
#     # generated_prices = price_generator.gen_single_jump_gbm(S0, mu, sigma, dt, T,jump_list,jump_direction,recovery_perc,recovery_speed)
#     # S = generated_prices
#     # plot GBM paths
#     # price_generator.plot_gbms(S,n_assets)

#     # Slippage
#     slippage_engine = Slippage()

#     # @TODO: change logspace index from power to number of tokens
#     # Linear Slippage
#     # slippage_engine.plot_lin_collat_slippage(low=-9,high=6,x_type="log")

#     # Multivariate Slippage
#     slippage_engine.plot_multi_var_collat_slippage(
#         low_tokens=-9,
#         high_tokens=5,
#         low_vol=0,
#         high_vol=0.4,
#         x0_type="log",
#         x1_type="lin",
#     )


def main():
    pass


if __name__ == "__main__":
    main()
