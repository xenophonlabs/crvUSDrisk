import llamma as lm
import controller as cntrlr
import oracle as orcl
import liquidator as lqdtr
import pegkeeper as pk
import plotly.express as px
import numpy as np

def gen_gbm(S0,mu,sigma, dt, T):
    W = np.random.normal(loc=0, scale=np.sqrt(dt), size=int(T / dt))
    S = S0 * np.exp(np.cumsum((mu - 0.5 * sigma ** 2) * dt + sigma * W))
    return(S)

# Graphing
def graph(df,y1: str,y2: int =False):
    if y2!=False:
        fig = px.line(df,x=df.index,y=y1,labels={'X':'Timestep',"Y":y1})
        fig.add_trace(go.Scatter(x=df.index, y=df[y2], mode='lines',name=y2,yaxis="y2"))
        fig.update_layout(yaxis2=dict(overlaying='y',side='right'))
        fig.show()
    else:
        fig = px.line(df,x=df.index,y=y1,labels={'X':'Timestep',"Y":y1})
        fig.show()

def calc_p_impact(x,y,original_swap_x,fee):
    # x = 2e6 
    # y = 1e3
    k = x*y
    original_price = x/y
    # original_swap_x = 10e3
    # fee=0.00
    swap_x = original_swap_x*(1-fee)
    new_x = x + swap_x
    new_y = k/new_x
    swap_y = y-new_y
    trade_price = swap_x/swap_y
    new_price = new_x/new_y

    return((trade_price-original_price)/original_price)

def sim(
        T, # number of time periods
        dt, # resolution of time steps
        collat_base_price,
        collat_mu,
        collat_sigma
    ):
    # NOTE: For now assume Gas is 0? But DO create the Gas variable, and set it to 0.
    # NOTE: Eventually optimize as just DataFrame operations.

    dfs = [] # Store data from run

    # Generate collateral distribution <- This means Borrowers will create loans
    llamma = lm.LLAMMA(A=100,base_price=1800,oracle=orcl.Oracle(),fee=0.01,admin_fee=0.01)
    controller = cntrlr.Controller()
    controller.create() # NOTE: Might want to just query subgraph?

    # Gen from GBM. The price is collateral/USD price
    spot_collateral_prices = gen_gbm(S0=collat_base_price,mu=collat_mu, sigma=collat_sigma, dt=dt,T=T) 
    
    # TODO: Eventually, this price will be a function of Collateral/USDC and Collateral/USDT -> crvUSD/USD (from PKs) -> turn into Collateral/USD
    # ETH/USDC and crvUSD/USDC (Tricrypto) -> ETH/crvUSD (PK pool)
    # ETH/USDT and crvUSD/USDT (Tricrypto) -> ETH/crvUSD (PK pool)
    # LWA -> p = ETH/crvUSD
    # p_s = Aggregator price crvUSD/USD (all PK pools)
    # p = p * p_s -> ETH/USD <- this is the oracle price
    # for now, just generate ETH/USD from GBM?
    # Ultimately, will need to generate 6 price paths (+ crvUSD price path?)

    # Create external slippage Curve for ETH and crvUSD?
    liquidity = calc_p_impact(x=2e6,y=1e3,original_swap_x=10e3,fee=0.005) 

    # This loops through timesteps
    for t in range(int(T/dt)):    

        # Get price
        p_spot = spot_collateral_prices[t]

        # First: update Peg Keepers. For now: pass <- this involves arbitrage and the update() function
        # This mints/burns crvUSD
        # TODO: implement peg keeper        
        pegkeeper = pk.PegKeeper()
        pegkeeper.update()

        # Update oracle price <- This updates position healths
        oracle = orcl.Oracle()
        p_oracle = oracle.price()

        # Liquidators liquidate positions or arbitrage LLAMMA <- This updates LLAMMA/Controller
        # NOTE: Liquidators do whatever is most profitable < check hard liquidations first, then arbs (soft liquidations)
        # NOTE: This is where slippage/liquidity is important
        

        # Borrowers update positions or create new loans <- This updates LLAMMA/Controller
        # TODO: How will borrowers update positions?
        # Try to have distribution be fixed (e.g. Normally around current price)

        # Update metrics in dfs <- e.g., calculate loss/bad debt


def main():
    sim(T=1,dt=1/365,collat_base_price=1500,collat_mu=0.05,collat_sigma=0.2)

if __name__ == "__main__":
    main()