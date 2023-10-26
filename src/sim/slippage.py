import numpy as np
import matplotlib.pyplot as plt
import random
from itertools import product
import pandas as pd
import plotly.express as px

class Slippage:
    def __init__(self):
       pass     

    def lin_collat_output(self,x):
        return min(.99,1.081593506690093e-06*x+0.0004379110082802476)
    
    def multi_var_collat_output(self,tokens_in,volatility):
        beta_vals = [0.00044410620128718933,-1.2023638988922835e-05,1.0899943426418682e-06]
        output = beta_vals[0] + beta_vals[1] * volatility + beta_vals[2] * tokens_in
        return min(.99,output)

    def collateral_auction(self,tokens_in,price,price_path=[]):
        # price defined as token0/token1 eg ETHUSD so amount of USD per ETH
        # tokens_in defined as amount of WETH being sold
        perc_loss = self.lin_output(tokens_in)
        volatility=pd.DataFrame(price_path).rolling(window=min(9,len(price_path))).std().to_numpy().flatten()[-1]
        return self.multi_var_collat_output(tokens_in,volatility)*price
    
    def stable_auction(self,tokens_in,price,price_path):
        # defined as token0/token1 eg crvUSD-USDC so amount of USDC per crvUSD
        price
        return tokens_in*price*.995
    
    def plot_lin_collat_slippage(self,low,high,x_type="lin"):
        if x_type=="log":
            x = np.logspace(low, high, endpoint=True, base=10.0, dtype=None, axis=0)
        else:
            x = np.linspace(low,high,100)
        y = [self.lin_collat_output(x_i) for x_i in x]
        plt.plot(x, y, color='blue')
        plt.show()
        pass
    
    def plot_multi_var_collat_slippage(self,low_tokens,high_tokens,low_vol,high_vol,x0_type="lin",x1_type="lin"):
        if x0_type=="log":
            x0 = np.logspace(low_tokens, high_tokens, endpoint=True, base=10.0, dtype=None, axis=0)
        else:
            x0 = np.linspace(low_tokens,high_tokens,100)
        
        if x1_type=="log":
            x1 = np.logspace(low_vol, high_vol, endpoint=True, base=10.0, dtype=None, axis=0)
        else:
            x1 = np.linspace(low_vol,high_vol,100)

        combinations = list(product(x0, x1))
        df = pd.DataFrame(combinations, columns=['tokens', 'volatility'])
        df["price_impact"] = df.apply(lambda row: self.multi_var_collat_output(row["tokens"],row["volatility"]), axis=1)
        fig = px.scatter_3d(df,x="tokens",y="volatility",z="price_impact",color="price_impact")
        fig.show()
        pass