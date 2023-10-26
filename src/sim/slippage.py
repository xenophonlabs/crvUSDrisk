import numpy as np
import matplotlib.pyplot as plt
import random

class Slippage:
    def __init__(self):
       pass     

    def lin_collat_output(self,x):
        return min(1,1.081593506690093e-06*x+0.0004379110082802476)
    
    def multi_var_collat_output(self,tokens_in,volatility):
        return 0.4157 + 0.0313 * volatility + 0.0650 * tokens_in

    def collateral_auction(self,tokens_in,price,price_path=[]):
        # price defined as token0/token1 eg ETHUSD so amount of USD per ETH
        # tokens_in defined as amount of WETH being sold
        perc_loss = self.lin_output(tokens_in)
        volatility = .2        
        return tokens_in*price
    
    def stable_auction(self,tokens_in,price,price_path):
        # defined as token0/token1 eg crvUSD-USDC so amount of USDC per crvUSD
        price
        return tokens_in*price
    
    def plot_lin_collat_slippage(self,low,high,x_type="lin"):
        if x_type=="log":
            x = np.logspace(low, high, endpoint=True, base=10.0, dtype=None, axis=0)
        else:
            x = np.linspace(low,high,100)
        y = [self.lin_collat_output(x_i) for x_i in x]
        plt.plot(x, y, color='blue')
        plt.show()
        pass
    