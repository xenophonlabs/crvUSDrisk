import numpy as np
import matplotlib.pyplot as plt
import random
import pprint

class PriceGenerator:
    def __init__(self):
        pass

    def gen_cor_matrix(self, n_assets, sparse_cor):
        cor_matrix = np.identity(n_assets)
        for i in range(n_assets):
            for j in range(n_assets):
                pair = sorted([i, j])
                if i == j:
                    cor_matrix[i][j] = 1.0
                else:
                    cor_matrix[i][j] = sparse_cor[str(pair[0])][str(pair[1])]
        return cor_matrix

    def gen_cor_jump_gbm2(self,assets,cor_matrix,T,dt):
        cor_matrix=self.gen_cor_matrix(len(assets),cor_matrix)
        
        n_steps = int(T / dt)
        n_assets = len(assets)            
        
        # Generate uncorrelated Brownian motions
        dW = np.sqrt(dt) * np.random.randn(n_steps,n_assets)
        # Apply Cholesky decomposition to get correlated Brownian motions
        L = np.linalg.cholesky(cor_matrix)                                
        # get the dot product of the weiner processes and the transpose of the cholesky matrix
        dW_correlated = dW.dot(L.T)
        
        # Initialize asset prices
        for index,asset in enumerate(assets):
            # jump_data = "size","prob","rec_perc","rec_speed","limit","count"
            asset["jump_data"]=sorted(asset["jump_data"], key=lambda x: x["annual_prob"])
            S = np.zeros(n_steps)
            S[0] = asset["S0"]
            asset["S"] = S
            asset["recovery_period"] = 0
            asset["jump_to_recover"] = 0
        
        # Iterate over each time step
        for t in range(1, n_steps):
            for index,asset in enumerate(assets):
                rand_num = np.random.rand()

                if asset["recovery_period"] > 0:
                    asset["recovery_period"] -= 1                    
                    asset["S"][t] = (asset["S"][t-1] + (asset["jump_to_recover"])) * np.exp(( asset["mu"] - 0.5 *  asset["sigma"]**2) * dt +  asset["sigma"] * dW_correlated[t][index])
                else:                
                    # jump diffusion based on poisson process
                    for jump in asset["jump_data"]:
                        lag = jump["lag_days"]*24
                        if rand_num < (jump["annual_prob"]/(365*24)) and jump["count"] < asset["jump_limit"] and t>lag:
                            asset["S"][t] = asset["S"][t-1] * (1 + jump["size"])
                            asset["recovery_period"] = (jump["rec_speed_days"]*24)                            
                            asset["jump_to_recover"] = (-1*jump["rec_perc"]*jump["size"]*asset["S"][t-1])/(asset["recovery_period"])
                            jump["count"]=1+jump["count"]
                            break
                        else:
                             asset["S"][t] =  asset["S"][t - 1] * np.exp(( asset["mu"] - 0.5 *  asset["sigma"]**2) * dt +  asset["sigma"] * dW_correlated[t][index])
        return assets

    
    def gen_jump_gbm2(self,assets,T,dt):
        n_steps = int(T / dt)
        n_assets = len(assets)            
        
        # Generate uncorrelated Brownian motions
        dW = np.sqrt(dt) * np.random.randn(n_steps,n_assets)
        
        # Initialize asset prices
        for index,asset in enumerate(assets):
            # jump_data = "size","prob","rec_perc","rec_speed","limit","count"
            asset["jump_data"]=sorted(asset["jump_data"], key=lambda x: x["annual_prob"])
            S = np.zeros(n_steps)
            S[0] = asset["S0"]
            asset["S"] = S
            asset["recovery_period"] = 0
            asset["jump_to_recover"] = 0
        
        # Iterate over each time step
        for t in range(1, n_steps):
            for index,asset in enumerate(assets):
                rand_num = np.random.rand()
                W = np.random.normal(loc=0, scale=np.sqrt(dt))
                if asset["recovery_period"] > 0:
                    asset["recovery_period"] -= 1                    
                    asset["S"][t] = (asset["S"][t-1] + (asset["jump_to_recover"])) * np.exp(( asset["mu"] - 0.5 *  asset["sigma"]**2) * dt +  asset["sigma"]*W)
                else:                
                    # jump diffusion based on poisson process
                    for jump in asset["jump_data"]:
                        lag = jump["lag_days"]*24
                        if rand_num < (jump["annual_prob"]/(365*24)) and jump["count"] < asset["jump_limit"] and t>lag:
                            asset["S"][t] = asset["S"][t-1] * (1 + jump["size"])
                            asset["recovery_period"] = (jump["rec_speed_days"]*24)                            
                            asset["jump_to_recover"] = (-1*jump["rec_perc"]*jump["size"]*asset["S"][t-1])/(asset["recovery_period"])
                            jump["count"]=1+jump["count"]
                            break
                        else:
                             asset["S"][t] =  asset["S"][t - 1] * np.exp(( asset["mu"] - 0.5 *  asset["sigma"]**2) * dt +  asset["sigma"]*W)
        return assets
    
    def plot_gbms(self,T,dt,assets,title="Geometric Brownian Motion"):
        plt.figure(figsize=(12, 6))
        for index,asset in enumerate(assets):
                plt.plot(asset["S"], label=f'Asset {asset["name"]}')
        plt.xticks(np.arange(0, T/dt, step=24*10))
        plt.xticks(rotation=90)
        plt.title(title)
        plt.xlabel("Time Steps")
        plt.ylabel("Asset Price")
        plt.legend()
        plt.show()





#### Archive below this line ####
    # def gen_cor_jump_gbm(
    #     self,
    #     n_assets,
    #     T,
    #     dt,
    #     mu,
    #     sigma,
    #     S0,
    #     cor_matrix,
    #     jump_list,
    #     jump_direction,
    #     recovery_perc,
    #     recovery_speed,
    # ):
    #     n_steps = int(T / dt)
    #     # Generate uncorrelated Brownian motions
    #     dW = np.sqrt(dt) * np.random.randn(n_steps, n_assets)

    #     # List of ordered pairs (jump_size, probability)
    #     jump_list = sorted(
    #         jump_list, key=lambda x: x[1]
    #     )  # Sort by jump probability ascending

    #     # Apply Cholesky decomposition to get correlated Brownian motions
    #     L = np.linalg.cholesky(cor_matrix)
    #     dW_correlated = dW.dot(L.T)

    #     # Initialize asset prices
    #     S = np.zeros((n_steps, n_assets))
    #     S[0] = S0

    #     recovery_period = 0
    #     jump_to_recover = 0
    #     jump_up_or_down = 0

    #     # Generate GBM paths
    #     for t in range(1, n_steps):
    #         if recovery_period == 0:
    #             # Generate a random number to decide if a jump occurs
    #             rand_num = np.random.rand()

    #             # jump diffusion based on poisson process
    #             # @TODO: need to optimize the loop below
    #             for size, prob in jump_list:
    #                 if rand_num < prob:
    #                     jump_up_or_down = random.choice(jump_direction)
    #                     S[t] = S[t - 1] * (1 + size * jump_up_or_down)
    #                     jump_to_recover = -1 * jump_up_or_down * size
    #                     recovery_period = recovery_speed
    #                     break
    #                 else:
    #                     S[t] = S[t - 1] * np.exp(
    #                         (mu - 0.5 * sigma**2) * dt + sigma * dW_correlated[t]
    #                     )
    #         else:
    #             recovery_period -= 1
    #             S[t] = S[t - 1] * (1 + recovery_perc * jump_to_recover / recovery_speed)

    #     return S
    
    # def gen_single_gbm(self, S0, mu, sigma, dt, T):
    #     W = np.random.normal(loc=0, scale=np.sqrt(dt), size=int(T / dt))
    #     S = S0 * np.exp(np.cumsum((mu - 0.5 * sigma**2) * dt + sigma * W))
    #     return S

    # def gen_single_jump_gbm(
    #     self,
    #     S0,
    #     mu,
    #     sigma,
    #     dt,
    #     T,
    #     jump_list,
    #     jump_direction,
    #     recovery_perc,
    #     recovery_speed,
    # ):
    #     n_steps = int(T / dt)
    #     # List of ordered pairs (jump_size, probability)
    #     jump_list = sorted(
    #         jump_list, key=lambda x: x[1]
    #     )  # Sort by jump probability ascending

    #     # Initialize asset prices
    #     S = np.zeros(n_steps)
    #     S[0] = S0

    #     recovery_period = 0
    #     jump_to_recover = 0
    #     jump_up_or_down = 0

    #     # Generate GBM paths
    #     for t in range(1, n_steps):
    #         # @ TODO: need to optimize the loop below and decide whether we want to limit number of jumps
    #         if recovery_period == 0:
    #             for size, prob in jump_list:
    #                 # Generate a random number to decide if a jump occurs
    #                 rand_num = np.random.rand()
    #                 if rand_num < prob:
    #                     # 1 = up, -1 = down
    #                     jump_up_or_down = random.choice(jump_direction)
    #                     S[t] = S[t - 1] * (1 + size * jump_up_or_down)
    #                     jump_to_recover = -1 * jump_up_or_down * size
    #                     recovery_period = recovery_speed
    #                     break
    #                 else:
    #                     W = np.random.normal(loc=0, scale=np.sqrt(dt))
    #                     S[t] = S[t - 1] * np.exp((mu - 0.5 * sigma**2) * dt + sigma * W)
    #         else:
    #             recovery_period -= 1
    #             S[t] = S[t - 1] * (1 + recovery_perc * jump_to_recover / recovery_speed)
    #     return S

    # def gen_cor_gbm(self, n_assets, n_steps, dt, mu, sigma, S0, cor_matrix):
    #     # Generate uncorrelated Brownian motions
    #     dW = np.sqrt(dt) * np.random.randn(n_steps, n_assets)

    #     # Apply Cholesky decomposition to get correlated Brownian motions
    #     L = np.linalg.cholesky(cor_matrix)
    #     dW_correlated = dW.dot(L.T)

    #     # Initialize asset prices
    #     S = np.zeros((n_steps, n_assets))
    #     S[0] = S0

    #     # Generate GBM paths
    #     for t in range(1, n_steps):
    #         S[t] = S[t - 1] * np.exp(
    #             (mu - 0.5 * sigma**2) * dt + sigma * dW_correlated[t]
    #         )

    #     return S