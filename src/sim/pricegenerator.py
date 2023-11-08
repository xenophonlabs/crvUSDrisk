import numpy as np
import matplotlib.pyplot as plt

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
    
    def plot_gbms(self, T, dt, assets, title="Geometric Brownian Motion"):
        # Set the style parameters similar to your ETH/BTC graph
        plt.rcParams["font.family"] = "serif"
        plt.rcParams["font.size"] = 10
        plt.rcParams["axes.spines.top"] = False
        plt.rcParams["axes.spines.right"] = True  # Enable right spine for secondary y-axis
        plt.rcParams["grid.color"] = "grey"
        plt.rcParams["grid.linestyle"] = "--"
        plt.rcParams["grid.linewidth"] = 0.5

        # Create the figure and primary axis objects
        fig, ax_left = plt.subplots(figsize=(12, 6))
        ax_right = ax_left.twinx()  # Secondary y-axis for the right side

        # Initialize right axis usage flag
        right_axis_used = False

        # Plot each asset in the assets list
        for index, asset in enumerate(assets):
            if asset['plot_left']=="False":
                # Plot on the left axis
                ax_left.plot(asset["S"], label=f'{asset["name"]}', color=f'C{index}')  # Use a consistent color cycle
                ax_left.set_ylabel("Asset Price", color=f'C{index}')
                ax_left.tick_params(axis='y', labelcolor=f'C{index}')
            else:
                # Plot on the right axis
                ax_right.plot(asset["S"], label=f'{asset["name"]}', color=f'C{index}')  # Use a consistent color cycle
                ax_right.set_ylabel("Asset Price", color=f'C{index}')
                ax_right.tick_params(axis='y', labelcolor=f'C{index}')
                right_axis_used = True

        # Set ticks and format dates on the x-axis if needed
        # Assuming that 'T' is total time and 'dt' is the time step
        total_hours = int(T * 365 * 24)
        time_steps = np.arange(0, total_hours + 1, 1)  # Every hour
        ticks_per_day = 24
        day_interval = ticks_per_day / dt  # Number of ticks per day
        # tick_labels = [f"Day {int(i/ticks_per_day)}" for i in range(0, total_hours + 1, int(day_interval))]
        # ax_left.set_xticks(range(0, total_hours + 1, int(day_interval)))
        # ax_left.set_xticklabels(tick_labels, rotation=45)

        # Add labels and title
        ax_left.set_xlabel("Time Steps")
        ax_left.set_title(title)

        # Enable the grid
        ax_left.grid(True)

        # Add a combined legend for both axes if the right axis is used, else just add for the left
        if right_axis_used:
            lines, labels = ax_left.get_legend_handles_labels()
            lines2, labels2 = ax_right.get_legend_handles_labels()
            ax_left.legend(lines + lines2, labels + labels2)
        else:
            ax_left.legend()

        # Adjust the subplot to fit the figure area
        plt.tight_layout()

        # Show the plot
        plt.show()
        return 0

