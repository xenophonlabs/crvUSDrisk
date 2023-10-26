import numpy as np
import matplotlib.pyplot as plt
import random


class PriceGenerator:
    # __slots__ = (
    #     "pool",  # CurvePool object
    #     "is_inverse",
    # )

    def __init__(self, pool_type="collateral", n=1):
        self.pool_type = pool_type
        self.n = n

    def gen_single_gbm(self, S0, mu, sigma, dt, T):
        W = np.random.normal(loc=0, scale=np.sqrt(dt), size=int(T / dt))
        S = S0 * np.exp(np.cumsum((mu - 0.5 * sigma**2) * dt + sigma * W))
        return S

    def gen_single_jump_gbm(
        self,
        S0,
        mu,
        sigma,
        dt,
        T,
        jump_list,
        jump_direction,
        recovery_perc,
        recovery_speed,
    ):
        n_steps = int(T / dt)
        # List of ordered pairs (jump_size, probability)
        jump_list = sorted(
            jump_list, key=lambda x: x[1]
        )  # Sort by jump probability ascending

        # Initialize asset prices
        S = np.zeros(n_steps)
        S[0] = S0

        recovery_period = 0
        jump_to_recover = 0
        jump_up_or_down = 0

        # Generate GBM paths
        for t in range(1, n_steps):
            # @ TODO: need to optimize the loop below and decide whether we want to limit number of jumps
            if recovery_period == 0:
                for size, prob in jump_list:
                    # Generate a random number to decide if a jump occurs
                    rand_num = np.random.rand()
                    if rand_num < prob:
                        # 1 = up, -1 = down
                        jump_up_or_down = random.choice(jump_direction)
                        S[t] = S[t - 1] * (1 + size * jump_up_or_down)
                        jump_to_recover = -1 * jump_up_or_down * size
                        recovery_period = recovery_speed
                        break
                    else:
                        W = np.random.normal(loc=0, scale=np.sqrt(dt))
                        S[t] = S[t - 1] * np.exp((mu - 0.5 * sigma**2) * dt + sigma * W)
            else:
                recovery_period -= 1
                S[t] = S[t - 1] * (1 + recovery_perc * jump_to_recover / recovery_speed)
        return S

    def gen_cor_matrix(self, n_assets, sparse_cor):
        cor_matrix = np.identity(n_assets)
        for i in range(n_assets):
            for j in range(n_assets):
                pair = sorted([i, j])
                if i == j:
                    cor_matrix[i][j] = 1.0
                else:
                    cor_matrix[i][j] = sparse_cor[pair[0]][pair[1]]
        return cor_matrix

    def gen_cor_jump_gbm(
        self,
        n_assets,
        T,
        dt,
        mu,
        sigma,
        S0,
        cor_matrix,
        jump_list,
        jump_direction,
        recovery_perc,
        recovery_speed,
    ):
        n_steps = int(T / dt)
        # Generate uncorrelated Brownian motions
        dW = np.sqrt(dt) * np.random.randn(n_steps, n_assets)

        # List of ordered pairs (jump_size, probability)
        jump_list = sorted(
            jump_list, key=lambda x: x[1]
        )  # Sort by jump probability ascending

        # Apply Cholesky decomposition to get correlated Brownian motions
        L = np.linalg.cholesky(cor_matrix)
        dW_correlated = dW.dot(L.T)

        # Initialize asset prices
        S = np.zeros((n_steps, n_assets))
        S[0] = S0

        recovery_period = 0
        jump_to_recover = 0
        jump_up_or_down = 0

        # Generate GBM paths
        for t in range(1, n_steps):
            if recovery_period == 0:
                # Generate a random number to decide if a jump occurs
                rand_num = np.random.rand()

                # jump diffusion based on poisson process
                # @TODO: need to optimize the loop below
                for size, prob in jump_list:
                    if rand_num < prob:
                        jump_up_or_down = random.choice(jump_direction)
                        S[t] = S[t - 1] * (1 + size * jump_up_or_down)
                        jump_to_recover = -1 * jump_up_or_down * size
                        recovery_period = recovery_speed
                        break
                    else:
                        S[t] = S[t - 1] * np.exp(
                            (mu - 0.5 * sigma**2) * dt + sigma * dW_correlated[t]
                        )
            else:
                recovery_period -= 1
                S[t] = S[t - 1] * (1 + recovery_perc * jump_to_recover / recovery_speed)

        return S

    def gen_cor_gbm(self, n_assets, n_steps, dt, mu, sigma, S0, cor_matrix):
        # Generate uncorrelated Brownian motions
        dW = np.sqrt(dt) * np.random.randn(n_steps, n_assets)

        # Apply Cholesky decomposition to get correlated Brownian motions
        L = np.linalg.cholesky(cor_matrix)
        dW_correlated = dW.dot(L.T)

        # Initialize asset prices
        S = np.zeros((n_steps, n_assets))
        S[0] = S0

        # Generate GBM paths
        for t in range(1, n_steps):
            S[t] = S[t - 1] * np.exp(
                (mu - 0.5 * sigma**2) * dt + sigma * dW_correlated[t]
            )

        return S

    def plot_gbms(self, S, n_assets):
        plt.figure(figsize=(12, 6))
        if n_assets > 1:
            for i in range(n_assets):
                plt.plot(S[:, i], label=f"Asset {i+1}")
        else:
            plt.plot(S, label="Asset")
        plt.title("Correlated Geometric Brownian Motions")
        plt.xlabel("Time Steps")
        plt.ylabel("Asset Price")
        plt.legend()
        plt.show()
