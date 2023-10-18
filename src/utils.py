import numpy as np
import matplotlib.pyplot as plt

def gen_gbm(S0,mu,sigma, dt, T):
    W = np.random.normal(loc=0, scale=np.sqrt(dt), size=int(T / dt))
    S = S0 * np.exp(np.cumsum((mu - 0.5 * sigma ** 2) * dt + sigma * W))
    return(S)

def external_swap(
        x,
        y,
        swap,
        fee,
        y_in
    ):
    """
    @notice account for slippage when trading against open market
    Assuming that we trade against Uniswap is a conservative assumption
    since not accounting for CEX. 
    However, we are not accounting for crvUSD liquidity in Curve pools yet.
    """
    # TODO: Need to incorporate concentrated liquidity (e.g., no more liquidity beyond ]a, b[)
    # TODO: Need to incorproate the StableSwap invariant for crvUSD pool liquidity
    k = x*y

    if y_in:
        new_y = y + swap
        new_x = k/new_y
        out = (x - new_x) * (1-fee)
    else:
        new_x = x + swap
        new_y = k/new_x
        out = (y - new_y) * (1-fee)

    return out

# === PLOTTING === #

plt.rcParams["font.family"] = "serif"
plt.rcParams.update({'font.size': 10})
plt.rcParams["axes.spines.top"] = False
plt.rcParams["axes.spines.right"] = False

plt.rcParams['axes.facecolor'] = '#f5f5f5'
plt.rcParams['grid.color'] = 'white'
plt.rcParams['grid.linestyle'] = '-'
plt.rcParams['grid.linewidth'] = 2

def _plot_reserves(llamma):
    band_range = range(llamma.min_band, llamma.max_band+1)
    bands_x = [llamma.bands_x[i] for i in band_range]
    bands_y = [llamma.bands_y[i] * llamma.p_o for i in band_range]
    band_edges = [llamma.p_o_down(i) for i in band_range]
    band_widths = [llamma.band_width(i)*0.9 for i in band_range]

    f, ax = plt.subplots(figsize=(8, 5))

    ax.bar(band_edges, bands_y, color='royalblue', width=band_widths, label='Collateral')
    ax.bar(band_edges, bands_x, bottom=bands_y, color='indianred', width=band_widths, label='crvUSD')
    ax.set_xlabel('p_o_down[n] (USD)')
    ax.set_ylabel('Reserves (USD)')
    ax.set_title('LLAMMA Collateral Distribution')
    ax.axvline(llamma.p_o, color='black', linestyle='--', label='Oracle price')
    # ax.xticks([round(i) for i in band_edges], rotation=45)
    ax.legend()
    f.tight_layout()

def _plot_borrowers(borrowers, price):
    f, ax = plt.subplots(3, figsize=(10, 10))
    n_bins = len(borrowers) // 2
    ax[0].hist(borrowers[:,0] * price / 1e6, bins=n_bins, color='darkblue')
    ax[0].set_title("Collateral Distribution")
    ax[0].set_xlabel("Collateral (Mn USD)")
    ax[1].hist(borrowers[:,1] / 1e6, bins=n_bins, color='darkblue')
    ax[1].set_title("Debt Distribution")
    ax[1].set_xlabel("Debt (Mn USD)")
    ax[2].hist(borrowers[:,2], bins=np.unique(borrowers[:,2]), color='darkblue')
    ax[2].set_title("N Distribution")
    ax[2].set_xlabel("N")
    f.tight_layout()

