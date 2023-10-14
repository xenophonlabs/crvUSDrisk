import numpy as np
import matplotlib.pyplot as plt

plt.rcParams["font.family"] = "serif"
plt.rcParams.update({'font.size': 10})
plt.rcParams["axes.spines.top"] = False
plt.rcParams["axes.spines.right"] = False

def _plot_reserves(llamma):
    band_range = range(llamma.min_band, llamma.max_band+1)
    bands_x = [llamma.bands_x[i] for i in band_range]
    bands_y = [llamma.bands_y[i] * llamma.p_o for i in band_range]
    band_edges = [llamma.p_o_down(i) for i in band_range]
    band_widths = [llamma.band_width(i)*0.9 for i in band_range]

    f, ax = plt.subplots()

    ax.bar(band_edges, bands_y, color='darkblue', width=band_widths, label='Collateral')
    ax.bar(band_edges, bands_x, bottom=bands_y, color='darkred', width=band_widths, label='crvUSD')
    ax.set_xlabel('p_o_down[n] (USD)')
    ax.set_ylabel('Reserves (USD)')
    ax.set_title('LLAMMA Collateral Distribution')
    ax.axvline(llamma.p_o, color='black', linestyle='--', label='p_o')
    # ax.xticks([round(i) for i in band_edges], rotation=45)
    ax.legend()
    f.tight_layout()

def _plot_borrowers(borrowers, price):
    f, ax = plt.subplots(3, figsize=(10, 10))
    n_bins = len(borrowers) // 2
    ax[0].hist(borrowers[:,0] * price / 1e6, bins=n_bins)
    ax[0].set_title("Collateral Distribution")
    ax[0].set_xlabel("Collateral (Mn USD)")
    ax[1].hist(borrowers[:,1] / 1e6, bins=n_bins)
    ax[1].set_title("Debt Distribution")
    ax[1].set_xlabel("Debt (Mn USD)")
    ax[2].hist(borrowers[:,2], bins=np.unique(borrowers[:,2]))
    ax[2].set_title("N Distribution")
    ax[2].set_xlabel("N")
    f.tight_layout()