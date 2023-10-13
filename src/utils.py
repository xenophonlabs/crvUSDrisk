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

    plt.bar(band_edges, bands_y, color='darkblue', width=band_widths, label='Collateral')
    plt.bar(band_edges, bands_x, bottom=bands_y, color='darkred', width=band_widths, label='crvUSD')
    plt.xlabel('p_o_down[n] (USD)')
    plt.ylabel('Reserves (USD)')
    plt.title('LLAMMA Collateral Distribution')
    plt.xticks([round(i) for i in band_edges], rotation=45)
    plt.show()
