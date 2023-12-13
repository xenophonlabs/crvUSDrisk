"""Provides plotting functions for the crvusdsim package."""
import warnings
import matplotlib.pyplot as plt
from .sim import plot_prices
from .oneinch import plot_quotes, plot_predictions, plot_regression

warnings.simplefilter("ignore", UserWarning)

plt.rcParams["font.family"] = "serif"
plt.rcParams.update({"font.size": 10})
plt.rcParams["axes.spines.top"] = False
plt.rcParams["axes.spines.right"] = False

plt.rcParams["grid.color"] = "white"
plt.rcParams["grid.linestyle"] = "-"
plt.rcParams["grid.linewidth"] = 2

__all__ = ["plot_prices", "plot_quotes", "plot_predictions", "plot_regression"]
