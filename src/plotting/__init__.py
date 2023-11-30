import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import imageio
import warnings
from PIL import Image
from ..utils import get_crvUSD_index
from ..modules import ExternalMarket

# FIXME remove this
warnings.simplefilter("ignore", UserWarning)

FPS = 3

FRAMES_PATH = "../figs/frames/test_"
GIFS_PATH = "../figs/gifs/test_"

fn_frames_stableswap_bals = FRAMES_PATH + "stableswap_bals_{}.png"
fn_gif_stableswap_bals = GIFS_PATH + "stableswap_bals.gif"

fn_frames_reserves = FRAMES_PATH + "reserves_{}.png"
fn_gif_reserves = GIFS_PATH + "reserves.gif"

fn_frames_actions = FRAMES_PATH + "actions_{}.png"
fn_gif_actions = GIFS_PATH + "actions.gif"

fn_frames_bad_debt = FRAMES_PATH + "bad_debt_{}.png"
fn_gif_bad_debt = GIFS_PATH + "bad_debt.gif"

fn_frames_system_health = FRAMES_PATH + "system_health_{}.png"
fn_gif_system_health = GIFS_PATH + "system_health.gif"

plt.rcParams["font.family"] = "serif"
plt.rcParams.update({"font.size": 10})
plt.rcParams["axes.spines.top"] = False
plt.rcParams["axes.spines.right"] = False

# plt.rcParams["axes.facecolor"] = "#f5f5f5"
plt.rcParams["grid.color"] = "white"
plt.rcParams["grid.linestyle"] = "-"
plt.rcParams["grid.linewidth"] = 2


def make_gif(frames_fn, gif_fn, n, fps=FPS):
    frames = [imageio.v2.imread(frames_fn.format(i)) for i in range(n)]
    imageio.mimsave(gif_fn, frames, fps=fps)


def plot_stableswap_balances(pools, bals, width=0.25, fn=None, ylim=None):
    """
    Simple barchart of pool balances.

    Note
    ----
    We take in `pools` to get the index of crvUSD. We don't use it
    for balances since we are plotting the previous balances passed in
    by the `bals` array.
    """
    n = len(pools)
    width = 1 / (2 * n)
    ind = np.arange(n)

    pool_names = [
        p.metadata["name"].replace("Curve.fi Factory Plain Pool: ", "") for p in pools
    ]

    f, ax = plt.subplots(figsize=(8, 5))

    crvUSD_balances = []
    other_balances = []

    for i in range(n):
        # Assume bals = [pool_0_coin_0, pool_0_coin_1, pool_1_coin_0, ..., pool_n_coin_1]
        # so crvUSD idx for pool i is 2*i + get_crvUSD_index(pool i)
        crvUSD_idx = get_crvUSD_index(pools[i])
        other_idx = 2 * i + crvUSD_idx ^ 1
        crvUSD_idx = 2 * i + crvUSD_idx
        crvUSD_balances.append(bals[crvUSD_idx] / 1e6)
        other_balances.append(bals[other_idx] / 1e6)

    ax.bar(ind, crvUSD_balances, width, label="crvUSD Balance", color="indianred")
    ax.bar(ind + width, other_balances, width, label="Other Balance", color="royalblue")

    ax.set_xticks(ind + width / 2, pool_names)
    ax.set_ylabel("Token Balance (Mns)")
    ax.set_xlabel("Pool Tokens")
    ax.set_title("crvUSD Pool Balances")

    if ylim:
        ax.set_ylim(0, ylim / 1e6)

    f.legend(loc="upper center", bbox_to_anchor=(0.5, 0), ncol=2)
    f.tight_layout()

    if fn:
        plt.savefig(fn, bbox_inches="tight", dpi=300)
        plt.close()  # don't show

    return f


def plot_combined_2(fps=FPS):
    """
    FIXME better name
    FIXME just make it like combined_1 with all the frames being gifs.
    """
    # Read frames from both GIFs
    reader = imageio.get_reader(fn_gif_stableswap_bals)

    # Initialize list to store new frames
    new_frames = []

    # These are static
    img2 = Image.open("../figs/test_arbitrage_profits.png")
    img3 = Image.open("../figs/test_prices.png")
    img4 = Image.open("../figs/test_agg_price.png")

    # Combine frames
    for frame in reader:
        img1 = Image.fromarray(frame)

        # Concatenate images
        new_img = Image.new("RGB", (img1.width + img2.width, img1.height + img3.height))
        new_img.paste(img1, (0, 0))
        new_img.paste(img2, (img1.width, 0))
        new_img.paste(img3, (0, img1.height))
        new_img.paste(img4, (img1.width, img1.height))

        # Convert back to array and append to new frames
        new_frames.append(np.array(new_img))

    # Create new GIF
    imageio.mimsave("../figs/gifs/combined_2.gif", new_frames, fps=fps)


def plot_combined(fn):
    buffer_x = 150
    buffer_y = 150
    background = "white"

    # Read frames from both GIFs
    reader1 = imageio.get_reader(fn_gif_reserves)
    reader2 = imageio.get_reader(fn_gif_actions)
    reader3 = imageio.get_reader(fn_gif_bad_debt)
    reader4 = imageio.get_reader(fn_gif_system_health)

    # Initialize list to store new frames
    new_frames = []

    # Combine frames
    for frame1, frame2, frame3, frame4 in zip(reader1, reader2, reader3, reader4):
        img1 = Image.fromarray(frame1)
        img2 = Image.fromarray(frame2)
        img3 = Image.fromarray(frame3)
        img4 = Image.fromarray(frame4)

        # Concatenate images
        new_img = Image.new(
            "RGB",
            (img1.width + img2.width + buffer_x, img1.height + img3.height + buffer_y),
            background,
        )
        new_img.paste(img1, (0, 0))
        new_img.paste(img2, (img1.width + buffer_x, 0))
        new_img.paste(img3, (0, img1.height + buffer_y))
        new_img.paste(img4, (img1.width + buffer_x, img1.height + buffer_y))

        # Convert back to array and append to new frames
        new_frames.append(np.array(new_img))

    # Create new GIF
    imageio.mimsave(fn, new_frames, fps=FPS)


def plot_actions(
    df, i, min_time, max_time, min_price, max_price, min_pnl, max_pnl, fn, alpha=0.75
):
    f, ax = plt.subplots(figsize=(8, 5))

    ax2 = ax.twinx()

    ndf = df.loc[:i]

    ax.plot(ndf.index, ndf["spot"], label="Spot", color="black", linestyle="--", lw=1)
    ax.plot(ndf.index, ndf["oracle"], label="Oracle", color="black", lw=1)

    ax.set_title("Simulated Agent Behavior: Liquidator")
    ax.set_ylabel("Collateral Price (USD)")
    ax.tick_params(axis="x", rotation=45)
    ax.set_ylim(min_price, max_price)
    ax.set_xlim(min_time, max_time)

    width = (df.index[1] - df.index[0]) * 0.75

    ax2.bar(
        ndf.index,
        ndf["liquidation_pnl"],
        label="Liquidation PnL",
        color="indianred",
        width=width,
        alpha=alpha,
        zorder=2,
    )
    ax2.bar(
        ndf.index,
        ndf["arbitrage_pnl"],
        bottom=ndf["liquidation_pnl"],
        label="Arbitrage PnL",
        color="royalblue",
        width=width,
        alpha=alpha,
        zorder=2,
    )

    ax2.set_xlim(min_time, max_time)
    ax2.set_ylim(min_pnl, max_pnl)
    ax2.set_ylabel("Liquidator PnL (USD)")

    f.legend(loc="upper center", bbox_to_anchor=(0.5, 0), ncol=4)
    f.tight_layout()

    plt.savefig(fn, bbox_inches="tight", dpi=300)
    plt.close()  # don't show


def plot_bad_debt(
    df, i, min_time, max_time, min_price, max_price, min_bad_debt, max_bad_debt, fn
):
    f, ax = plt.subplots(figsize=(8, 5))

    ax2 = ax.twinx()

    ndf = df.loc[:i]

    ax.plot(ndf.index, ndf["spot"], label="Spot", color="black", linestyle="--", lw=1)
    ax.plot(ndf.index, ndf["oracle"], label="Oracle", color="black", lw=1)

    ax.set_title("Simulated Bad Debt")
    ax.set_ylabel("Collateral Price (USD)")
    ax.tick_params(axis="x", rotation=45)
    ax.set_ylim(min_price, max_price)
    ax.set_xlim(min_time, max_time)

    ax2.plot(ndf.index, ndf["bad_debt"], label="Bad Debt", color="indianred", lw=1)

    ax2.set_xlim(min_time, max_time)
    ax2.set_ylim(min_bad_debt, max_bad_debt)
    ax2.set_ylabel("Bad Debt (USD)")

    f.legend(loc="upper center", bbox_to_anchor=(0.5, 0), ncol=3)
    f.tight_layout()

    plt.savefig(fn, bbox_inches="tight", dpi=300)
    plt.close()  # don't show


def plot_system_health(
    df,
    i,
    min_time,
    max_time,
    min_price,
    max_price,
    min_system_health,
    max_system_health,
    fn,
):
    f, ax = plt.subplots(figsize=(8, 5))

    ax2 = ax.twinx()

    ndf = df.loc[:i]

    ax.plot(ndf.index, ndf["spot"], label="Spot", color="black", linestyle="--", lw=1)
    ax.plot(ndf.index, ndf["oracle"], label="Oracle", color="black", lw=1)

    ax.set_title("Simulated System Health")
    ax.set_ylabel("Collateral Price (USD)")
    ax.tick_params(axis="x", rotation=45)
    ax.set_ylim(min_price, max_price)
    ax.set_xlim(min_time, max_time)

    ax2.plot(
        ndf.index, ndf["system_health"], label="System Health", color="indianred", lw=1
    )

    ax2.set_xlim(min_time, max_time)
    ax2.set_ylim(min_system_health, max_system_health)
    ax2.set_ylabel("System Health")

    f.legend(loc="upper center", bbox_to_anchor=(0.5, 0), ncol=3)
    f.tight_layout()

    plt.savefig(fn, bbox_inches="tight", dpi=300)
    plt.close()  # don't show


def plot_sim(df, scale=0.1) -> None:
    # Define plot bounds
    min_time, max_time = df.index.min(), df.index.max()
    min_price, max_price = (
        df["spot"].min() * (1 - scale),
        df["spot"].max() * (1 + scale),
    )
    min_pnl, max_pnl = (
        0,
        max(df["liquidation_pnl"].max(), df["arbitrage_pnl"].max()) * (1 + scale),
    )
    min_bad_debt, max_bad_debt = 0, df["bad_debt"].max() * (1 + scale)
    min_system_health, max_system_health = 0, df["system_health"].max() * (1 + scale)

    frame = 0
    for i, _ in df.iterrows():
        plot_actions(
            df,
            i,
            min_time,
            max_time,
            min_price,
            max_price,
            min_pnl,
            max_pnl,
            fn_frames_actions.format(frame),
        )
        plot_bad_debt(
            df,
            i,
            min_time,
            max_time,
            min_price,
            max_price,
            min_bad_debt,
            max_bad_debt,
            fn_frames_bad_debt.format(frame),
        )
        plot_system_health(
            df,
            i,
            min_time,
            max_time,
            min_price,
            max_price,
            min_system_health,
            max_system_health,
            fn_frames_system_health.format(frame),
        )
        frame += 1

    n = len(df)
    make_gif(fn_frames_actions, fn_gif_actions, n)
    make_gif(fn_frames_bad_debt, fn_gif_bad_debt, n)
    make_gif(fn_frames_system_health, fn_gif_system_health, n)


def plot_reserves(llamma, fn=None, max_y=None):
    band_range = range(llamma.min_band, llamma.max_band + 1)
    bands_x = [llamma.bands_x[i] for i in band_range]
    bands_y = [llamma.bands_y[i] * llamma.p_o for i in band_range]
    band_edges = [llamma.p_o_down(i) for i in band_range]
    band_widths = [llamma.band_width(i) * 0.9 for i in band_range]

    f, ax = plt.subplots(figsize=(8, 5))

    ax.bar(
        band_edges, bands_y, color="royalblue", width=band_widths, label="Collateral"
    )
    ax.bar(
        band_edges,
        bands_x,
        bottom=bands_y,
        color="indianred",
        width=band_widths,
        label="crvUSD",
    )
    ax.set_xlabel("p_o_down[n] (USD)")
    ax.set_ylabel("Reserves (USD)")
    ax.set_title("LLAMMA Collateral Distribution")
    ax.axvline(llamma.p_o, color="black", linestyle="--", label="Oracle price")
    ax.axvline(llamma.p, color="green", linestyle="--", label="AMM price")
    # ax.xticks([round(i) for i in band_edges], rotation=45)

    if max_y:
        ax.set_ylim(0, max_y)

    # f.legend(loc="upper center", bbox_to_anchor=(0.5, 0), ncol=4)
    ax.legend()
    f.tight_layout()

    if fn:
        plt.savefig(fn, dpi=300)
        plt.close()  # don't show

    return f


def plot_borrowers(borrowers, price, fn=None):
    f, ax = plt.subplots(3, figsize=(10, 10))
    n_bins = len(borrowers) // 2
    ax[0].hist(borrowers[:, 0] * price / 1e6, bins=n_bins, color="darkblue")
    ax[0].set_title("Collateral Distribution")
    ax[0].set_xlabel("Collateral (Mn USD)")
    ax[1].hist(borrowers[:, 1] / 1e6, bins=n_bins, color="darkblue")
    ax[1].set_title("Debt Distribution")
    ax[1].set_xlabel("Debt (Mn USD)")
    ax[2].hist(borrowers[:, 2], bins=np.unique(borrowers[:, 2]), color="darkblue")
    ax[2].set_title("N Distribution")
    ax[2].set_xlabel("N")
    f.tight_layout()
    if fn:
        plt.savefig(fn, bbox_inches="tight", dpi=300)
        plt.close()  # don't show
    return f


# def plot_predictions(df0, df1, fn=None):
#     f, axs = plt.subplots(1, 2, figsize=(10, 5))

#     axs[0].scatter(
#         df0["amount0_adjusted"],
#         -df0["amount1_adjusted"],
#         label="True",
#         c="indianred",
#         s=1,
#     )
#     axs[0].scatter(
#         df0["amount0_adjusted"], -df0["predicted"], label="Pred", c="royalblue", s=1
#     )
#     axs[0].set_xlabel("Token In")
#     axs[0].set_ylabel("Token Out")

#     axs[1].scatter(
#         df1["amount1_adjusted"], -df1["amount0_adjusted"], c="indianred", s=1
#     )
#     axs[1].scatter(df1["amount1_adjusted"], -df1["predicted"], c="royalblue", s=1)
#     axs[1].set_xlabel("Token In")
#     axs[1].set_ylabel("Token Out")

#     f.legend(loc="upper center", bbox_to_anchor=(0.5, 0), ncol=2)
#     f.tight_layout()

#     axs[0].set_title("Predicted vs Actual Token Out")
#     axs[1].set_title("Predicted vs Actual Token Out")

#     if fn:
#         plt.savefig(fn, dpi=300)
#         plt.close()  # don't show

#     return f


def plot_price_impact_prediction_error(df, amt_col, fn=None):
    f, ax = plt.subplots()

    ax.scatter(
        df[amt_col],
        -df["price_impact"],
        s=df["pct_error"] * 100,
        c="royalblue",
    )
    ax.set_xlabel(amt_col)
    ax.set_ylabel("Price impact")
    ax.set_title("Predicted Price Impact Error")

    f.tight_layout()

    if fn:
        plt.savefig(fn, dpi=300)
        plt.close()  # don't show

    return f


def plot_prices(df, df2=None, fn=None):
    """
    Plot prices in df. Assumes that each col
    in the df is a coin.

    Parameters
    ----------
    df : pd.DataFrame
        Dataframe with prices
    fn : Optional[str]
        Filename to save plot to
    """
    # Get coin names
    cols = [col for col in df.columns if col != "timestamp"]

    n = len(cols)
    n, m = int(n**0.5), n // int(n**0.5) + (n % int(n**0.5) > 0)
    # now n, m are the dimensions of the grid

    f, axs = plt.subplots(n, m, figsize=(15, 15))

    for i in range(n):
        for j in range(m):
            col = cols.pop()
            ax = axs[i, j]
            ax.plot(df.index, df[col], lw=1, c="royalblue", label="Real")
            ax.set_title(f"{col} Prices")
            ax.set_ylabel("Price (USD)")
            ax.tick_params(axis="x", rotation=45)

            if df2 is not None:
                ax.plot(df.index, df2[col], lw=1, c="indianred", label="Simulated")
                ax.legend()

    f.tight_layout()

    if fn:
        plt.savefig(fn, dpi=300)
        plt.close()

    return f


def plot_jumps(df, recovery_threshold, fn=None):
    f, ax = plt.subplots(figsize=(8, 5))

    ax.plot(df["weighted_avg"], color="royalblue", lw=1, label="WAVG Price")
    ax.plot(
        df["recovery_mean"] * (1 + recovery_threshold),
        color="black",
        linestyle="--",
        lw=1,
        label="Recovery Threshold",
    )
    ax.plot(
        df["recovery_mean"] * (1 - recovery_threshold),
        color="black",
        linestyle="--",
        lw=1,
    )
    ax.set_ylabel("ETH Price (USD)")
    ax.set_title("Detecting ETH Jumps")

    for i, row in df.iterrows():
        left_edge = i - pd.to_timedelta(0.5, unit="D")
        right_edge = i + pd.to_timedelta(0.5, unit="D")
        if row["jump_state"]:
            ax.axvspan(left_edge, right_edge, color="indianred")

    ax.axvspan(None, None, color="indianred", label="Jump")
    ax.tick_params(axis="x", rotation=45)

    f.legend(loc="upper center", bbox_to_anchor=(0.5, 0), ncol=3)
    f.tight_layout()

    if fn:
        f.savefig(fn, bbox_inches="tight", dpi=300)
        plt.close()  # don't show

    return f


def plot_price_impact_1inch(res, in_token, out_token, fn: str = None):
    prices = [r.price for r in res]
    base_price = max(prices)  # price = out/in
    price_impact = [100 * (base_price - p) / base_price for p in prices]
    in_amounts = [r.in_amount / (10**r.in_decimals) for r in res]

    f, ax = plt.subplots(figsize=(10, 5))
    ax.scatter(in_amounts, price_impact, s=10, c="royalblue")
    ax.set_xscale("log")
    ax.set_title(f"Price Impact for Swapping {in_token} into {out_token}")
    ax.set_ylabel("Price Impact (%)")
    ax.set_xlabel(f"Trade Size ({in_token})")

    f.tight_layout()

    if fn:
        plt.savefig(fn, bbox_inches="tight", dpi=300)
        plt.close()  # don't show

    return f


def plot_price_impact(df, in_token, out_token, fn: str = None):
    f, ax = plt.subplots(figsize=(10, 5))
    ax.scatter(df["in_amount"], df["price_impact"], s=10, c="royalblue")
    ax.set_xscale("log")
    ax.set_title(f"Price Impact for Swapping {in_token} into {out_token}")
    ax.set_ylabel("Price Impact (%)")
    ax.set_xlabel(f"Trade Size ({in_token})")

    f.tight_layout()

    if fn:
        plt.savefig(fn, bbox_inches="tight", dpi=300)
        plt.close()  # don't show

    return f


S = 5


def plot_price_1inch(df, in_token, out_token, fn: str = None):
    dt = pd.to_datetime(df["hour"], unit="s")

    f, ax = plt.subplots(figsize=(10, 5))

    scatter = ax.scatter(df["in_amount"], df["price"], s=S, c=dt, cmap="viridis")

    ax.set_xscale("log")
    ax.set_title(f"Prices for Swapping {in_token} into {out_token}")
    ax.set_ylabel("Exchange Rate (out/in)")
    ax.set_xlabel(f"Trade Size ({in_token})")

    cbar = plt.colorbar(scatter, ax=ax)
    cbar.ax.set_yticklabels(dt.dt.strftime("%d %b %Y"))
    cbar.set_label("Date")

    f.tight_layout()

    if fn:
        plt.savefig(fn, bbox_inches="tight", dpi=300)
        plt.close()  # don't show

    return f


def plot_regression(
    df: pd.DataFrame,
    i: int,
    j: int,
    market: ExternalMarket,
    fn: str = None,
    scale: str = "log",
    xlim: float = None,
):
    in_token = market.coins[i]
    out_token = market.coins[j]

    dt = pd.to_datetime(df["timestamp"], unit="s")
    X = np.geomspace(df["in_amount"].min(), df["in_amount"].max(), 100)
    y = market.price_impact(i, j, X) * 100

    f, ax = plt.subplots(figsize=(10, 5))
    scatter = ax.scatter(
        df["in_amount"] / 10**in_token.decimals,
        df["price_impact"] * 100,
        c=dt,
        s=S,
        label="1inch Quotes",
    )
    ax.plot(X / 10**in_token.decimals, y, label="Prediction", c="indianred", lw=1)
    ax.set_xscale(scale)
    ax.legend()
    ax.set_xlabel(f"Amount in ({in_token.symbol})")
    ax.set_ylabel("Price Impact %")
    ax.set_title(f"{in_token.symbol} -> {out_token.symbol} Price Impact")

    cbar = plt.colorbar(scatter, ax=ax)
    cbar.ax.set_yticklabels(dt.dt.strftime("%d %b %Y"))
    cbar.set_label("Date")

    if xlim:
        ax.set_xlim(0, xlim)
        ax.set_ylim(0, df[df["in_amount"] < xlim]["price_impact"].max() * 100)

    f.tight_layout()

    if fn:
        plt.savefig(fn, bbox_inches="tight", dpi=300)
        plt.close()  # don't show

    return f


def plot_predictions(
    df: pd.DataFrame,
    i: int,
    j: int,
    market: ExternalMarket,
    fn: str = None,
    scale: str = "log",
    xlim: float = None,
):
    in_token = market.coins[i]
    out_token = market.coins[j]

    dt = pd.to_datetime(df["timestamp"], unit="s")

    X = np.geomspace(df["in_amount"].min(), df["in_amount"].max(), 100)
    y = np.array([market.trade(i, j, x) for x in X])

    f, ax = plt.subplots(figsize=(10, 5))
    scatter = ax.scatter(
        df["in_amount"] / 10**in_token.decimals,
        df["out_amount"] / 10**out_token.decimals,
        c=dt,
        s=S,
        label="1inch Quotes",
    )
    ax.plot(
        X / 10**in_token.decimals,
        y / 10**out_token.decimals,
        label="Prediction",
        c="indianred",
        lw=1,
    )
    ax.set_xscale(scale)
    ax.legend()
    ax.set_xlabel(f"Amount in ({in_token.symbol})")
    ax.set_ylabel(f"Amount out ({out_token.symbol})")
    ax.set_title(f"{in_token.symbol} -> {out_token.symbol} Quotes")

    cbar = plt.colorbar(scatter, ax=ax)
    cbar.ax.set_yticklabels(dt.dt.strftime("%d %b %Y"))
    cbar.set_label("Date")

    if xlim:
        ax.set_xlim(0, xlim)
        ax.set_ylim(0, df[df["in_amount"] < xlim]["out_amount"].max())

    f.tight_layout()

    if fn:
        plt.savefig(fn, bbox_inches="tight", dpi=300)
        plt.close()  # don't show

    return f
