import numpy as np
import matplotlib.pyplot as plt
import imageio
from PIL import Image
from .utils import get_crvUSD_index

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

plt.rcParams["axes.facecolor"] = "#f5f5f5"
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

    f, ax = plt.subplots()

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


def plot_combined():
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
        new_img = Image.new("RGB", (img1.width + img2.width, img1.height + img3.height))
        new_img.paste(img1, (0, 0))
        new_img.paste(img2, (img1.width, 0))
        new_img.paste(img3, (0, img1.height))
        new_img.paste(img4, (img1.width, img1.height))

        # Convert back to array and append to new frames
        new_frames.append(np.array(new_img))

    # Create new GIF
    imageio.mimsave("./figs/gifs/combined.gif", new_frames, fps=FPS)


def plot_actions(df, i, min_time, max_time, min_price, max_price, min_pnl, max_pnl, fn):
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

    ax2.scatter(
        ndf.index,
        ndf["liquidation_pnl"],
        label="Liquidation PnL",
        color="indianred",
        s=20,
    )
    ax2.scatter(
        ndf.index, ndf["arbitrage_pnl"], label="Arbitrage PnL", color="royalblue", s=20
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
    make_gif(fn_gif_system_health, fn_frames_system_health, n)


def _plot_reserves(llamma, fn=None):
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

    f.legend(loc="upper center", bbox_to_anchor=(0.5, 0), ncol=4)
    f.tight_layout()

    if fn:
        plt.savefig(fn, bbox_inches="tight", dpi=300)
        plt.close()  # don't show

    return f


def _plot_borrowers(borrowers, price):
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
