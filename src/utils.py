import numpy as np
import matplotlib.pyplot as plt
import imageio
from PIL import Image


def gen_gbm(S0, mu, sigma, dt, T):
    W = np.random.normal(loc=0, scale=np.sqrt(dt), size=int(T / dt))
    S = S0 * np.exp(np.cumsum((mu - 0.5 * sigma**2) * dt + sigma * W))
    return S

def slippage(size, sigma):
    # TODO understand what the actual trades would be
    # for the arbitrage. Would they
    # FIXME Ignoring volatility for now
    m = 1.081593506690093e-06
    b = 0.0004379110082802476
    return m * size + b

def get_crvUSD_index(pool):
    """
    Return index of crvUSD in pool.
    """
    return pool.metadata["coins"]["names"].index("crvUSD")

def external_swap(x, y, swap, fee, y_in):
    """
    @notice account for slippage when trading against open market
    Assuming that we trade against Uniswap is a conservative assumption
    since not accounting for CEX.
    However, we are not accounting for crvUSD liquidity in Curve pools yet.
    # Fit this curve
    # f(x | sigma) = c * sigma * x <- Gauntlet found this empirically in Compound report
    # f(x | sigma) = c * sigma * x**0.5 <- TradFi empirical finding
    # f(x | sigma) = c * sigma * x**2 <- Perhaps more like a simple CFMM
    """
    # TODO: Need to incorporate concentrated liquidity (e.g., no more liquidity beyond ]a, b[)
    # TODO: Need to incorproate the StableSwap invariant for crvUSD pool liquidity
    k = x * y

    if y_in:
        new_y = y + swap
        new_x = k / new_y
        out = (x - new_x) * (1 - fee)
    else:
        new_x = x + swap
        new_y = k / new_x
        out = (y - new_y) * (1 - fee)

    return out


# === PLOTTING === #

FPS = 3

fn_frames_reserves = "./figs/frames/test_reserves_{}.png"
fn_gif_reserves = "./figs/gifs/test_reserves.gif"

fn_frames_actions = "./figs/frames/test_actions_{}.png"
fn_gif_actions = "./figs/gifs/test_actions.gif"

fn_frames_bad_debt = "./figs/frames/test_bad_debt_{}.png"
fn_gif_bad_debt = "./figs/gifs/test_bad_debt.gif"

fn_frames_system_health = "./figs/frames/test_system_health_{}.png"
fn_gif_system_health = "./figs/gifs/test_system_health.gif"

plt.rcParams["font.family"] = "serif"
plt.rcParams.update({"font.size": 10})
plt.rcParams["axes.spines.top"] = False
plt.rcParams["axes.spines.right"] = False

plt.rcParams["axes.facecolor"] = "#f5f5f5"
plt.rcParams["grid.color"] = "white"
plt.rcParams["grid.linestyle"] = "-"
plt.rcParams["grid.linewidth"] = 2


def plot_combined(df):
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
    min_price, max_price = df["spot"].min() * (1 - scale), df["spot"].max() * (
        1 + scale
    )
    min_pnl, max_pnl = 0, max(
        df["liquidation_pnl"].max(), df["arbitrage_pnl"].max()
    ) * (1 + scale)
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

    imageio.mimsave(
        fn_gif_actions,
        [imageio.v2.imread(fn_frames_actions.format(i)) for i in range(len(df))],
        fps=FPS,
    )
    imageio.mimsave(
        fn_gif_bad_debt,
        [imageio.v2.imread(fn_frames_bad_debt.format(i)) for i in range(len(df))],
        fps=FPS,
    )
    imageio.mimsave(
        fn_gif_system_health,
        [imageio.v2.imread(fn_frames_system_health.format(i)) for i in range(len(df))],
        fps=FPS,
    )


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
