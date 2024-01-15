"""Utitlities for plotting."""

from typing import Tuple, List
import matplotlib.pyplot as plt
from matplotlib.axes import Axes


def save(fig: plt.Figure, fn: str | None) -> plt.Figure:
    """Save plot."""
    fig.tight_layout()
    if fn:
        plt.savefig(fn, bbox_inches="tight", dpi=300)
        plt.close()
    return fig


def make_square(n: int) -> Tuple[int, int]:
    """
    Make square subplots figure.
    """
    return int(n**0.5), n // int(n**0.5) + (n % int(n**0.5) > 0)


def remove_spines_and_ticks(
    ax: Axes,
    skip: List[str] | None = None,
    keep_x_ticks: bool = False,
    keep_y_ticks: bool = False,
) -> Axes:
    """
    Remove the spines and ticks from an axis.
    """
    # Remove spines and ticks from ax_histx
    for spine in ax.spines.values():
        if skip and spine.spine_type in skip:
            continue
        spine.set_visible(False)
    if not keep_x_ticks:
        ax.set_xticks([])
    if not keep_y_ticks:
        ax.set_yticks([])
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    return ax
