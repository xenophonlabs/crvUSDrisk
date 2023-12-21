"""Utitlities for plotting."""

from typing import Tuple
import matplotlib.pyplot as plt


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
