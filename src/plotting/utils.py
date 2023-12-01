"""Utitlities for plotting."""
import matplotlib.pyplot as plt


def save(fig: plt.Figure, fn: str | None) -> plt.Figure:
    """Save plot."""
    fig.tight_layout()
    if fn:
        plt.savefig(fn, bbox_inches="tight", dpi=300)
        plt.close()
    return fig
