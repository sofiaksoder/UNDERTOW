"""Shared matplotlib style: the same palette as the web demo."""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

BG, INK, MUTED = "#04070b", "#e4ece9", "#7f9590"
ACCENT, DEEP, CORAL = "#9ed0c4", "#4682aa", "#e8735a"
ROLL, GARCH, EWMA = "#8fb8d6", "#d9c9a6", "#b49fd6"


def apply() -> None:
    plt.rcParams.update({
        "figure.facecolor": BG, "axes.facecolor": BG, "savefig.facecolor": BG,
        "axes.edgecolor": "#2f4450", "axes.labelcolor": MUTED, "text.color": INK,
        "xtick.color": MUTED, "ytick.color": MUTED, "grid.color": "#1c2a33",
        "axes.grid": True, "grid.linewidth": 0.6, "font.size": 9,
        "axes.spines.top": False, "axes.spines.right": False,
        "legend.frameon": False, "legend.labelcolor": INK,
    })
