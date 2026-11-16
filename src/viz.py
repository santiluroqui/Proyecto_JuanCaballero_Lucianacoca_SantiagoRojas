"""Estilo y utilidades de gráficos para que todo el repo se vea consistente."""
from __future__ import annotations

import matplotlib.pyplot as plt
import seaborn as sns

from . import config as cfg

# Paleta categórica fija (apta para daltonismo) reutilizable por todo el equipo
PALETTE = ["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B3",
           "#937860", "#DA8BC3", "#8C8C8C", "#CCB974", "#64B5CD"]


def set_style():
    sns.set_theme(style="whitegrid", context="notebook", palette=PALETTE)
    plt.rcParams.update({
        "figure.dpi": 110,
        "savefig.dpi": 200,
        "savefig.bbox": "tight",
        "axes.titleweight": "bold",
        "axes.titlesize": 13,
        "axes.spines.top": False,
        "axes.spines.right": False,
    })


def save_fig(fig, name: str):
    """Guarda en reports/figures/<name>.png y devuelve la ruta."""
    cfg.FIGURES.mkdir(parents=True, exist_ok=True)
    path = cfg.FIGURES / f"{name}.png"
    fig.savefig(path)
    return path
