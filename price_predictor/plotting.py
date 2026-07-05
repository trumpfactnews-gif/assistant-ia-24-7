"""
plotting
========

Génération du graphique matplotlib de synthèse comportant :
    - la courbe de prix sur 5 ans (+ EMA 50/200) ;
    - les zones d'événements marquants annotées ;
    - les niveaux de support / résistance actuels ;
    - une barre indicatrice de la probabilité directionnelle par horizon.
"""

from __future__ import annotations

from typing import List

import matplotlib

# Backend non interactif : permet de sauvegarder même sans écran (headless).
matplotlib.use("Agg")
import matplotlib.dates as mdates  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

from .analysis import MarketStructure  # noqa: E402
from .events import Event  # noqa: E402
from .probability import HorizonResult  # noqa: E402


# Couleurs par type d'événement.
_EVENT_COLORS = {
    "earnings": "#8e44ad",
    "dividend": "#16a085",
    "split": "#f39c12",
    "fomc": "#c0392b",
    "cpi": "#2980b9",
    "nfp": "#27ae60",
}


def plot_analysis(
    ticker: str,
    df: pd.DataFrame,
    structure: MarketStructure,
    events: List[Event],
    results: List[HorizonResult],
    out_path: str = "analysis.png",
) -> str:
    """
    Construit et sauvegarde la figure d'analyse.

    Returns
    -------
    str
        Le chemin du fichier PNG généré.
    """
    fig = plt.figure(figsize=(15, 9))
    gs = fig.add_gridspec(2, 1, height_ratios=[3, 1], hspace=0.28)
    ax = fig.add_subplot(gs[0])
    ax_prob = fig.add_subplot(gs[1])

    # --- Courbe de prix + moyennes mobiles -------------------------------
    ax.plot(df.index, df["Close"], color="#2c3e50", lw=1.2, label="Clôture")
    if "ema50" in df:
        ax.plot(df.index, df["ema50"], color="#2980b9", lw=1.0, alpha=0.8, label="EMA 50")
    if "ema200" in df:
        ax.plot(df.index, df["ema200"], color="#e67e22", lw=1.0, alpha=0.8, label="EMA 200")

    # --- Niveaux de support / résistance ---------------------------------
    for s in structure.supports:
        ax.axhline(s, color="#27ae60", ls="--", lw=0.8, alpha=0.6)
        ax.text(df.index[0], s, f" S {s:.2f}", color="#27ae60", va="center", fontsize=8)
    for r in structure.resistances:
        ax.axhline(r, color="#c0392b", ls="--", lw=0.8, alpha=0.6)
        ax.text(df.index[0], r, f" R {r:.2f}", color="#c0392b", va="center", fontsize=8)

    # --- Zones d'événements marquants ------------------------------------
    # Pour éviter la surcharge, on n'annote que les événements « titre »
    # (earnings, split, dividende) et on ombre légèrement les macro.
    seen_labels = set()
    for ev in events:
        if ev.date not in df.index:
            continue
        color = _EVENT_COLORS.get(ev.kind, "#7f8c8d")
        if ev.kind in ("earnings", "split"):
            y = float(df.loc[ev.date, "Close"])
            ax.scatter([ev.date], [y], color=color, s=28, zorder=5,
                       edgecolors="white", linewidths=0.5)
            lbl = ev.kind if ev.kind not in seen_labels else None
            if lbl:
                ax.scatter([], [], color=color, s=28, label=f"Événement: {ev.kind}")
                seen_labels.add(ev.kind)
        elif ev.kind in ("fomc",):
            ax.axvline(ev.date, color=color, lw=0.4, alpha=0.15)

    ax.set_title(f"{ticker} — Analyse probabiliste des prix (5 ans)", fontsize=13, weight="bold")
    ax.set_ylabel("Prix")
    ax.legend(loc="upper left", fontsize=8, ncol=2)
    ax.grid(True, alpha=0.2)
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

    # --- Barre des probabilités directionnelles par horizon --------------
    horizons = [r.horizon for r in results]
    ups = [r.prob_up for r in results]
    downs = [r.prob_down for r in results]
    x = range(len(horizons))

    ax_prob.bar(x, ups, color="#27ae60", label="Hausse")
    ax_prob.bar(x, downs, bottom=ups, color="#c0392b", label="Baisse")
    ax_prob.axhline(50, color="black", ls=":", lw=0.8, alpha=0.6)

    for i, r in enumerate(results):
        ax_prob.text(i, r.prob_up / 2, f"{r.prob_up:.0f}%", ha="center",
                     va="center", color="white", fontsize=8, weight="bold")
        ax_prob.text(i, r.prob_up + r.prob_down / 2, f"{r.prob_down:.0f}%",
                     ha="center", va="center", color="white", fontsize=8, weight="bold")

    ax_prob.set_xticks(list(x))
    ax_prob.set_xticklabels(horizons)
    ax_prob.set_ylim(0, 100)
    ax_prob.set_ylabel("Probabilité (%)")
    ax_prob.set_title("Probabilité directionnelle par horizon", fontsize=11)
    ax_prob.legend(loc="upper right", fontsize=8)

    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return out_path
