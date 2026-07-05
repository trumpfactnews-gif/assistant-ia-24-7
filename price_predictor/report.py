"""
report
======

Formatage du tableau de résultats textuel (console).
"""

from __future__ import annotations

from typing import List

import pandas as pd

from .analysis import MarketStructure
from .events import Event
from .probability import HorizonResult, overall_confidence


def _fmt_pct(v: float) -> str:
    return f"{v:.0f} %"


def build_key_factors(
    structure: MarketStructure, recent_events: List[Event]
) -> List[str]:
    """
    Construit la liste des facteurs clés (signaux techniques + événements).
    """
    factors: List[str] = []

    factors.append(
        f"Tendance {structure.trend} (force {structure.trend_strength:.0%})"
    )
    factors.append(f"Régime de volatilité {structure.volatility_regime}")
    if structure.rsi_state != "neutre":
        factors.append(f"RSI {structure.rsi_state}")
    for p in structure.patterns:
        factors.append(f"Pattern : {p}")
    if structure.supports:
        factors.append(
            "Support proche : " + ", ".join(f"{s:.2f}" for s in structure.supports[:2])
        )
    if structure.resistances:
        factors.append(
            "Résistance proche : "
            + ", ".join(f"{r:.2f}" for r in structure.resistances[:2])
        )
    for ev in recent_events[:4]:
        factors.append(
            f"Événement {ev.date.date()} : {ev.label}"
        )
    return factors


def format_report(
    ticker: str,
    name: str,
    analysis_date: pd.Timestamp,
    last_price: float,
    results: List[HorizonResult],
    structure: MarketStructure,
    recent_events: List[Event],
) -> str:
    """
    Produit le rapport texte complet, prêt à être imprimé en console.
    """
    lines: List[str] = []
    header = (
        f"Ticker : {ticker} ({name})  |  "
        f"Date d'analyse : {analysis_date.date()}  |  "
        f"Prix actuel : {last_price:.2f}"
    )
    lines.append("=" * len(header))
    lines.append(header)
    lines.append("=" * len(header))
    lines.append("")

    # Tableau des probabilités.
    lines.append(
        f"{'Horizon':<11}| {'Proba HAUSSE':^18}| "
        f"{'Proba BAISSE':^18}| {'Signal dominant':<16}"
    )
    lines.append(f"{'-'*11}|{'-'*19}|{'-'*19}|{'-'*17}")
    for r in results:
        lines.append(
            f"{r.horizon:<11}| {_fmt_pct(r.prob_up):^18}| "
            f"{_fmt_pct(r.prob_down):^18}| {r.signal:<16}"
        )
    lines.append("")

    # Facteurs clés.
    factors = build_key_factors(structure, recent_events)
    lines.append("Facteurs clés détectés :")
    for f in factors:
        lines.append(f"  • {f}")
    lines.append("")

    # Confiance globale.
    conf = overall_confidence(results)
    lines.append(
        f"Niveau de confiance du modèle : {conf:.0f} % "
        f"(précision historique moyenne du modèle en validation croisée)"
    )
    lines.append("")
    lines.append(
        "⚠️  Avertissement : cet outil est fourni à titre informatif et éducatif. "
        "Il ne constitue pas un conseil en investissement. Les performances "
        "passées ne préjugent pas des performances futures."
    )
    return "\n".join(lines)
