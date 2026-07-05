"""
analysis
========

Analyse de la structure de la courbe de prix :

    - **Tendance** (haussière / baissière / latérale) déterminée par la position
      relative des EMA et la pente de l'EMA50 ;
    - **Supports / résistances** détectés par extrema locaux (pivots) puis
      regroupés en niveaux (clustering simple) ;
    - **Patterns récurrents** : breakouts, reversals, consolidations, identifiés
      à partir des Bollinger, de l'ATR et des cassures de niveaux.

Ces éléments servent à définir le « régime » de marché courant, base du calcul
de probabilités « en conditions similaires ».
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List

import numpy as np
import pandas as pd


@dataclass
class MarketStructure:
    """Résumé de la structure de marché à la date d'analyse."""

    trend: str                       # 'haussière' | 'baissière' | 'latérale'
    trend_strength: float            # 0..1
    volatility_regime: str           # 'faible' | 'normale' | 'élevée'
    supports: List[float] = field(default_factory=list)
    resistances: List[float] = field(default_factory=list)
    patterns: List[str] = field(default_factory=list)
    rsi_state: str = "neutre"        # 'suracheté' | 'survendu' | 'neutre'


def detect_trend(df: pd.DataFrame) -> tuple[str, float]:
    """
    Détermine la tendance courante et sa force (0..1).

    Règles :
        - haussière  : Close > EMA50 > EMA200 et pente EMA50 positive ;
        - baissière  : Close < EMA50 < EMA200 et pente EMA50 négative ;
        - latérale   : sinon.
    La force combine l'écartement des EMA et la pente normalisée.
    """
    last = df.iloc[-1]
    close = last["Close"]
    ema50, ema200 = last["ema50"], last["ema200"]

    # Pente de l'EMA50 sur ~20 séances, normalisée par le prix.
    ema50_series = df["ema50"].dropna()
    if len(ema50_series) >= 21:
        slope = (ema50_series.iloc[-1] - ema50_series.iloc[-21]) / 20.0
        slope_norm = slope / close
    else:
        slope_norm = 0.0

    spread = abs(ema50 - ema200) / close if close else 0.0
    strength = float(np.clip(spread * 8 + abs(slope_norm) * 200, 0.0, 1.0))

    if close > ema50 > ema200 and slope_norm > 0:
        return "haussière", strength
    if close < ema50 < ema200 and slope_norm < 0:
        return "baissière", strength
    # Cas mixtes : on tranche par la position vs EMA200 mais avec force réduite.
    if close > ema200 and slope_norm >= 0:
        return "haussière", strength * 0.5
    if close < ema200 and slope_norm <= 0:
        return "baissière", strength * 0.5
    return "latérale", strength * 0.4


def volatility_regime(df: pd.DataFrame) -> str:
    """Classe le régime de volatilité courant vs son historique (terciles ATR%)."""
    atr_pct = df["atr_pct"].dropna()
    if atr_pct.empty:
        return "normale"
    current = atr_pct.iloc[-1]
    low, high = atr_pct.quantile(0.33), atr_pct.quantile(0.66)
    if current <= low:
        return "faible"
    if current >= high:
        return "élevée"
    return "normale"


def find_pivots(df: pd.DataFrame, order: int = 5) -> tuple[list[int], list[int]]:
    """
    Détecte les pivots hauts et bas.

    Un pivot haut à l'indice i : High[i] est le max strict de la fenêtre
    [i-order, i+order]. Idem symétriquement pour les pivots bas sur Low.
    """
    highs = df["High"].values
    lows = df["Low"].values
    n = len(df)
    pivot_high, pivot_low = [], []
    for i in range(order, n - order):
        window_h = highs[i - order : i + order + 1]
        window_l = lows[i - order : i + order + 1]
        if highs[i] == window_h.max() and (window_h.argmax() == order):
            pivot_high.append(i)
        if lows[i] == window_l.min() and (window_l.argmin() == order):
            pivot_low.append(i)
    return pivot_high, pivot_low


def cluster_levels(prices: list[float], tol: float = 0.02) -> list[float]:
    """
    Regroupe des niveaux proches (à ``tol`` relatif près) en niveaux consolidés.

    Renvoie la moyenne de chaque cluster, trié.
    """
    if not prices:
        return []
    prices = sorted(prices)
    clusters: list[list[float]] = [[prices[0]]]
    for p in prices[1:]:
        if abs(p - clusters[-1][-1]) / clusters[-1][-1] <= tol:
            clusters[-1].append(p)
        else:
            clusters.append([p])
    return [float(np.mean(c)) for c in clusters]


def support_resistance(
    df: pd.DataFrame, order: int = 5, max_levels: int = 4
) -> tuple[list[float], list[float]]:
    """
    Calcule les niveaux de support (sous le prix) et résistance (au-dessus).

    On ne considère que les pivots des ~2 dernières années pour rester pertinent.
    """
    recent = df.iloc[-504:] if len(df) > 504 else df
    ph, pl = find_pivots(recent, order=order)
    highs = [float(recent["High"].iloc[i]) for i in ph]
    lows = [float(recent["Low"].iloc[i]) for i in pl]

    levels = cluster_levels(highs + lows, tol=0.02)
    price = float(df["Close"].iloc[-1])

    supports = sorted([l for l in levels if l < price], reverse=True)[:max_levels]
    resistances = sorted([l for l in levels if l > price])[:max_levels]
    return supports, resistances


def detect_patterns(df: pd.DataFrame) -> list[str]:
    """
    Identifie des patterns récents simples sur les ~30 dernières séances.

    Patterns détectés :
        - « Breakout haussier » / « Breakdown baissier » (cassure de bande) ;
        - « Squeeze de volatilité » (bande de Bollinger resserrée) ;
        - « Reversal potentiel » (divergence RSI / croisement MACD) ;
        - « Consolidation » (faible amplitude, ATR bas).
    """
    patterns: list[str] = []
    tail = df.iloc[-30:]
    last = df.iloc[-1]

    # Squeeze de volatilité : largeur de bande dans le décile bas de l'historique.
    bw = df["bb_width"].dropna()
    if not bw.empty and last["bb_width"] <= bw.quantile(0.15):
        patterns.append("Squeeze de volatilité (compression)")

    # Breakout / breakdown : clôture au-delà des bandes de Bollinger.
    if last["Close"] > last["bb_upper"]:
        patterns.append("Breakout haussier (au-dessus bande sup.)")
    elif last["Close"] < last["bb_lower"]:
        patterns.append("Breakdown baissier (sous bande inf.)")

    # Croisement MACD récent.
    macd_hist = df["macd_hist"].dropna()
    if len(macd_hist) >= 2:
        if macd_hist.iloc[-2] < 0 <= macd_hist.iloc[-1]:
            patterns.append("Croisement MACD haussier")
        elif macd_hist.iloc[-2] > 0 >= macd_hist.iloc[-1]:
            patterns.append("Croisement MACD baissier")

    # Consolidation : faible amplitude relative sur la fenêtre.
    rng = (tail["High"].max() - tail["Low"].min()) / last["Close"]
    if rng < 0.06:
        patterns.append("Consolidation (range serré)")

    return patterns


def rsi_state(df: pd.DataFrame) -> str:
    """Qualifie l'état du RSI courant."""
    r = df["rsi"].iloc[-1]
    if pd.isna(r):
        return "neutre"
    if r >= 70:
        return "suracheté"
    if r <= 30:
        return "survendu"
    return "neutre"


def analyze_structure(df: pd.DataFrame) -> MarketStructure:
    """Point d'entrée : produit un ``MarketStructure`` complet."""
    trend, strength = detect_trend(df)
    supports, resistances = support_resistance(df)
    return MarketStructure(
        trend=trend,
        trend_strength=strength,
        volatility_regime=volatility_regime(df),
        supports=supports,
        resistances=resistances,
        patterns=detect_patterns(df),
        rsi_state=rsi_state(df),
    )
