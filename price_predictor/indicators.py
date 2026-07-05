"""
indicators
==========

Calcul des indicateurs techniques utilisés comme facteurs de pondération :
    - RSI (Relative Strength Index)
    - MACD (ligne, signal, histogramme)
    - Bandes de Bollinger (+ %B et largeur de bande)
    - Moyennes mobiles exponentielles EMA 20 / 50 / 200
    - ATR (Average True Range) — mesure de volatilité
    - Statistiques de volume (moyenne mobile, ratio)
    - Rendements et volatilité réalisée

Toutes les fonctions renvoient des ``pandas.Series`` alignées sur l'index du prix.
Implémentation « maison » (numpy/pandas) pour éviter toute dépendance lourde.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def ema(series: pd.Series, span: int) -> pd.Series:
    """Moyenne mobile exponentielle."""
    return series.ewm(span=span, adjust=False).mean()


def sma(series: pd.Series, window: int) -> pd.Series:
    """Moyenne mobile simple."""
    return series.rolling(window=window, min_periods=max(2, window // 2)).mean()


def rsi(close: pd.Series, period: int = 14) -> pd.Series:
    """
    RSI de Wilder.

    Valeurs > 70 : suracheté ; < 30 : survendu.
    """
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)
    # Lissage exponentiel « à la Wilder » (alpha = 1/period).
    avg_gain = gain.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    out = 100 - (100 / (1 + rs))
    # Si avg_loss == 0 (aucune baisse), le RSI vaut 100.
    out = out.where(avg_loss != 0, 100.0)
    return out


def macd(
    close: pd.Series,
    fast: int = 12,
    slow: int = 26,
    signal: int = 9,
) -> pd.DataFrame:
    """
    MACD classique.

    Returns un DataFrame avec les colonnes ``macd``, ``signal`` et ``hist``.
    """
    macd_line = ema(close, fast) - ema(close, slow)
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    hist = macd_line - signal_line
    return pd.DataFrame(
        {"macd": macd_line, "signal": signal_line, "hist": hist}
    )


def bollinger(
    close: pd.Series,
    window: int = 20,
    n_std: float = 2.0,
) -> pd.DataFrame:
    """
    Bandes de Bollinger.

    Renvoie ``mid`` (SMA), ``upper``, ``lower``, ``pct_b`` (position relative dans
    la bande, 0 = bande basse, 1 = bande haute) et ``bandwidth`` (largeur relative).
    """
    mid = sma(close, window)
    std = close.rolling(window=window, min_periods=max(2, window // 2)).std()
    upper = mid + n_std * std
    lower = mid - n_std * std
    width = (upper - lower)
    pct_b = (close - lower) / width.replace(0, np.nan)
    bandwidth = width / mid.replace(0, np.nan)
    return pd.DataFrame(
        {
            "mid": mid,
            "upper": upper,
            "lower": lower,
            "pct_b": pct_b,
            "bandwidth": bandwidth,
        }
    )


def atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """
    Average True Range : mesure de volatilité basée sur les mèches.
    Nécessite les colonnes High, Low, Close.
    """
    high, low, close = df["High"], df["Low"], df["Close"]
    prev_close = close.shift(1)
    tr = pd.concat(
        [
            (high - low),
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    return tr.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()


def realized_volatility(close: pd.Series, window: int = 20) -> pd.Series:
    """Volatilité réalisée annualisée (écart-type des rendements log)."""
    log_ret = np.log(close / close.shift(1))
    return log_ret.rolling(window=window, min_periods=max(2, window // 2)).std() * np.sqrt(252)


def compute_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calcule l'ensemble des indicateurs et les concatène au DataFrame OHLCV.

    Parameters
    ----------
    df : pd.DataFrame
        Données OHLCV nettoyées.

    Returns
    -------
    pd.DataFrame
        Copie enrichie de toutes les colonnes d'indicateurs.
    """
    out = df.copy()
    close = out["Close"]

    # Rendements journaliers (simple + log).
    out["ret"] = close.pct_change()
    out["log_ret"] = np.log(close / close.shift(1))

    # Moyennes mobiles.
    out["ema20"] = ema(close, 20)
    out["ema50"] = ema(close, 50)
    out["ema200"] = ema(close, 200)

    # Oscillateurs.
    out["rsi"] = rsi(close, 14)
    macd_df = macd(close)
    out["macd"] = macd_df["macd"]
    out["macd_signal"] = macd_df["signal"]
    out["macd_hist"] = macd_df["hist"]

    # Bollinger.
    bb = bollinger(close, 20, 2.0)
    out["bb_mid"] = bb["mid"]
    out["bb_upper"] = bb["upper"]
    out["bb_lower"] = bb["lower"]
    out["bb_pct"] = bb["pct_b"]
    out["bb_width"] = bb["bandwidth"]

    # Volatilité.
    out["atr"] = atr(out, 14)
    out["atr_pct"] = out["atr"] / close  # ATR normalisé par le prix
    out["volatility"] = realized_volatility(close, 20)

    # Volume.
    if "Volume" in out.columns:
        out["vol_sma20"] = sma(out["Volume"], 20)
        out["vol_ratio"] = out["Volume"] / out["vol_sma20"].replace(0, np.nan)
    else:
        out["vol_sma20"] = np.nan
        out["vol_ratio"] = np.nan

    # Momentum sur différentes fenêtres (utile en features ML).
    for w in (5, 10, 20):
        out[f"mom_{w}"] = close.pct_change(w)

    # Distance relative aux moyennes mobiles (features de tendance).
    out["dist_ema20"] = (close - out["ema20"]) / out["ema20"]
    out["dist_ema50"] = (close - out["ema50"]) / out["ema50"]
    out["dist_ema200"] = (close - out["ema200"]) / out["ema200"]

    return out
