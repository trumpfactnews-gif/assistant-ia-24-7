"""
data_loader
===========

Récupération et nettoyage des données de marché OHLCV via yfinance.

Le module gère :
    - le téléchargement des 5 dernières années de données quotidiennes ;
    - la gestion des données manquantes et des gaps de marché ;
    - la récupération des méta-données de l'entreprise (secteur, nom) ;
    - la récupération des dividendes, splits et dates d'earnings.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Optional

import numpy as np
import pandas as pd

try:
    import yfinance as yf
except ImportError:  # pragma: no cover - dépendance optionnelle au moment de l'import
    yf = None


# Nombre de jours calendaires correspondant à ~5 ans de données.
FIVE_YEARS_DAYS = 365 * 5 + 2


@dataclass
class MarketData:
    """Conteneur pour l'ensemble des données récupérées sur un ticker."""

    ticker: str
    ohlcv: pd.DataFrame                      # colonnes: Open, High, Low, Close, Volume
    info: dict = field(default_factory=dict)
    dividends: pd.Series = field(default_factory=lambda: pd.Series(dtype=float))
    splits: pd.Series = field(default_factory=lambda: pd.Series(dtype=float))
    earnings_dates: pd.DatetimeIndex = field(
        default_factory=lambda: pd.DatetimeIndex([])
    )

    @property
    def last_price(self) -> float:
        """Dernier prix de clôture disponible."""
        return float(self.ohlcv["Close"].iloc[-1])

    @property
    def last_date(self) -> pd.Timestamp:
        """Dernière date disponible dans la série."""
        return self.ohlcv.index[-1]

    @property
    def name(self) -> str:
        """Nom lisible du sous-jacent (ou le ticker à défaut)."""
        return self.info.get("shortName") or self.info.get("longName") or self.ticker


def _clean_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
    """
    Nettoie une trame OHLCV brute.

    - normalise les noms de colonnes ;
    - supprime les lignes entièrement vides ;
    - comble les gaps de marché sur les prix par forward-fill (les jours fériés/
      week-ends ne sont de toute façon pas présents en données quotidiennes) ;
    - remet le volume manquant à 0 ;
    - supprime les doublons d'index et trie chronologiquement.
    """
    if df is None or df.empty:
        raise ValueError("Aucune donnée reçue pour ce ticker.")

    # yfinance peut renvoyer un MultiIndex de colonnes (Field, Ticker) ;
    # on aplatit pour ne garder que le champ.
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    # On garde uniquement les colonnes utiles quand elles existent.
    keep = [c for c in ["Open", "High", "Low", "Close", "Volume"] if c in df.columns]
    df = df[keep].copy()

    # Index temporel propre, trié, sans doublon, sans fuseau horaire.
    df.index = pd.to_datetime(df.index)
    if df.index.tz is not None:
        df.index = df.index.tz_localize(None)
    df = df[~df.index.duplicated(keep="last")].sort_index()

    # Suppression des lignes totalement vides puis comblement des trous de prix.
    df = df.dropna(how="all")
    price_cols = [c for c in ["Open", "High", "Low", "Close"] if c in df.columns]
    df[price_cols] = df[price_cols].ffill()
    if "Volume" in df.columns:
        df["Volume"] = df["Volume"].fillna(0)

    # Après le forward-fill, il peut rester des NaN en tête de série : on les retire.
    df = df.dropna(subset=price_cols)

    # Filtre de sécurité : on écarte les prix nuls ou négatifs (données corrompues).
    df = df[(df[price_cols] > 0).all(axis=1)]

    if df.empty:
        raise ValueError("Données OHLCV vides après nettoyage.")
    return df


def load_market_data(ticker: str, years: int = 5) -> MarketData:
    """
    Télécharge et prépare l'ensemble des données pour un ticker.

    Parameters
    ----------
    ticker : str
        Symbole reconnu par yfinance (action, ETF, crypto type ``BTC-USD``...).
    years : int
        Profondeur d'historique en années (5 par défaut).

    Returns
    -------
    MarketData
        Conteneur avec OHLCV nettoyé, méta-données et événements sociétés.
    """
    if yf is None:
        raise ImportError(
            "yfinance n'est pas installé. Installez les dépendances avec "
            "`pip install -r requirements.txt`."
        )

    ticker = ticker.strip().upper()
    end = datetime.utcnow()
    start = end - timedelta(days=int(365 * years) + 5)

    tk = yf.Ticker(ticker)

    # Téléchargement des prix quotidiens ajustés (splits/dividendes pris en compte).
    raw = tk.history(start=start, end=end, interval="1d", auto_adjust=True)
    ohlcv = _clean_ohlcv(raw)

    # Méta-données (peut échouer selon la connectivité : on reste tolérant).
    info: dict = {}
    try:
        info = tk.get_info() if hasattr(tk, "get_info") else tk.info
    except Exception:
        info = {}

    # Dividendes et splits (séries indexées par date).
    dividends = pd.Series(dtype=float)
    splits = pd.Series(dtype=float)
    try:
        dividends = _strip_tz(tk.dividends)
        splits = _strip_tz(tk.splits)
    except Exception:
        pass

    # Dates d'earnings passées (peut ne pas exister pour un ETF/crypto).
    earnings_dates = pd.DatetimeIndex([])
    try:
        ed = tk.get_earnings_dates(limit=40)
        if ed is not None and not ed.empty:
            idx = pd.to_datetime(ed.index)
            if idx.tz is not None:
                idx = idx.tz_localize(None)
            earnings_dates = pd.DatetimeIndex(idx)
    except Exception:
        pass

    return MarketData(
        ticker=ticker,
        ohlcv=ohlcv,
        info=info or {},
        dividends=dividends,
        splits=splits,
        earnings_dates=earnings_dates,
    )


def _strip_tz(series: pd.Series) -> pd.Series:
    """Retire le fuseau horaire de l'index d'une série (robuste au None)."""
    if series is None or len(series) == 0:
        return pd.Series(dtype=float)
    series = series.copy()
    idx = pd.to_datetime(series.index)
    if idx.tz is not None:
        idx = idx.tz_localize(None)
    series.index = idx
    return series
