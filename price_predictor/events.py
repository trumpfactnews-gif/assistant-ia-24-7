"""
events
======

Détection et pondération des événements externes ayant influencé le prix.

Deux familles d'événements :

1. **Événements propres au titre** (issus de yfinance) :
   - dates d'earnings (publications de résultats) ;
   - dividendes ;
   - splits.

2. **Événements macroéconomiques récurrents** approximés par un calendrier
   modélisé (les API macro temps réel ne sont pas garanties dans tous les
   environnements) :
   - réunions FOMC / décisions de taux de la Fed (~toutes les 6 semaines) ;
   - publications CPI (inflation, mensuel) ;
   - NFP / rapport emploi (premier vendredi du mois).

Chaque jour de la série reçoit :
   - un drapeau ``is_event`` (True si dans la fenêtre d'un événement) ;
   - une liste des événements actifs ce jour-là.

La séparation « jours normaux » / « jours d'événement » permet de ne pas
biaiser le calcul des probabilités en conditions normales de marché.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List

import numpy as np
import pandas as pd

from .data_loader import MarketData


@dataclass
class Event:
    """Un événement horodaté avec un type et une description."""

    date: pd.Timestamp
    kind: str          # 'earnings' | 'dividend' | 'split' | 'fomc' | 'cpi' | 'nfp'
    label: str


# Fenêtre (en jours de trading) autour d'un événement considérée comme « perturbée ».
EVENT_WINDOW = 2


def _monthly_cpi_dates(index: pd.DatetimeIndex) -> List[pd.Timestamp]:
    """
    Approxime les dates de publication du CPI américain.

    Le CPI est publié mensuellement, généralement entre le 10 et le 15 du mois.
    On retient, pour chaque mois couvert, le jour de trading le plus proche du 12.
    """
    dates: List[pd.Timestamp] = []
    months = pd.date_range(index.min(), index.max(), freq="MS")
    for m in months:
        target = m + pd.Timedelta(days=11)  # ~ le 12 du mois
        nearest = _nearest_trading_day(index, target)
        if nearest is not None:
            dates.append(nearest)
    return dates


def _monthly_nfp_dates(index: pd.DatetimeIndex) -> List[pd.Timestamp]:
    """
    Approxime les dates du rapport emploi (NFP) : premier vendredi du mois.
    """
    dates: List[pd.Timestamp] = []
    months = pd.date_range(index.min(), index.max(), freq="MS")
    for m in months:
        # Premier vendredi : weekday() == 4
        d = m
        while d.weekday() != 4:
            d += pd.Timedelta(days=1)
        nearest = _nearest_trading_day(index, d)
        if nearest is not None:
            dates.append(nearest)
    return dates


def _fomc_dates(index: pd.DatetimeIndex) -> List[pd.Timestamp]:
    """
    Approxime les réunions du FOMC : environ 8 par an, espacées de ~6 semaines.

    On génère une grille toutes les 6 semaines à partir d'un point d'ancrage
    (fin janvier), puis on colle chaque date au jour de trading le plus proche.
    """
    dates: List[pd.Timestamp] = []
    anchor = pd.Timestamp(year=index.min().year, month=1, day=29)
    d = anchor
    end = index.max()
    while d <= end:
        if d >= index.min():
            nearest = _nearest_trading_day(index, d)
            if nearest is not None:
                dates.append(nearest)
        d += pd.Timedelta(weeks=6)
    return dates


def _nearest_trading_day(
    index: pd.DatetimeIndex, target: pd.Timestamp, tol_days: int = 5
) -> pd.Timestamp | None:
    """Renvoie le jour de trading de l'index le plus proche de ``target``."""
    if len(index) == 0:
        return None
    diffs = np.abs((index - target).days)
    pos = int(diffs.argmin())
    if diffs[pos] <= tol_days:
        return index[pos]
    return None


def build_event_calendar(data: MarketData) -> List[Event]:
    """
    Construit la liste complète des événements sur la période de la série.
    """
    index = data.ohlcv.index
    events: List[Event] = []

    # --- Événements propres au titre -------------------------------------
    for d in data.earnings_dates:
        nd = _nearest_trading_day(index, pd.Timestamp(d))
        if nd is not None:
            events.append(Event(nd, "earnings", "Publication de résultats (earnings)"))

    for d, amount in data.dividends.items():
        nd = _nearest_trading_day(index, pd.Timestamp(d))
        if nd is not None:
            events.append(Event(nd, "dividend", f"Dividende ({amount:.2f})"))

    for d, ratio in data.splits.items():
        nd = _nearest_trading_day(index, pd.Timestamp(d))
        if nd is not None:
            events.append(Event(nd, "split", f"Split {ratio:g}:1"))

    # --- Événements macroéconomiques modélisés ---------------------------
    for d in _fomc_dates(index):
        events.append(Event(d, "fomc", "Réunion FOMC / décision de taux Fed"))
    for d in _monthly_cpi_dates(index):
        events.append(Event(d, "cpi", "Publication CPI (inflation)"))
    for d in _monthly_nfp_dates(index):
        events.append(Event(d, "nfp", "Rapport emploi NFP"))

    events.sort(key=lambda e: e.date)
    return events


def annotate_events(
    df: pd.DataFrame, events: List[Event], window: int = EVENT_WINDOW
) -> pd.DataFrame:
    """
    Ajoute au DataFrame les colonnes d'événements.

    Colonnes ajoutées :
        - ``is_event``       : bool, True dans une fenêtre +/- ``window`` jours ;
        - ``event_kinds``    : chaîne des types d'événements actifs ce jour ;
        - ``dist_to_event``  : distance (en jours de trading) au plus proche event.

    La fenêtre marque « perturbés » les jours autour d'un événement, ce qui
    permet ensuite de séparer conditions normales / conditions d'événement.
    """
    out = df.copy()
    index = out.index
    out["is_event"] = False
    out["event_kinds"] = ""

    pos_of = {ts: i for i, ts in enumerate(index)}
    n = len(index)

    for ev in events:
        if ev.date not in pos_of:
            continue
        center = pos_of[ev.date]
        lo, hi = max(0, center - window), min(n - 1, center + window)
        for i in range(lo, hi + 1):
            ts = index[i]
            out.at[ts, "is_event"] = True
            existing = out.at[ts, "event_kinds"]
            if ev.kind not in existing.split(","):
                out.at[ts, "event_kinds"] = (
                    f"{existing},{ev.kind}" if existing else ev.kind
                )

    return out


def top_recent_events(
    events: List[Event], reference: pd.Timestamp, n: int = 6
) -> List[Event]:
    """Renvoie les ``n`` événements les plus récents avant la date de référence."""
    past = [e for e in events if e.date <= reference]
    return sorted(past, key=lambda e: e.date, reverse=True)[:n]
