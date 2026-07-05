"""
predictor
=========

Orchestrateur haut niveau. Enchaîne les étapes du pipeline :

    1. téléchargement + nettoyage des données (data_loader) ;
    2. calcul des indicateurs techniques (indicators) ;
    3. construction et annotation du calendrier d'événements (events) ;
    4. analyse de la structure de marché (analysis) ;
    5. calcul des probabilités par horizon (probability) ;
    6. formatage du rapport et génération du graphique.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

import pandas as pd

from . import analysis as analysis_mod
from . import events as events_mod
from . import indicators as indicators_mod
from . import plotting as plotting_mod
from . import probability as probability_mod
from . import report as report_mod
from .data_loader import load_market_data, MarketData
from .probability import HORIZONS, HorizonResult, overall_confidence


@dataclass
class PredictionOutput:
    """Résultat complet d'une analyse."""

    ticker: str
    name: str
    analysis_date: pd.Timestamp
    last_price: float
    results: List[HorizonResult]
    structure: analysis_mod.MarketStructure
    recent_events: List[events_mod.Event]
    confidence: float
    report_text: str = ""
    chart_path: Optional[str] = None


class PricePredictor:
    """
    Interface principale du système d'analyse probabiliste.

    Exemple
    -------
    >>> predictor = PricePredictor("AAPL")
    >>> output = predictor.run(chart_path="aapl.png")
    >>> print(output.report_text)
    """

    def __init__(self, ticker: str, years: int = 5):
        self.ticker = ticker.strip().upper()
        self.years = years
        self._data: Optional[MarketData] = None
        self._df: Optional[pd.DataFrame] = None
        self._events: List[events_mod.Event] = []

    # ------------------------------------------------------------------ #
    # Étapes du pipeline
    # ------------------------------------------------------------------ #
    def load(self) -> "PricePredictor":
        """Étape 1 : télécharge et prépare les données."""
        self._data = load_market_data(self.ticker, years=self.years)
        return self

    def prepare(self) -> "PricePredictor":
        """Étapes 2-3 : indicateurs + calendrier d'événements annoté."""
        assert self._data is not None, "Appelez load() avant prepare()."
        df = indicators_mod.compute_indicators(self._data.ohlcv)
        self._events = events_mod.build_event_calendar(self._data)
        df = events_mod.annotate_events(df, self._events)
        self._df = df
        return self

    def run(self, chart_path: Optional[str] = "analysis.png") -> PredictionOutput:
        """
        Exécute l'ensemble du pipeline et renvoie le résultat complet.

        Parameters
        ----------
        chart_path : str | None
            Chemin de sauvegarde du graphique. Si None, aucun graphique généré.
        """
        if self._data is None:
            self.load()
        if self._df is None:
            self.prepare()

        assert self._data is not None and self._df is not None

        # Étape 4 : structure de marché.
        structure = analysis_mod.analyze_structure(self._df)

        # Étape 5 : probabilités par horizon.
        results = probability_mod.compute_probabilities(self._df, structure)
        confidence = overall_confidence(results)

        # Événements récents (pour le rapport).
        recent_events = events_mod.top_recent_events(
            self._events, self._data.last_date, n=6
        )

        # Étape 6 : rapport texte.
        report_text = report_mod.format_report(
            ticker=self.ticker,
            name=self._data.name,
            analysis_date=self._data.last_date,
            last_price=self._data.last_price,
            results=results,
            structure=structure,
            recent_events=recent_events,
        )

        # Graphique.
        chart_out = None
        if chart_path:
            chart_out = plotting_mod.plot_analysis(
                ticker=self.ticker,
                df=self._df,
                structure=structure,
                events=self._events,
                results=results,
                out_path=chart_path,
            )

        return PredictionOutput(
            ticker=self.ticker,
            name=self._data.name,
            analysis_date=self._data.last_date,
            last_price=self._data.last_price,
            results=results,
            structure=structure,
            recent_events=recent_events,
            confidence=confidence,
            report_text=report_text,
            chart_path=chart_out,
        )
