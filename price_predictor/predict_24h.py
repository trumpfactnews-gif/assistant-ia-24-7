"""
predict_24h
===========

Prédiction **24h uniquement**, pensée pour être branchée dans une application
externe (ex. le site FLUXEL). Elle réutilise tout le moteur de calcul du package
mais ne renvoie qu'un seul résultat : le sens le plus probable à 24h et sa
probabilité, basé **uniquement sur des calculs** (aucune opinion).

Fonction principale :
    predict_24h(ticker) -> dict

Le dictionnaire renvoyé est directement sérialisable en JSON.
"""

from __future__ import annotations

from typing import Dict, Optional

import numpy as np

from . import analysis as analysis_mod
from . import events as events_mod
from . import indicators as indicators_mod
from . import probability as probability_mod
from .data_loader import load_market_data


# Horizon fixé à 1 séance de bourse (~24h).
_HORIZON_DAYS = 1


def predict_24h(ticker: str, years: int = 5) -> Dict:
    """
    Calcule la probabilité de hausse/baisse à 24h pour un ticker.

    Parameters
    ----------
    ticker : str
        Symbole reconnu par yfinance (AAPL, BTC-USD, ...).
    years : int
        Profondeur d'historique (5 ans par défaut).

    Returns
    -------
    dict
        {
          "ticker": "AAPL",
          "prix": 308.63,
          "date": "2026-07-02",
          "sens": "HAUSSE" | "BAISSE" | "NEUTRE",
          "proba": 61,                # probabilité du sens dominant, en %
          "proba_hausse": 61,
          "proba_baisse": 39,
          "confiance": 53,            # précision historique du modèle, en %
          "facteurs": [...],          # signaux techniques détectés
          "ok": True
        }
        En cas d'erreur : {"ok": False, "erreur": "..."}.
    """
    try:
        # 1. Données + indicateurs + événements.
        data = load_market_data(ticker, years=years)
        df = indicators_mod.compute_indicators(data.ohlcv)
        events = events_mod.build_event_calendar(data)
        df = events_mod.annotate_events(df, events)

        # 2. Structure de marché (pour l'appariement d'analogues).
        structure = analysis_mod.analyze_structure(df)

        # 3. Les trois sources de probabilité, pour l'horizon 24h uniquement.
        np.random.seed(42)  # reproductibilité de la simulation Monte Carlo
        p_analog = probability_mod.historical_analog_probability(
            df, _HORIZON_DAYS, structure
        )
        p_ml, accuracy = probability_mod.ml_probability(df, _HORIZON_DAYS)
        p_mc = probability_mod.monte_carlo_probability(df, _HORIZON_DAYS)

        # 4. Fusion pondérée (poids court terme) + normalisation à 100 %.
        weights = probability_mod._horizon_weights(_HORIZON_DAYS)
        prob_up = probability_mod._blend(
            {"analog": p_analog, "ml": p_ml, "mc": p_mc}, weights
        )
        prob_up = float(np.clip(prob_up, 1.0, 99.0))
        prob_down = 100.0 - prob_up

        # 5. Sens dominant + probabilité associée.
        signal = probability_mod._signal_from_prob(prob_up)
        sens = {"Haussier": "HAUSSE", "Baissier": "BAISSE", "Neutre": "NEUTRE"}[signal]
        proba_dominante = round(max(prob_up, prob_down), 1)

        # 6. Facteurs techniques détectés (résumé court).
        facteurs = [
            f"Tendance {structure.trend}",
            f"Volatilité {structure.volatility_regime}",
        ]
        if structure.rsi_state != "neutre":
            facteurs.append(f"RSI {structure.rsi_state}")
        facteurs.extend(structure.patterns[:2])

        return {
            "ok": True,
            "ticker": data.ticker,
            "nom": data.name,
            "prix": round(data.last_price, 2),
            "date": str(data.last_date.date()),
            "sens": sens,
            "proba": proba_dominante,
            "proba_hausse": round(prob_up, 1),
            "proba_baisse": round(prob_down, 1),
            "confiance": round(accuracy, 1),
            "facteurs": facteurs,
        }

    except Exception as exc:  # renvoie une erreur exploitable côté appelant
        return {"ok": False, "ticker": ticker.upper(), "erreur": str(exc)}


if __name__ == "__main__":
    # Petit test en ligne de commande : `python -m price_predictor.predict_24h AAPL`
    import json
    import sys

    symbol = sys.argv[1] if len(sys.argv) > 1 else "AAPL"
    print(json.dumps(predict_24h(symbol), ensure_ascii=False, indent=2))
