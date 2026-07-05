"""
price_predictor
===============

Système d'analyse probabiliste des mouvements de prix d'un ticker financier.

Ce package calcule, à partir des 5 dernières années de données historiques,
la probabilité (sur 100 %) qu'un prix monte ou descende sur plusieurs horizons
temporels (24h, 48h, 72h, 5 jours, 1 semaine, 6 semaines).

Modules :
    - data_loader : récupération et nettoyage des données OHLCV (yfinance)
    - indicators  : indicateurs techniques (RSI, MACD, Bollinger, EMA, ATR...)
    - events      : détection et pondération des événements externes
    - analysis    : structure de courbe (tendance, supports/résistances, patterns)
    - probability : moteur probabiliste (historique + ML + Monte Carlo)
    - plotting    : génération du graphique matplotlib
    - report      : formatage du tableau de résultats
    - predictor   : orchestrateur haut niveau
"""

from .predictor import PricePredictor, HORIZONS

__all__ = ["PricePredictor", "HORIZONS"]
__version__ = "1.0.0"
