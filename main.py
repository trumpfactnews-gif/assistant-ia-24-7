#!/usr/bin/env python3
"""
main.py
=======

Point d'entrée en ligne de commande du système d'analyse probabiliste de prix.

Le script demande un ticker à l'utilisateur (ou l'accepte en argument), lance
l'analyse complète, affiche le tableau de résultats et sauvegarde un graphique.

Usage :
    python main.py                 # mode interactif (demande le ticker)
    python main.py AAPL            # ticker en argument
    python main.py BTC-USD --no-chart
    python main.py TSLA --out tsla.png
"""

from __future__ import annotations

import argparse
import sys

from price_predictor import PricePredictor


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyse probabiliste des mouvements de prix d'un ticker."
    )
    parser.add_argument(
        "ticker",
        nargs="?",
        default=None,
        help="Symbole du ticker (ex: AAPL, MSFT, BTC-USD, SPY). "
        "Si omis, il sera demandé de manière interactive.",
    )
    parser.add_argument(
        "--out",
        default="analysis.png",
        help="Chemin de sauvegarde du graphique (défaut: analysis.png).",
    )
    parser.add_argument(
        "--no-chart",
        action="store_true",
        help="Ne pas générer de graphique.",
    )
    parser.add_argument(
        "--years",
        type=int,
        default=5,
        help="Profondeur d'historique en années (défaut: 5).",
    )
    return parser.parse_args(argv)


def prompt_ticker() -> str:
    """Demande interactivement un ticker à l'utilisateur."""
    try:
        ticker = input("Entrez le ticker à analyser (ex: AAPL, BTC-USD) : ").strip()
    except (EOFError, KeyboardInterrupt):
        print("\nAnnulé.")
        sys.exit(1)
    if not ticker:
        print("Aucun ticker fourni. Fin.")
        sys.exit(1)
    return ticker


def main(argv=None) -> int:
    args = parse_args(argv)
    ticker = args.ticker or prompt_ticker()

    print(f"\n⏳ Analyse de {ticker.upper()} en cours "
          f"(téléchargement de {args.years} ans de données)...\n")

    try:
        predictor = PricePredictor(ticker, years=args.years)
        output = predictor.run(chart_path=None if args.no_chart else args.out)
    except ImportError as exc:
        print(f"❌ Dépendance manquante : {exc}")
        return 2
    except ValueError as exc:
        print(f"❌ Erreur de données : {exc}")
        print("   Vérifiez que le ticker est valide et reconnu par yfinance.")
        return 3
    except Exception as exc:  # filet de sécurité générique
        print(f"❌ Erreur inattendue : {exc}")
        return 1

    print(output.report_text)

    if output.chart_path:
        print(f"\n📈 Graphique sauvegardé : {output.chart_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
