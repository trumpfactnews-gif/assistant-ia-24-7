# 📊 Système d'analyse probabiliste des prix

Système complet en Python qui estime, à partir des **5 dernières années** de
données historiques, la **probabilité (sur 100 %)** qu'un ticker financier
**monte ou descende** sur plusieurs horizons temporels :

> **24h · 48h · 72h · 5 jours · 1 semaine · 6 semaines**

Compatible avec n'importe quel ticker reconnu par `yfinance` : actions, ETF,
et cryptos (ex. `AAPL`, `MSFT`, `SPY`, `BTC-USD`).

---

## 🚀 Installation

```bash
pip install -r requirements.txt
```

## ▶️ Utilisation

```bash
# Mode interactif : le script demande le ticker
python main.py

# Ticker en argument
python main.py AAPL

# Crypto, sans graphique
python main.py BTC-USD --no-chart

# Choisir le fichier de sortie du graphique
python main.py TSLA --out tsla.png
```

### En tant que bibliothèque

```python
from price_predictor import PricePredictor

output = PricePredictor("AAPL").run(chart_path="aapl.png")
print(output.report_text)
for r in output.results:
    print(r.horizon, r.prob_up, r.prob_down, r.signal)
```

---

## 🧠 Comment ça marche

Le pipeline enchaîne six étapes (un module par responsabilité) :

| Module | Rôle |
|--------|------|
| `data_loader.py` | Téléchargement OHLCV (yfinance), nettoyage des gaps/valeurs manquantes, méta-données, dividendes, splits, dates d'earnings |
| `indicators.py`  | Indicateurs techniques : RSI, MACD, Bandes de Bollinger, EMA 20/50/200, ATR, volume, momentum, volatilité |
| `events.py`      | Calendrier d'événements : earnings, dividendes, splits + macro modélisée (FOMC, CPI, NFP). Les jours d'événement sont marqués pour ne pas biaiser les probabilités « en conditions normales » |
| `analysis.py`    | Structure de courbe : tendance (haussière/baissière/latérale), régime de volatilité, supports/résistances (pivots + clustering), patterns (breakout, reversal, squeeze, consolidation) |
| `probability.py` | Moteur probabiliste (voir ci-dessous) |
| `plotting.py` / `report.py` | Graphique matplotlib + tableau texte |

### Moteur probabiliste

Pour **chaque horizon**, trois sources de signal sont fusionnées :

1. **Analogues historiques** — fréquence de hausse sur les jours passés
   partageant le **même régime** (tendance, volatilité, zone RSI), *hors jours
   d'événement*.
2. **Apprentissage supervisé** — ensemble **Régression logistique + Random
   Forest** classant « monte/descend » à partir des indicateurs. La précision
   est estimée par **validation croisée temporelle** (walk-forward) et sert de
   niveau de confiance.
3. **Monte Carlo** — simulation par **bootstrap des rendements** historiques
   (poids croissant sur les horizons longs).

Les poids de fusion dépendent de l'horizon (le ML/analogues dominent à court
terme, Monte Carlo à long terme). Le résultat est **toujours normalisé** :
`P(hausse) + P(baisse) = 100 %`.

---

## 📋 Exemple de sortie

```
Ticker : AAPL (Apple Inc.)  |  Date d'analyse : 2025-01-15  |  Prix actuel : 233.79

Horizon    |    Proba HAUSSE   |    Proba BAISSE   | Signal dominant
-----------|-------------------|-------------------|-----------------
24h        |        51 %       |        49 %       | Neutre
48h        |        60 %       |        40 %       | Haussier
72h        |        56 %       |        44 %       | Haussier
5 jours    |        57 %       |        43 %       | Haussier
1 semaine  |        53 %       |        47 %       | Neutre
6 semaines |        57 %       |        43 %       | Haussier

Facteurs clés détectés :
  • Tendance haussière (force 100%)
  • Régime de volatilité normale
  • Support proche : 329.52, 309.18
  • Événement : Publication de résultats (earnings)

Niveau de confiance du modèle : 54 %
```

Un graphique (`analysis.png`) est également généré : prix sur 5 ans, EMA 50/200,
zones d'événements annotées, niveaux de support/résistance, et barres de
probabilité directionnelle par horizon.

---

## 📁 Structure du projet

```
.
├── main.py                    # Point d'entrée CLI (demande le ticker)
├── requirements.txt
├── README.md
└── price_predictor/
    ├── __init__.py
    ├── data_loader.py         # 1. données OHLCV + nettoyage
    ├── indicators.py          # 2. indicateurs techniques
    ├── events.py              # 3. événements externes
    ├── analysis.py            # 4. structure de marché
    ├── probability.py         # 5. moteur probabiliste
    ├── plotting.py            # 6a. graphique
    ├── report.py              # 6b. tableau texte
    └── predictor.py           # orchestrateur
```

---

## ⚙️ Note sur l'accès réseau

Le script a besoin d'un accès sortant à **Yahoo Finance** (via `yfinance`).
Dans un environnement où le trafic sortant est filtré (proxy/allowlist), le
téléchargement peut échouer avec une erreur `403 CONNECT tunnel failed` :
autorisez alors `query1.finance.yahoo.com` / `query2.finance.yahoo.com` dans la
politique réseau, ou exécutez le script depuis une machine à accès Internet
ouvert.

---

## ⚠️ Avertissement

Cet outil est fourni à titre **informatif et éducatif uniquement**. Il ne
constitue **pas un conseil en investissement**. Les performances passées ne
préjugent pas des performances futures.
