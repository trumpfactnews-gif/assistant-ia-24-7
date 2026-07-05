"""
probability
===========

Moteur probabiliste. Pour chaque horizon, il combine trois sources de signal :

1. **Probabilité historique en conditions similaires** (analogues) — on filtre
   l'historique sur le régime courant (tendance, régime de volatilité, zone RSI)
   *en excluant les jours d'événement*, puis on mesure la fréquence de hausse.

2. **Modèle d'apprentissage supervisé** — régression logistique et Random Forest
   qui classent « monte / descend » à partir des indicateurs techniques. La
   précision est estimée par validation croisée temporelle (walk-forward) et
   sert d'indice de confiance.

3. **Simulation Monte Carlo** — pour les horizons plus longs, on simule des
   trajectoires par bootstrap des rendements historiques et on mesure la part
   de trajectoires terminant au-dessus du prix courant.

Les trois probabilités sont fusionnées par pondération dépendante de l'horizon.
Le résultat final est toujours normalisé pour que P(hausse) + P(baisse) = 100 %.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List

import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .analysis import MarketStructure, detect_trend, volatility_regime


# Horizons demandés, exprimés en séances de bourse.
# (1 semaine ≈ 7 jours calendaires ; 6 semaines ≈ 30 séances.)
HORIZONS: Dict[str, int] = {
    "24h": 1,
    "48h": 2,
    "72h": 3,
    "5 jours": 5,
    "1 semaine": 7,
    "6 semaines": 30,
}

# Colonnes d'indicateurs utilisées comme variables explicatives (features).
FEATURE_COLS: List[str] = [
    "rsi",
    "macd",
    "macd_signal",
    "macd_hist",
    "bb_pct",
    "bb_width",
    "atr_pct",
    "volatility",
    "vol_ratio",
    "mom_5",
    "mom_10",
    "mom_20",
    "dist_ema20",
    "dist_ema50",
    "dist_ema200",
    "ret",
]


@dataclass
class HorizonResult:
    """Résultat probabiliste pour un horizon donné."""

    horizon: str
    days: int
    prob_up: float                    # en %
    prob_down: float                  # en %
    signal: str                       # 'Haussier' | 'Baissier' | 'Neutre'
    model_accuracy: float             # précision historique du modèle (%)
    components: Dict[str, float] = field(default_factory=dict)  # détail des sources


def _signal_from_prob(prob_up: float, neutral_band: float = 5.0) -> str:
    """Convertit une probabilité de hausse en signal directionnel."""
    if prob_up >= 50 + neutral_band:
        return "Haussier"
    if prob_up <= 50 - neutral_band:
        return "Baissier"
    return "Neutre"


def _rsi_bucket(rsi: float) -> int:
    """Discrétise le RSI en zones (0: survendu ... 3: suracheté)."""
    if pd.isna(rsi):
        return 1
    if rsi < 30:
        return 0
    if rsi < 50:
        return 1
    if rsi < 70:
        return 2
    return 3


def historical_analog_probability(
    df: pd.DataFrame, horizon_days: int, structure: MarketStructure
) -> float | None:
    """
    Probabilité de hausse mesurée sur les jours historiques « analogues ».

    Conditions d'analogie :
        - même label de tendance ;
        - même régime de volatilité (tercile ATR%) ;
        - même zone de RSI.
    On exclut les jours d'événement pour rester en conditions normales, et on
    calcule la fréquence de hausse sur ``horizon_days`` séances.

    Returns
    -------
    float | None
        Probabilité de hausse en %, ou None si l'échantillon est trop faible.
    """
    work = df.copy()
    # Rendement futur sur l'horizon.
    future = work["Close"].shift(-horizon_days)
    work["up"] = (future > work["Close"]).astype(float)
    work = work.iloc[:-horizon_days] if horizon_days > 0 else work

    # Reconstruit le régime jour par jour de façon vectorisée et robuste.
    # Zone RSI.
    work["rsi_bucket"] = work["rsi"].apply(_rsi_bucket)

    # Régime de volatilité par terciles ATR% (global).
    atr_pct = work["atr_pct"]
    low_q, high_q = atr_pct.quantile(0.33), atr_pct.quantile(0.66)

    def vol_bucket(v: float) -> str:
        if pd.isna(v):
            return "normale"
        if v <= low_q:
            return "faible"
        if v >= high_q:
            return "élevée"
        return "normale"

    work["vol_bucket"] = work["atr_pct"].apply(vol_bucket)

    # Label de tendance simplifié par position vs EMA (proxy vectoriel).
    def trend_label(row) -> str:
        c, e50, e200 = row["Close"], row["ema50"], row["ema200"]
        if pd.isna(e200):
            return "latérale"
        if c > e50 > e200:
            return "haussière"
        if c < e50 < e200:
            return "baissière"
        return "latérale"

    work["trend_label"] = work.apply(trend_label, axis=1)

    # Filtre : conditions normales (hors événement) + régime courant.
    cur_rsi_bucket = _rsi_bucket(df["rsi"].iloc[-1])
    mask = (
        (~work.get("is_event", pd.Series(False, index=work.index)).astype(bool))
        & (work["trend_label"] == structure.trend)
        & (work["vol_bucket"] == structure.volatility_regime)
        & (work["rsi_bucket"] == cur_rsi_bucket)
    )
    sample = work.loc[mask, "up"].dropna()

    # On exige un minimum d'observations pour être significatif.
    if len(sample) < 20:
        # Relâche la contrainte RSI si trop peu d'analogues.
        mask2 = (
            (~work.get("is_event", pd.Series(False, index=work.index)).astype(bool))
            & (work["trend_label"] == structure.trend)
            & (work["vol_bucket"] == structure.volatility_regime)
        )
        sample = work.loc[mask2, "up"].dropna()
    if len(sample) < 15:
        return None
    return float(sample.mean() * 100.0)


def _build_xy(df: pd.DataFrame, horizon_days: int):
    """Construit la matrice de features X et la cible binaire y pour un horizon."""
    cols = [c for c in FEATURE_COLS if c in df.columns]
    X = df[cols].copy()
    future = df["Close"].shift(-horizon_days)
    y = (future > df["Close"]).astype(int)

    # Optionnel : exclure les jours d'événement de l'apprentissage « normal ».
    if "is_event" in df.columns:
        normal = ~df["is_event"].astype(bool)
    else:
        normal = pd.Series(True, index=df.index)

    valid = X.notna().all(axis=1) & future.notna() & normal
    return X.loc[valid], y.loc[valid], cols


def ml_probability(df: pd.DataFrame, horizon_days: int) -> tuple[float | None, float]:
    """
    Probabilité de hausse issue d'un ensemble (LogReg + RandomForest).

    Returns
    -------
    (prob_up_pct, accuracy_pct)
        La probabilité prédite pour la dernière observation et la précision
        moyenne estimée par validation croisée temporelle. Renvoie (None, 50)
        si l'échantillon est insuffisant.
    """
    X, y, cols = _build_xy(df, horizon_days)
    if len(X) < 120 or y.nunique() < 2:
        return None, 50.0

    # Deux modèles complémentaires : linéaire régularisé + forêt non linéaire.
    logreg = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=1000, C=0.5)),
        ]
    )
    rf = RandomForestClassifier(
        n_estimators=250,
        max_depth=5,
        min_samples_leaf=25,
        random_state=42,
        n_jobs=-1,
    )

    # --- Estimation de la précision par walk-forward (TimeSeriesSplit) -----
    accuracies: List[float] = []
    n_splits = min(5, max(2, len(X) // 200))
    tscv = TimeSeriesSplit(n_splits=n_splits)
    for train_idx, test_idx in tscv.split(X):
        if y.iloc[train_idx].nunique() < 2:
            continue
        X_tr, X_te = X.iloc[train_idx], X.iloc[test_idx]
        y_tr, y_te = y.iloc[train_idx], y.iloc[test_idx]
        fold_probs = []
        for model in (logreg, rf):
            model.fit(X_tr, y_tr)
            fold_probs.append(model.predict_proba(X_te)[:, 1])
        ens = np.mean(fold_probs, axis=0)
        pred = (ens >= 0.5).astype(int)
        accuracies.append(float((pred == y_te.values).mean()))

    accuracy = float(np.mean(accuracies) * 100.0) if accuracies else 50.0

    # --- Ré-entraînement final sur tout l'historique, prédiction du jour J ---
    last_row = df[cols].iloc[[-1]]
    if last_row.isna().any(axis=1).iloc[0]:
        return None, accuracy

    probs = []
    for model in (logreg, rf):
        model.fit(X, y)
        probs.append(float(model.predict_proba(last_row)[0, 1]))
    prob_up = float(np.mean(probs) * 100.0)
    return prob_up, accuracy


def monte_carlo_probability(
    df: pd.DataFrame, horizon_days: int, n_sims: int = 5000
) -> float | None:
    """
    Probabilité de hausse par simulation Monte Carlo (bootstrap des rendements).

    On rééchantillonne les rendements log historiques (hors événements quand
    l'info existe) pour générer ``n_sims`` trajectoires de ``horizon_days``
    séances, et on mesure la part terminant au-dessus du prix actuel.
    """
    log_ret = df["log_ret"].dropna()
    if "is_event" in df.columns:
        normal_mask = ~df["is_event"].reindex(log_ret.index).fillna(False).astype(bool)
        pool = log_ret[normal_mask]
        if len(pool) < 100:
            pool = log_ret
    else:
        pool = log_ret

    if len(pool) < 60:
        return None

    pool_values = pool.values
    # Tirage vectorisé : matrice (n_sims x horizon_days) de rendements bootstrap.
    draws = np.random.choice(pool_values, size=(n_sims, horizon_days), replace=True)
    cumulative = draws.sum(axis=1)  # rendement log cumulé par trajectoire
    prob_up = float((cumulative > 0).mean() * 100.0)
    return prob_up


def _horizon_weights(horizon_days: int) -> Dict[str, float]:
    """
    Pondère les trois sources selon l'horizon.

    - Court terme : le modèle ML et les analogues dominent.
    - Long terme : Monte Carlo (dérive/vol) prend plus de poids.
    """
    if horizon_days <= 3:
        return {"analog": 0.35, "ml": 0.45, "mc": 0.20}
    if horizon_days <= 7:
        return {"analog": 0.30, "ml": 0.40, "mc": 0.30}
    return {"analog": 0.25, "ml": 0.30, "mc": 0.45}


def _blend(prob_map: Dict[str, float | None], weights: Dict[str, float]) -> float:
    """Fusionne les probabilités disponibles selon les poids (renormalisés)."""
    num, den = 0.0, 0.0
    for key, prob in prob_map.items():
        if prob is None:
            continue
        w = weights.get(key, 0.0)
        num += w * prob
        den += w
    if den == 0:
        return 50.0
    return num / den


def compute_probabilities(
    df: pd.DataFrame, structure: MarketStructure
) -> List[HorizonResult]:
    """
    Calcule les probabilités fusionnées pour tous les horizons.

    Parameters
    ----------
    df : pd.DataFrame
        Données enrichies d'indicateurs et d'annotations d'événements.
    structure : MarketStructure
        Structure de marché courante (pour l'appariement d'analogues).

    Returns
    -------
    List[HorizonResult]
        Un résultat par horizon, avec probabilités normalisées à 100 %.
    """
    # Graine fixe pour la reproductibilité des simulations Monte Carlo.
    np.random.seed(42)

    results: List[HorizonResult] = []
    for name, days in HORIZONS.items():
        p_analog = historical_analog_probability(df, days, structure)
        p_ml, accuracy = ml_probability(df, days)
        p_mc = monte_carlo_probability(df, days)

        weights = _horizon_weights(days)
        prob_up = _blend({"analog": p_analog, "ml": p_ml, "mc": p_mc}, weights)

        # Garde-fous : borne raisonnable et normalisation hausse+baisse = 100 %.
        prob_up = float(np.clip(prob_up, 1.0, 99.0))
        prob_down = 100.0 - prob_up

        results.append(
            HorizonResult(
                horizon=name,
                days=days,
                prob_up=round(prob_up, 1),
                prob_down=round(prob_down, 1),
                signal=_signal_from_prob(prob_up),
                model_accuracy=round(accuracy, 1),
                components={
                    "analog": None if p_analog is None else round(p_analog, 1),
                    "ml": None if p_ml is None else round(p_ml, 1),
                    "mc": None if p_mc is None else round(p_mc, 1),
                },
            )
        )
    return results


def overall_confidence(results: List[HorizonResult]) -> float:
    """Niveau de confiance global = précision ML moyenne sur les horizons."""
    accs = [r.model_accuracy for r in results if r.model_accuracy]
    return round(float(np.mean(accs)), 1) if accs else 50.0
