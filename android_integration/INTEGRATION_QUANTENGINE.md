# Intégration du moteur quantitatif local réel (QuantEngine)

Ce document décrit l'intégration de l'analyse probabiliste **100 % calcul** dans
l'onglet **Quantique** de SEC Analyst Pro.

## Ce qui a changé

Avant, l'onglet Quantique, lorsque le serveur VPS et Gemini étaient injoignables,
retombait sur une **simulation locale à biais codés en dur** (ex. `PLTR → +1.8`) —
sans aucune valeur analytique.

Désormais, un **vrai moteur de calcul sur l'appareil** (`QuantEngine`) est la
**source par défaut**. Il calcule les probabilités à partir de **5 ans d'OHLCV
réels** (API Yahoo), sans biais codé en dur ni génération par IA.

### Nouvel ordre de priorité des sources (`QuantumService.getPredictions`)

1. **`LOCAL_REAL_QUANT`** — moteur local réel (QuantEngine). *Nouveau, par défaut.*
2. `QUANT_SERVER_REAL` — serveur VPS quantitatif (si configuré/joignable).
3. `GEMINI_GENERATED` — inférence Gemini (fallback IA).
4. `LOCAL_SIMULATED` — simulation de secours (dernier recours, inchangée).

## Fichiers ajoutés

| Fichier | Rôle |
|---------|------|
| `service/QuantEngine.kt` | Moteur pur Kotlin : indicateurs (RSI, MACD, Bollinger, EMA, ATR), probabilités par analogues historiques, régression logistique (descente de gradient), Monte Carlo, mise à l'échelle intraday (racine du temps). Aucune dépendance Android. |
| `data/repository/PriceHistoryRepository.kt` | Récupère 5 ans d'OHLCV quotidiens via l'API Yahoo Chart (`range=5y&interval=1d`), ajuste des splits (adjclose), nettoie les gaps. |

## Fichiers modifiés

| Fichier | Modification |
|---------|--------------|
| `service/QuantumService.kt` | Réécrit en pipeline ordonné ; ajout de `computeLocalReal()`. Logique VPS/Gemini/simulé conservée à l'identique dans des fonctions dédiées. |
| `ui/screens/QuantumScreen.kt` | Ajout de l'affichage honnête de la provenance `LOCAL_REAL_QUANT` (carte d'en-tête émeraude, puce de traçabilité, sous-titre). |

## Comment fonctionne le calcul (par horizon `1min/5min/30min/1d/5d`)

1. **Régime courant** : tendance (position vs EMA50/EMA200), régime de volatilité
   (terciles ATR%), zone RSI.
2. **Analogues historiques** : on collecte les rendements passés à l'horizon,
   pour les jours partageant le régime courant (contrainte relâchée si trop peu
   d'échantillons). `nNeighbors` = nombre réel d'analogues trouvés.
3. **Repli Monte Carlo** : si < 30 analogues, bootstrap des rendements réels.
4. **Direction** : probabilité de hausse issue des moments empiriques, mélangée
   (30 %) à une **régression logistique** entraînée sur les indicateurs pour les
   horizons journaliers.
5. **Intraday** : mise à l'échelle des moments par la racine du temps
   (`σ·√f`, `μ·f`) — pas de biais arbitraire.
6. **Sortie** : `bins` (4 seuils), `probabilities` (5, somme = 1), `expected`,
   IC 95 %, tous cohérents avec le schéma `QuantumPrediction` existant.

## Validation effectuée

`QuantEngine.kt` a été **compilé avec kotlinc et exécuté** (hors Android, car pur
Kotlin). Invariants vérifiés sur données synthétiques :

- probabilités toujours normalisées à 1.0 ;
- `bins` strictement croissants ;
- direction cohérente (tendance haussière → `pUp` et `expected` montent avec
  l'horizon ; tendance baissière → `pUp` 5j < 0.5) ;
- historique < 120 barres → refus propre (`null`) ;
- série plate → aucun crash, sorties valides ;
- incertitude intraday croissante avec l'horizon (scaling racine du temps).

## Compilation de l'app

Aucune nouvelle dépendance Gradle : `QuantEngine` n'utilise que `kotlin.math`,
et `PriceHistoryRepository` réutilise OkHttp + org.json déjà présents. Ouvrir le
projet dans Android Studio et lancer normalement.

## ⚠️ Avertissement

Les probabilités sont des indicateurs statistiques d'aide à la décision, sans
garantie. Les marchés restent imprévisibles ; la précision directionnelle à court
terme reste proche de 50 %. À afficher comme information, jamais comme certitude.
