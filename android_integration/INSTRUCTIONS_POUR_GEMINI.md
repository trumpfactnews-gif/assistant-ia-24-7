# Instructions d'intégration — à donner à l'app Gemini (AI Studio)

Copie-colle le message ci-dessous à Gemini dans AI Studio, puis ajoute les
4 fichiers joints quand il te le demande.

---

## 📋 Message à coller dans Gemini

> Je veux intégrer un **moteur quantitatif local réel** dans l'onglet Quantique
> de mon app (SEC Analyst Pro). Il remplace le fallback simulé à biais codés en
> dur par un vrai calcul probabiliste sur 5 ans d'OHLCV réels (API Yahoo).
> Effectue exactement ces 4 opérations, sans rien casser d'autre :
>
> **1. AJOUTE un nouveau fichier** `app/src/main/java/com/example/service/QuantEngine.kt`
> (contenu fourni). C'est du Kotlin pur (aucune dépendance nouvelle).
>
> **2. AJOUTE un nouveau fichier** `app/src/main/java/com/example/data/repository/PriceHistoryRepository.kt`
> (contenu fourni). Il utilise OkHttp et org.json, déjà présents.
>
> **3. REMPLACE intégralement** le fichier
> `app/src/main/java/com/example/service/QuantumService.kt` par le contenu fourni.
> La nouvelle version garde exactement les mêmes signatures publiques
> (`suspend fun getPredictions(ticker): List<QuantumPrediction>`) et le même
> modèle de données `QuantumPrediction`. Elle ajoute simplement le moteur local
> comme source prioritaire ; le VPS, Gemini et la simulation restent des
> fallbacks inchangés.
>
> **4. APPLIQUE ces petites modifications** à
> `app/src/main/java/com/example/ui/screens/QuantumScreen.kt` (voir le fichier
> `QuantumScreen.provenance.patch`) : elles ajoutent l'affichage de la nouvelle
> provenance `"LOCAL_REAL_QUANT"` (une carte verte + 2 libellés) et remplacent
> un libellé de badge codé en dur. Ce sont des ajouts de branches `when` et un
> changement de texte, rien de structurel.
>
> Ne modifie aucun autre fichier. Ne change pas le schéma Room ni le modèle
> `QuantumPrediction`. Après intégration, l'app doit compiler telle quelle.

---

## 🔎 Détail des changements

### Nouveau flux (dans `QuantumService.getPredictions`)

Ordre de priorité des sources :

1. **`LOCAL_REAL_QUANT`** — moteur local réel `QuantEngine` (nouveau, par défaut).
2. `QUANT_SERVER_REAL` — serveur VPS (si joignable).
3. `GEMINI_GENERATED` — fallback IA Gemini.
4. `LOCAL_SIMULATED` — simulation de secours (dernier recours).

### Les 4 modifications de `QuantumScreen.kt` (déjà dans le .patch)

1. Nouvelle branche `"LOCAL_REAL_QUANT" -> { … }` dans le `when (provenance)` des
   cartes d'en-tête (carte émeraude « MOTEUR QUANTITATIF LOCAL – CALCUL RÉEL »).
2. Nouvelle ligne dans le `when (provenance)` du libellé de la puce de traçabilité :
   `"LOCAL_REAL_QUANT" -> "CALCUL LOCAL RÉEL (OHLCV 5 ANS)"`.
3. Nouvelle ligne dans le `when (provenance)` du sous-titre :
   `"LOCAL_REAL_QUANT" -> "Basé sur … états historiques analogues réels …"`.
4. Le badge de fraîcheur : `"source: VAE + FAISS | fraîcheur: …"` devient
   `"mise à jour: …"` (l'ancien texte était trompeur pour la source réelle).

> ℹ️ Optionnel : le bloc « Encodeur VAE / Recherche HNSW / FAISS 50 M d'états »
> de la carte méthodologique est un texte hérité affiché en permanence. Tu peux
> demander à Gemini de le rendre cohérent avec la source réelle, mais ce n'est
> pas nécessaire pour que ça fonctionne.

## ✅ Ce qui a été vérifié

`QuantEngine.kt` a été compilé avec le compilateur Kotlin et exécuté : les
probabilités totalisent 100 %, les seuils sont croissants, la direction est
cohérente avec la tendance, l'historique trop court est refusé proprement, et
une série plate ne provoque aucun crash.

## ⚠️ Avertissement

Indicateur statistique d'aide à la décision, sans garantie. La précision
directionnelle à court terme reste proche de 50 % — c'est la réalité des marchés.
À afficher comme information, jamais comme certitude.
