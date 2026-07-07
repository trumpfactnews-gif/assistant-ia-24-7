# Scanner Polymarket — Sport, probabilité implicite ≥ 80 %

Outil pour analyser la section **Sport** de Polymarket et extraire les issues
dont le prix (= probabilité implicite) dépasse un seuil, afin de les valider
ensuite avec des statistiques historiques.

## Utilisation

```bash
python3 polymarket_sport_scanner.py                     # seuil 80 %, volume 24h ≥ 1 000 $
python3 polymarket_sport_scanner.py --min-prob 0.85     # seuil 85 %
python3 polymarket_sport_scanner.py --keyword wimbledon # filtrer un tournoi
python3 polymarket_sport_scanner.py --json resultats.json
```

Aucune dépendance : Python 3.8+ suffit. Nécessite un accès réseau à
`gamma-api.polymarket.com`.

## Point structurel important

Polymarket ne propose **pas** de marchés de « props » statistiques
(nombre d'aces, doubles fautes, corners, cartons jaunes, tirs cadrés…).
La section Sport contient essentiellement des marchés **vainqueur
(moneyline)**, quelques **totaux/handicaps** et des **futures** (vainqueur
de tournoi). Un pari « plus de 4 aces pour X » n'existe donc pas sur cette
plateforme — ce type de marché se trouve chez les bookmakers classiques.

La démarche correcte sur Polymarket est donc :

1. Scanner les marchés dont le prix ≥ 0,80 $ (ce script).
2. Valider chaque candidat avec des **données mesurables** externes :
   - Tennis : Tennis Abstract / ATP-WTA stats (taux de jeux de service
     gagnés, % de victoires vs joueurs hors top 50, historique face-à-face).
   - Football : FBref / Opta (xG, corners, cartons par match, forme sur
     10 matchs).
   - Baseball : Baseball Reference / FanGraphs.
3. Ne retenir que les issues où la fréquence historique de l'événement
   dépasse 80 % sur un échantillon suffisant (≥ 10 observations récentes).

## Avertissement

Un prix de 80 ¢ signifie que le marché estime déjà la probabilité à 80 % :
gagner 20 % dans 80 % des cas n'est **pas** un « pari sûr » et n'a une
espérance positive que si votre probabilité réelle, justifiée par les
données, est *supérieure* au prix payé (frais et slippage inclus).
Aucun pari n'est garanti.
