#!/usr/bin/env python3
"""Scanner Polymarket — section Sport.

Récupère tous les événements sportifs actifs via l'API Gamma, puis liste
chaque issue (outcome) dont la probabilité implicite (prix) dépasse un
seuil donné (défaut : 80 %). Le résultat sert de liste de candidats à
valider ensuite avec des statistiques historiques externes.

Aucune dépendance externe : Python 3.8+ (stdlib uniquement).

Usage :
    python3 polymarket_sport_scanner.py
    python3 polymarket_sport_scanner.py --min-prob 0.85 --keyword wimbledon
    python3 polymarket_sport_scanner.py --min-volume 10000 --json out.json
"""

import argparse
import json
import sys
import urllib.parse
import urllib.request

GAMMA = "https://gamma-api.polymarket.com"


def http_get(url: str):
    req = urllib.request.Request(url, headers={"User-Agent": "sport-scanner/1.0"})
    with urllib.request.urlopen(req, timeout=30) as resp:
        return json.loads(resp.read().decode())


def fetch_sport_events(limit_pages: int = 30):
    """Pagine /events filtré sur le tag « sports » (événements ouverts)."""
    events, offset = [], 0
    while limit_pages > 0:
        params = urllib.parse.urlencode({
            "closed": "false",
            "active": "true",
            "archived": "false",
            "tag_slug": "sports",
            "limit": 100,
            "offset": offset,
            "order": "volume24hr",
            "ascending": "false",
        })
        batch = http_get(f"{GAMMA}/events?{params}")
        if not batch:
            break
        events.extend(batch)
        if len(batch) < 100:
            break
        offset += 100
        limit_pages -= 1
    return events


def scan(min_prob: float, min_volume: float, keyword: str):
    candidates = []
    for ev in fetch_sport_events():
        title = ev.get("title", "")
        if keyword and keyword.lower() not in json.dumps(ev, default=str).lower():
            continue
        for m in ev.get("markets", []):
            if m.get("closed") or not m.get("active", True):
                continue
            try:
                outcomes = json.loads(m.get("outcomes") or "[]")
                prices = [float(p) for p in json.loads(m.get("outcomePrices") or "[]")]
            except (ValueError, TypeError):
                continue
            vol24 = float(m.get("volume24hr") or 0)
            if vol24 < min_volume:
                continue
            for outcome, price in zip(outcomes, prices):
                if price >= min_prob:
                    candidates.append({
                        "event": title,
                        "market": m.get("question", ""),
                        "outcome": outcome,
                        "prix": price,
                        "prob_implicite_pct": round(price * 100, 1),
                        "volume_24h_usd": round(vol24),
                        "liquidite_usd": round(float(m.get("liquidity") or 0)),
                        "fin": m.get("endDate", ""),
                        "url": f"https://polymarket.com/event/{ev.get('slug', '')}",
                    })
    candidates.sort(key=lambda c: (-c["volume_24h_usd"], -c["prix"]))
    return candidates


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--min-prob", type=float, default=0.80,
                    help="probabilité implicite minimale (défaut 0.80)")
    ap.add_argument("--min-volume", type=float, default=1000,
                    help="volume 24h minimal en USD pour écarter les marchés illiquides")
    ap.add_argument("--keyword", default="",
                    help="filtre texte (ex: wimbledon, mlb, ligue)")
    ap.add_argument("--json", metavar="FICHIER", default="",
                    help="écrire aussi le résultat brut en JSON")
    args = ap.parse_args()

    try:
        candidates = scan(args.min_prob, args.min_volume, args.keyword)
    except urllib.error.URLError as e:
        sys.exit(f"Erreur réseau vers {GAMMA} : {e}. "
                 "Vérifiez que gamma-api.polymarket.com est accessible.")

    if args.json:
        with open(args.json, "w") as f:
            json.dump(candidates, f, ensure_ascii=False, indent=2)

    if not candidates:
        print("Aucun marché sportif au-dessus du seuil avec ce volume minimal.")
        return

    print(f"{len(candidates)} issues ≥ {args.min_prob:.0%} (volume 24h ≥ {args.min_volume:.0f} $)\n")
    for c in candidates[:40]:
        print(f"  {c['prob_implicite_pct']:5.1f}%  {c['event']} — {c['market']}"
              f" → {c['outcome']}  (vol 24h {c['volume_24h_usd']:,} $)")
        print(f"         {c['url']}")


if __name__ == "__main__":
    main()
