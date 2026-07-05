#!/usr/bin/env python3
"""
api.py
======

Mini-API HTTP exposant la prédiction 24h, à déployer sur le VPS pour alimenter
la page FLUXEL (/marches).

Endpoints
---------
    GET /health
        -> {"ok": true, "service": "fluxel-predict-24h"}

    GET /predict?ticker=AAPL
        -> résultat 24h en JSON (voir price_predictor.predict_24h)

Caractéristiques
----------------
    - CORS activé (la page web peut appeler l'API depuis le navigateur) ;
    - cache mémoire avec expiration (TTL) pour ne pas re-télécharger les données
      à chaque requête (utile car yfinance est lent).

Lancement (développement) :
    python api.py                 # écoute sur http://0.0.0.0:8000

Lancement (production, recommandé) :
    pip install gunicorn
    gunicorn -w 2 -b 0.0.0.0:8000 api:app
"""

from __future__ import annotations

import os
import time
from threading import Lock

from flask import Flask, jsonify, request

from price_predictor.predict_24h import predict_24h

app = Flask(__name__)

# --- Cache mémoire simple avec TTL --------------------------------------
# On garde le dernier résultat par ticker pendant CACHE_TTL secondes.
CACHE_TTL = int(os.environ.get("CACHE_TTL", "900"))  # 15 min par défaut
_cache: dict[str, tuple[float, dict]] = {}
_cache_lock = Lock()


def _get_cached(ticker: str) -> dict:
    """Renvoie le résultat en cache s'il est encore frais, sinon recalcule."""
    now = time.time()
    with _cache_lock:
        hit = _cache.get(ticker)
        if hit and (now - hit[0]) < CACHE_TTL:
            return hit[1]

    # Calcul hors verrou (peut être long) puis mise en cache.
    result = predict_24h(ticker)
    with _cache_lock:
        _cache[ticker] = (now, result)
    return result


@app.after_request
def _add_cors_headers(response):
    """Autorise les appels depuis le navigateur (page FLUXEL)."""
    response.headers["Access-Control-Allow-Origin"] = "*"
    response.headers["Access-Control-Allow-Methods"] = "GET, OPTIONS"
    response.headers["Access-Control-Allow-Headers"] = "Content-Type"
    return response


@app.route("/health", methods=["GET"])
def health():
    """Vérification de vie du service."""
    return jsonify({"ok": True, "service": "fluxel-predict-24h"})


@app.route("/predict", methods=["GET", "OPTIONS"])
def predict():
    """
    Prédiction 24h.

    Paramètre de requête :
        ticker (obligatoire) : symbole à analyser, ex. ?ticker=AAPL
    """
    if request.method == "OPTIONS":  # pré-vol CORS
        return ("", 204)

    ticker = (request.args.get("ticker") or "").strip()
    if not ticker:
        return jsonify({"ok": False, "erreur": "Paramètre 'ticker' manquant."}), 400

    result = _get_cached(ticker.upper())
    status = 200 if result.get("ok") else 502
    return jsonify(result), status


if __name__ == "__main__":
    port = int(os.environ.get("PORT", "8000"))
    # host 0.0.0.0 pour être joignable depuis l'extérieur du VPS.
    app.run(host="0.0.0.0", port=port)
