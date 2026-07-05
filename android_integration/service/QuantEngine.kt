package com.example.service

import kotlin.math.abs
import kotlin.math.exp
import kotlin.math.ln
import kotlin.math.sqrt

/**
 * QuantEngine
 * ===========
 *
 * Moteur quantitatif **100 % calcul**, porté depuis le système Python d'analyse
 * probabiliste. Il ne contient aucun biais codé en dur : toutes les probabilités
 * sont dérivées de l'historique réel (5 ans d'OHLCV) fourni en entrée.
 *
 * Pour chaque horizon, il combine trois sources de signal :
 *   1. Analogues historiques — distribution des rendements des jours passés
 *      partageant le régime courant (tendance, volatilité, zone RSI).
 *   2. Régression logistique — entraînée par descente de gradient sur les
 *      indicateurs techniques (direction à 1 séance).
 *   3. Volatilité réalisée — pour la mise à l'échelle des horizons intraday
 *      (racine du temps) et le calcul des bins/CI.
 *
 * La sortie respecte le schéma de QuantumPrediction (bins, probabilities,
 * expected, ci_low, ci_high, nNeighbors) attendu par l'écran Quantique.
 */
object QuantEngine {

    /** Une barre OHLCV quotidienne (déjà ajustée des splits si possible). */
    data class OhlcvBar(
        val time: Long,
        val open: Double,
        val high: Double,
        val low: Double,
        val close: Double,
        val volume: Double
    )

    /** Prévision pour un horizon, compatible avec QuantumPrediction. */
    data class HorizonForecast(
        val bins: List<Double>,           // 4 seuils de rendement (en %)
        val probabilities: List<Double>,  // 5 probabilités, somme = 1.0
        val expected: Double,             // rendement attendu (en %)
        val ciLow: Double,                // borne basse IC 95 % (en %)
        val ciHigh: Double,               // borne haute IC 95 % (en %)
        val pUp: Double,                  // probabilité de hausse (0..1)
        val nNeighbors: Int               // nb d'états analogues réels utilisés
    )

    // Horizons de l'app -> durée en séances de bourse.
    // Les valeurs < 1 correspondent aux horizons intraday (~390 min par séance).
    private val HORIZON_TRADING_DAYS = linkedMapOf(
        "1min" to 1.0 / 390.0,
        "5min" to 5.0 / 390.0,
        "30min" to 30.0 / 390.0,
        "1d" to 1.0,
        "5d" to 5.0
    )

    /**
     * Point d'entrée : analyse un historique et renvoie une prévision par horizon.
     *
     * @param bars historique OHLCV trié chronologiquement (le plus ancien d'abord).
     * @return map horizon -> HorizonForecast, ou null si l'historique est insuffisant.
     */
    fun analyze(bars: List<OhlcvBar>): Map<String, HorizonForecast>? {
        // On exige un minimum d'historique pour des statistiques significatives.
        if (bars.size < 120) return null

        val close = DoubleArray(bars.size) { bars[it].close }
        val high = DoubleArray(bars.size) { bars[it].high }
        val low = DoubleArray(bars.size) { bars[it].low }

        // --- Indicateurs techniques -------------------------------------
        val ema20 = ema(close, 20)
        val ema50 = ema(close, 50)
        val ema200 = ema(close, 200)
        val rsi = rsi(close, 14)
        val macdHist = macdHistogram(close)
        val bbPct = bollingerPctB(close, 20, 2.0)
        val atrPct = atrPercent(high, low, close, 14)
        val mom5 = momentum(close, 5)
        val mom10 = momentum(close, 10)
        val mom20 = momentum(close, 20)
        val distEma20 = distance(close, ema20)
        val distEma50 = distance(close, ema50)
        val distEma200 = distance(close, ema200)

        // Rendements quotidiens (simples) pour les analogues et le Monte Carlo.
        val dailyReturns = DoubleArray(close.size) { i ->
            if (i == 0 || close[i - 1] == 0.0) 0.0 else close[i] / close[i - 1] - 1.0
        }

        val n = close.size
        val last = n - 1

        // --- Régime courant (dernière séance) ---------------------------
        val curTrend = trendBucket(close[last], ema50[last], ema200[last])
        val volTerciles = terciles(atrPct)
        val curVol = volBucket(atrPct[last], volTerciles.first, volTerciles.second)
        val curRsiB = rsiBucket(rsi[last])

        // --- Régime historique jour par jour (pour l'appariement) -------
        val trendHist = IntArray(n) { trendBucket(close[it], ema50[it], ema200[it]) }
        val volHist = IntArray(n) { volBucket(atrPct[it], volTerciles.first, volTerciles.second) }
        val rsiHist = IntArray(n) { rsiBucket(rsi[it]) }

        // --- Régression logistique (direction à 1 séance) ---------------
        val logisticUp = trainAndPredictLogistic(
            features = arrayOf(rsi, macdHist, bbPct, atrPct, mom5, mom10, mom20, distEma20, distEma50, distEma200),
            close = close,
            startIndex = 200
        )

        // --- Prévision par horizon --------------------------------------
        val result = LinkedHashMap<String, HorizonForecast>()
        for ((label, hDays) in HORIZON_TRADING_DAYS) {
            result[label] = forecastHorizon(
                hDays = hDays,
                close = close,
                dailyReturns = dailyReturns,
                trendHist = trendHist, volHist = volHist, rsiHist = rsiHist,
                curTrend = curTrend, curVol = curVol, curRsi = curRsiB,
                logisticUp = logisticUp
            )
        }
        return result
    }

    // ================================================================== //
    //  Construction d'une prévision pour un horizon
    // ================================================================== //
    private fun forecastHorizon(
        hDays: Double,
        close: DoubleArray,
        dailyReturns: DoubleArray,
        trendHist: IntArray, volHist: IntArray, rsiHist: IntArray,
        curTrend: Int, curVol: Int, curRsi: Int,
        logisticUp: Double?
    ): HorizonForecast {
        // Horizon entier utilisé pour l'échantillon d'analogues (>= 1 séance).
        val baseDays = if (hDays < 1.0) 1 else Math.round(hDays).toInt()

        // 1) Échantillon d'analogues : rendements cumulés à baseDays, régime courant.
        val analog = collectAnalogReturns(close, trendHist, volHist, rsiHist, curTrend, curVol, curRsi, baseDays)
        var sample = analog.first
        val nNeighbors = analog.second

        // Repli Monte Carlo si trop peu d'analogues : bootstrap des rendements réels.
        if (sample.size < 30) {
            sample = monteCarloReturns(dailyReturns, baseDays, 4000)
        }

        // 2) Moments empiriques (en fraction), puis conversion en pourcentage.
        var mean = mean(sample)
        var sd = std(sample, mean)
        if (sd <= 1e-9) sd = 1e-4

        // 3) Mise à l'échelle intraday par la racine du temps (si horizon < 1 jour).
        if (hDays < 1.0) {
            val f = hDays // fraction de séance
            mean *= f
            sd *= sqrt(f)
        }

        // 4) Probabilité de hausse empirique, mélangée avec la logistique (prior).
        //    Le prior logistique est un signal *directionnel journalier* : on ne
        //    l'injecte que sur les horizons journaliers courts (1d/2d), jamais sur
        //    l'intraday (qui reste un pur scaling racine-du-temps des moments).
        var pUp = normalCdf(mean / sd, 0.0, 1.0) // P(rendement > 0) sous l'hypothèse normale des moments
        if (logisticUp != null && hDays >= 1.0 && baseDays <= 2) {
            pUp = 0.70 * pUp + 0.30 * logisticUp
        }
        pUp = pUp.coerceIn(0.02, 0.98)

        // 5) Moyenne ajustée pour rester cohérente avec pUp mélangée.
        val meanAdj = sd * invNormalCdf(pUp)

        // 6) Bins symétriques autour de 0 (en %), échelle = écart-type.
        val sdPct = sd * 100.0
        val meanPct = meanAdj * 100.0
        val rawBins = listOf(-1.5 * sdPct, -0.5 * sdPct, 0.5 * sdPct, 1.5 * sdPct)
        val bins = monotonic(rawBins)

        // 7) Probabilités sur les 5 régions via la CDF normale (moyenne = meanPct).
        val probs = probabilitiesFromNormal(bins, meanPct, sdPct)

        // 8) Intervalle de confiance à 95 %.
        val ciLow = meanPct - 1.96 * sdPct
        val ciHigh = meanPct + 1.96 * sdPct

        return HorizonForecast(
            bins = bins,
            probabilities = probs,
            expected = meanPct,
            ciLow = ciLow,
            ciHigh = ciHigh,
            pUp = pUp,
            nNeighbors = nNeighbors
        )
    }

    /**
     * Collecte les rendements cumulés à [hDays] séances pour les jours passés
     * partageant le régime courant. Relâche progressivement les contraintes
     * (RSI puis volatilité) si l'échantillon est trop faible.
     *
     * @return (échantillon de rendements, nombre d'analogues réels trouvés).
     */
    private fun collectAnalogReturns(
        close: DoubleArray,
        trendHist: IntArray, volHist: IntArray, rsiHist: IntArray,
        curTrend: Int, curVol: Int, curRsi: Int,
        hDays: Int
    ): Pair<DoubleArray, Int> {
        // Niveaux de contrainte décroissants.
        val levels = listOf(3, 2, 1)
        for (level in levels) {
            val out = ArrayList<Double>()
            val limit = close.size - hDays
            for (i in 0 until limit) {
                if (trendHist[i] != curTrend) continue
                if (level >= 2 && volHist[i] != curVol) continue
                if (level >= 3 && rsiHist[i] != curRsi) continue
                if (close[i] == 0.0) continue
                out.add(close[i + hDays] / close[i] - 1.0)
            }
            if (out.size >= 30) {
                return Pair(out.toDoubleArray(), out.size)
            }
            // Au niveau le plus bas, on renvoie ce qu'on a (même si < 30).
            if (level == 1) {
                return Pair(out.toDoubleArray(), out.size)
            }
        }
        return Pair(DoubleArray(0), 0)
    }

    /**
     * Simulation Monte Carlo : rééchantillonne les rendements log quotidiens
     * réels et compose sur [hDays] séances. Renvoie les rendements simples cumulés.
     */
    private fun monteCarloReturns(dailyReturns: DoubleArray, hDays: Int, nSims: Int): DoubleArray {
        // Passe en log-rendements pour une composition additive stable.
        val logs = ArrayList<Double>(dailyReturns.size)
        for (i in 1 until dailyReturns.size) {
            val r = dailyReturns[i]
            if (r > -0.99) logs.add(ln(1.0 + r))
        }
        if (logs.isEmpty()) return DoubleArray(0)

        // Générateur déterministe (reproductibilité) — pas de dépendance à l'heure.
        var state = 0x2545F4914F6CDD1DL
        fun nextIndex(bound: Int): Int {
            // xorshift64
            state = state xor (state shl 13)
            state = state xor (state ushr 7)
            state = state xor (state shl 17)
            val v = (state ushr 1).toInt()
            return (v % bound + bound) % bound
        }

        val out = DoubleArray(nSims)
        for (s in 0 until nSims) {
            var acc = 0.0
            for (k in 0 until hDays) acc += logs[nextIndex(logs.size)]
            out[s] = exp(acc) - 1.0 // retour au rendement simple
        }
        return out
    }

    // ================================================================== //
    //  Régression logistique (descente de gradient, standardisation)
    // ================================================================== //
    /**
     * Entraîne une régression logistique sur la direction à 1 séance et renvoie
     * la probabilité de hausse prédite pour la dernière observation.
     * Renvoie null si l'échantillon est insuffisant.
     */
    private fun trainAndPredictLogistic(
        features: Array<DoubleArray>,
        close: DoubleArray,
        startIndex: Int
    ): Double? {
        val n = close.size
        val m = features.size
        if (n - 1 - startIndex < 100) return null

        // Construction de la matrice X et de la cible y (direction à J+1).
        val rows = ArrayList<DoubleArray>()
        val ys = ArrayList<Double>()
        for (i in startIndex until n - 1) {
            val row = DoubleArray(m)
            var ok = true
            for (j in 0 until m) {
                val v = features[j][i]
                if (v.isNaN() || v.isInfinite()) { ok = false; break }
                row[j] = v
            }
            if (!ok) continue
            rows.add(row)
            ys.add(if (close[i + 1] > close[i]) 1.0 else 0.0)
        }
        if (rows.size < 100) return null

        // Standardisation par colonne.
        val means = DoubleArray(m)
        val stds = DoubleArray(m)
        for (j in 0 until m) {
            var s = 0.0
            for (r in rows) s += r[j]
            val mu = s / rows.size
            var v = 0.0
            for (r in rows) v += (r[j] - mu) * (r[j] - mu)
            means[j] = mu
            stds[j] = sqrt(v / rows.size).coerceAtLeast(1e-9)
        }

        // Descente de gradient (avec petite régularisation L2).
        val w = DoubleArray(m + 1) // w[0] = biais
        val lr = 0.1
        val l2 = 1e-3
        val iters = 400
        val nRows = rows.size
        for (it in 0 until iters) {
            val grad = DoubleArray(m + 1)
            for (idx in 0 until nRows) {
                val r = rows[idx]
                var z = w[0]
                for (j in 0 until m) z += w[j + 1] * ((r[j] - means[j]) / stds[j])
                val p = 1.0 / (1.0 + exp(-z))
                val err = p - ys[idx]
                grad[0] += err
                for (j in 0 until m) grad[j + 1] += err * ((r[j] - means[j]) / stds[j])
            }
            w[0] -= lr * grad[0] / nRows
            for (j in 1..m) w[j] -= lr * (grad[j] / nRows + l2 * w[j])
        }

        // Prédiction sur la dernière ligne complète.
        val lastRow = features.map { it[n - 1] }
        var z = w[0]
        for (j in 0 until m) {
            val v = lastRow[j]
            if (v.isNaN() || v.isInfinite()) return null
            z += w[j + 1] * ((v - means[j]) / stds[j])
        }
        return 1.0 / (1.0 + exp(-z))
    }

    // ================================================================== //
    //  Indicateurs techniques
    // ================================================================== //
    private fun ema(x: DoubleArray, span: Int): DoubleArray {
        val out = DoubleArray(x.size)
        if (x.isEmpty()) return out
        val a = 2.0 / (span + 1.0)
        out[0] = x[0]
        for (i in 1 until x.size) out[i] = a * x[i] + (1 - a) * out[i - 1]
        return out
    }

    private fun rsi(close: DoubleArray, period: Int): DoubleArray {
        val out = DoubleArray(close.size) { Double.NaN }
        if (close.size < period + 1) return out
        var avgGain = 0.0
        var avgLoss = 0.0
        for (i in 1..period) {
            val d = close[i] - close[i - 1]
            if (d > 0) avgGain += d else avgLoss += -d
        }
        avgGain /= period
        avgLoss /= period
        out[period] = if (avgLoss == 0.0) 100.0 else 100.0 - 100.0 / (1.0 + avgGain / avgLoss)
        for (i in period + 1 until close.size) {
            val d = close[i] - close[i - 1]
            val g = if (d > 0) d else 0.0
            val l = if (d < 0) -d else 0.0
            avgGain = (avgGain * (period - 1) + g) / period
            avgLoss = (avgLoss * (period - 1) + l) / period
            out[i] = if (avgLoss == 0.0) 100.0 else 100.0 - 100.0 / (1.0 + avgGain / avgLoss)
        }
        // Comble le début par la première valeur valide pour éviter les NaN en features.
        val firstValid = out.indexOfFirst { !it.isNaN() }
        if (firstValid > 0) for (i in 0 until firstValid) out[i] = 50.0
        return out
    }

    private fun macdHistogram(close: DoubleArray): DoubleArray {
        val fast = ema(close, 12)
        val slow = ema(close, 26)
        val macd = DoubleArray(close.size) { fast[it] - slow[it] }
        val signal = ema(macd, 9)
        return DoubleArray(close.size) { macd[it] - signal[it] }
    }

    /** %B des bandes de Bollinger : 0 = bande basse, 1 = bande haute. */
    private fun bollingerPctB(close: DoubleArray, window: Int, nStd: Double): DoubleArray {
        val out = DoubleArray(close.size) { 0.5 }
        for (i in close.indices) {
            if (i < window - 1) continue
            var s = 0.0
            for (k in i - window + 1..i) s += close[k]
            val mean = s / window
            var v = 0.0
            for (k in i - window + 1..i) v += (close[k] - mean) * (close[k] - mean)
            val sd = sqrt(v / window)
            if (sd <= 1e-12) { out[i] = 0.5; continue }
            val upper = mean + nStd * sd
            val lower = mean - nStd * sd
            out[i] = ((close[i] - lower) / (upper - lower)).coerceIn(-0.5, 1.5)
        }
        return out
    }

    /** ATR normalisé par le prix (volatilité relative). */
    private fun atrPercent(high: DoubleArray, low: DoubleArray, close: DoubleArray, period: Int): DoubleArray {
        val n = close.size
        val tr = DoubleArray(n)
        tr[0] = high[0] - low[0]
        for (i in 1 until n) {
            val a = high[i] - low[i]
            val b = abs(high[i] - close[i - 1])
            val c = abs(low[i] - close[i - 1])
            tr[i] = maxOf(a, b, c)
        }
        val atr = ema(tr, period) // lissage exponentiel (proxy de Wilder)
        return DoubleArray(n) { if (close[it] != 0.0) atr[it] / close[it] else 0.0 }
    }

    private fun momentum(close: DoubleArray, w: Int): DoubleArray {
        return DoubleArray(close.size) { i ->
            if (i < w || close[i - w] == 0.0) 0.0 else close[i] / close[i - w] - 1.0
        }
    }

    private fun distance(close: DoubleArray, ma: DoubleArray): DoubleArray {
        return DoubleArray(close.size) { i ->
            if (ma[i] != 0.0) (close[i] - ma[i]) / ma[i] else 0.0
        }
    }

    // ================================================================== //
    //  Régimes / buckets
    // ================================================================== //
    private fun trendBucket(close: Double, ema50: Double, ema200: Double): Int {
        if (ema200 == 0.0) return 0
        if (close > ema50 && ema50 > ema200) return 1   // haussière
        if (close < ema50 && ema50 < ema200) return -1  // baissière
        return 0                                         // latérale
    }

    private fun rsiBucket(rsi: Double): Int {
        if (rsi.isNaN()) return 1
        if (rsi < 30) return 0
        if (rsi < 50) return 1
        if (rsi < 70) return 2
        return 3
    }

    private fun volBucket(v: Double, low: Double, high: Double): Int {
        if (v.isNaN()) return 1
        if (v <= low) return 0
        if (v >= high) return 2
        return 1
    }

    /** Terciles (33 % / 66 %) d'un tableau, en ignorant les NaN. */
    private fun terciles(x: DoubleArray): Pair<Double, Double> {
        val vals = x.filter { !it.isNaN() }.sorted()
        if (vals.isEmpty()) return Pair(0.0, 0.0)
        val lo = vals[(vals.size * 0.33).toInt().coerceIn(0, vals.size - 1)]
        val hi = vals[(vals.size * 0.66).toInt().coerceIn(0, vals.size - 1)]
        return Pair(lo, hi)
    }

    // ================================================================== //
    //  Statistiques & distributions
    // ================================================================== //
    private fun mean(x: DoubleArray): Double {
        if (x.isEmpty()) return 0.0
        var s = 0.0
        for (v in x) s += v
        return s / x.size
    }

    private fun std(x: DoubleArray, mean: Double): Double {
        if (x.size < 2) return 0.0
        var v = 0.0
        for (e in x) v += (e - mean) * (e - mean)
        return sqrt(v / (x.size - 1))
    }

    /** Garantit des seuils strictement croissants (bins). */
    private fun monotonic(bins: List<Double>): List<Double> {
        val out = bins.toMutableList()
        for (i in 1 until out.size) {
            if (out[i] <= out[i - 1]) out[i] = out[i - 1] + 1e-6
        }
        return out
    }

    /**
     * Probabilités sur les 5 régions définies par 4 bins, sous une loi normale
     * de moyenne [mean] et d'écart-type [std]. Somme normalisée à 1.0.
     */
    private fun probabilitiesFromNormal(bins: List<Double>, mean: Double, std: Double): List<Double> {
        val s = if (std <= 1e-9) 1e-6 else std
        val raw = ArrayList<Double>(bins.size + 1)
        raw.add(normalCdf(bins[0], mean, s))
        for (i in 0 until bins.size - 1) {
            raw.add((normalCdf(bins[i + 1], mean, s) - normalCdf(bins[i], mean, s)).coerceAtLeast(0.001))
        }
        raw.add((1.0 - normalCdf(bins.last(), mean, s)).coerceAtLeast(0.001))
        val sum = raw.sum()
        return raw.map { it / sum }
    }

    /** CDF de la loi normale via une approximation de la fonction d'erreur. */
    private fun normalCdf(x: Double, mean: Double, std: Double): Double {
        val z = (x - mean) / std
        return 0.5 * (1.0 + erf(z / sqrt(2.0)))
    }

    /** Fonction d'erreur (Abramowitz & Stegun 7.1.26). */
    private fun erf(x: Double): Double {
        val sign = if (x < 0) -1.0 else 1.0
        val ax = abs(x)
        val t = 1.0 / (1.0 + 0.3275911 * ax)
        val y = 1.0 - (((((1.061405429 * t - 1.453152027) * t) + 1.421413741) * t - 0.284496736) * t + 0.254829592) * t * exp(-ax * ax)
        return sign * y
    }

    /** Inverse de la CDF normale standard (algorithme d'Acklam). */
    private fun invNormalCdf(p: Double): Double {
        val pp = p.coerceIn(1e-6, 1 - 1e-6)
        val a = doubleArrayOf(-3.969683028665376e+01, 2.209460984245205e+02, -2.759285104469687e+02, 1.383577518672690e+02, -3.066479806614716e+01, 2.506628277459239e+00)
        val b = doubleArrayOf(-5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02, 6.680131188771972e+01, -1.328068155288572e+01)
        val c = doubleArrayOf(-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00, -2.549732539343734e+00, 4.374664141464968e+00, 2.938163982698783e+00)
        val d = doubleArrayOf(7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00, 3.754408661907416e+00)
        val plow = 0.02425
        val phigh = 1 - plow
        return when {
            pp < plow -> {
                val q = sqrt(-2.0 * ln(pp))
                (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) /
                    ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0)
            }
            pp <= phigh -> {
                val q = pp - 0.5
                val r = q * q
                (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q /
                    (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1.0)
            }
            else -> {
                val q = sqrt(-2.0 * ln(1.0 - pp))
                -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) /
                    ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0)
            }
        }
    }
}
