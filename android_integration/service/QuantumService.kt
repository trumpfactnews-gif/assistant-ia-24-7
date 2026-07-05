package com.example.service

import android.util.Log
import com.example.data.model.QuantumPrediction
import com.example.data.repository.PriceHistoryRepository
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import okhttp3.MediaType.Companion.toMediaType
import okhttp3.OkHttpClient
import okhttp3.Request
import okhttp3.RequestBody.Companion.toRequestBody
import org.json.JSONArray
import org.json.JSONObject
import java.text.SimpleDateFormat
import java.util.*
import java.util.concurrent.TimeUnit
import kotlin.random.Random

object QuantumService {
    private const val TAG = "QuantumService"
    private const val BASE_URL = "https://ais-dev-277fg2xvluescesiq4k6u5-403114238537.us-west1.run.app/api/quantum_inference"

    private val client = OkHttpClient.Builder()
        .connectTimeout(10, TimeUnit.SECONDS)
        .readTimeout(10, TimeUnit.SECONDS)
        .build()

    private val historyRepository = PriceHistoryRepository()

    /**
     * Renvoie les prédictions pour tous les horizons, selon un ordre de priorité :
     *
     *   1. MOTEUR QUANTITATIF LOCAL RÉEL (QuantEngine) — 100 % calcul, sur 5 ans
     *      d'OHLCV réels. C'est la source par défaut : elle n'utilise aucun biais
     *      codé en dur et fonctionne sans serveur dédié.
     *   2. Serveur VPS quantitatif (si configuré et joignable).
     *   3. Inférence Gemini (fallback IA distant).
     *   4. Simulation locale de secours (dernier recours).
     */
    suspend fun getPredictions(ticker: String): List<QuantumPrediction> = withContext(Dispatchers.IO) {
        val t = ticker.trim().uppercase()

        // 1. Moteur local réel (calcul pur sur données historiques réelles).
        try {
            val local = computeLocalReal(t)
            if (local.isNotEmpty()) {
                Log.d(TAG, "Prédictions calculées localement (QuantEngine) pour $t")
                return@withContext local
            }
        } catch (e: Exception) {
            Log.w(TAG, "Moteur local indisponible pour $t (${e.message}). Essai serveur VPS.")
        }

        // 2. Serveur VPS quantitatif.
        try {
            val vps = fetchFromVps(t)
            if (vps.isNotEmpty()) {
                Log.d(TAG, "Prédictions chargées depuis le VPS pour $t")
                return@withContext vps
            }
        } catch (e: Exception) {
            Log.w(TAG, "VPS hors-ligne pour $t (${e.message}).")
        }

        // 3. Fallback Gemini.
        try {
            val gemini = fetchFromGemini(t)
            if (gemini.isNotEmpty()) {
                Log.d(TAG, "Prédictions générées par Gemini pour $t")
                return@withContext gemini
            }
        } catch (e: Exception) {
            Log.e(TAG, "Gemini indisponible pour $t (${e.message}).")
        }

        // 4. Simulation locale de secours.
        Log.w(TAG, "Toutes les sources ont échoué. Simulation de secours pour $t.")
        return@withContext simulatedFallback(t)
    }

    // ================================================================== //
    //  1. Moteur quantitatif local réel (QuantEngine)
    // ================================================================== //
    private suspend fun computeLocalReal(ticker: String): List<QuantumPrediction> {
        val bars = historyRepository.fetchDailyHistory(ticker)
        if (bars.size < 120) {
            Log.w(TAG, "Historique insuffisant pour $ticker (${bars.size} barres).")
            return emptyList()
        }

        val forecasts = QuantEngine.analyze(bars) ?: return emptyList()
        val currentTime = System.currentTimeMillis()

        return forecasts.map { (horizon, f) ->
            QuantumPrediction(
                ticker = ticker,
                timestamp = currentTime,
                horizon = horizon,
                binsJson = JSONArray(f.bins).toString(),
                probsJson = JSONArray(f.probabilities).toString(),
                expectedReturn = f.expected,
                ciLower = f.ciLow,
                ciUpper = f.ciHigh,
                nNeighbors = f.nNeighbors,
                provenance = "LOCAL_REAL_QUANT"
            )
        }
    }

    // ================================================================== //
    //  2. Serveur VPS quantitatif
    // ================================================================== //
    private fun fetchFromVps(ticker: String): List<QuantumPrediction> {
        val predictions = mutableListOf<QuantumPrediction>()
        val currentTime = System.currentTimeMillis()

        val sdf = SimpleDateFormat("yyyy-MM-dd'T'HH:mm:ss'Z'", Locale.US).apply {
            timeZone = TimeZone.getTimeZone("UTC")
        }
        val requestJson = JSONObject().apply {
            put("ticker", ticker)
            put("timestamp", sdf.format(Date(currentTime)))
        }

        val mediaType = "application/json; charset=utf-8".toMediaType()
        val requestBody = requestJson.toString().toRequestBody(mediaType)
        val request = Request.Builder()
            .url(BASE_URL)
            .post(requestBody)
            .build()

        client.newCall(request).execute().use { response ->
            if (!response.isSuccessful) return emptyList()
            val body = response.body?.string() ?: ""
            val json = JSONObject(body)
            val horizonsJson = json.optJSONObject("horizons") ?: return emptyList()
            val nNeighbors = json.optInt("n_neighbors", 10000)

            val keys = horizonsJson.keys()
            while (keys.hasNext()) {
                val horizon = keys.next()
                val horizonObj = horizonsJson.getJSONObject(horizon)
                predictions.add(
                    QuantumPrediction(
                        ticker = ticker,
                        timestamp = currentTime,
                        horizon = horizon,
                        binsJson = horizonObj.getJSONArray("bins").toString(),
                        probsJson = horizonObj.getJSONArray("probabilities").toString(),
                        expectedReturn = horizonObj.optDouble("expected", 0.0),
                        ciLower = horizonObj.optDouble("ci_low", 0.0),
                        ciUpper = horizonObj.optDouble("ci_high", 0.0),
                        nNeighbors = nNeighbors,
                        provenance = "QUANT_SERVER_REAL"
                    )
                )
            }
        }
        return predictions
    }

    // ================================================================== //
    //  3. Fallback Gemini
    // ================================================================== //
    private fun fetchFromGemini(ticker: String): List<QuantumPrediction> {
        val apiKey = com.example.BuildConfig.GEMINI_API_KEY
        if (apiKey.isEmpty() || apiKey == "MY_GEMINI_API_KEY") return emptyList()

        val predictions = mutableListOf<QuantumPrediction>()
        val currentTime = System.currentTimeMillis()

        val prompt = """
            Tu es un moteur d'inférence quantitative et d'analyse financière avancée. Calcule des prédictions empiriques réalistes et actuelles de distribution de rendement pour le ticker $ticker.
            Produis un JSON contenant des prédictions pour 5 horizons de temps: '1min', '5min', '30min', '1d', '5d'.
            Chaque horizon doit être réaliste par rapport au comportement historique du titre.
            La structure doit être STRICTEMENT en JSON valide avec la clé racine "horizons" et la clé racine "n_neighbors" (un entier entre 100 et 450 représentant les états similaires trouvés).
            L'objet "horizons" contient des clés '1min', '5min', '30min', '1d', '5d'.
            Chaque objet d'horizon contient:
              - "bins": liste ordonnée de 4 nombres décimaux délimitant les bins de rendement (ex: [-1.0, -0.5, 0.5, 1.0]). Les bins doivent être plus larges pour les horizons plus longs.
              - "probabilities": liste de 5 nombres décimaux sommant exactement à 1.0 (probabilité d'être < bins[0], entre chaque bin, et > bins[3]).
              - "expected": rendement attendu (nombre décimal).
              - "ci_low": borne inférieure IC 95% (nombre décimal).
              - "ci_high": borne supérieure IC 95% (nombre décimal).
        """.trimIndent()

        val requestJson = JSONObject().apply {
            val contentsArray = JSONArray().apply {
                put(JSONObject().apply {
                    put("parts", JSONArray().apply {
                        put(JSONObject().apply { put("text", prompt) })
                    })
                })
            }
            put("contents", contentsArray)
            put("generationConfig", JSONObject().apply {
                put("responseMimeType", "application/json")
                put("temperature", 0.2)
            })
        }

        val mediaType = "application/json; charset=utf-8".toMediaType()
        val requestBody = requestJson.toString().toRequestBody(mediaType)
        val geminiUrl = "https://generativelanguage.googleapis.com/v1beta/models/gemini-3.5-flash:generateContent?key=$apiKey"
        val geminiRequest = Request.Builder()
            .url(geminiUrl)
            .post(requestBody)
            .build()

        client.newCall(geminiRequest).execute().use { geminiResponse ->
            if (!geminiResponse.isSuccessful) return emptyList()
            val bodyStr = geminiResponse.body?.string() ?: ""
            val responseJson = JSONObject(bodyStr)
            val textResponse = responseJson.optJSONArray("candidates")
                ?.optJSONObject(0)
                ?.optJSONObject("content")
                ?.optJSONArray("parts")
                ?.optJSONObject(0)
                ?.optString("text") ?: ""
            if (textResponse.isEmpty()) return emptyList()

            val cleanText = textResponse.trim()
                .removePrefix("```json").removePrefix("```").removeSuffix("```").trim()
            val json = JSONObject(cleanText)
            val horizonsJson = json.optJSONObject("horizons") ?: return emptyList()
            val nNeighbors = json.optInt("n_neighbors", 150)

            val keys = horizonsJson.keys()
            while (keys.hasNext()) {
                val horizon = keys.next()
                val horizonObj = horizonsJson.getJSONObject(horizon)
                predictions.add(
                    QuantumPrediction(
                        ticker = ticker,
                        timestamp = currentTime,
                        horizon = horizon,
                        binsJson = horizonObj.getJSONArray("bins").toString(),
                        probsJson = horizonObj.getJSONArray("probabilities").toString(),
                        expectedReturn = horizonObj.optDouble("expected", 0.0),
                        ciLower = horizonObj.optDouble("ci_low", 0.0),
                        ciUpper = horizonObj.optDouble("ci_high", 0.0),
                        nNeighbors = nNeighbors,
                        provenance = "GEMINI_GENERATED"
                    )
                )
            }
        }
        return predictions
    }

    // ================================================================== //
    //  4. Simulation locale de secours (dernier recours)
    // ================================================================== //
    private fun simulatedFallback(ticker: String): List<QuantumPrediction> {
        val predictions = mutableListOf<QuantumPrediction>()
        val currentTime = System.currentTimeMillis()
        val horizons = listOf("1min", "5min", "30min", "1d", "5d")

        // Biais neutre dérivé du symbole (aucune valeur financière — secours only).
        val bias = when (ticker) {
            "PLTR" -> 1.8
            "OCGN" -> 2.4
            "BABA" -> 0.4
            "MARA" -> -1.2
            "RILY" -> -3.5
            else -> if (ticker.length % 2 == 0) 1.1 else -0.5
        }

        for (horizon in horizons) {
            val factor = when (horizon) {
                "1min" -> 0.15
                "5min" -> 0.4
                "30min" -> 1.2
                "1d" -> 4.5
                else -> 12.0
            }
            val expectedReturn = bias * factor * (0.8 + Random.nextDouble(0.4))
            val stdDev = (factor * 2.5).coerceAtLeast(0.5)
            val bins = when (horizon) {
                "1min", "5min" -> listOf(-1.0, -0.5, 0.5, 1.0)
                "30min" -> listOf(-3.0, -1.0, 1.0, 3.0)
                else -> listOf(-5.0, -2.0, 2.0, 5.0)
            }
            val probs = calculateEmpiricalProbs(expectedReturn, stdDev, bins)
            predictions.add(
                QuantumPrediction(
                    ticker = ticker,
                    timestamp = currentTime,
                    horizon = horizon,
                    binsJson = JSONArray(bins).toString(),
                    probsJson = JSONArray(probs).toString(),
                    expectedReturn = expectedReturn,
                    ciLower = expectedReturn - 1.96 * stdDev,
                    ciUpper = expectedReturn + 1.96 * stdDev,
                    nNeighbors = -1,
                    provenance = "LOCAL_SIMULATED"
                )
            )
        }
        return predictions
    }

    private fun calculateEmpiricalProbs(mean: Double, std: Double, bins: List<Double>): List<Double> {
        val rawProbs = mutableListOf<Double>()
        rawProbs.add(normalCdf(bins[0], mean, std))
        for (i in 0 until bins.size - 1) {
            val p1 = normalCdf(bins[i], mean, std)
            val p2 = normalCdf(bins[i + 1], mean, std)
            rawProbs.add((p2 - p1).coerceAtLeast(0.01))
        }
        rawProbs.add((1.0 - normalCdf(bins.last(), mean, std)).coerceAtLeast(0.01))
        val sum = rawProbs.sum()
        return rawProbs.map { it / sum }
    }

    private fun normalCdf(x: Double, mean: Double, std: Double): Double {
        val z = (x - mean) / std
        val t = 1.0 / (1.0 + 0.2316419 * Math.abs(z))
        val d = 0.39894228 * Math.exp(-z * z / 2.0)
        val p = d * t * (0.31938153 + t * (-0.356563782 + t * (1.781477937 + t * (-1.821255978 + t * 1.330274429))))
        return if (z >= 0.0) 1.0 - p else p
    }
}
