package com.example.data.repository

import android.util.Log
import com.example.service.QuantEngine
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import okhttp3.OkHttpClient
import okhttp3.Request
import org.json.JSONObject
import java.util.concurrent.TimeUnit

/**
 * PriceHistoryRepository
 * ======================
 *
 * Récupère l'historique OHLCV quotidien (5 ans) via l'API Chart de Yahoo Finance,
 * la même source que MarketDataRepository. Les barres sont ajustées des splits
 * (via adjclose) pour être cohérentes avec un historique « auto-adjusted ».
 *
 * Le résultat alimente QuantEngine pour un calcul probabiliste 100 % local.
 */
class PriceHistoryRepository {

    private val client = OkHttpClient.Builder()
        .connectTimeout(15, TimeUnit.SECONDS)
        .readTimeout(20, TimeUnit.SECONDS)
        .build()

    /**
     * Télécharge ~5 ans de barres quotidiennes pour [ticker].
     * @return liste triée (ancien -> récent) ou liste vide en cas d'échec.
     */
    suspend fun fetchDailyHistory(ticker: String): List<QuantEngine.OhlcvBar> = withContext(Dispatchers.IO) {
        val clean = ticker.uppercase().trim()
        val url = "https://query1.finance.yahoo.com/v8/finance/chart/$clean?range=5y&interval=1d"
        val request = Request.Builder()
            .url(url)
            .header(
                "User-Agent",
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
            )
            .build()

        try {
            client.newCall(request).execute().use { response ->
                if (!response.isSuccessful) {
                    Log.w("PriceHistoryRepository", "Chart API HTTP ${response.code} pour $clean")
                    return@withContext emptyList()
                }
                val body = response.body?.string() ?: return@withContext emptyList()
                return@withContext parseChart(body)
            }
        } catch (e: Exception) {
            Log.e("PriceHistoryRepository", "Échec récupération historique $clean: ${e.message}")
            return@withContext emptyList()
        }
    }

    /** Parse la réponse JSON de l'API Chart en barres OHLCV nettoyées. */
    private fun parseChart(body: String): List<QuantEngine.OhlcvBar> {
        val root = JSONObject(body)
        val result = root.optJSONObject("chart")
            ?.optJSONArray("result")
            ?.optJSONObject(0) ?: return emptyList()

        val timestamps = result.optJSONArray("timestamp") ?: return emptyList()
        val indicators = result.optJSONObject("indicators") ?: return emptyList()
        val quote = indicators.optJSONArray("quote")?.optJSONObject(0) ?: return emptyList()

        val opens = quote.optJSONArray("open")
        val highs = quote.optJSONArray("high")
        val lows = quote.optJSONArray("low")
        val closes = quote.optJSONArray("close")
        val volumes = quote.optJSONArray("volume")

        // adjclose : permet d'ajuster les splits (cohérence des rendements).
        val adjClose = indicators.optJSONArray("adjclose")?.optJSONObject(0)?.optJSONArray("adjclose")

        val bars = ArrayList<QuantEngine.OhlcvBar>(timestamps.length())
        for (i in 0 until timestamps.length()) {
            val rawClose = closes?.optDouble(i, Double.NaN) ?: Double.NaN
            if (rawClose.isNaN() || rawClose <= 0.0) continue // gap de marché -> on saute

            val o = opens?.optDouble(i, rawClose) ?: rawClose
            val h = highs?.optDouble(i, rawClose) ?: rawClose
            val l = lows?.optDouble(i, rawClose) ?: rawClose
            val v = volumes?.optDouble(i, 0.0) ?: 0.0

            // Facteur d'ajustement split = adjClose / close (1.0 si indisponible).
            val adj = adjClose?.optDouble(i, rawClose) ?: rawClose
            val factor = if (rawClose != 0.0 && !adj.isNaN() && adj > 0.0) adj / rawClose else 1.0

            bars.add(
                QuantEngine.OhlcvBar(
                    time = timestamps.optLong(i, 0L),
                    open = if (o.isNaN()) adj else o * factor,
                    high = if (h.isNaN()) adj else h * factor,
                    low = if (l.isNaN()) adj else l * factor,
                    close = adj,
                    volume = if (v.isNaN()) 0.0 else v
                )
            )
        }
        return bars
    }
}
