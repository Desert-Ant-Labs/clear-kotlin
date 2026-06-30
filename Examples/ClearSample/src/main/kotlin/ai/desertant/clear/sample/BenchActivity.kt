package ai.desertant.clear.sample

import android.app.Activity
import android.graphics.Color
import android.graphics.Typeface
import android.os.Build
import android.os.Bundle
import android.text.method.ScrollingMovementMethod
import android.view.Gravity
import android.view.View
import android.view.WindowManager
import android.widget.Button
import android.widget.LinearLayout
import android.widget.ProgressBar
import android.widget.ScrollView
import android.widget.TextView
import ai.desertant.clear.Clear
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.SupervisorJob
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import java.io.File
import kotlin.math.max
import kotlin.math.min

/**
 * Bench harness — runs bundled clips through `Clear.enhance()` N
 * times each and reports cold/warm timings + per-stage breakdown.
 */
class BenchActivity : Activity() {

    private val scope = CoroutineScope(SupervisorJob() + Dispatchers.Main)

    private lateinit var hud: TextView
    private lateinit var log: TextView
    private lateinit var progress: ProgressBar
    private lateinit var runBtn: Button

    // (filename, iterations). Long clips get fewer iterations since each pass is ~15s.
    private val clips = listOf(
        "bench_short.wav" to 5,
        "bench_mid.wav" to 5,
        "bench_long.wav" to 5,
        // Cross-platform parity clip — bit-identical with the iOS bench asset.
        "bench_reference_60s.m4a" to 5,
        "bench_reference_mono.m4a" to 3,
        "bench_reference_stereo.m4a" to 3,
    )
    private val variants = listOf(Clear.ModelVariant.ClearStudio, Clear.ModelVariant.ClearNatural)
    private var totalRuns: Int = clips.sumOf { it.second } * variants.size
    private var completed: Int = 0

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        // Keep the screen on through long bench runs; allow adb-launched runs
        // over the lock screen.
        window.addFlags(
            WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON or
            WindowManager.LayoutParams.FLAG_TURN_SCREEN_ON or
            WindowManager.LayoutParams.FLAG_SHOW_WHEN_LOCKED or
            WindowManager.LayoutParams.FLAG_DISMISS_KEYGUARD
        )
        setContentView(buildLayout())
        hud.text = deviceHeader()
    }

    private fun buildLayout(): View {
        val root = LinearLayout(this).apply {
            orientation = LinearLayout.VERTICAL
            setPadding(40, 40, 40, 40)
            setBackgroundColor(Color.parseColor("#0B0F1A"))
        }
        hud = TextView(this).apply {
            text = "—"
            setTextColor(Color.parseColor("#E6EDF3"))
            textSize = 14f
            typeface = Typeface.MONOSPACE
            setPadding(0, 0, 0, 24)
        }
        progress = ProgressBar(this, null, android.R.attr.progressBarStyleHorizontal).apply {
            isIndeterminate = false
            max = 100
            visibility = View.GONE
        }
        runBtn = Button(this).apply {
            text = "Run benchmark"
            setOnClickListener { runBenchmark() }
        }
        log = TextView(this).apply {
            text = ""
            setTextColor(Color.parseColor("#8B949E"))
            textSize = 11f
            typeface = Typeface.MONOSPACE
            setPadding(0, 24, 0, 0)
            gravity = Gravity.START or Gravity.TOP
            movementMethod = ScrollingMovementMethod()
            isVerticalScrollBarEnabled = true
        }
        val scroll = ScrollView(this).apply { addView(log) }
        root.addView(hud)
        root.addView(progress)
        root.addView(runBtn)
        root.addView(scroll, LinearLayout.LayoutParams(
            LinearLayout.LayoutParams.MATCH_PARENT, 0, 1f))
        return root
    }

    private fun deviceHeader(): String {
        val abi = Build.SUPPORTED_ABIS.firstOrNull() ?: "?"
        return """
            ${Build.MANUFACTURER} ${Build.MODEL}
            Android ${Build.VERSION.RELEASE} (API ${Build.VERSION.SDK_INT})  abi=$abi
            Ready. Tap Run.
        """.trimIndent()
    }

    private fun runBenchmark() {
        runBtn.isEnabled = false
        progress.visibility = View.VISIBLE
        progress.progress = 0
        log.text = ""
        completed = 0
        appendLog("=== benchmark start ===")

        scope.launch {
            try {
                // Stage bundled assets to internal storage.
                val files = withContext(Dispatchers.IO) {
                    clips.map { (name, iter) ->
                        val f = File(filesDir, name)
                        if (!f.exists()) {
                            assets.open(name).use { input ->
                                f.outputStream().use { input.copyTo(it) }
                            }
                        }
                        f to iter
                    }
                }

                // Iterate variants × clips × iterations; HUD shows variants side-by-side.
                val variantResults = mutableMapOf<Clear.ModelVariant, VariantBlock>()
                for (variant in variants) {
                    appendLog("")
                    appendLog("### variant: ${variant.rawValue} ###")
                    val loadStart = System.nanoTime()
                    val clear = Clear.create(applicationContext, variant = variant)
                    val loadSec = (System.nanoTime() - loadStart) / 1e9
                    appendLog("model loaded in %.2fs".format(loadSec))
                    val block = VariantBlock(loadSec = loadSec)
                    variantResults[variant] = block
                    runOnUiThread { updateHud(variantResults) }

                    for ((clipIdx, fp) in files.withIndex()) {
                        val (file, iter) = fp
                        val clipName = file.name
                        appendLog("")
                        appendLog("[$clipIdx] $clipName  ($iter iter)")
                        val br = BenchResults()
                        block.clips[clipName] = br
                        for (i in 0 until iter) {
                            System.gc()
                            Thread.sleep(50)
                            val t0 = System.nanoTime()
                            val r = clear.enhance(file.path)
                            val wall = (System.nanoTime() - t0) / 1e9
                            val rtf = r.realtimeFactor
                            val tag = if (i == 0) "cold" else "warm"
                            appendLog(("  iter ${i+1}: %.3fs (%.1f× rt)  " +
                                "decode=%.3f onnx=%.3f master=%.3f encode=%.3f  [%s]").format(
                                wall, rtf,
                                r.phaseTimings.decodeResampleSec, r.phaseTimings.onnxPredictSec,
                                r.phaseTimings.r128AndLimiterSec, r.phaseTimings.writeSec, tag))
                            if (i == 0) br.cold = Sample(wall, rtf, r)
                            else br.warm.add(Sample(wall, rtf, r))
                            completed++
                            runOnUiThread {
                                progress.progress = (completed * 100 / totalRuns)
                                updateHud(variantResults)
                            }
                        }
                    }
                    clear.close()
                }
                appendLog("")
                appendLog("=== benchmark complete ===")
                updateHud(variantResults)
            } catch (t: Throwable) {
                appendLog("FAILED: ${t.message}")
            } finally {
                runBtn.isEnabled = true
            }
        }
    }

    private fun appendLog(line: String) {
        runOnUiThread {
            log.append("$line\n")
        }
    }

    private fun updateHud(variantResults: Map<Clear.ModelVariant, VariantBlock>) {
        val sb = StringBuilder()
        sb.appendLine("${Build.MANUFACTURER} ${Build.MODEL}")
        val abi = Build.SUPPORTED_ABIS.firstOrNull() ?: "?"
        sb.appendLine("Android ${Build.VERSION.RELEASE} (API ${Build.VERSION.SDK_INT})  abi=$abi")
        for ((variant, block) in variantResults) {
            sb.appendLine()
            sb.appendLine("--- ${variant.rawValue} (load %.2fs) ---".format(block.loadSec))
            if (block.clips.isNotEmpty()) {
                sb.appendLine("clip               cold    warm avg   warm best   warm rtf")
                for ((clip, r) in block.clips) {
                    val shortName = clip.removePrefix("bench_").removeSuffix(".wav").removeSuffix(".m4a").padEnd(16)
                    val cold = r.cold?.let { "%.2fs".format(it.wall) } ?: "—"
                    val warmAvg = if (r.warm.isNotEmpty()) "%.2fs".format(r.warm.map { it.wall }.average()) else "—"
                    val warmBest = if (r.warm.isNotEmpty()) "%.2fs".format(r.warm.minOf { it.wall }) else "—"
                    val warmRtf = if (r.warm.isNotEmpty())
                        "%.1f×".format(r.warm.map { it.rtf }.average()) else "—"
                    sb.appendLine("$shortName ${cold.padEnd(7)} ${warmAvg.padEnd(10)} ${warmBest.padEnd(10)} $warmRtf")
                }
            }
            val allWarm = block.clips.values.flatMap { it.warm }
            if (allWarm.isNotEmpty()) {
                val avg = allWarm.map { it.rtf }.average()
                val best = allWarm.maxOf { it.rtf }
                sb.appendLine("aggregate warm: avg %.1f× rt  best %.1f× rt  n=%d".format(
                    avg, best, allWarm.size))
            }
        }
        runOnUiThread { hud.text = sb.toString() }
    }

    private data class Sample(val wall: Double, val rtf: Double, val r: Clear.Result)
    private class BenchResults {
        var cold: Sample? = null
        val warm = mutableListOf<Sample>()
    }
    private class VariantBlock(val loadSec: Double) {
        val clips: MutableMap<String, BenchResults> = linkedMapOf()
    }
}
