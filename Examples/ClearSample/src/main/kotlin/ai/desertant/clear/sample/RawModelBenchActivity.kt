package ai.desertant.clear.sample

import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import android.app.Activity
import android.graphics.Color
import android.graphics.Typeface
import android.os.Build
import android.os.Bundle
import android.text.method.ScrollingMovementMethod
import android.view.View
import android.view.WindowManager
import android.widget.Button
import android.widget.LinearLayout
import android.widget.ScrollView
import android.widget.TextView
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.SupervisorJob
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import java.nio.FloatBuffer

/**
 * Raw model latency bench. Loads clear-studio.onnx and runs N predict()
 * calls with zero-filled inputs, reports min/median/max wall time
 * per call. Isolates the model's intrinsic latency from the DSP glue.
 */
class RawModelBenchActivity : Activity() {

    private val scope = CoroutineScope(SupervisorJob() + Dispatchers.Main)
    private lateinit var hud: TextView
    private lateinit var log: TextView
    private lateinit var runBtn: Button

    private val nRuns = 50
    private val tShape: LongArray = longArrayOf(1, 1, 200, 481, 2)
    private val erbShape: LongArray = longArrayOf(1, 1, 200, 32)
    private val featSpecShape: LongArray = longArrayOf(1, 1, 200, 96, 2)

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        window.addFlags(
            WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON or
            WindowManager.LayoutParams.FLAG_TURN_SCREEN_ON or
            WindowManager.LayoutParams.FLAG_SHOW_WHEN_LOCKED or
            WindowManager.LayoutParams.FLAG_DISMISS_KEYGUARD
        )
        setContentView(buildLayout())
        hud.text = "${Build.MANUFACTURER} ${Build.MODEL}  abi=${Build.SUPPORTED_ABIS.firstOrNull() ?: "?"}\nReady. Tap Run."
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
        runBtn = Button(this).apply {
            text = "Run raw model bench"
            setOnClickListener { runBench() }
        }
        log = TextView(this).apply {
            text = ""
            setTextColor(Color.parseColor("#8B949E"))
            textSize = 11f
            typeface = Typeface.MONOSPACE
            setPadding(0, 24, 0, 0)
            movementMethod = ScrollingMovementMethod()
        }
        root.addView(hud)
        root.addView(runBtn)
        root.addView(ScrollView(this).apply { addView(log) },
            LinearLayout.LayoutParams(LinearLayout.LayoutParams.MATCH_PARENT, 0, 1f))
        return root
    }

    private fun runBench() {
        runBtn.isEnabled = false
        log.text = ""
        scope.launch {
            try {
                appendLog("Loading clear-studio.onnx…")
                val (env, session, loadSec) = withContext(Dispatchers.IO) {
                    val bytes = assets.open("clear-studio.onnx").use { it.readBytes() }
                    val t0 = System.nanoTime()
                    val e = OrtEnvironment.getEnvironment()
                    val opts = OrtSession.SessionOptions().apply {
                        setIntraOpNumThreads(Runtime.getRuntime().availableProcessors().coerceAtMost(4))
                        setOptimizationLevel(OrtSession.SessionOptions.OptLevel.BASIC_OPT)
                    }
                    val s = e.createSession(bytes, opts)
                    Triple(e, s, (System.nanoTime() - t0) / 1e9)
                }
                appendLog("Loaded in %.2fs".format(loadSec))

                val spec = FloatArray(200 * 481 * 2)
                val erb = FloatArray(200 * 32)
                val featSpec = FloatArray(200 * 96 * 2)

                val timesMs = DoubleArray(nRuns)
                for (i in 0 until nRuns) {
                    val t0 = System.nanoTime()
                    OnnxTensor.createTensor(env, FloatBuffer.wrap(spec), tShape).use { t1 ->
                        OnnxTensor.createTensor(env, FloatBuffer.wrap(erb), erbShape).use { t2 ->
                            OnnxTensor.createTensor(env, FloatBuffer.wrap(featSpec), featSpecShape).use { t3 ->
                                val inputs = mapOf("spec" to t1, "feat_erb" to t2, "feat_spec" to t3)
                                session.run(inputs).use { /* discard */ }
                            }
                        }
                    }
                    timesMs[i] = (System.nanoTime() - t0) / 1e6
                    if (i < 3 || i == nRuns - 1) appendLog("run %2d: %.2f ms".format(i + 1, timesMs[i]))
                }

                // Drop the first 3 runs as warm-up.
                val warm = timesMs.copyOfRange(3, nRuns)
                warm.sort()
                val median = warm[warm.size / 2]
                val best = warm.first()
                val worst = warm.last()
                val mean = warm.average()

                // 200-frame chunk at hop=480 ≈ 1 s of audio, so 1000/ms is the
                // predict-only realtime factor.
                val rtfMedian = 1000.0 / median
                val rtfBest = 1000.0 / best

                hud.text = """
                    ${Build.MANUFACTURER} ${Build.MODEL}
                    Android ${Build.VERSION.RELEASE} (API ${Build.VERSION.SDK_INT})

                    Raw model: clear-studio.onnx ($nRuns runs, dropped 3 warm-up)
                    median  %.2f ms   (%.1f× realtime, predict-only)
                    best    %.2f ms   (%.1f× realtime)
                    mean    %.2f ms
                    worst   %.2f ms
                """.trimIndent().format(median, rtfMedian, best, rtfBest, mean, worst)
                appendLog("")
                appendLog("median %.2f ms / best %.2f ms / worst %.2f ms".format(median, best, worst))
            } catch (t: Throwable) {
                appendLog("FAILED: ${t.message}")
            } finally {
                runBtn.isEnabled = true
            }
        }
    }

    private fun appendLog(line: String) = runOnUiThread { log.append("$line\n") }
}
