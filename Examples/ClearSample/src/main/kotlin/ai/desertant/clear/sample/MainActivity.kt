package ai.desertant.clear.sample

import android.app.Activity
import android.content.Intent
import android.media.MediaPlayer
import android.os.Bundle
import android.view.View
import android.widget.Button
import android.widget.LinearLayout
import android.widget.TextView
import ai.desertant.clear.Clear
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.Job
import kotlinx.coroutines.SupervisorJob
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import java.io.File

/** Sample app that runs `Clear.enhance()` on a bundled WAV and plays the result. */
class MainActivity : Activity() {

    private val scope = CoroutineScope(SupervisorJob() + Dispatchers.Main)
    private lateinit var status: TextView
    private lateinit var enhanceBtn: Button
    private lateinit var playRawBtn: Button
    private lateinit var playEnhancedBtn: Button

    private var clear: Clear? = null
    private var rawPath: String? = null
    private var enhancedPath: String? = null
    private var player: MediaPlayer? = null

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(buildLayout())

        // Copy assets/sample.wav to a regular file path the library can read.
        scope.launch {
            val sampleFile = File(filesDir, "sample.wav")
            if (!sampleFile.exists()) {
                withContext(Dispatchers.IO) {
                    assets.open("sample.wav").use { input ->
                        sampleFile.outputStream().use { output -> input.copyTo(output) }
                    }
                }
            }
            rawPath = sampleFile.path
            status.text = "Ready. Loading model…"
            try {
                clear = Clear.create(applicationContext)
                status.text = "Model loaded. Tap Enhance."
                enhanceBtn.isEnabled = true
                playRawBtn.isEnabled = true
            } catch (t: Throwable) {
                status.text = "Model load FAILED: ${t.message}"
            }
        }
    }

    private fun buildLayout(): View {
        val root = LinearLayout(this).apply {
            orientation = LinearLayout.VERTICAL
            setPadding(48, 48, 48, 48)
        }
        status = TextView(this).apply {
            text = "Starting…"
            textSize = 16f
        }
        enhanceBtn = Button(this).apply {
            text = "Enhance"
            isEnabled = false
            setOnClickListener { runEnhance() }
        }
        playRawBtn = Button(this).apply {
            text = "Play raw"
            isEnabled = false
            setOnClickListener { rawPath?.let { play(it) } }
        }
        playEnhancedBtn = Button(this).apply {
            text = "Play enhanced"
            isEnabled = false
            setOnClickListener { enhancedPath?.let { play(it) } }
        }
        val benchBtn = Button(this).apply {
            text = "Run benchmark (full pipeline)"
            setOnClickListener {
                startActivity(Intent(this@MainActivity, BenchActivity::class.java))
            }
        }
        val rawBtn = Button(this).apply {
            text = "Run raw model bench (predict-only)"
            setOnClickListener {
                startActivity(Intent(this@MainActivity, RawModelBenchActivity::class.java))
            }
        }
        root.addView(status)
        root.addView(enhanceBtn)
        root.addView(playRawBtn)
        root.addView(playEnhancedBtn)
        root.addView(benchBtn)
        root.addView(rawBtn)
        return root
    }

    private fun runEnhance() {
        val src = rawPath ?: return
        val c = clear ?: return
        enhanceBtn.isEnabled = false
        status.text = "Enhancing…"
        scope.launch {
            val start = System.nanoTime()
            try {
                val result = c.enhance(src) { p ->
                    val pct = (p.fraction * 100).toInt()
                    val phase = p::class.simpleName?.removeSuffix("Class") ?: "?"
                    runOnUiThread { status.text = "$phase $pct%" }
                }
                val wallSec = (System.nanoTime() - start) / 1e9
                enhancedPath = result.outputPath
                val rtf = "%.1f".format(result.realtimeFactor)
                val lufs = result.measuredLufs?.let { "%.1f".format(it) } ?: "?"
                val peak = result.measuredTruePeakDbfs?.let { "%.1f".format(it) } ?: "?"
                status.text = """
                    ✓ Enhanced in %.2fs (%s× rt)
                    duration: %.2fs
                    LUFS: %s, peak: %s dBFS
                    ONNX: %.2fs
                    → ${result.outputPath.substringAfterLast('/')}
                """.trimIndent().format(wallSec, rtf, result.durationSec, lufs, peak, result.onnxSec)
                playEnhancedBtn.isEnabled = true
            } catch (t: Throwable) {
                status.text = "Enhance FAILED: ${t.message}"
            } finally {
                enhanceBtn.isEnabled = true
            }
        }
    }

    private fun play(path: String) {
        try {
            player?.release()
            player = MediaPlayer().apply {
                setDataSource(path)
                prepare()
                start()
            }
        } catch (t: Throwable) {
            status.text = "Playback FAILED: ${t.message}"
        }
    }

    override fun onDestroy() {
        super.onDestroy()
        player?.release()
        (clear as? java.io.Closeable)?.close()
    }
}
