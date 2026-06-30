package ai.desertant.clear.pipeline

import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import ai.desertant.clear.internal.Constants
import ai.desertant.clear.internal.dsp.FeatureExtractor
import ai.desertant.clear.internal.dsp.Inference
import ai.desertant.clear.internal.dsp.ModelBridge
import ai.desertant.clear.internal.dsp.Stft
import ai.desertant.clear.internal.io.Wav
import ai.desertant.clear.internal.mastering.R128
import java.io.File
import java.nio.FloatBuffer
import org.junit.Assume.assumeTrue
import org.junit.Test

/**
 * Batch-renders a directory of reference clips through the Kotlin
 * pipeline for A/B comparison against the Swift `clear-studio` output.
 *
 * Point `CLEAR_AUDITION_CLIPS` at a directory of 48 kHz WAVs to run;
 * the test skips itself when the env var is unset (i.e. CI). Output
 * goes to `/tmp/kotlin-clear-studio/`.
 *
 * Invocation:
 *   CLEAR_AUDITION_CLIPS=/path/to/clips ./gradlew :dsp:test \
 *       --tests "*AuditionBatchRender*"
 *   ls /tmp/kotlin-clear-studio/
 */
class AuditionBatchRender {

    @Test
    fun `render audition clips through Kotlin pipeline`() {
        val rawDirPath = System.getenv("CLEAR_AUDITION_CLIPS")
        assumeTrue("CLEAR_AUDITION_CLIPS not set", rawDirPath != null)
        val rawDir = File(rawDirPath!!)
        assumeTrue("audition raw clips not present at ${rawDir.path}", rawDir.isDirectory)
        val modelFile = File(File(System.getProperty("user.dir")).parentFile,
            "library/src/main/assets/clear-studio.onnx")
        assumeTrue("clear-studio.onnx not present at ${modelFile.path}", modelFile.exists())

        val outDir = File("/tmp/kotlin-clear-studio")
        outDir.deleteRecursively(); outDir.mkdirs()

        val env = OrtEnvironment.getEnvironment()
        val session = env.createSession(modelFile.absolutePath, OrtSession.SessionOptions())
        val bridge = JvmBridge(env, session)
        val stft = Stft()
        val inference = Inference(bridge)

        val clips = rawDir.listFiles { f -> f.name.endsWith(".wav") }?.sortedBy { it.name }
            ?: emptyList()
        println("Rendering ${clips.size} clips → ${outDir.path}")

        var totalAudioSec = 0.0
        var totalWallSec = 0.0
        for ((idx, clip) in clips.withIndex()) {
            val start = System.nanoTime()
            try {
                val audio = Wav.read(clip)
                if (audio.sampleRate != 48_000) {
                    println("[${idx + 1}/${clips.size}] ${clip.name} — SKIP (sr=${audio.sampleRate})")
                    continue
                }
                val nCh = audio.channels
                val outChannels = Array(nCh) { c ->
                    val samples = audio.samples[c]
                    val fwd = stft.forward(samples)
                    if (fwd.nFrames == 0) return@Array samples
                    val feats = FeatureExtractor.compute(fwd.real, fwd.imag, fwd.nFrames)
                    val enhanced = inference.run(
                        fwd.real, fwd.imag,
                        feats.featErb, feats.featSpecReal, feats.featSpecImag,
                        fwd.nFrames)
                    val rec = stft.inverse(enhanced.real, enhanced.imag, enhanced.nFrames)
                    if (rec.size > samples.size) rec.copyOfRange(0, samples.size) else rec
                }
                // Apple Podcasts master
                R128.normalize(outChannels, targetLufs = -19.0, truePeakDbtp = -1.5,
                    sampleRate = audio.sampleRate)
                val outFile = File(outDir, "${clip.nameWithoutExtension}_clear.wav")
                Wav.write(outFile, outChannels, audio.sampleRate, nCh)

                val wall = (System.nanoTime() - start) / 1e9
                totalAudioSec += audio.durationSec
                totalWallSec += wall
                val rtf = audio.durationSec / wall
                println("[${idx + 1}/${clips.size}] ${clip.name} → ${outFile.name}  " +
                    "(${"%.1f".format(audio.durationSec)}s in ${"%.2f".format(wall)}s, " +
                    "${"%.1f".format(rtf)}× rt)")
            } catch (t: Throwable) {
                println("[${idx + 1}/${clips.size}] ${clip.name} — FAILED: ${t.message}")
            }
        }
        session.close()

        val overallRtf = totalAudioSec / totalWallSec.coerceAtLeast(1e-9)
        println("")
        println("Total: ${"%.1f".format(totalAudioSec)}s of audio in " +
            "${"%.2f".format(totalWallSec)}s wall = ${"%.1f".format(overallRtf)}× realtime")
        println("Outputs in: ${outDir.path}")
    }

    private class JvmBridge(
        private val env: OrtEnvironment,
        private val session: OrtSession,
    ) : ModelBridge {
        private val tShape = longArrayOf(1, 1, Constants.CHUNK_LEN.toLong(), Constants.N_FREQ.toLong(), 2)
        private val erbShape = longArrayOf(1, 1, Constants.CHUNK_LEN.toLong(), Constants.N_ERB.toLong())
        private val featSpecShape = longArrayOf(1, 1, Constants.CHUNK_LEN.toLong(), Constants.N_DF.toLong(), 2)
        private val outElementCount = Constants.CHUNK_LEN * Constants.N_FREQ * 2

        override fun runChunk(spec: FloatArray, featErb: FloatArray, featSpec: FloatArray): FloatArray {
            OnnxTensor.createTensor(env, FloatBuffer.wrap(spec), tShape).use { specT ->
                OnnxTensor.createTensor(env, FloatBuffer.wrap(featErb), erbShape).use { erbT ->
                    OnnxTensor.createTensor(env, FloatBuffer.wrap(featSpec), featSpecShape).use { fsT ->
                        val inputs = mapOf("spec" to specT, "feat_erb" to erbT, "feat_spec" to fsT)
                        session.run(inputs).use { result ->
                            val out = result[0] as OnnxTensor
                            val arr = FloatArray(outElementCount)
                            out.floatBuffer.get(arr)
                            return arr
                        }
                    }
                }
            }
        }
    }
}
