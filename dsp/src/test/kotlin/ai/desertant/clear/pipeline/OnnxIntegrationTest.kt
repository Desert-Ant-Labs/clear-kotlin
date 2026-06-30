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
import java.io.File
import java.nio.FloatBuffer
import java.nio.file.Files
import kotlin.math.PI
import kotlin.math.sin
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test

/**
 * Integration test: load the real clear-studio.onnx, run the full
 * pipeline against a synthetic input, verify the model produces
 * finite output of the right shape.
 *
 * Runs on the desktop JVM via `com.microsoft.onnxruntime:onnxruntime`
 * (the non-Android variant). This proves the ONNX export, model
 * loading, and pipeline wiring all work without needing an emulator
 * — strong signal the Android path will too, since the model and
 * the API surface are the same.
 */
class OnnxIntegrationTest {

    private fun modelFile(): File {
        // The library bundles the ONNX in assets/. JVM tests in :dsp
        // resolve it via the source path.
        val repoRoot = File(System.getProperty("user.dir")).parentFile  // android/
        val candidate = File(repoRoot, "library/src/main/assets/clear-studio.onnx")
        if (candidate.exists()) return candidate
        error("clear-studio.onnx not found at ${candidate.path} — drop the asset into library assets first.")
    }

    @Test
    fun `clear-studio ONNX loads and runs through the full pipeline`() {
        val sr = 48_000
        val n = 24_000  // 0.5 s — exercises 1 inference chunk
        // Speech-like input: 220 Hz fundamental + first 3 harmonics
        val signal = FloatArray(n) { i ->
            val t = i.toDouble() / sr
            0.3f * (sin(2 * PI * 220 * t) + 0.5 * sin(2 * PI * 440 * t) +
                    0.3 * sin(2 * PI * 880 * t)).toFloat()
        }

        val outFile = Files.createTempFile("clear-onnx-out", ".wav").toFile()
        val env = OrtEnvironment.getEnvironment()
        val session: OrtSession = env.createSession(modelFile().absolutePath, OrtSession.SessionOptions())
        try {
            // Build a ModelBridge over the JVM ORT session.
            val bridge = JvmOnnxBridge(env, session)

            val stft = Stft()
            val fwd = stft.forward(signal)
            assertTrue("STFT produced 0 frames", fwd.nFrames > 0)
            val features = FeatureExtractor.compute(fwd.real, fwd.imag, fwd.nFrames)
            val enhanced = Inference(bridge).run(
                fwd.real, fwd.imag,
                features.featErb, features.featSpecReal, features.featSpecImag,
                fwd.nFrames)

            // Verify shape preserved
            assertEquals(fwd.nFrames, enhanced.nFrames)
            assertEquals(fwd.real.size, enhanced.real.size)

            // Verify the model didn't emit NaN/Inf.
            for (i in enhanced.real.indices) {
                val r = enhanced.real[i]
                val ii = enhanced.imag[i]
                assertTrue("real[$i] = $r is not finite", r.isFinite())
                assertTrue("imag[$i] = $ii is not finite", ii.isFinite())
            }

            // Reconstruct and write.
            val reconstructed = stft.inverse(enhanced.real, enhanced.imag, enhanced.nFrames)
            val trimmed = if (reconstructed.size > signal.size)
                reconstructed.copyOfRange(0, signal.size) else reconstructed
            Wav.writeMono(outFile, trimmed, sr)

            // The output should have non-trivial energy (model didn't
            // silence everything).
            var energy = 0.0
            for (x in trimmed) energy += x.toDouble() * x.toDouble()
            energy /= trimmed.size
            assertTrue("output energy $energy too low — model may have zeroed everything",
                energy > 1e-6)

            println("OnnxIntegrationTest: ${trimmed.size} samples, energy=$energy")
        } finally {
            session.close()
            outFile.delete()
        }
    }

    private class JvmOnnxBridge(
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
                            val buf = out.floatBuffer
                            val arr = FloatArray(outElementCount)
                            buf.get(arr)
                            return arr
                        }
                    }
                }
            }
        }
    }
}
