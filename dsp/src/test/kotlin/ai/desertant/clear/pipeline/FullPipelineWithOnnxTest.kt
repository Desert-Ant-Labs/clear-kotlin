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
import java.nio.file.Files
import kotlin.math.PI
import kotlin.math.sin
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test

/**
 * Full Clear-equivalent pipeline test: WAV → DSP → ONNX model → DSP → R128
 * mastering → WAV. Mirrors what `Clear.enhance(audioPath)` does in
 * production, minus the Android Context. Verifies the end-to-end loudness
 * normalization lands within ±0.5 LU of the target.
 *
 * Runs on the desktop JVM via the `onnxruntime` (non-Android) artifact.
 * The Android variant uses the same C++ ORT under the hood, so a green
 * pass here is a strong predictor for on-device behavior.
 */
class FullPipelineWithOnnxTest {

    private fun modelFile(): File {
        val repoRoot = File(System.getProperty("user.dir")).parentFile
        return File(repoRoot, "library/src/main/assets/clear-studio.onnx")
    }

    @Test
    fun `Clear-equivalent pipeline outputs WAV at target LUFS within tolerance`() {
        val sr = 48_000
        val n = sr * 3   // 3 s, enough for the LUFS integrator
        // Speech-like signal: 220 Hz fundamental + a few harmonics, modulated.
        val signal = FloatArray(n) { i ->
            val t = i.toDouble() / sr
            val env = (1.0 + 0.5 * sin(2 * PI * 4 * t))   // 4 Hz tremolo
            (0.3 * env * (sin(2 * PI * 220 * t) +
                          0.5 * sin(2 * PI * 440 * t) +
                          0.3 * sin(2 * PI * 660 * t))).toFloat()
        }

        val inFile = Files.createTempFile("clear-full-in", ".wav").toFile()
        val outFile = Files.createTempFile("clear-full-out", ".wav").toFile()
        val env = OrtEnvironment.getEnvironment()
        val session = env.createSession(modelFile().absolutePath, OrtSession.SessionOptions())
        try {
            Wav.writeMono(inFile, signal, sr)

            // Read
            val audio = Wav.read(inFile)
            assertEquals(sr, audio.sampleRate)

            // Pipeline — same as Clear.enhanceChannel + mastering chain
            val stft = Stft()
            val bridge = JvmBridge(env, session)
            val inference = Inference(bridge)

            val samples = audio.samples[0]
            val fwd = stft.forward(samples)
            val features = FeatureExtractor.compute(fwd.real, fwd.imag, fwd.nFrames)
            val enhanced = inference.run(
                fwd.real, fwd.imag,
                features.featErb, features.featSpecReal, features.featSpecImag,
                fwd.nFrames)
            val reconstructed = stft.inverse(enhanced.real, enhanced.imag, enhanced.nFrames)
            val trimmed = if (reconstructed.size > samples.size)
                reconstructed.copyOfRange(0, samples.size) else reconstructed

            // Master to -19 LUFS / -1 dBTP (Apple Podcasts)
            val out = arrayOf(trimmed)
            R128.normalize(out, targetLufs = -19.0, truePeakDbtp = -1.0, sampleRate = sr)
            Wav.writeMono(outFile, out[0], sr)

            // Re-read and measure
            val outAudio = Wav.read(outFile)
            val finalMeasurement = R128.measure(arrayOf(outAudio.samples[0]), sr)
            println("FullPipeline output: integrated=${finalMeasurement.integratedLufs} LUFS, " +
                "truePeak=${finalMeasurement.truePeakDbfs} dBFS")

            // Should be at -19 LUFS within 0.5 LU (gain-only norm is exact)
            assertEquals(-19.0, finalMeasurement.integratedLufs, 0.5)
            // Peak at or below -1 dBTP
            assertTrue("peak ${finalMeasurement.truePeakDbfs} > -1 dBTP",
                finalMeasurement.truePeakDbfs <= -1.0 + 0.05)
            // Output length matches input
            assertEquals(samples.size, outAudio.frames)
        } finally {
            session.close()
            inFile.delete()
            outFile.delete()
        }
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
