package ai.desertant.clear.pipeline

import ai.desertant.clear.internal.dsp.FeatureExtractor
import ai.desertant.clear.internal.dsp.IdentityModelBridge
import ai.desertant.clear.internal.dsp.Inference
import ai.desertant.clear.internal.dsp.Stft
import ai.desertant.clear.internal.io.Wav
import java.io.File
import java.nio.file.Files
import kotlin.math.PI
import kotlin.math.abs
import kotlin.math.sin
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test

/**
 * End-to-end smoke test of the DSP pipeline using the identity model
 * bridge. Proves all the wiring works without needing Android or ONNX
 * runtime:
 *   read WAV → STFT.forward → features → Inference(identity) → STFT.inverse
 *   → write WAV → re-read → verify content
 *
 * Identity model output ≈ input (modulo STFT round-trip rounding), so
 * the reconstruction should be close to the original within reasonable
 * float32 tolerance.
 */
class EndToEndTest {

    @Test
    fun `full pipeline reconstructs input within reasonable tolerance using identity model`() {
        val sr = 48_000
        val n = 24_000  // 0.5 s
        val freq = 1000.0
        val amplitude = 0.5f
        val signal = FloatArray(n) { i ->
            amplitude * sin(2 * PI * freq * i / sr).toFloat()
        }

        val inFile = Files.createTempFile("clear-e2e-in", ".wav").toFile()
        val outFile = Files.createTempFile("clear-e2e-out", ".wav").toFile()
        try {
            Wav.writeMono(inFile, signal, sr)

            // The actual pipeline — same shape as Clear.enhanceChannel()
            // but without Android Context.
            val stft = Stft()
            val inference = Inference(IdentityModelBridge())
            val audio = Wav.read(inFile)
            val samples = audio.samples[0]
            val fwd = stft.forward(samples)
            val features = FeatureExtractor.compute(fwd.real, fwd.imag, fwd.nFrames)
            val enhanced = inference.run(
                fwd.real, fwd.imag,
                features.featErb, features.featSpecReal, features.featSpecImag,
                fwd.nFrames,
            )
            val reconstructed = stft.inverse(enhanced.real, enhanced.imag, enhanced.nFrames)
            val trimmed = if (reconstructed.size > samples.size)
                reconstructed.copyOfRange(0, samples.size) else reconstructed
            Wav.writeMono(outFile, trimmed, sr)

            // Re-read and verify
            val outAudio = Wav.read(outFile)
            assertEquals(sr, outAudio.sampleRate)
            assertEquals(1, outAudio.channels)
            assertEquals(n, outAudio.frames)

            // The STFT round-trip with identity model is exact within
            // float32 rounding for the inner samples. There's a small
            // edge region from the analysis window. Check the middle.
            val out = outAudio.samples[0]
            val startIdx = 1000
            val endIdx = n - 1000
            var maxErr = 0f
            for (i in startIdx until endIdx) {
                val err = abs(out[i] - signal[i])
                if (err > maxErr) maxErr = err
            }
            assertTrue("max reconstruction error $maxErr too large", maxErr < 1e-4f)
        } finally {
            inFile.delete()
            outFile.delete()
        }
    }
}
