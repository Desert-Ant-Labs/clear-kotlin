package ai.desertant.clear.internal.dsp

import ai.desertant.clear.internal.Constants
import kotlin.math.PI
import kotlin.math.sin
import org.junit.Assert.assertArrayEquals
import org.junit.Assert.assertEquals
import org.junit.Test

/**
 * Streaming-vs-batch equivalence tests. For any partition of the same
 * input into chunks, the streaming variant must produce exactly the
 * same output as the batch version. This is the core correctness
 * invariant of the streaming refactor.
 */
class StreamingTest {

    private fun sweepSignal(n: Int = 12_000): FloatArray {
        val sr = 48_000.0
        val duration = n / sr
        val f0 = 100.0; val f1 = 8000.0
        val logK = kotlin.math.ln(f1 / f0)
        return FloatArray(n) { i ->
            val t = i / sr
            val phase = 2 * PI * f0 * duration / logK * (Math.pow(f1 / f0, t / duration) - 1)
            (0.3 * sin(phase)).toFloat()
        }
    }

    @Test
    fun `streaming STFT forward matches batch on arbitrary chunk sizes`() {
        val signal = sweepSignal()
        val batchResult = Stft().forward(signal)

        // Try several chunk sizes including non-aligned ones.
        for (chunkSize in listOf(481, 480, 1024, 480 * 3 + 7, signal.size, 1)) {
            if (chunkSize > signal.size) continue
            val sft = StreamingStft()
            val real = mutableListOf<Float>()
            val imag = mutableListOf<Float>()
            var totalFrames = 0
            var off = 0
            while (off < signal.size) {
                val n = minOf(chunkSize, signal.size - off)
                val chunk = sft.process(signal, off, n)
                if (chunk.nFrames > 0) {
                    for (v in chunk.real) real.add(v)
                    for (v in chunk.imag) imag.add(v)
                    totalFrames += chunk.nFrames
                }
                off += n
            }
            assertEquals("chunkSize=$chunkSize nFrames", batchResult.nFrames, totalFrames)
            assertArrayEquals("chunkSize=$chunkSize real",
                batchResult.real, real.toFloatArray(), 0f)
            assertArrayEquals("chunkSize=$chunkSize imag",
                batchResult.imag, imag.toFloatArray(), 0f)
        }
    }

    @Test
    fun `streaming STFT inverse matches batch`() {
        val signal = sweepSignal()
        val fwd = Stft().forward(signal)
        val batchOut = Stft().inverse(fwd.real, fwd.imag, fwd.nFrames)

        // Feed all frames in one streaming call + flush — should match batch.
        run {
            val istft = StreamingIStft()
            val out = istft.process(fwd.real, fwd.imag, fwd.nFrames)
            val tail = istft.flush()
            val combined = out + tail
            assertEquals("single-call size", batchOut.size, combined.size)
            assertArrayEquals("single-call data", batchOut, combined, 0f)
        }

        // Now in multiple frame chunks — process + flush at the end.
        for (framesPerChunk in listOf(1, 50, 100, 200)) {
            val istft = StreamingIStft()
            val accum = mutableListOf<Float>()
            var f = 0
            while (f < fwd.nFrames) {
                val n = minOf(framesPerChunk, fwd.nFrames - f)
                val real = FloatArray(n * Constants.N_FREQ)
                val imag = FloatArray(n * Constants.N_FREQ)
                System.arraycopy(fwd.real, f * Constants.N_FREQ, real, 0, n * Constants.N_FREQ)
                System.arraycopy(fwd.imag, f * Constants.N_FREQ, imag, 0, n * Constants.N_FREQ)
                val out = istft.process(real, imag, n)
                for (v in out) accum.add(v)
                f += n
            }
            // Final flush emits the OLA carry tail.
            for (v in istft.flush()) accum.add(v)
            assertEquals("framesPerChunk=$framesPerChunk size", batchOut.size, accum.size)
            assertArrayEquals("framesPerChunk=$framesPerChunk",
                batchOut, accum.toFloatArray(), 1e-5f)
        }
    }

    @Test
    fun `streaming features match batch on arbitrary frame partitions`() {
        val signal = sweepSignal(48_000)  // 1s = 100 frames
        val fwd = Stft().forward(signal)
        val batch = FeatureExtractor.compute(fwd.real, fwd.imag, fwd.nFrames)

        for (framesPerChunk in listOf(1, 7, 50, 100, fwd.nFrames)) {
            val sf = StreamingFeatureExtractor()
            val erbAccum = FloatArray(batch.featErb.size)
            val realAccum = FloatArray(batch.featSpecReal.size)
            val imagAccum = FloatArray(batch.featSpecImag.size)
            var written = 0
            var f = 0
            while (f < fwd.nFrames) {
                val n = minOf(framesPerChunk, fwd.nFrames - f)
                val realSlice = FloatArray(n * Constants.N_FREQ)
                val imagSlice = FloatArray(n * Constants.N_FREQ)
                System.arraycopy(fwd.real, f * Constants.N_FREQ, realSlice, 0, n * Constants.N_FREQ)
                System.arraycopy(fwd.imag, f * Constants.N_FREQ, imagSlice, 0, n * Constants.N_FREQ)
                val out = sf.process(realSlice, imagSlice, n)
                System.arraycopy(out.featErb, 0, erbAccum, f * Constants.N_ERB, n * Constants.N_ERB)
                System.arraycopy(out.featSpecReal, 0, realAccum, f * Constants.N_DF, n * Constants.N_DF)
                System.arraycopy(out.featSpecImag, 0, imagAccum, f * Constants.N_DF, n * Constants.N_DF)
                written += n
                f += n
            }
            assertArrayEquals("framesPerChunk=$framesPerChunk featErb",
                batch.featErb, erbAccum, 0f)
            assertArrayEquals("framesPerChunk=$framesPerChunk featSpecReal",
                batch.featSpecReal, realAccum, 0f)
            assertArrayEquals("framesPerChunk=$framesPerChunk featSpecImag",
                batch.featSpecImag, imagAccum, 0f)
        }
    }
}
