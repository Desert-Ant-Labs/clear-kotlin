package ai.desertant.clear.internal.dsp

import ai.desertant.clear.internal.Constants
import org.junit.Assert.assertArrayEquals
import org.junit.Assert.assertEquals
import org.junit.Test

/**
 * Streaming inference parity vs batch [Inference]. Identity model
 * is enough to verify the slicing/lookahead/flush bookkeeping —
 * if the streaming wrapper passes the right windows to the model,
 * the batch and streaming outputs must agree exactly.
 */
class StreamingInferenceTest {

    @Test
    fun `streaming inference matches batch for various input chunk sizes`() {
        val nFreq = Constants.N_FREQ
        val nErb = Constants.N_ERB
        val nDf = Constants.N_DF
        val nFrames = 470  // 2.35 chunks worth — exercises the boundary
        val real = FloatArray(nFrames * nFreq) { it.toFloat() * 0.001f }
        val imag = FloatArray(nFrames * nFreq) { it.toFloat() * -0.001f }
        val featErb = FloatArray(nFrames * nErb) { (it + 1) * 0.01f }
        val featSpecReal = FloatArray(nFrames * nDf) { it.toFloat() * 0.002f }
        val featSpecImag = FloatArray(nFrames * nDf) { it.toFloat() * -0.002f }

        val batch = Inference(IdentityModelBridge()).run(
            real, imag, featErb, featSpecReal, featSpecImag, nFrames)
        assertEquals(nFrames, batch.nFrames)

        for (framesPerCall in listOf(1, 50, 100, 200, 250, nFrames)) {
            val streaming = StreamingInference(IdentityModelBridge())
            val outReal = FloatArray(nFrames * nFreq)
            val outImag = FloatArray(nFrames * nFreq)
            var emitted = 0

            var off = 0
            while (off < nFrames) {
                val n = minOf(framesPerCall, nFrames - off)
                val r = streaming.process(
                    real.copyOfRange(off * nFreq, (off + n) * nFreq),
                    imag.copyOfRange(off * nFreq, (off + n) * nFreq),
                    featErb.copyOfRange(off * nErb, (off + n) * nErb),
                    featSpecReal.copyOfRange(off * nDf, (off + n) * nDf),
                    featSpecImag.copyOfRange(off * nDf, (off + n) * nDf),
                    n,
                )
                if (r.nFrames > 0) {
                    System.arraycopy(r.real, 0, outReal, emitted * nFreq, r.nFrames * nFreq)
                    System.arraycopy(r.imag, 0, outImag, emitted * nFreq, r.nFrames * nFreq)
                    emitted += r.nFrames
                }
                off += n
            }
            // Flush the trailing partial chunk.
            val tail = streaming.flush()
            if (tail.nFrames > 0) {
                System.arraycopy(tail.real, 0, outReal, emitted * nFreq, tail.nFrames * nFreq)
                System.arraycopy(tail.imag, 0, outImag, emitted * nFreq, tail.nFrames * nFreq)
                emitted += tail.nFrames
            }

            assertEquals("framesPerCall=$framesPerCall total emitted", nFrames, emitted)
            // Identity model passes spec through unchanged, so streaming
            // output should match the input real/imag arrays.
            assertArrayEquals("framesPerCall=$framesPerCall real",
                real, outReal, 0f)
            assertArrayEquals("framesPerCall=$framesPerCall imag",
                imag, outImag, 0f)
        }
    }
}
