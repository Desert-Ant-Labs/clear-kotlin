package ai.desertant.clear.internal.dsp

import ai.desertant.clear.internal.Constants
import org.junit.Assert.assertArrayEquals
import org.junit.Assert.assertEquals
import org.junit.Test

class InferenceTest {

    @Test
    fun `identity model passes spec through unchanged`() {
        val nFreq = Constants.N_FREQ
        val nErb = Constants.N_ERB
        val nDf = Constants.N_DF
        val nFrames = 250  // forces a second chunk

        // Deterministic ramp signal — easy to check element-wise.
        val real = FloatArray(nFrames * nFreq) { it.toFloat() * 0.001f }
        val imag = FloatArray(nFrames * nFreq) { it.toFloat() * -0.001f }
        val featErb = FloatArray(nFrames * nErb)
        val featSpecReal = FloatArray(nFrames * nDf)
        val featSpecImag = FloatArray(nFrames * nDf)

        val out = Inference(IdentityModelBridge()).run(
            real, imag, featErb, featSpecReal, featSpecImag, nFrames)

        // First nFrames - convLookahead frames should match exactly. The
        // lookahead window can leave the last `convLookahead` frames of
        // the second chunk un-populated past `nFrames`, but the chunk's
        // valid window writes the spec verbatim through identity.
        assertEquals(nFrames, out.nFrames)
        assertArrayEquals(real, out.real, 0f)
        assertArrayEquals(imag, out.imag, 0f)
    }

    @Test
    fun `inference handles a single sub-chunk-length input`() {
        val nFreq = Constants.N_FREQ
        val nErb = Constants.N_ERB
        val nDf = Constants.N_DF
        val nFrames = 25
        val real = FloatArray(nFrames * nFreq) { 1f }
        val imag = FloatArray(nFrames * nFreq) { -1f }
        val featErb = FloatArray(nFrames * nErb)
        val featSpecReal = FloatArray(nFrames * nDf)
        val featSpecImag = FloatArray(nFrames * nDf)

        val out = Inference(IdentityModelBridge()).run(
            real, imag, featErb, featSpecReal, featSpecImag, nFrames)
        assertEquals(nFrames, out.nFrames)
        assertArrayEquals(real, out.real, 0f)
        assertArrayEquals(imag, out.imag, 0f)
    }
}
