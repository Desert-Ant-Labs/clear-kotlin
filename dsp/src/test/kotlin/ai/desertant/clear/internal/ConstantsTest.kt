package ai.desertant.clear.internal

import org.junit.Assert.assertEquals
import org.junit.Assert.assertNotEquals
import org.junit.Test

/**
 * Sanity tests for the DSP constants. Anchors the values against the
 * Swift reference so a typo in `Constants.kt` fails CI immediately.
 *
 * These are not the parity tests — those load fixtures dumped from the
 * Swift CLI. This file is the first thing that must build + pass when
 * the scaffold is wired up: `cd android && ./gradlew :library:test`.
 */
class ConstantsTest {

    @Test
    fun `STFT geometry matches Swift Inference`() {
        assertEquals(960, Constants.FFT_SIZE)
        assertEquals(480, Constants.HOP_SIZE)
        assertEquals(481, Constants.N_FREQ)
        assertEquals(Constants.FFT_SIZE / 2 + 1, Constants.N_FREQ)
        assertEquals(Constants.FFT_SIZE - Constants.HOP_SIZE, Constants.STFT_PRE_PAD)
    }

    @Test
    fun `feature dimensions match Swift reference`() {
        assertEquals(32, Constants.N_ERB)
        assertEquals(96, Constants.N_DF)
    }

    @Test
    fun `chunk and lookahead match ONNX export`() {
        assertEquals(200, Constants.CHUNK_LEN)
        assertEquals(2, Constants.CONV_LOOKAHEAD)
        assertEquals(2, Constants.DF_LOOKAHEAD)
        assertEquals(5, Constants.DF_ORDER)
    }

    @Test
    fun `EMA smoothing matches Swift`() {
        assertEquals(0.99f, Constants.NORM_ALPHA, 0f)
    }

    @Test
    fun `STFT forward gain matches Vorbis spec`() {
        // Swift: wnorm = 2*hop / fftSize²  →  for 960/480 this is 1/960.
        assertEquals(1f / 960f, Constants.WNORM, 1e-9f)
    }

    @Test
    fun `erbState init is the documented ramp not zeros`() {
        val s = EmaInit.erbState()
        assertEquals(Constants.N_ERB, s.size)
        assertEquals(-60f, s[0], 1e-6f)
        assertEquals(-90f, s.last(), 1e-5f)
        // Monotonically decreasing — every step is negative.
        for (i in 1 until s.size) {
            assertNotEquals(s[i - 1], s[i])
        }
        // Crucial — must not be all zeros (the documented footgun).
        assertNotEquals(0f, s[0])
        assertNotEquals(0f, s.last())
    }

    @Test
    fun `unitNormState init is the documented ramp not zeros`() {
        val s = EmaInit.unitNormState()
        assertEquals(Constants.N_DF, s.size)
        assertEquals(0.001f, s[0], 1e-9f)
        assertEquals(0.0001f, s.last(), 1e-9f)
        // All entries strictly positive — required for the `1/sqrt(s)` op.
        s.forEach { v -> assert(v > 0f) { "unitNormState has non-positive entry: $v" } }
    }
}
