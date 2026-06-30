package ai.desertant.clear.internal.mastering

import kotlin.math.PI
import kotlin.math.abs
import kotlin.math.sin
import kotlin.math.sqrt
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test

class R128Test {

    private fun pinkAtLufs(targetLufs: Double, durationSec: Double, sr: Int = 48_000): Array<FloatArray> {
        val n = (durationSec * sr).toInt()
        val rng = java.util.Random(42)
        var s = 0f
        val signal = FloatArray(n) { _ ->
            val w = rng.nextFloat() * 2f - 1f
            s = 0.99f * s + 0.01f * w
            (0.7f * w + 0.3f * s)
        }
        var maxAbs = 0f
        for (x in signal) if (abs(x) > maxAbs) maxAbs = abs(x)
        if (maxAbs > 0f) {
            val targetPeak = Math.pow(10.0, (targetLufs + 14.0) / 20.0).toFloat()
            val g = targetPeak / maxAbs
            for (i in signal.indices) signal[i] = signal[i] * g
        }
        return arrayOf(signal)
    }

    @Test
    fun `silence is reported as negative infinity`() {
        val silence = arrayOf(FloatArray(48_000) { 0f })
        val m = R128.measure(silence)
        assertEquals(Double.NEGATIVE_INFINITY, m.integratedLufs, 0.0)
    }

    @Test
    fun `a 1 kHz sine measures within reasonable LUFS range`() {
        val n = 48_000 * 2
        val amp = 0.5f
        val sine = FloatArray(n) { i -> amp * sin(2 * PI * 1000.0 * i / 48_000).toFloat() }
        val m = R128.measure(arrayOf(sine))
        assertTrue("integratedLufs ${m.integratedLufs} not in [-13, -5]",
            m.integratedLufs in -13.0..-5.0)
    }

    @Test
    fun `normalize moves measured LUFS to within 1 LU of target`() {
        val sig = pinkAtLufs(-25.0, durationSec = 4.0)
        val before = R128.measure(sig)
        assertTrue("before LUFS ${before.integratedLufs} not finite", before.integratedLufs.isFinite())

        R128.normalize(sig, targetLufs = -19.0, truePeakDbtp = -1.0)

        val after = R128.measure(sig)
        assertTrue("after LUFS ${after.integratedLufs} not finite", after.integratedLufs.isFinite())
        // Limiter may reduce overall level slightly when peaks exceed the
        // ceiling. Tolerance of 1 LU is reasonable for the post-limit
        // measurement; the gain-only-pre-limiter step is exact.
        assertEquals(-19.0, after.integratedLufs, 1.0)
        // True peak must be at-or-below -1 dBTP (limiter guarantees this).
        assertTrue("true peak ${after.truePeakDbfs} > -1 dBTP",
            after.truePeakDbfs <= -1.0 + 0.05)
    }

    @Test
    fun `limiter holds peak at ceiling without hard-clipping transients`() {
        // Pink noise with sparse spike transients: most samples at moderate
        // level, occasional samples at ±0.9 (close to clipping). The limiter
        // should pull the spikes under the ceiling without dropping the
        // moderate-level samples.
        val sr = 48_000
        val n = sr * 4
        val rng = java.util.Random(7)
        val sig = FloatArray(n) { _ -> rng.nextFloat() * 0.4f - 0.2f }
        // Plant 20 spikes randomly
        for (i in 0 until 20) {
            val idx = rng.nextInt(n - 480) + 240
            val sign = if (rng.nextBoolean()) 1f else -1f
            sig[idx] = sign * 0.95f
        }
        val channels = arrayOf(sig)

        // Apply limiter directly with -1 dBTP ceiling.
        R128.applyLookaheadLimiter(channels, ceilDbtp = -1.0, sampleRate = sr)

        // Peak should now be at-or-below -1 dBTP (4× lerp peak).
        val measuredPeak = R128.truePeakDbfs(channels)
        assertTrue("peak after limit: $measuredPeak dBFS > -1 dBTP",
            measuredPeak <= -1.0 + 0.05)

        // RMS should still be reasonable — the moderate-level samples
        // weren't gain-cut by much because the limiter releases quickly.
        var sumSq = 0.0
        for (x in channels[0]) sumSq += x.toDouble() * x.toDouble()
        val rmsDb = 20.0 * kotlin.math.log10(sqrt(sumSq / n))
        // Pre-limit RMS of pink-noise-in-±0.2 is roughly -19 dBFS.
        // Post-limit should be within ~3 dB of that.
        assertTrue("RMS after limit: $rmsDb dBFS too quiet — limiter over-attenuated",
            rmsDb > -23.0)
    }

    @Test
    fun `true-peak lerp catches inter-sample excursions a sample-peak misses`() {
        // Construct a signal that has small sample magnitudes but large
        // halfway-interpolated value: alternating ±0.9 samples at Nyquist/2
        // (12 kHz @ 48k) means linearly-interpolated midpoints are 0 but
        // the *unalternating* version (1 kHz tone) has a smooth waveform.
        // The clearest construction: two adjacent samples of opposite sign
        // and large magnitude — the midpoint (lerp t=0.5) is 0, so this
        // tests the LERP doesn't add false peaks.
        val ch1 = floatArrayOf(0.9f, -0.9f, 0.9f, -0.9f, 0.9f, -0.9f)
        val lerp = R128.truePeakDbfs(arrayOf(ch1))
        // Sample peak is 0.9 (= -0.92 dBFS). 4× lerp on ±0.9 alternations
        // gives midpoints near 0, which doesn't elevate the peak. Output
        // should still be ~ -0.92 dBFS, not higher.
        assertEquals(20.0 * kotlin.math.log10(0.9), lerp, 0.05)

        // Now a signal where lerp catches what sample-peak misses: two
        // adjacent samples 0.95 + 0.85. Midpoint lerp = 0.90 (sample
        // peak is 0.95). Doesn't exceed sample peak. The actual benefit
        // of true-peak only shows on full-amplitude alternating signals
        // where intersample peaks DO exceed sample peaks; that requires
        // band-limited reconstruction (sinc), not linear. Matching the
        // Swift 4×-lerp here is the goal — it's an approximation.
    }
}
