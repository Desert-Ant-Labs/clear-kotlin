package ai.desertant.clear.internal.mastering

import kotlin.math.exp
import kotlin.math.log10
import kotlin.math.max

/**
 * EBU R128 / ITU-R BS.1770-4 loudness measurement + look-ahead limiter
 * mastering chain. Batch-only — see [StreamingMeter] and
 * [StreamingLimiter] for streaming variants.
 *
 * - Integrated LUFS with -70 LUFS absolute + -10 LU relative gates.
 * - 4× linear-interp true-peak detection.
 * - Loudness normalization to a target LUFS, ceilinged at the dBTP cap.
 * - Joint soft-knee look-ahead limiter (5 ms lookahead, 50 ms release).
 */
object R128 {

    data class Measurement(
        /** Integrated loudness in LUFS. NEGATIVE_INFINITY on silence. */
        val integratedLufs: Double,
        /** 4×-lerp true-peak max across all channels in dBFS. NEGATIVE_INFINITY on silence. */
        val truePeakDbfs: Double,
    )

    // K-weighting biquad coefficients for 48 kHz.
    private val PRE_B = floatArrayOf(1.53512485958697f, -2.69169618940638f, 1.19839281085285f)
    private val PRE_A = floatArrayOf(1.0f, -1.69065929318241f, 0.73248077421585f)
    private val RLB_B = floatArrayOf(1.0f, -2.0f, 1.0f)
    private val RLB_A = floatArrayOf(1.0f, -1.99004745483398f, 0.99007225036621f)

    fun measure(channels: Array<FloatArray>, sampleRate: Int = 48_000): Measurement {
        require(channels.isNotEmpty()) { "no channels" }
        val nFrames = channels[0].size
        if (nFrames == 0) return Measurement(Double.NEGATIVE_INFINITY, Double.NEGATIVE_INFINITY)

        val weighted = Array(channels.size) { c ->
            // K-weighting: pre-filter shelf + RLB high-pass.
            val pre = Biquad(PRE_B, PRE_A)
            val rlb = Biquad(RLB_B, RLB_A)
            rlb.process(pre.process(channels[c]))
        }

        val blockLen = (0.400 * sampleRate).toInt()
        val hop = (0.100 * sampleRate).toInt()
        if (nFrames < blockLen) {
            return Measurement(Double.NEGATIVE_INFINITY, truePeakDbfs(channels))
        }

        val blockMs = mutableListOf<Double>()
        var blockStart = 0
        while (blockStart + blockLen <= nFrames) {
            var sumSq = 0.0
            for (ch in weighted) {
                for (i in blockStart until blockStart + blockLen) {
                    val v = ch[i].toDouble()
                    sumSq += v * v
                }
            }
            blockMs.add(sumSq / blockLen)
            blockStart += hop
        }
        if (blockMs.isEmpty()) return Measurement(Double.NEGATIVE_INFINITY, truePeakDbfs(channels))

        // BS.1770 gating
        val absGate = Math.pow(10.0, (-70.0 + 0.691) / 10.0)
        val gatedAbs = blockMs.filter { it > absGate }
        if (gatedAbs.isEmpty()) return Measurement(Double.NEGATIVE_INFINITY, truePeakDbfs(channels))

        val meanAbs = gatedAbs.average()
        val ungated = -0.691 + 10.0 * log10(meanAbs)
        val relThreshold = ungated - 10.0
        val relMs = Math.pow(10.0, (relThreshold + 0.691) / 10.0)
        val gatedFinal = blockMs.filter { it > absGate && it > relMs }
        if (gatedFinal.isEmpty()) return Measurement(ungated, truePeakDbfs(channels))

        val integrated = -0.691 + 10.0 * log10(gatedFinal.average())
        return Measurement(integrated, truePeakDbfs(channels))
    }

    /** Per-channel integrated LUFS — same gating as joint [measure], run per channel. */
    fun measurePerChannel(channels: Array<FloatArray>, sampleRate: Int = 48_000): List<Double> {
        return channels.map { ch -> measure(arrayOf(ch), sampleRate).integratedLufs }
    }

    /** Apply per-channel pre-gain so each channel hits `targetLufs` independently. */
    fun balanceChannels(
        channels: Array<FloatArray>,
        targetLufs: Double,
        sampleRate: Int = 48_000,
    ) {
        val measurements = measurePerChannel(channels, sampleRate)
        for ((c, lufs) in measurements.withIndex()) {
            if (!lufs.isFinite()) continue
            val gain = Math.pow(10.0, (targetLufs - lufs) / 20.0).toFloat()
            val ch = channels[c]
            for (i in ch.indices) ch[i] *= gain
        }
    }

    /** 4× linear-interp true-peak across all channels in dBFS. */
    fun truePeakDbfs(channels: Array<FloatArray>): Double {
        var maxAbs = 0f
        for (ch in channels) {
            if (ch.isEmpty()) continue
            // Phase 0 (sample peak) + 3 lerp phases between prev and cur.
            var prev = 0f
            for (cur in ch) {
                val absCur = if (cur < 0f) -cur else cur
                if (absCur > maxAbs) maxAbs = absCur
                val a1 = 0.75f * prev + 0.25f * cur
                val a2 = 0.5f * prev + 0.5f * cur
                val a3 = 0.25f * prev + 0.75f * cur
                val m1 = if (a1 < 0f) -a1 else a1
                val m2 = if (a2 < 0f) -a2 else a2
                val m3 = if (a3 < 0f) -a3 else a3
                if (m1 > maxAbs) maxAbs = m1
                if (m2 > maxAbs) maxAbs = m2
                if (m3 > maxAbs) maxAbs = m3
                prev = cur
            }
        }
        if (maxAbs == 0f) return Double.NEGATIVE_INFINITY
        return 20.0 * log10(maxAbs.toDouble())
    }

    /**
     * Normalize `channels` (in-place) to `targetLufs`: pre-gain then
     * look-ahead limiter at the dBTP ceiling. Returns the
     * pre-normalization measurement.
     */
    fun normalize(
        channels: Array<FloatArray>,
        targetLufs: Double,
        truePeakDbtp: Double,
        sampleRate: Int = 48_000,
        maxLoudnessGainDb: Double = 9.0,
    ): Measurement {
        val meas = measure(channels, sampleRate)
        if (!meas.integratedLufs.isFinite()) return meas

        // Cap UPWARD gain at maxLoudnessGainDb — keeps quiet inputs from
        // being amplified past the model's noise floor. Downward gain
        // (attenuating loud inputs) is unaffected.
        val requestedGainDb = targetLufs - meas.integratedLufs
        val effectiveGainDb = minOf(requestedGainDb, maxLoudnessGainDb)
        val gain = Math.pow(10.0, effectiveGainDb / 20.0).toFloat()
        for (ch in channels) for (i in ch.indices) ch[i] *= gain

        applyLookaheadLimiter(channels, truePeakDbtp, sampleRate)
        return meas
    }

    /**
     * Joint look-ahead soft-knee limiter — 5 ms lookahead, 50 ms release.
     * Sliding-window max via a monotonic deque; instantaneous attack,
     * exponential release.
     */
    fun applyLookaheadLimiter(
        channels: Array<FloatArray>,
        ceilDbtp: Double,
        sampleRate: Int = 48_000,
    ) {
        val n = channels.firstOrNull()?.size ?: 0
        if (n == 0) return
        val nCh = channels.size
        val ceil = Math.pow(10.0, ceilDbtp / 20.0).toFloat()
        val lookahead = (0.005 * sampleRate).toInt().coerceAtMost(n)
        val releaseCoef = exp(-1.0 / (sampleRate * 0.050)).toFloat()

        // Joint |x| per sample (max across channels).
        val absX = FloatArray(n)
        for (i in 0 until n) {
            var m = 0f
            for (c in 0 until nCh) {
                val v = channels[c][i]
                val a = if (v < 0f) -v else v
                if (a > m) m = a
            }
            absX[i] = m
        }

        // Ring-buffer deque sized to the live window. The previous IntArray(n+1)
        // allocated up to ~115 MB for a 5-min stereo file where only `lookahead+1`
        // slots are ever live at once.
        val dqCap = lookahead + 2
        val dq = IntArray(dqCap)
        var dqHead = 0
        var dqTail = 0
        fun dqSize() = dqTail - dqHead
        for (j in 0 until lookahead) {
            val vj = absX[j]
            while (dqSize() > 0 && absX[dq[(dqTail - 1).mod(dqCap)]] <= vj) dqTail -= 1
            dq[dqTail.mod(dqCap)] = j
            dqTail += 1
        }

        var env = 1f
        for (i in 0 until n) {
            val end = i + lookahead
            if (end in lookahead until n) {
                val ve = absX[end]
                while (dqSize() > 0 && absX[dq[(dqTail - 1).mod(dqCap)]] <= ve) dqTail -= 1
                dq[dqTail.mod(dqCap)] = end
                dqTail += 1
            }
            while (dqSize() > 0 && dq[dqHead.mod(dqCap)] < i) dqHead += 1
            val maxAhead = if (dqSize() > 0) absX[dq[dqHead.mod(dqCap)]] else 0f
            if (maxAhead > ceil) {
                val required = ceil / maxAhead
                if (required < env) env = required
            }
            for (c in 0 until nCh) channels[c][i] *= env
            env = 1f - (1f - env) * releaseCoef
        }
    }
}

/** Direct-form-I biquad, batch — processes a full array per call. */
internal class Biquad(b: FloatArray, a: FloatArray) {
    private val b0: Float
    private val b1: Float
    private val b2: Float
    private val a1: Float
    private val a2: Float

    init {
        require(b.size == 3 && a.size == 3) { "biquad needs 3-tap b/a" }
        val a0 = a[0]
        b0 = b[0] / a0
        b1 = b[1] / a0
        b2 = b[2] / a0
        a1 = a[1] / a0
        a2 = a[2] / a0
    }

    fun process(x: FloatArray): FloatArray {
        val n = x.size
        val y = FloatArray(n)
        var xm1 = 0f; var xm2 = 0f
        var ym1 = 0f; var ym2 = 0f
        for (i in 0 until n) {
            val xi = x[i]
            val yi = b0 * xi + b1 * xm1 + b2 * xm2 - a1 * ym1 - a2 * ym2
            y[i] = yi
            xm2 = xm1; xm1 = xi
            ym2 = ym1; ym1 = yi
        }
        return y
    }
}
