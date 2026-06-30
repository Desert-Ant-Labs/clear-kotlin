package ai.desertant.clear.internal.mastering

import kotlin.math.exp
import kotlin.math.log10

/**
 * Streaming K-weighted loudness meter. Consume audio chunks via
 * [consume]; [finalize] applies BS.1770 gating and returns the
 * integrated LUFS + true peak.
 */
class StreamingMeter(private val sampleRate: Int = 48_000) {

    private val blockLen: Int = (0.400 * sampleRate).toInt()
    private val blockHop: Int = (0.100 * sampleRate).toInt()

    private val preB = floatArrayOf(1.53512485958697f, -2.69169618940638f, 1.19839281085285f)
    private val preA = floatArrayOf(1.0f, -1.69065929318241f, 0.73248077421585f)
    private val rlbB = floatArrayOf(1.0f, -2.0f, 1.0f)
    private val rlbA = floatArrayOf(1.0f, -1.99004745483398f, 0.99007225036621f)

    private var preFilters: Array<StreamingBiquad>? = null
    private var rlbFilters: Array<StreamingBiquad>? = null

    // K-weighted samples pending block emission, per channel.
    private var buffered: Array<FloatArray>? = null
    private var bufferedFill: Int = 0
    private val blockMs = mutableListOf<Double>()

    private var truePeakAbs: Float = 0f
    private var truePeakTail: FloatArray? = null

    /** Consume one chunk of audio. `channels` is `[channels][frames]`. */
    fun consume(channels: Array<FloatArray>) {
        val n = channels.firstOrNull()?.size ?: 0
        if (n == 0) return
        val nCh = channels.size

        if (preFilters == null) {
            preFilters = Array(nCh) { StreamingBiquad(preB, preA) }
            rlbFilters = Array(nCh) { StreamingBiquad(rlbB, rlbA) }
            buffered = Array(nCh) { FloatArray(blockLen + n) }
            bufferedFill = 0
            truePeakTail = FloatArray(nCh)
        }

        // 4× lerp true peak across all channels.
        for (c in 0 until nCh) {
            val ch = channels[c]
            var prev = truePeakTail!![c]
            for (i in 0 until n) {
                val cur = ch[i]
                val a0 = if (cur < 0f) -cur else cur
                if (a0 > truePeakAbs) truePeakAbs = a0
                val v1 = 0.75f * prev + 0.25f * cur
                val v2 = 0.5f * prev + 0.5f * cur
                val v3 = 0.25f * prev + 0.75f * cur
                val m1 = if (v1 < 0f) -v1 else v1
                val m2 = if (v2 < 0f) -v2 else v2
                val m3 = if (v3 < 0f) -v3 else v3
                if (m1 > truePeakAbs) truePeakAbs = m1
                if (m2 > truePeakAbs) truePeakAbs = m2
                if (m3 > truePeakAbs) truePeakAbs = m3
                prev = cur
            }
            truePeakTail!![c] = prev
        }

        val weighted = Array(nCh) { c ->
            rlbFilters!![c].process(preFilters!![c].process(channels[c]))
        }

        ensureBufferedCapacity(bufferedFill + n)
        for (c in 0 until nCh) {
            System.arraycopy(weighted[c], 0, buffered!![c], bufferedFill, n)
        }
        bufferedFill += n

        while (bufferedFill >= blockLen) {
            var sumSq = 0.0
            for (c in 0 until nCh) {
                val ch = buffered!![c]
                for (i in 0 until blockLen) {
                    val v = ch[i].toDouble()
                    sumSq += v * v
                }
            }
            blockMs.add(sumSq / blockLen)
            // Slide each per-channel buffer forward by hop.
            for (c in 0 until nCh) {
                System.arraycopy(buffered!![c], blockHop, buffered!![c], 0, bufferedFill - blockHop)
            }
            bufferedFill -= blockHop
        }
    }

    private fun ensureBufferedCapacity(needed: Int) {
        val buf = buffered ?: return
        if (buf[0].size >= needed) return
        val grown = ((needed * 3 + 1) / 2).coerceAtLeast(blockLen * 2)
        for (c in buf.indices) {
            val n = FloatArray(grown)
            System.arraycopy(buf[c], 0, n, 0, bufferedFill)
            buf[c] = n
        }
    }

    /** Finalize with BS.1770 gating; resets internal state for reuse. */
    fun finalize(): R128.Measurement {
        if (blockMs.isEmpty()) {
            reset()
            return R128.Measurement(Double.NEGATIVE_INFINITY, truePeakDbfs())
        }
        val absGate = Math.pow(10.0, (-70.0 + 0.691) / 10.0)
        val gatedAbs = blockMs.filter { it > absGate }
        if (gatedAbs.isEmpty()) {
            val tp = truePeakDbfs()
            reset()
            return R128.Measurement(Double.NEGATIVE_INFINITY, tp)
        }
        val meanAbs = gatedAbs.average()
        val ungated = -0.691 + 10.0 * log10(meanAbs)
        val relThreshold = ungated - 10.0
        val relMs = Math.pow(10.0, (relThreshold + 0.691) / 10.0)
        val gatedFinal = blockMs.filter { it > absGate && it > relMs }
        val integrated = if (gatedFinal.isEmpty()) ungated
        else -0.691 + 10.0 * log10(gatedFinal.average())
        val tp = truePeakDbfs()
        reset()
        return R128.Measurement(integrated, tp)
    }

    private fun truePeakDbfs(): Double =
        if (truePeakAbs <= 0f) Double.NEGATIVE_INFINITY
        else 20.0 * log10(truePeakAbs.toDouble())

    private fun reset() {
        blockMs.clear()
        truePeakAbs = 0f
        bufferedFill = 0
        preFilters = null
        rlbFilters = null
        buffered = null
        truePeakTail = null
    }
}

/** Streaming biquad — same math as [Biquad], history carried across calls. */
internal class StreamingBiquad(b: FloatArray, a: FloatArray) {
    private val b0: Float; private val b1: Float; private val b2: Float
    private val a1: Float; private val a2: Float
    private var xm1 = 0f; private var xm2 = 0f
    private var ym1 = 0f; private var ym2 = 0f

    init {
        require(b.size == 3 && a.size == 3)
        val a0 = a[0]
        b0 = b[0] / a0; b1 = b[1] / a0; b2 = b[2] / a0
        a1 = a[1] / a0; a2 = a[2] / a0
    }

    fun process(x: FloatArray): FloatArray {
        val n = x.size
        val y = FloatArray(n)
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

/**
 * Streaming joint look-ahead limiter — same math as
 * [R128.applyLookaheadLimiter], state carried across arbitrary chunk
 * boundaries. Holds up to 5 ms of lookahead per channel; the tail is
 * emitted on the next call or via [flush] at end-of-stream.
 */
class StreamingLimiter(
    ceilDbtp: Double,
    private val sampleRate: Int = 48_000,
) {
    private val ceil: Float = Math.pow(10.0, ceilDbtp / 20.0).toFloat()
    private val lookahead: Int = (0.005 * sampleRate).toInt()
    private val releaseCoef: Float = exp(-1.0 / (sampleRate * 0.050)).toFloat()
    private var envGain: Float = 1f
    private var pending: Array<FloatArray> = arrayOf()

    /** Process an audio chunk; the last `lookahead` samples stay pending. */
    fun process(chunk: Array<FloatArray>): Array<FloatArray> {
        val nCh = chunk.size
        if (nCh == 0) return arrayOf()
        val chunkLen = chunk.firstOrNull()?.size ?: 0
        if (pending.isEmpty()) {
            pending = Array(nCh) { FloatArray(0) }
        }
        val joined = Array(nCh) { c ->
            val p = pending[c]
            val total = p.size + chunkLen
            val out = FloatArray(total)
            System.arraycopy(p, 0, out, 0, p.size)
            System.arraycopy(chunk[c], 0, out, p.size, chunkLen)
            out
        }
        val total = joined[0].size
        val finishable = maxOf(0, total - lookahead)
        val out = Array(nCh) { FloatArray(finishable) }
        applyLimiter(joined, total, finishable, out, nCh)
        val carry = total - finishable
        pending = Array(nCh) { c ->
            val tail = FloatArray(carry)
            System.arraycopy(joined[c], finishable, tail, 0, carry)
            tail
        }
        return out
    }

    /** Emit any pending samples after the last `process` call. */
    fun flush(): Array<FloatArray> {
        val nCh = pending.size
        if (nCh == 0) return arrayOf()
        val n = pending[0].size
        if (n == 0) {
            pending = arrayOf()
            return Array(nCh) { FloatArray(0) }
        }
        val out = Array(nCh) { FloatArray(n) }
        applyLimiter(pending, n, n, out, nCh)
        pending = arrayOf()
        return out
    }

    private fun applyLimiter(
        joined: Array<FloatArray>,
        total: Int,
        finishable: Int,
        out: Array<FloatArray>,
        nCh: Int,
    ) {
        if (total == 0 || finishable == 0) return
        val absX = FloatArray(total)
        for (i in 0 until total) {
            var m = 0f
            for (c in 0 until nCh) {
                val v = joined[c][i]
                val a = if (v < 0f) -v else v
                if (a > m) m = a
            }
            absX[i] = m
        }
        val dq = IntArray(total + 1)
        var dqHead = 0; var dqTail = 0
        val seedN = minOf(lookahead, total)
        for (j in 0 until seedN) {
            val vj = absX[j]
            while (dqTail > dqHead && absX[dq[dqTail - 1]] <= vj) dqTail -= 1
            dq[dqTail] = j
            dqTail += 1
        }
        var env = envGain
        for (i in 0 until finishable) {
            val end = i + lookahead
            if (end in lookahead until total) {
                val ve = absX[end]
                while (dqTail > dqHead && absX[dq[dqTail - 1]] <= ve) dqTail -= 1
                dq[dqTail] = end
                dqTail += 1
            }
            while (dqHead < dqTail && dq[dqHead] < i) dqHead += 1
            val maxAhead = if (dqHead < dqTail) absX[dq[dqHead]] else 0f
            if (maxAhead > ceil) {
                val required = ceil / maxAhead
                if (required < env) env = required
            }
            for (c in 0 until nCh) out[c][i] = joined[c][i] * env
            env = 1f - (1f - env) * releaseCoef
        }
        envGain = env
    }
}
