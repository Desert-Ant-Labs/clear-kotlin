package ai.desertant.clear.internal.dsp

import ai.desertant.clear.internal.Constants
import ai.desertant.clear.internal.EmaInit
import kotlin.math.log10
import kotlin.math.sqrt

/**
 * Streaming feature extraction — same math as the batch
 * [FeatureExtractor], EMA state carried across calls. Numerically
 * equivalent to a single batch `compute()` for any partition.
 */
class StreamingFeatureExtractor {
    private val nFreq = Constants.N_FREQ
    private val nErb = Constants.N_ERB
    private val nDf = Constants.N_DF
    private val alpha = Constants.NORM_ALPHA
    private val oneMinus = 1f - alpha

    // EMA state, carried across process() calls. Ramp-init — see EmaInit.
    private val erbState: FloatArray = EmaInit.erbState()
    private val s: FloatArray = EmaInit.unitNormState()

    data class FeaturesChunk(
        val featErb: FloatArray,
        val featSpecReal: FloatArray,
        val featSpecImag: FloatArray,
        val nFrames: Int,
    )

    /** Process `nFrames` worth of spectrum; returns the corresponding feature frames. */
    fun process(real: FloatArray, imag: FloatArray, nFrames: Int): FeaturesChunk {
        if (nFrames == 0) return FeaturesChunk(FloatArray(0), FloatArray(0), FloatArray(0), 0)
        require(real.size >= nFrames * nFreq && imag.size >= nFrames * nFreq)

        val power = FloatArray(nFrames * nFreq)
        for (i in 0 until nFrames * nFreq) {
            val r = real[i]
            val im = imag[i]
            power[i] = r * r + im * im
        }

        val erbPower = ErbFilterbank.projectPower(power, nFrames)

        val erbDB = FloatArray(erbPower.size)
        for (i in erbDB.indices) {
            erbDB[i] = (10.0 * log10((erbPower[i] + 1e-10f).toDouble())).toFloat()
        }

        val featErb = FloatArray(nFrames * nErb)
        for (t in 0 until nFrames) {
            val off = t * nErb
            for (f in 0 until nErb) {
                val v = erbDB[off + f]
                erbState[f] = v * oneMinus + erbState[f] * alpha
                featErb[off + f] = (v - erbState[f]) / 40f
            }
        }

        val featSpecReal = FloatArray(nFrames * nDf)
        val featSpecImag = FloatArray(nFrames * nDf)
        for (t in 0 until nFrames) {
            val inOff = t * nFreq
            val outOff = t * nDf
            for (f in 0 until nDf) {
                val r = real[inOff + f]
                val im = imag[inOff + f]
                val mag = sqrt((r * r + im * im).toDouble()).toFloat()
                s[f] = mag * oneMinus + s[f] * alpha
                val inv = (1.0 / sqrt(s[f].toDouble())).toFloat()
                featSpecReal[outOff + f] = r * inv
                featSpecImag[outOff + f] = im * inv
            }
        }

        return FeaturesChunk(featErb, featSpecReal, featSpecImag, nFrames)
    }

    /** Reset internal state — call between independent streams. */
    fun reset() {
        val freshErb = EmaInit.erbState()
        val freshS = EmaInit.unitNormState()
        System.arraycopy(freshErb, 0, erbState, 0, erbState.size)
        System.arraycopy(freshS, 0, s, 0, s.size)
    }
}
