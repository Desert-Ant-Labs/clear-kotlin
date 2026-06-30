package ai.desertant.clear.internal.dsp

import ai.desertant.clear.internal.Constants
import ai.desertant.clear.internal.EmaInit
import kotlin.math.log10
import kotlin.math.sqrt

/**
 * Feature extraction. Returns the two normalized feature tensors the
 * model expects:
 * - `featErb`        [nFrames, N_ERB]   — ERB-band log-energy, per-band EMA
 * - `featSpec` (R,I) [nFrames, N_DF, 2] — complex spectrum, per-band √RMS EMA
 *
 * Both EMAs use ramp initializers (NOT zero) — see `EmaInit`.
 */
object FeatureExtractor {

    data class Features(
        val featErb: FloatArray,
        val featSpecReal: FloatArray,
        val featSpecImag: FloatArray,
        val nFrames: Int,
    )

    fun compute(real: FloatArray, imag: FloatArray, nFrames: Int): Features {
        val nFreq = Constants.N_FREQ
        val nErb = Constants.N_ERB
        val nDf = Constants.N_DF
        val alpha = Constants.NORM_ALPHA
        val oneMinus = 1f - alpha

        require(real.size == nFrames * nFreq && imag.size == nFrames * nFreq) {
            "real/imag size ≠ nFrames × nFreq"
        }

        val power = FloatArray(nFrames * nFreq)
        for (i in power.indices) {
            val r = real[i]
            val im = imag[i]
            power[i] = r * r + im * im
        }

        val erbPower = ErbFilterbank.projectPower(power, nFrames)

        // erbDB = 10 · log10(erbPower + 1e-10).
        val erbDB = FloatArray(erbPower.size)
        for (i in erbDB.indices) {
            erbDB[i] = (10.0 * log10((erbPower[i] + 1e-10f).toDouble())).toFloat()
        }

        val featErb = FloatArray(nFrames * nErb)
        val erbState = EmaInit.erbState()
        for (t in 0 until nFrames) {
            val off = t * nErb
            for (f in 0 until nErb) {
                val v = erbDB[off + f]
                erbState[f] = v * oneMinus + erbState[f] * alpha
                featErb[off + f] = (v - erbState[f]) / 40f
            }
        }

        // unit-norm pathway: per-frame √EMA of |spec| over the first nDf bins.
        val featSpecReal = FloatArray(nFrames * nDf)
        val featSpecImag = FloatArray(nFrames * nDf)
        val s = EmaInit.unitNormState()

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

        return Features(featErb, featSpecReal, featSpecImag, nFrames)
    }
}
