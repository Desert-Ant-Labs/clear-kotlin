package ai.desertant.clear.internal

/**
 * DSP constants — must stay bit-identical to Swift
 * `Sources/Clear/Inference.swift`. Changes propagate to every parity test.
 */
object Constants {
    const val FFT_SIZE: Int = 960
    const val HOP_SIZE: Int = 480
    const val N_FREQ: Int = 481
    const val N_ERB: Int = 32
    const val N_DF: Int = 96
    const val CHUNK_LEN: Int = 200
    const val CONV_LOOKAHEAD: Int = 2
    const val DF_LOOKAHEAD: Int = 2
    const val DF_ORDER: Int = 5
    const val NORM_ALPHA: Float = 0.99f
    /** Forward STFT gain: 2·hop / fftSize². */
    const val WNORM: Float = 2f * HOP_SIZE / (FFT_SIZE.toFloat() * FFT_SIZE.toFloat())
    /** Pre-pad applied to the input before STFT framing. fftSize − hopSize. */
    const val STFT_PRE_PAD: Int = FFT_SIZE - HOP_SIZE
}

/**
 * EMA state initializers. NOT zero-init — zero-init makes the first
 * ~200 frames of every clip audibly wrong.
 */
internal object EmaInit {
    /** erbState: 32 bands, linspace(−60, −90). */
    fun erbState(): FloatArray {
        val step = (-90f - -60f) / (Constants.N_ERB - 1)
        return FloatArray(Constants.N_ERB) { f -> -60f + f * step }
    }

    /** unit-norm `s`: 96 bands, linspace(0.001, 0.0001). */
    fun unitNormState(): FloatArray {
        val step = (0.0001f - 0.001f) / (Constants.N_DF - 1)
        return FloatArray(Constants.N_DF) { f -> 0.001f + f * step }
    }
}
