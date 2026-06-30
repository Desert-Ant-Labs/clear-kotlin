package ai.desertant.clear.internal.dsp

/**
 * ERB filterbank — variable-width per-band averaging. Widths are
 * fixed at training time and must match Swift's `ERBFilterbank.widths`
 * exactly. Sum to 481 (= nFreq).
 */
internal object ErbFilterbank {
    val widths: IntArray = intArrayOf(
        2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
        5, 5, 7, 7, 8, 10, 12, 13, 15, 18, 20,
        24, 28, 31, 37, 42, 50, 56, 67
    )
    const val N_ERB: Int = 32
    val N_FREQ: Int = widths.sum()  // 481

    private val widthRecip: FloatArray = FloatArray(N_ERB) { f -> 1f / widths[f].toFloat() }

    init {
        require(widths.size == N_ERB) { "widths length ${widths.size} ≠ N_ERB $N_ERB" }
        require(N_FREQ == 481) { "widths sum $N_FREQ ≠ 481" }
    }

    /**
     * Mean per-bin power per band.
     *
     * @param power row-major `[nFrames, N_FREQ]`
     * @return      row-major `[nFrames, N_ERB]`
     */
    fun projectPower(power: FloatArray, nFrames: Int): FloatArray {
        require(power.size == nFrames * N_FREQ) {
            "power size ${power.size} ≠ nFrames($nFrames) × N_FREQ($N_FREQ)"
        }
        val out = FloatArray(nFrames * N_ERB)
        for (t in 0 until nFrames) {
            val rowIn = t * N_FREQ
            val rowOut = t * N_ERB
            var off = 0
            for (band in 0 until N_ERB) {
                val w = widths[band]
                var sum = 0f
                for (k in 0 until w) sum += power[rowIn + off + k]
                out[rowOut + band] = sum * widthRecip[band]
                off += w
            }
        }
        return out
    }
}
