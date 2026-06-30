package ai.desertant.clear.internal.dsp

import ai.desertant.clear.internal.Constants

/**
 * Chunked inference loop. The model sees fixed-size T=200 chunks; the
 * full spectrum + features are partitioned, packed into [1, 1, T, F, ...]
 * tensors (features shifted by `convLookahead`), and the enhanced
 * output is scattered back into the global buffer.
 *
 * The model is plugged in via the [ModelBridge] interface — :dsp has
 * no dependency on a model runtime.
 */
class Inference(
    private val model: ModelBridge,
    private val chunkLen: Int = Constants.CHUNK_LEN,
) {
    private val nFreq = Constants.N_FREQ
    private val nErb = Constants.N_ERB
    private val nDf = Constants.N_DF
    private val convLookahead = Constants.CONV_LOOKAHEAD

    private val specScratch = FloatArray(chunkLen * nFreq * 2)
    private val erbScratch = FloatArray(chunkLen * nErb)
    private val featSpecScratch = FloatArray(chunkLen * nDf * 2)

    data class Enhanced(val real: FloatArray, val imag: FloatArray, val nFrames: Int) {
        override fun equals(other: Any?): Boolean =
            other is Enhanced && nFrames == other.nFrames &&
            real.contentEquals(other.real) && imag.contentEquals(other.imag)
        override fun hashCode(): Int =
            31 * (31 * real.contentHashCode() + imag.contentHashCode()) + nFrames
    }

    fun run(
        specReal: FloatArray,
        specImag: FloatArray,
        featErb: FloatArray,
        featSpecReal: FloatArray,
        featSpecImag: FloatArray,
        nFrames: Int,
        onChunkProgress: ((Float) -> Unit)? = null,
    ): Enhanced {
        require(specReal.size == nFrames * nFreq && specImag.size == nFrames * nFreq) {
            "spec real/imag size mismatch"
        }
        require(featErb.size == nFrames * nErb) { "featErb size mismatch" }
        require(featSpecReal.size == nFrames * nDf) { "featSpecReal size mismatch" }
        require(featSpecImag.size == nFrames * nDf) { "featSpecImag size mismatch" }

        val outReal = FloatArray(nFrames * nFreq)
        val outImag = FloatArray(nFrames * nFreq)
        val nChunks = (nFrames + chunkLen - 1) / chunkLen
        for (c in 0 until nChunks) {
            val chunkStart = c * chunkLen
            val chunkEnd = minOf(chunkStart + chunkLen, nFrames)
            runChunk(chunkStart, chunkEnd, nFrames,
                featErb, featSpecReal, featSpecImag,
                specReal, specImag, outReal, outImag)
            onChunkProgress?.invoke((c + 1).toFloat() / nChunks)
        }
        return Enhanced(outReal, outImag, nFrames)
    }

    private fun runChunk(
        start: Int, end: Int, nFrames: Int,
        featErb: FloatArray, featSpecReal: FloatArray, featSpecImag: FloatArray,
        specReal: FloatArray, specImag: FloatArray,
        outReal: FloatArray, outImag: FloatArray,
    ) {
        val T = chunkLen
        val validFrames = end - start

        // Frames outside the valid range stay zero-padded.
        specScratch.fill(0f)
        erbScratch.fill(0f)
        featSpecScratch.fill(0f)

        // Feature window (shifted by convLookahead).
        val tStart = maxOf(0, -start - convLookahead)
        val tEnd = minOf(T, nFrames - start - convLookahead)
        if (tEnd > tStart) {
            val n = tEnd - tStart
            val srcFrameStart = start + tStart + convLookahead
            System.arraycopy(
                featErb, srcFrameStart * nErb,
                erbScratch, tStart * nErb,
                n * nErb
            )
            for (i in 0 until n * nDf) {
                val srcBin = (srcFrameStart * nDf) + i
                val dstBin = (tStart * nDf) + i
                featSpecScratch[dstBin * 2] = featSpecReal[srcBin]
                featSpecScratch[dstBin * 2 + 1] = featSpecImag[srcBin]
            }
        }

        // Spec window (no lookahead shift).
        val sStart = maxOf(0, -start)
        val sEnd = minOf(T, nFrames - start)
        if (sEnd > sStart) {
            val n = sEnd - sStart
            val srcFrameStart = start + sStart
            for (i in 0 until n * nFreq) {
                val srcBin = (srcFrameStart * nFreq) + i
                val dstBin = (sStart * nFreq) + i
                specScratch[dstBin * 2] = specReal[srcBin]
                specScratch[dstBin * 2 + 1] = specImag[srcBin]
            }
        }

        val enhanced = model.runChunk(specScratch, erbScratch, featSpecScratch)
        require(enhanced.size == T * nFreq * 2) {
            "model output size ${enhanced.size} ≠ ${T * nFreq * 2}"
        }

        // Scatter [T, F, 2] interleaved back into parallel real/imag globals.
        if (validFrames > 0) {
            val totalBins = validFrames * nFreq
            for (i in 0 until totalBins) {
                val dstBin = start * nFreq + i
                outReal[dstBin] = enhanced[i * 2]
                outImag[dstBin] = enhanced[i * 2 + 1]
            }
        }
    }
}

/**
 * Bridge to the speech-enhancement model. :dsp never imports a model
 * runtime; :library provides the ONNX Runtime implementation.
 */
fun interface ModelBridge {
    /**
     * @param spec      [1, 1, T, nFreq, 2] flattened, interleaved real/imag
     * @param featErb   [1, 1, T, nErb] flattened
     * @param featSpec  [1, 1, T, nDf, 2] flattened, interleaved real/imag
     * @return enhanced spec, same shape as `spec`
     */
    fun runChunk(spec: FloatArray, featErb: FloatArray, featSpec: FloatArray): FloatArray
}

/** Identity ModelBridge for tests — returns the input spec unchanged. */
class IdentityModelBridge : ModelBridge {
    override fun runChunk(spec: FloatArray, featErb: FloatArray, featSpec: FloatArray): FloatArray =
        spec.copyOf()
}
