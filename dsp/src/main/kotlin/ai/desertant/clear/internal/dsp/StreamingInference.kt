package ai.desertant.clear.internal.dsp

import ai.desertant.clear.internal.Constants

/**
 * Streaming inference. Buffers (spec, features); whenever
 * `chunkLen + convLookahead` frames are available, runs one chunk and
 * slides the buffer forward by `chunkLen`. [flush] drains the tail
 * with the missing lookahead zero-padded. Numerically equivalent to
 * [Inference.run] for any partition.
 */
class StreamingInference(
    private val model: ModelBridge,
    private val chunkLen: Int = Constants.CHUNK_LEN,
    private val convLookahead: Int = Constants.CONV_LOOKAHEAD,
) {
    private val nFreq = Constants.N_FREQ
    private val nErb = Constants.N_ERB
    private val nDf = Constants.N_DF

    // Growable buffers — initial cap covers 2 chunks worth.
    private var specReal = FloatArray((chunkLen + convLookahead + chunkLen) * nFreq)
    private var specImag = FloatArray(specReal.size)
    private var featErb = FloatArray((chunkLen + convLookahead + chunkLen) * nErb)
    private var featSpecReal = FloatArray((chunkLen + convLookahead + chunkLen) * nDf)
    private var featSpecImag = FloatArray(featSpecReal.size)
    private var bufFrames = 0

    private val specTensor = FloatArray(chunkLen * nFreq * 2)
    private val erbTensor = FloatArray(chunkLen * nErb)
    private val featSpecTensor = FloatArray(chunkLen * nDf * 2)

    /** Append `nFrames` of spec + features; emit any complete chunks. */
    fun process(
        real: FloatArray,
        imag: FloatArray,
        featErbIn: FloatArray,
        featSpecRealIn: FloatArray,
        featSpecImagIn: FloatArray,
        nFrames: Int,
    ): Inference.Enhanced {
        if (nFrames == 0) return Inference.Enhanced(FloatArray(0), FloatArray(0), 0)

        val needFrames = bufFrames + nFrames
        ensureCapacity(needFrames)
        System.arraycopy(real, 0, specReal, bufFrames * nFreq, nFrames * nFreq)
        System.arraycopy(imag, 0, specImag, bufFrames * nFreq, nFrames * nFreq)
        System.arraycopy(featErbIn, 0, featErb, bufFrames * nErb, nFrames * nErb)
        System.arraycopy(featSpecRealIn, 0, featSpecReal, bufFrames * nDf, nFrames * nDf)
        System.arraycopy(featSpecImagIn, 0, featSpecImag, bufFrames * nDf, nFrames * nDf)
        bufFrames = needFrames

        var emittedFrames = 0
        var outReal: FloatArray? = null
        var outImag: FloatArray? = null
        while (bufFrames >= chunkLen + convLookahead) {
            val (cReal, cImag) = runOneChunk()
            if (outReal == null) {
                outReal = FloatArray(chunkLen * nFreq * 2)
                outImag = FloatArray(chunkLen * nFreq * 2)
            }
            if ((emittedFrames + chunkLen) * nFreq > outReal!!.size) {
                outReal = outReal.copyOf((emittedFrames + chunkLen) * nFreq * 2)
                outImag = outImag!!.copyOf((emittedFrames + chunkLen) * nFreq * 2)
            }
            System.arraycopy(cReal, 0, outReal, emittedFrames * nFreq, chunkLen * nFreq)
            System.arraycopy(cImag, 0, outImag!!, emittedFrames * nFreq, chunkLen * nFreq)
            emittedFrames += chunkLen
            slideBuffers(chunkLen)
        }
        if (emittedFrames == 0) return Inference.Enhanced(FloatArray(0), FloatArray(0), 0)
        return Inference.Enhanced(
            outReal!!.copyOfRange(0, emittedFrames * nFreq),
            outImag!!.copyOfRange(0, emittedFrames * nFreq),
            emittedFrames
        )
    }

    /**
     * Drain the remaining buffer at end-of-stream. Missing
     * `convLookahead` feature frames are zero-padded.
     */
    fun flush(): Inference.Enhanced {
        if (bufFrames == 0) return Inference.Enhanced(FloatArray(0), FloatArray(0), 0)
        val totalFrames = bufFrames
        val outRealAll = FloatArray(totalFrames * nFreq)
        val outImagAll = FloatArray(totalFrames * nFreq)
        var written = 0
        var start = 0
        while (start < totalFrames) {
            val validIn = totalFrames - start
            packTensorsAt(start, validFrames = validIn)
            val enhancedFlat = model.runChunk(specTensor, erbTensor, featSpecTensor)
            val emit = minOf(chunkLen, validIn)
            for (i in 0 until emit * nFreq) {
                outRealAll[written * nFreq + i] = enhancedFlat[i * 2]
                outImagAll[written * nFreq + i] = enhancedFlat[i * 2 + 1]
            }
            written += emit
            start += chunkLen
        }
        bufFrames = 0
        return Inference.Enhanced(outRealAll, outImagAll, totalFrames)
    }

    /** Run one chunk on the first `chunkLen` frames of the buffer. */
    private fun runOneChunk(): Pair<FloatArray, FloatArray> {
        for (i in 0 until chunkLen * nFreq) {
            specTensor[i * 2] = specReal[i]
            specTensor[i * 2 + 1] = specImag[i]
        }
        // Features shifted by convLookahead.
        val erbSrcOff = convLookahead * nErb
        System.arraycopy(featErb, erbSrcOff, erbTensor, 0, chunkLen * nErb)
        val fsSrcOff = convLookahead * nDf
        for (i in 0 until chunkLen * nDf) {
            featSpecTensor[i * 2] = featSpecReal[fsSrcOff + i]
            featSpecTensor[i * 2 + 1] = featSpecImag[fsSrcOff + i]
        }
        val enhancedFlat = model.runChunk(specTensor, erbTensor, featSpecTensor)
        val outReal = FloatArray(chunkLen * nFreq)
        val outImag = FloatArray(chunkLen * nFreq)
        for (i in 0 until chunkLen * nFreq) {
            outReal[i] = enhancedFlat[i * 2]
            outImag[i] = enhancedFlat[i * 2 + 1]
        }
        return outReal to outImag
    }

    /**
     * Pack the model's input tensors for a chunk at frame `start`,
     * zero-filling past `validFrames`. Used by [flush] for the tail.
     */
    private fun packTensorsAt(start: Int, validFrames: Int) {
        java.util.Arrays.fill(specTensor, 0f)
        java.util.Arrays.fill(erbTensor, 0f)
        java.util.Arrays.fill(featSpecTensor, 0f)
        val specCount = minOf(chunkLen, validFrames)
        for (t in 0 until specCount) {
            val srcOff = (start + t) * nFreq
            val dstOff = t * nFreq * 2
            for (f in 0 until nFreq) {
                specTensor[dstOff + f * 2] = specReal[srcOff + f]
                specTensor[dstOff + f * 2 + 1] = specImag[srcOff + f]
            }
        }
        // Features read window clamped to available range; rest stays zero.
        val featStart = start + convLookahead
        val featEnd = minOf(featStart + chunkLen, bufFrames)
        val featValid = (featEnd - featStart).coerceAtLeast(0)
        if (featValid > 0) {
            System.arraycopy(featErb, featStart * nErb, erbTensor, 0, featValid * nErb)
            for (t in 0 until featValid) {
                val srcOff = (featStart + t) * nDf
                val dstOff = t * nDf * 2
                for (f in 0 until nDf) {
                    featSpecTensor[dstOff + f * 2] = featSpecReal[srcOff + f]
                    featSpecTensor[dstOff + f * 2 + 1] = featSpecImag[srcOff + f]
                }
            }
        }
    }

    private fun ensureCapacity(frames: Int) {
        if (specReal.size >= frames * nFreq) return
        val grow = { src: FloatArray, perFrame: Int ->
            val n = FloatArray(((frames * perFrame) * 3 + 1) / 2)
            System.arraycopy(src, 0, n, 0, bufFrames * perFrame)
            n
        }
        specReal = grow(specReal, nFreq)
        specImag = grow(specImag, nFreq)
        featErb = grow(featErb, nErb)
        featSpecReal = grow(featSpecReal, nDf)
        featSpecImag = grow(featSpecImag, nDf)
    }

    private fun slideBuffers(byFrames: Int) {
        val keep = bufFrames - byFrames
        System.arraycopy(specReal, byFrames * nFreq, specReal, 0, keep * nFreq)
        System.arraycopy(specImag, byFrames * nFreq, specImag, 0, keep * nFreq)
        System.arraycopy(featErb, byFrames * nErb, featErb, 0, keep * nErb)
        System.arraycopy(featSpecReal, byFrames * nDf, featSpecReal, 0, keep * nDf)
        System.arraycopy(featSpecImag, byFrames * nDf, featSpecImag, 0, keep * nDf)
        bufFrames = keep
    }
}
