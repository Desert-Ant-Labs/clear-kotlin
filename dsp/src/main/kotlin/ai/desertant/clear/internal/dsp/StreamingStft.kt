package ai.desertant.clear.internal.dsp

import ai.desertant.clear.internal.Constants
import org.jtransforms.fft.FloatFFT_1D

/**
 * Streaming forward STFT — same math as [Stft.forward] but accepts
 * audio chunks of arbitrary size. Numerically equivalent to the batch
 * version for any partition of the input.
 */
class StreamingStft(
    private val fftSize: Int = Constants.FFT_SIZE,
    private val hopSize: Int = Constants.HOP_SIZE,
) {
    val nFreq: Int = fftSize / 2 + 1

    private val window: FloatArray = Stft.buildVorbisWindow(fftSize)
    private val wnorm: Float = 2f * hopSize / (fftSize.toFloat() * fftSize.toFloat())
    private val fft = FloatFFT_1D(fftSize.toLong())
    private val rscratch = FloatArray(fftSize)

    // Sliding accumulator, pre-seeded with the analysis pre-pad.
    private val carry = FloatArray(fftSize)
    private var carryFill: Int = fftSize - hopSize

    data class FrameChunk(val real: FloatArray, val imag: FloatArray, val nFrames: Int) {
        override fun equals(other: Any?): Boolean =
            other is FrameChunk && nFrames == other.nFrames &&
            real.contentEquals(other.real) && imag.contentEquals(other.imag)
        override fun hashCode(): Int =
            31 * (31 * real.contentHashCode() + imag.contentHashCode()) + nFrames
    }

    /**
     * Append `length` samples to the accumulator; emit any complete
     * STFT frames. Returns the frames produced for this call.
     */
    fun process(samples: FloatArray, offset: Int = 0, length: Int = samples.size - offset): FrameChunk {
        if (length <= 0) return FrameChunk(FloatArray(0), FloatArray(0), 0)
        require(offset >= 0 && offset + length <= samples.size) { "offset/length out of range" }

        val combined = carryFill + length
        val nFrames = if (combined >= fftSize) (combined - fftSize) / hopSize + 1 else 0

        if (nFrames == 0) {
            // No emission — just accumulate. combined < fftSize == carry.size.
            System.arraycopy(samples, offset, carry, carryFill, length)
            carryFill = combined
            return FrameChunk(FloatArray(0), FloatArray(0), 0)
        }

        val real = FloatArray(nFrames * nFreq)
        val imag = FloatArray(nFrames * nFreq)
        val nyq = fftSize / 2
        for (t in 0 until nFrames) {
            val start = t * hopSize
            for (n in 0 until fftSize) {
                val idx = start + n
                val sample = if (idx < carryFill) carry[idx] else samples[offset + (idx - carryFill)]
                rscratch[n] = sample * window[n]
            }
            fft.realForward(rscratch)
            val base = t * nFreq
            real[base] = rscratch[0] * wnorm
            imag[base] = 0f
            for (k in 1 until nyq) {
                real[base + k] = rscratch[2 * k] * wnorm
                imag[base + k] = rscratch[2 * k + 1] * wnorm
            }
            real[base + nyq] = rscratch[1] * wnorm
            imag[base + nyq] = 0f
        }

        // Carry samples that haven't been fully windowed yet.
        val srcStart = nFrames * hopSize
        val newCarryFill = combined - srcStart
        if (srcStart > 0 || newCarryFill > 0) {
            val temp = FloatArray(newCarryFill)
            for (i in 0 until newCarryFill) {
                val srcIdx = srcStart + i
                temp[i] = if (srcIdx < carryFill) carry[srcIdx] else samples[offset + (srcIdx - carryFill)]
            }
            System.arraycopy(temp, 0, carry, 0, newCarryFill)
        }
        carryFill = newCarryFill
        return FrameChunk(real, imag, nFrames)
    }

    /** Reset internal state. Call between independent streams. */
    fun reset() {
        java.util.Arrays.fill(carry, 0f)
        carryFill = fftSize - hopSize
    }
}

/**
 * Streaming inverse STFT — accumulates frames into an OLA buffer and
 * emits `nFrames * hopSize` samples per call. Numerically equivalent
 * to [Stft.inverse] for any partition.
 */
class StreamingIStft(
    private val fftSize: Int = Constants.FFT_SIZE,
    private val hopSize: Int = Constants.HOP_SIZE,
) {
    val nFreq: Int = fftSize / 2 + 1

    private val window: FloatArray = Stft.buildVorbisWindow(fftSize)
    private val fft = FloatFFT_1D(fftSize.toLong())
    private val rscratch = FloatArray(fftSize)

    // The (fftSize - hopSize) tail of the previous emission's OLA buffer —
    // upcoming frames will OLA into these samples before they emit.
    private val carry = FloatArray(fftSize - hopSize)

    private var prePadRemaining: Int = fftSize - hopSize

    /** Process `nFrames` STFT frames; returns the time-domain samples fully formed so far. */
    fun process(real: FloatArray, imag: FloatArray, nFrames: Int): FloatArray {
        if (nFrames <= 0) return FloatArray(0)
        require(real.size >= nFrames * nFreq && imag.size >= nFrames * nFreq) {
            "real/imag too small for nFrames=$nFrames"
        }
        val olaLen = (nFrames - 1) * hopSize + fftSize
        val ola = FloatArray(olaLen)
        // Seed with the previous chunk's pending tail.
        System.arraycopy(carry, 0, ola, 0, carry.size)

        val nyq = fftSize / 2
        for (t in 0 until nFrames) {
            val base = t * nFreq
            rscratch[0] = real[base]
            rscratch[1] = real[base + nyq]
            for (k in 1 until nyq) {
                rscratch[2 * k] = real[base + k]
                rscratch[2 * k + 1] = imag[base + k]
            }
            fft.realInverse(rscratch, /* scale = */ false)
            val off = t * hopSize
            for (n in 0 until fftSize) {
                ola[off + n] += rscratch[n] * window[n]
            }
        }

        System.arraycopy(ola, nFrames * hopSize, carry, 0, fftSize - hopSize)

        val emitLenRaw = nFrames * hopSize
        return if (prePadRemaining > 0) {
            val drop = minOf(prePadRemaining, emitLenRaw)
            prePadRemaining -= drop
            ola.copyOfRange(drop, emitLenRaw)
        } else {
            ola.copyOfRange(0, emitLenRaw)
        }
    }

    /**
     * Emit the remaining OLA tail at end-of-stream. Call exactly once
     * to match the sample count from batch [Stft.inverse].
     */
    fun flush(): FloatArray {
        val out = carry.copyOf()
        java.util.Arrays.fill(carry, 0f)
        return out
    }

    fun reset() {
        java.util.Arrays.fill(carry, 0f)
        prePadRemaining = fftSize - hopSize
    }
}
