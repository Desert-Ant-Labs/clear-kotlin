package ai.desertant.clear.internal.dsp

import ai.desertant.clear.internal.Constants
import org.jtransforms.fft.FloatFFT_1D
import kotlin.math.PI
import kotlin.math.sin

/**
 * Forward / inverse STFT. 960-point FFT, hop 480 (50% overlap),
 * Vorbis window, forward gain `wnorm = 2·hop/N² = 1/960`. The analysis
 * pass pre-pads `fftSize - hopSize` zeros to compensate for the
 * analysis-synthesis delay; the synthesis pass drops the same count.
 * Must match Swift `Sources/Clear/STFT.swift` within abs 1e-5.
 */
class Stft(
    private val fftSize: Int = Constants.FFT_SIZE,
    private val hopSize: Int = Constants.HOP_SIZE,
) {
    val nFreq: Int = fftSize / 2 + 1

    private val window: FloatArray = buildVorbisWindow(fftSize)
    private val wnorm: Float = 2f * hopSize / (fftSize.toFloat() * fftSize.toFloat())

    private val fft = FloatFFT_1D(fftSize.toLong())
    // JTransforms half-complex layout for real-input FFT:
    //   a[0]=Re[0], a[1]=Re[N/2], a[2k]=Re[k], a[2k+1]=Im[k] (k=1..N/2-1).
    //   Im[0] and Im[N/2] are 0.
    private val rscratch = FloatArray(fftSize)

    data class ForwardResult(val real: FloatArray, val imag: FloatArray, val nFrames: Int) {
        // FloatArray equals defaults to identity — override for content equality in tests.
        override fun equals(other: Any?): Boolean =
            other is ForwardResult && nFrames == other.nFrames &&
            real.contentEquals(other.real) && imag.contentEquals(other.imag)

        override fun hashCode(): Int =
            31 * (31 * real.contentHashCode() + imag.contentHashCode()) + nFrames
    }

    /**
     * Forward STFT. Returns `(real, imag, nFrames)` in row-major
     * `[nFrames, nFreq]` layout.
     */
    fun forward(audio: FloatArray): ForwardResult {
        val prePad = fftSize - hopSize
        val paddedLen = prePad + audio.size
        if (paddedLen < fftSize) return ForwardResult(FloatArray(0), FloatArray(0), 0)

        val padded = FloatArray(paddedLen)
        System.arraycopy(audio, 0, padded, prePad, audio.size)

        val nFrames = (paddedLen - fftSize) / hopSize + 1
        val real = FloatArray(nFrames * nFreq)
        val imag = FloatArray(nFrames * nFreq)

        val nyq = fftSize / 2
        for (t in 0 until nFrames) {
            val off = t * hopSize
            for (n in 0 until fftSize) {
                rscratch[n] = padded[off + n] * window[n]
            }
            fft.realForward(rscratch)
            val base = t * nFreq
            // Unpack half-complex layout into parallel real/imag, scaled by wnorm.
            real[base] = rscratch[0] * wnorm
            imag[base] = 0f
            for (k in 1 until nyq) {
                real[base + k] = rscratch[2 * k] * wnorm
                imag[base + k] = rscratch[2 * k + 1] * wnorm
            }
            real[base + nyq] = rscratch[1] * wnorm
            imag[base + nyq] = 0f
        }
        return ForwardResult(real, imag, nFrames)
    }

    /** Inverse STFT. Real-iFFT + windowed overlap-add, prePad trimmed from the head. */
    fun inverse(real: FloatArray, imag: FloatArray, nFrames: Int): FloatArray {
        if (nFrames <= 0) return FloatArray(0)
        val prePad = fftSize - hopSize
        val rawLen = (nFrames - 1) * hopSize + fftSize
        val out = FloatArray(rawLen)

        val nyq = fftSize / 2
        for (t in 0 until nFrames) {
            val base = t * nFreq
            rscratch[0] = real[base]            // Re[0]
            rscratch[1] = real[base + nyq]      // Re[N/2] (Nyquist)
            for (k in 1 until nyq) {
                rscratch[2 * k] = real[base + k]
                rscratch[2 * k + 1] = imag[base + k]
            }
            fft.realInverse(rscratch, /* scale = */ false)
            val off = t * hopSize
            for (n in 0 until fftSize) {
                out[off + n] += rscratch[n] * window[n]
            }
        }

        return if (rawLen > prePad) out.copyOfRange(prePad, rawLen) else out
    }

    companion object {
        /**
         * Vorbis window: `w[n] = sin(0.5π · sin²(0.5π · (n + 0.5) / (N/2)))`.
         * Do not substitute Hann or Hamming — the Swift reference uses Vorbis.
         */
        fun buildVorbisWindow(fftSize: Int): FloatArray {
            val halfN = fftSize / 2.0
            return FloatArray(fftSize) { n ->
                val s = sin(0.5 * PI * (n + 0.5) / halfN)
                sin(0.5 * PI * s * s).toFloat()
            }
        }
    }
}
