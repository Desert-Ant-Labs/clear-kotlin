package ai.desertant.clear.internal.io

import android.media.MediaCodec
import android.media.MediaCodec.BufferInfo
import android.media.MediaExtractor
import android.media.MediaFormat
import java.io.Closeable
import java.io.File

/**
 * Streaming PCM decoder. Yields chunks of `[channels][frames]` Float32
 * samples from any Android-supported audio container — constant peak
 * memory regardless of file length.
 */
internal class StreamingAudioDecoder(file: File) : Closeable {

    val sampleRate: Int
    val channels: Int
    val durationSec: Double

    private val extractor = MediaExtractor()
    private val codec: MediaCodec
    private val isFloat: Boolean
    private var inputDone = false
    private var outputDone = false

    init {
        extractor.setDataSource(file.path)
        var trackIdx = -1
        var mime = ""
        for (i in 0 until extractor.trackCount) {
            val m = extractor.getTrackFormat(i).getString(MediaFormat.KEY_MIME) ?: continue
            if (m.startsWith("audio/")) { trackIdx = i; mime = m; break }
        }
        require(trackIdx >= 0) { "no audio track in ${file.path}" }
        extractor.selectTrack(trackIdx)
        val inFmt = extractor.getTrackFormat(trackIdx)
        sampleRate = inFmt.getInteger(MediaFormat.KEY_SAMPLE_RATE)
        channels = inFmt.getInteger(MediaFormat.KEY_CHANNEL_COUNT)
        durationSec =
            if (inFmt.containsKey(MediaFormat.KEY_DURATION))
                inFmt.getLong(MediaFormat.KEY_DURATION) / 1_000_000.0
            else 0.0
        codec = MediaCodec.createDecoderByType(mime)
        val outFmt = MediaFormat.createAudioFormat(mime, sampleRate, channels).apply {
            try { setInteger(MediaFormat.KEY_PCM_ENCODING, AUDIO_FORMAT_PCM_FLOAT) } catch (_: Throwable) {}
        }
        inFmt.getByteBuffer("csd-0")?.let { outFmt.setByteBuffer("csd-0", it) }
        inFmt.getByteBuffer("csd-1")?.let { outFmt.setByteBuffer("csd-1", it) }
        codec.configure(outFmt, null, null, 0)
        codec.start()

        // PCM-encoding detected lazily on the codec's first output-format event.
        isFloat = false
    }

    private var _outputIsFloat: Boolean = false
    private val info = BufferInfo()
    private val inputTimeoutUs = 0L
    private val outputTimeoutUs = 5_000L

    /** Returns next PCM chunk, or null at end-of-stream. */
    fun next(): Array<FloatArray>? {
        while (!outputDone) {
            if (!inputDone) {
                val inIdx = codec.dequeueInputBuffer(inputTimeoutUs)
                if (inIdx >= 0) {
                    val buf = codec.getInputBuffer(inIdx)
                    if (buf == null) {
                        codec.queueInputBuffer(inIdx, 0, 0, 0, 0)
                    } else {
                        buf.clear()
                        val size = extractor.readSampleData(buf, 0)
                        if (size < 0) {
                            codec.queueInputBuffer(inIdx, 0, 0, 0, MediaCodec.BUFFER_FLAG_END_OF_STREAM)
                            inputDone = true
                        } else {
                            codec.queueInputBuffer(inIdx, 0, size, extractor.sampleTime, 0)
                            extractor.advance()
                        }
                    }
                }
            }
            val outIdx = codec.dequeueOutputBuffer(info, outputTimeoutUs)
            when {
                outIdx == MediaCodec.INFO_OUTPUT_FORMAT_CHANGED -> {
                    val fmt = codec.outputFormat
                    _outputIsFloat = (fmt.getIntegerOr(MediaFormat.KEY_PCM_ENCODING,
                        AUDIO_FORMAT_PCM_16BIT) == AUDIO_FORMAT_PCM_FLOAT)
                }
                outIdx >= 0 -> {
                    val buf = codec.getOutputBuffer(outIdx)
                    val chunk: Array<FloatArray>? = if (buf != null && info.size > 0) {
                        buf.position(info.offset)
                        buf.limit(info.offset + info.size)
                        decodePcm(buf, info.size)
                    } else null
                    codec.releaseOutputBuffer(outIdx, false)
                    if (info.flags and MediaCodec.BUFFER_FLAG_END_OF_STREAM != 0) outputDone = true
                    if (chunk != null) return chunk
                    if (outputDone) return null
                }
                outIdx == MediaCodec.INFO_TRY_AGAIN_LATER -> { /* loop */ }
            }
        }
        return null
    }

    private fun decodePcm(buf: java.nio.ByteBuffer, size: Int): Array<FloatArray> {
        buf.order(java.nio.ByteOrder.nativeOrder())
        val bytesPerSample = if (_outputIsFloat) 4 else 2
        val nFrames = size / (bytesPerSample * channels)
        val nSamples = nFrames * channels
        val out = Array(channels) { FloatArray(nFrames) }
        if (_outputIsFloat) {
            val flat = FloatArray(nSamples)
            buf.asFloatBuffer().get(flat, 0, nSamples)
            if (channels == 1) {
                System.arraycopy(flat, 0, out[0], 0, nFrames)
            } else {
                var r = 0
                for (f in 0 until nFrames) for (c in 0 until channels) out[c][f] = flat[r++]
            }
        } else {
            val flat = ShortArray(nSamples)
            buf.asShortBuffer().get(flat, 0, nSamples)
            val inv = 1f / 32768f
            if (channels == 1) {
                for (i in 0 until nFrames) out[0][i] = flat[i] * inv
            } else {
                var r = 0
                for (f in 0 until nFrames) for (c in 0 until channels) out[c][f] = flat[r++] * inv
            }
        }
        return out
    }

    override fun close() {
        try { codec.stop() } catch (_: Throwable) {}
        codec.release()
        extractor.release()
    }

    private companion object {
        const val AUDIO_FORMAT_PCM_16BIT = 2
        const val AUDIO_FORMAT_PCM_FLOAT = 4
    }
}
