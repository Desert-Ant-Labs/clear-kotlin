package ai.desertant.clear.internal.io

import android.media.MediaCodec
import android.media.MediaCodec.BufferInfo
import android.media.MediaCodecInfo
import android.media.MediaExtractor
import android.media.MediaFormat
import ai.desertant.clear.internal.io.Wav.Audio
import java.io.File

/**
 * Decode any Android-supported audio file (M4A/AAC, MP3, FLAC, OGG/Vorbis,
 * Opus, AMR) to PCM via MediaExtractor + MediaCodec. Returns planar
 * `[channels][frames]` float32 in [-1, 1] at the source sample rate.
 */
internal object AndroidAudioDecoder {

    fun decode(file: File): Audio {
        val extractor = MediaExtractor()
        extractor.setDataSource(file.path)
        try {
            return decode(extractor)
        } finally {
            extractor.release()
        }
    }

    private fun decode(extractor: MediaExtractor): Audio {
        val trackIndex = pickAudioTrack(extractor)
        require(trackIndex >= 0) { "no audio track in input" }
        extractor.selectTrack(trackIndex)
        val inputFormat = extractor.getTrackFormat(trackIndex)
        val mime = inputFormat.getString(MediaFormat.KEY_MIME)
            ?: error("track has no MIME type")
        val sampleRate = inputFormat.getInteger(MediaFormat.KEY_SAMPLE_RATE)
        val channels = inputFormat.getInteger(MediaFormat.KEY_CHANNEL_COUNT)

        val codec = MediaCodec.createDecoderByType(mime)
        // Request float PCM output; some vendors ignore this and emit int16
        // anyway, so the output path below handles both.
        val outFormat = MediaFormat.createAudioFormat(mime, sampleRate, channels).apply {
            try {
                setInteger(MediaFormat.KEY_PCM_ENCODING, AUDIO_FORMAT_PCM_FLOAT)
            } catch (_: Throwable) {}
        }
        // Carry over decoder-specific config (e.g. AAC CSD0).
        inputFormat.getByteBuffer("csd-0")?.let { outFormat.setByteBuffer("csd-0", it) }
        inputFormat.getByteBuffer("csd-1")?.let { outFormat.setByteBuffer("csd-1", it) }
        codec.configure(outFormat, null, null, 0)
        codec.start()

        val totalDurationUs = inputFormat.getLongOr(MediaFormat.KEY_DURATION, -1L)
        // Pre-size output when duration is known; otherwise default to 60 s and grow.
        val initialFramesPerCh = if (totalDurationUs > 0)
            ((totalDurationUs * sampleRate) / 1_000_000L).toInt().coerceAtLeast(1024)
        else 48_000 * 60

        val outChannels = Array(channels) { FloatArray(initialFramesPerCh) }
        var outFrames = 0
        var inputDone = false
        var outputDone = false
        val info = BufferInfo()
        val timeoutUs = 10_000L
        var actualOutputIsFloat = false
        var sawFirstOutput = false

        while (!outputDone) {
            if (!inputDone) {
                val inIdx = codec.dequeueInputBuffer(timeoutUs)
                if (inIdx >= 0) {
                    val inBuf = codec.getInputBuffer(inIdx) ?: continue
                    inBuf.clear()
                    val size = extractor.readSampleData(inBuf, 0)
                    if (size < 0) {
                        codec.queueInputBuffer(inIdx, 0, 0, 0, MediaCodec.BUFFER_FLAG_END_OF_STREAM)
                        inputDone = true
                    } else {
                        val pts = extractor.sampleTime
                        codec.queueInputBuffer(inIdx, 0, size, pts, 0)
                        extractor.advance()
                    }
                }
            }

            val outIdx = codec.dequeueOutputBuffer(info, timeoutUs)
            when {
                outIdx >= 0 -> {
                    if (!sawFirstOutput) {
                        val fmt = codec.outputFormat
                        actualOutputIsFloat = fmt.getIntegerOr(MediaFormat.KEY_PCM_ENCODING, AUDIO_FORMAT_PCM_16BIT) == AUDIO_FORMAT_PCM_FLOAT
                        sawFirstOutput = true
                    }
                    val outBuf = codec.getOutputBuffer(outIdx)
                    if (outBuf != null && info.size > 0) {
                        outBuf.position(info.offset)
                        outBuf.limit(info.offset + info.size)
                        val newFrames = appendOutput(outBuf, info.size, outChannels, outFrames, channels, actualOutputIsFloat)
                        outFrames += newFrames
                    }
                    codec.releaseOutputBuffer(outIdx, false)
                    if (info.flags and MediaCodec.BUFFER_FLAG_END_OF_STREAM != 0) {
                        outputDone = true
                    }
                }
                outIdx == MediaCodec.INFO_OUTPUT_FORMAT_CHANGED -> {
                    val fmt = codec.outputFormat
                    actualOutputIsFloat = fmt.getIntegerOr(MediaFormat.KEY_PCM_ENCODING, AUDIO_FORMAT_PCM_16BIT) == AUDIO_FORMAT_PCM_FLOAT
                }
                outIdx == MediaCodec.INFO_TRY_AGAIN_LATER -> { /* keep looping */ }
            }
        }

        codec.stop()
        codec.release()

        // Trim to actual frame count.
        val trimmed = Array(channels) { c ->
            if (outChannels[c].size == outFrames) outChannels[c]
            else outChannels[c].copyOf(outFrames)
        }
        return Audio(trimmed, sampleRate, channels)
    }

    /**
     * Append decoded PCM into the per-channel output arrays, growing
     * if needed. Bulk-copies via FloatBuffer / ShortBuffer — per-sample
     * buf.getFloat() is dramatically slower on Android.
     */
    private fun appendOutput(
        buf: java.nio.ByteBuffer,
        size: Int,
        out: Array<FloatArray>,
        startFrame: Int,
        channels: Int,
        isFloat: Boolean,
    ): Int {
        buf.order(java.nio.ByteOrder.nativeOrder())
        val bytesPerSample = if (isFloat) 4 else 2
        val newFrames = size / (bytesPerSample * channels)
        val nSamples = newFrames * channels
        val end = startFrame + newFrames
        for (c in 0 until channels) {
            if (out[c].size < end) {
                val grown = FloatArray((end * 3 + 1) / 2)
                System.arraycopy(out[c], 0, grown, 0, startFrame)
                out[c] = grown
            }
        }
        if (isFloat) {
            val flat = FloatArray(nSamples)
            buf.asFloatBuffer().get(flat, 0, nSamples)
            if (channels == 1) {
                System.arraycopy(flat, 0, out[0], startFrame, newFrames)
            } else {
                var r = 0
                for (f in 0 until newFrames) {
                    for (c in 0 until channels) {
                        out[c][startFrame + f] = flat[r++]
                    }
                }
            }
        } else {
            val flat = ShortArray(nSamples)
            buf.asShortBuffer().get(flat, 0, nSamples)
            val inv = 1f / 32768f
            if (channels == 1) {
                val dst = out[0]
                for (i in 0 until newFrames) dst[startFrame + i] = flat[i] * inv
            } else {
                var r = 0
                for (f in 0 until newFrames) {
                    for (c in 0 until channels) {
                        out[c][startFrame + f] = flat[r++] * inv
                    }
                }
            }
        }
        return newFrames
    }

    private fun pickAudioTrack(extractor: MediaExtractor): Int {
        for (i in 0 until extractor.trackCount) {
            val mime = extractor.getTrackFormat(i).getString(MediaFormat.KEY_MIME) ?: continue
            if (mime.startsWith("audio/")) return i
        }
        return -1
    }

    // Inline AudioFormat constants to avoid the android.media.AudioFormat import.
    private const val AUDIO_FORMAT_PCM_16BIT = 2
    private const val AUDIO_FORMAT_PCM_FLOAT = 4
}

internal fun MediaFormat.getLongOr(key: String, fallback: Long): Long =
    try { if (containsKey(key)) getLong(key) else fallback } catch (_: Throwable) { fallback }

internal fun MediaFormat.getIntegerOr(key: String, fallback: Int): Int =
    try { if (containsKey(key)) getInteger(key) else fallback } catch (_: Throwable) { fallback }

// Keeps the MediaCodecInfo import referenced (used by AndroidAudioEncoder).
private val _unused = MediaCodecInfo::class
