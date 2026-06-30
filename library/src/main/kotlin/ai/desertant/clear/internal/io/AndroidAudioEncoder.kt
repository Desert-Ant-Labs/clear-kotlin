package ai.desertant.clear.internal.io

import android.media.MediaCodec
import android.media.MediaCodec.BufferInfo
import android.media.MediaCodecInfo
import android.media.MediaFormat
import android.media.MediaMuxer
import ai.desertant.clear.internal.io.Wav.Audio
import java.io.File

/**
 * Encode planar float32 PCM to AAC inside an M4A container via
 * MediaCodec + MediaMuxer. Bit rate: 96 kbps mono / 128 kbps stereo,
 * matching Apple Podcasts speech recommendations.
 */
internal object AndroidAudioEncoder {

    fun writeM4A(file: File, audio: Audio, bitRate: Int = bitRateFor(audio.channels)) {
        writeAac(file, audio.samples, audio.sampleRate, audio.channels, bitRate, container = "m4a")
    }

    /** Generic AAC writer. `container` selects the muxer format ("m4a" for MPEG_4). */
    fun writeAac(
        file: File,
        samples: Array<FloatArray>,
        sampleRate: Int,
        channels: Int,
        bitRate: Int = bitRateFor(channels),
        container: String = "m4a",
    ) {
        require(samples.size == channels) {
            "samples ${samples.size} channels ≠ declared $channels"
        }
        val mime = MediaFormat.MIMETYPE_AUDIO_AAC

        val format = MediaFormat.createAudioFormat(mime, sampleRate, channels).apply {
            setInteger(MediaFormat.KEY_AAC_PROFILE, MediaCodecInfo.CodecProfileLevel.AACObjectLC)
            setInteger(MediaFormat.KEY_BIT_RATE, bitRate)
            setInteger(MediaFormat.KEY_PCM_ENCODING, AUDIO_FORMAT_PCM_FLOAT)
            setInteger(MediaFormat.KEY_MAX_INPUT_SIZE, 16 * 1024)
        }

        val codec = MediaCodec.createEncoderByType(mime)
        // Try float input; fall back to int16 if the encoder doesn't accept it.
        var inputIsFloat = true
        try {
            codec.configure(format, null, null, MediaCodec.CONFIGURE_FLAG_ENCODE)
        } catch (_: Throwable) {
            inputIsFloat = false
            format.setInteger(MediaFormat.KEY_PCM_ENCODING, AUDIO_FORMAT_PCM_16BIT)
            codec.configure(format, null, null, MediaCodec.CONFIGURE_FLAG_ENCODE)
        }
        codec.start()

        file.parentFile?.mkdirs()
        val muxerFormat = when (container) {
            "m4a" -> MediaMuxer.OutputFormat.MUXER_OUTPUT_MPEG_4
            else -> error("unsupported container '$container'")
        }
        val muxer = MediaMuxer(file.path, muxerFormat)
        var trackIndex = -1
        var muxerStarted = false

        val totalFrames = samples[0].size
        var srcFrame = 0
        val info = BufferInfo()
        // Non-blocking poll for input, short wait for output. Long waits on
        // every poll burn seconds across a multi-minute encode.
        val inputTimeoutUs = 0L
        val outputTimeoutUs = 100L
        var inputDone = false
        var outputDone = false

        // Reusable interleave scratch — bulk-copy through a FloatBuffer view
        // is dramatically faster than per-sample putFloat into a DirectByteBuffer.
        var interleavedFloats: FloatArray? = null
        var interleavedShorts: ShortArray? = null

        try {
            while (!outputDone) {
                if (!inputDone) {
                    val inIdx = codec.dequeueInputBuffer(inputTimeoutUs)
                    if (inIdx >= 0) {
                        val inBuf = codec.getInputBuffer(inIdx)
                        if (inBuf == null) {
                            codec.queueInputBuffer(inIdx, 0, 0, 0, 0)
                            continue
                        }
                        inBuf.clear()
                        val capacity = inBuf.capacity()
                        val bytesPerFrame = (if (inputIsFloat) 4 else 2) * channels
                        val framesThisBuf = minOf(totalFrames - srcFrame, capacity / bytesPerFrame)
                        val byteCount = framesThisBuf * bytesPerFrame
                        val ptsUs = (srcFrame.toLong() * 1_000_000L) / sampleRate
                        if (framesThisBuf == 0) {
                            codec.queueInputBuffer(inIdx, 0, 0, ptsUs, MediaCodec.BUFFER_FLAG_END_OF_STREAM)
                            inputDone = true
                        } else {
                            inBuf.order(java.nio.ByteOrder.nativeOrder())
                            val nSamples = framesThisBuf * channels
                            if (inputIsFloat) {
                                var scratch = interleavedFloats
                                if (scratch == null || scratch.size < nSamples) {
                                    scratch = FloatArray(nSamples)
                                    interleavedFloats = scratch
                                }
                                if (channels == 1) {
                                    System.arraycopy(samples[0], srcFrame, scratch, 0, framesThisBuf)
                                } else {
                                    var w = 0
                                    for (f in 0 until framesThisBuf) {
                                        for (c in 0 until channels) {
                                            scratch[w++] = samples[c][srcFrame + f]
                                        }
                                    }
                                }
                                inBuf.asFloatBuffer().put(scratch, 0, nSamples)
                                inBuf.position(0)
                                inBuf.limit(byteCount)
                            } else {
                                var scratch = interleavedShorts
                                if (scratch == null || scratch.size < nSamples) {
                                    scratch = ShortArray(nSamples)
                                    interleavedShorts = scratch
                                }
                                if (channels == 1) {
                                    val src = samples[0]
                                    for (i in 0 until framesThisBuf) {
                                        val v = src[srcFrame + i]
                                        val clamped = if (v > 1f) 1f else if (v < -1f) -1f else v
                                        scratch[i] = (clamped * 32767f).toInt().toShort()
                                    }
                                } else {
                                    var w = 0
                                    for (f in 0 until framesThisBuf) {
                                        for (c in 0 until channels) {
                                            val v = samples[c][srcFrame + f]
                                            val clamped = if (v > 1f) 1f else if (v < -1f) -1f else v
                                            scratch[w++] = (clamped * 32767f).toInt().toShort()
                                        }
                                    }
                                }
                                inBuf.asShortBuffer().put(scratch, 0, nSamples)
                                inBuf.position(0)
                                inBuf.limit(byteCount)
                            }
                            codec.queueInputBuffer(inIdx, 0, byteCount, ptsUs, 0)
                            srcFrame += framesThisBuf
                        }
                    }
                }

                val outIdx = codec.dequeueOutputBuffer(info, outputTimeoutUs)
                when {
                    outIdx == MediaCodec.INFO_OUTPUT_FORMAT_CHANGED -> {
                        check(!muxerStarted) { "format changed twice" }
                        trackIndex = muxer.addTrack(codec.outputFormat)
                        muxer.start()
                        muxerStarted = true
                    }
                    outIdx >= 0 -> {
                        val outBuf = codec.getOutputBuffer(outIdx)
                        if (outBuf != null && info.size > 0 && muxerStarted) {
                            outBuf.position(info.offset)
                            outBuf.limit(info.offset + info.size)
                            muxer.writeSampleData(trackIndex, outBuf, info)
                        }
                        codec.releaseOutputBuffer(outIdx, false)
                        if (info.flags and MediaCodec.BUFFER_FLAG_END_OF_STREAM != 0) {
                            outputDone = true
                        }
                    }
                    outIdx == MediaCodec.INFO_TRY_AGAIN_LATER -> { /* loop */ }
                }
            }
        } finally {
            try { codec.stop() } catch (_: Throwable) {}
            codec.release()
            try { if (muxerStarted) muxer.stop() } catch (_: Throwable) {}
            muxer.release()
        }
    }

    private fun bitRateFor(channels: Int): Int = when (channels) {
        1 -> 96_000
        else -> 128_000
    }

    private const val AUDIO_FORMAT_PCM_16BIT = 2
    private const val AUDIO_FORMAT_PCM_FLOAT = 4
}
