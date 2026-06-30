package ai.desertant.clear.internal.io

import android.media.MediaCodec
import android.media.MediaCodec.BufferInfo
import android.media.MediaCodecInfo
import android.media.MediaFormat
import android.media.MediaMuxer
import java.io.Closeable
import java.io.File

/**
 * Streaming AAC/M4A encoder — incremental writes via [write], constant
 * peak memory regardless of file length. Bit rate: 96 kbps mono / 128
 * kbps stereo.
 */
internal class StreamingAudioEncoder(
    file: File,
    private val sampleRate: Int,
    private val channels: Int,
    private val container: String = "m4a",
    private val bitRate: Int = bitRateFor(channels),
) : Closeable {

    private val codec: MediaCodec
    private val muxer: MediaMuxer
    private var trackIndex = -1
    private var muxerStarted = false
    private val info = BufferInfo()
    private var ptsFramesEmitted = 0L

    // Interleave scratch — reused across writes.
    private var interleavedShorts: ShortArray? = null

    init {
        require(channels in 1..32)
        file.parentFile?.mkdirs()
        val mime = MediaFormat.MIMETYPE_AUDIO_AAC
        val fmt = MediaFormat.createAudioFormat(mime, sampleRate, channels).apply {
            setInteger(MediaFormat.KEY_AAC_PROFILE, MediaCodecInfo.CodecProfileLevel.AACObjectLC)
            setInteger(MediaFormat.KEY_BIT_RATE, bitRate)
            setInteger(MediaFormat.KEY_MAX_INPUT_SIZE, 16 * 1024)
            setInteger(MediaFormat.KEY_PCM_ENCODING, AUDIO_FORMAT_PCM_16BIT)
        }
        codec = MediaCodec.createEncoderByType(mime)
        codec.configure(fmt, null, null, MediaCodec.CONFIGURE_FLAG_ENCODE)
        codec.start()
        val muxFormat = when (container) {
            "m4a" -> MediaMuxer.OutputFormat.MUXER_OUTPUT_MPEG_4
            else -> error("unsupported container '$container'")
        }
        muxer = MediaMuxer(file.path, muxFormat)
    }

    /** Encode and write a PCM chunk. Channel-major `[channels][frames]`. */
    fun write(chunk: Array<FloatArray>) {
        require(chunk.size == channels)
        val frames = chunk.firstOrNull()?.size ?: return
        if (frames == 0) return
        var srcFrame = 0
        while (srcFrame < frames) {
            val inIdx = codec.dequeueInputBuffer(5_000L)
            if (inIdx < 0) { drainOutput(false); continue }
            val inBuf = codec.getInputBuffer(inIdx) ?: continue
            inBuf.clear()
            val capacity = inBuf.capacity()
            // Always feed int16 PCM: several Android AAC encoders accept
            // configure() with float input but then read the 4-byte samples
            // as 2-byte PCM16, silently doubling the output length.
            val bytesPerFrame = 2 * channels
            val framesThisBuf = minOf(frames - srcFrame, capacity / bytesPerFrame)
            val byteCount = framesThisBuf * bytesPerFrame
            val ptsUs = (ptsFramesEmitted * 1_000_000L) / sampleRate
            inBuf.order(java.nio.ByteOrder.nativeOrder())
            val nSamples = framesThisBuf * channels
            run {
                var scratch = interleavedShorts
                if (scratch == null || scratch.size < nSamples) {
                    scratch = ShortArray(nSamples); interleavedShorts = scratch
                }
                if (channels == 1) {
                    val src = chunk[0]
                    for (i in 0 until framesThisBuf) {
                        val v = src[srcFrame + i]
                        val clamped = if (v > 1f) 1f else if (v < -1f) -1f else v
                        scratch[i] = (clamped * 32767f).toInt().toShort()
                    }
                } else {
                    var w = 0
                    for (f in 0 until framesThisBuf) for (c in 0 until channels) {
                        val v = chunk[c][srcFrame + f]
                        val clamped = if (v > 1f) 1f else if (v < -1f) -1f else v
                        scratch[w++] = (clamped * 32767f).toInt().toShort()
                    }
                }
                inBuf.asShortBuffer().put(scratch, 0, nSamples)
                inBuf.position(0); inBuf.limit(byteCount)
            }
            codec.queueInputBuffer(inIdx, 0, byteCount, ptsUs, 0)
            srcFrame += framesThisBuf
            ptsFramesEmitted += framesThisBuf
            drainOutput(false)
        }
    }

    private fun drainOutput(eos: Boolean) {
        val timeout = if (eos) 50_000L else 100L
        while (true) {
            val outIdx = codec.dequeueOutputBuffer(info, timeout)
            when {
                outIdx == MediaCodec.INFO_OUTPUT_FORMAT_CHANGED -> {
                    if (!muxerStarted) {
                        trackIndex = muxer.addTrack(codec.outputFormat)
                        muxer.start()
                        muxerStarted = true
                    }
                }
                outIdx >= 0 -> {
                    val outBuf = codec.getOutputBuffer(outIdx)
                    if (outBuf != null && info.size > 0 && muxerStarted) {
                        outBuf.position(info.offset)
                        outBuf.limit(info.offset + info.size)
                        muxer.writeSampleData(trackIndex, outBuf, info)
                    }
                    codec.releaseOutputBuffer(outIdx, false)
                    if (info.flags and MediaCodec.BUFFER_FLAG_END_OF_STREAM != 0) return
                }
                outIdx == MediaCodec.INFO_TRY_AGAIN_LATER -> { if (!eos) return }
            }
        }
    }

    override fun close() {
        // Send EOS as an empty input buffer with the EOS flag.
        try {
            val inIdx = codec.dequeueInputBuffer(100_000L)
            if (inIdx >= 0) {
                codec.queueInputBuffer(inIdx, 0, 0,
                    (ptsFramesEmitted * 1_000_000L) / sampleRate,
                    MediaCodec.BUFFER_FLAG_END_OF_STREAM)
            }
            drainOutput(true)
        } catch (_: Throwable) {}
        try { codec.stop() } catch (_: Throwable) {}
        codec.release()
        try { if (muxerStarted) muxer.stop() } catch (_: Throwable) {}
        muxer.release()
    }

    private companion object {
        const val AUDIO_FORMAT_PCM_16BIT = 2
        fun bitRateFor(channels: Int): Int = if (channels == 1) 96_000 else 128_000
    }
}
