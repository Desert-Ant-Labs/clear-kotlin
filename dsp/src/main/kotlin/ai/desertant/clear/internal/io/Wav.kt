package ai.desertant.clear.internal.io

import java.io.File
import java.nio.ByteBuffer
import java.nio.ByteOrder

/**
 * Minimal WAV codec.
 *
 * Reader: PCM int16/int24/int32 and IEEE float32/64. Returns planar
 * `[channels][frames]` FloatArray normalized to [-1, 1].
 *
 * Writer: IEEE float32 little-endian (codec tag 3).
 *
 * Reads load the whole file into memory — fine for the batch path
 * the Clear pipeline takes for WAV input.
 */
object Wav {

    data class Audio(
        val samples: Array<FloatArray>,   // [channels][frames]
        val sampleRate: Int,
        val channels: Int,
    ) {
        val frames: Int get() = samples.firstOrNull()?.size ?: 0
        val durationSec: Double get() = frames.toDouble() / sampleRate.coerceAtLeast(1)

        fun toMono(): FloatArray {
            if (channels == 1) return samples[0]
            val frames = this.frames
            val mono = FloatArray(frames)
            for (i in 0 until frames) {
                var sum = 0f
                for (c in 0 until channels) sum += samples[c][i]
                mono[i] = sum / channels
            }
            return mono
        }

        override fun equals(other: Any?): Boolean {
            if (other !is Audio) return false
            if (sampleRate != other.sampleRate || channels != other.channels) return false
            if (samples.size != other.samples.size) return false
            for (c in samples.indices) {
                if (!samples[c].contentEquals(other.samples[c])) return false
            }
            return true
        }
        override fun hashCode(): Int {
            var h = sampleRate.hashCode()
            h = 31 * h + channels
            for (c in samples) h = 31 * h + c.contentHashCode()
            return h
        }
    }

    class WavFormatException(msg: String) : RuntimeException(msg)

    fun read(file: File): Audio {
        require(file.exists()) { "WAV file not found: ${file.path}" }
        val bytes = file.readBytes()
        return readBytes(bytes, file.path)
    }

    fun readBytes(bytes: ByteArray, source: String = "<bytes>"): Audio {
        if (bytes.size < 44) throw WavFormatException("$source: too small to be a WAV (${bytes.size} bytes)")
        val buf = ByteBuffer.wrap(bytes).order(ByteOrder.LITTLE_ENDIAN)

        val riff = ByteArray(4); buf.get(riff)
        if (String(riff) != "RIFF") throw WavFormatException("$source: not RIFF (got '${String(riff)}')")
        buf.int  // file size minus 8 — not validated
        val wave = ByteArray(4); buf.get(wave)
        if (String(wave) != "WAVE") throw WavFormatException("$source: not WAVE")

        var codec: Int = -1
        var channels = 0
        var sampleRate = 0
        var bitsPerSample = 0
        var dataOffset = -1
        var dataSize = -1

        while (buf.remaining() >= 8) {
            val id = ByteArray(4); buf.get(id)
            val size = buf.int
            val chunkStart = buf.position()
            val chunkId = String(id)
            when (chunkId) {
                "fmt " -> {
                    if (size < 16) throw WavFormatException("$source: fmt chunk too small ($size)")
                    codec = buf.short.toInt() and 0xFFFF
                    channels = buf.short.toInt() and 0xFFFF
                    sampleRate = buf.int
                    buf.int     // byte rate — derived, ignore
                    buf.short   // block align — ignore
                    bitsPerSample = buf.short.toInt() and 0xFFFF
                    buf.position(chunkStart + size)
                    // Chunk size pads to even.
                    if (size % 2 != 0 && buf.remaining() > 0) buf.position(buf.position() + 1)
                }
                "data" -> {
                    dataOffset = chunkStart
                    dataSize = size
                    buf.position(chunkStart + size)
                    if (size % 2 != 0 && buf.remaining() > 0) buf.position(buf.position() + 1)
                }
                else -> {
                    buf.position(chunkStart + size)
                    if (size % 2 != 0 && buf.remaining() > 0) buf.position(buf.position() + 1)
                }
            }
        }

        if (codec < 0) throw WavFormatException("$source: no fmt chunk")
        if (dataOffset < 0) throw WavFormatException("$source: no data chunk")
        if (channels <= 0) throw WavFormatException("$source: invalid channel count $channels")
        if (sampleRate <= 0) throw WavFormatException("$source: invalid sample rate $sampleRate")

        val data = ByteBuffer.wrap(bytes, dataOffset, dataSize).order(ByteOrder.LITTLE_ENDIAN)
        val bytesPerSample = bitsPerSample / 8
        val frames = dataSize / (bytesPerSample * channels)
        val out = Array(channels) { FloatArray(frames) }

        when (codec) {
            1 -> {  // PCM integer
                when (bitsPerSample) {
                    16 -> {
                        for (f in 0 until frames) for (c in 0 until channels)
                            out[c][f] = data.short.toFloat() / 32768f
                    }
                    24 -> {
                        for (f in 0 until frames) for (c in 0 until channels) {
                            val b0 = data.get().toInt() and 0xFF
                            val b1 = data.get().toInt() and 0xFF
                            val b2 = data.get().toInt()  // signed
                            val s = (b2 shl 16) or (b1 shl 8) or b0
                            out[c][f] = s / 8388608f
                        }
                    }
                    32 -> {
                        for (f in 0 until frames) for (c in 0 until channels)
                            out[c][f] = data.int / 2147483648f
                    }
                    else -> throw WavFormatException("$source: PCM bits-per-sample $bitsPerSample not supported (16/24/32 only)")
                }
            }
            3 -> {  // IEEE float
                when (bitsPerSample) {
                    32 -> {
                        for (f in 0 until frames) for (c in 0 until channels)
                            out[c][f] = data.float
                    }
                    64 -> {
                        for (f in 0 until frames) for (c in 0 until channels)
                            out[c][f] = data.double.toFloat()
                    }
                    else -> throw WavFormatException("$source: float bits-per-sample $bitsPerSample not supported (32/64 only)")
                }
            }
            else -> throw WavFormatException("$source: codec $codec not supported (PCM=1 / IEEE float=3 only)")
        }

        return Audio(out, sampleRate, channels)
    }

    /** Write float32 little-endian WAV; channels are interleaved per WAV convention. */
    fun write(file: File, audio: Audio) {
        write(file, audio.samples, audio.sampleRate, audio.channels)
    }

    fun write(file: File, samples: Array<FloatArray>, sampleRate: Int, channels: Int) {
        require(channels in 1..32) { "channels out of range: $channels" }
        require(samples.size == channels) {
            "samples shape ${samples.size} ≠ channels $channels"
        }
        val frames = samples[0].size
        for (c in 1 until channels) {
            require(samples[c].size == frames) { "channel $c has ${samples[c].size} frames, expected $frames" }
        }

        val bytesPerSample = 4
        val dataSize = frames * channels * bytesPerSample
        val byteRate = sampleRate * channels * bytesPerSample
        val blockAlign = (channels * bytesPerSample).toShort()

        // Header layout: 12 (RIFF/WAVE) + 8 (fmt id+size) + 16 (fmt) + 8 (data id+size).
        val totalSize = 44 + dataSize
        val buf = ByteBuffer.allocate(totalSize).order(ByteOrder.LITTLE_ENDIAN)
        buf.put("RIFF".toByteArray())
        buf.putInt(totalSize - 8)
        buf.put("WAVE".toByteArray())
        buf.put("fmt ".toByteArray())
        buf.putInt(16)
        buf.putShort(3.toShort())                            // IEEE float
        buf.putShort(channels.toShort())
        buf.putInt(sampleRate)
        buf.putInt(byteRate)
        buf.putShort(blockAlign)
        buf.putShort((bytesPerSample * 8).toShort())
        buf.put("data".toByteArray())
        buf.putInt(dataSize)
        for (f in 0 until frames) {
            for (c in 0 until channels) {
                buf.putFloat(samples[c][f])
            }
        }

        file.parentFile?.mkdirs()
        file.outputStream().use { it.write(buf.array()) }
    }

    /** Convenience: write mono float buffer. */
    fun writeMono(file: File, samples: FloatArray, sampleRate: Int) {
        write(file, arrayOf(samples), sampleRate, 1)
    }
}
