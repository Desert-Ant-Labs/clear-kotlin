package ai.desertant.clear.internal.io

import java.io.File
import java.nio.file.Files
import kotlin.math.PI
import kotlin.math.sin
import org.junit.Assert.assertArrayEquals
import org.junit.Assert.assertEquals
import org.junit.Test

class WavTest {

    @Test
    fun `float32 mono round trips`() {
        val tmp = Files.createTempFile("wavtest", ".wav").toFile()
        try {
            val sr = 48_000
            val n = 4800
            val signal = FloatArray(n) { i -> 0.5f * sin(2 * PI * 440.0 * i / sr).toFloat() }
            Wav.writeMono(tmp, signal, sr)

            val read = Wav.read(tmp)
            assertEquals(sr, read.sampleRate)
            assertEquals(1, read.channels)
            assertEquals(n, read.frames)
            // float32 round-trip is exact
            assertArrayEquals(signal, read.samples[0], 0f)
        } finally { tmp.delete() }
    }

    @Test
    fun `float32 stereo round trips`() {
        val tmp = Files.createTempFile("wavtest", ".wav").toFile()
        try {
            val sr = 48_000
            val n = 1024
            val l = FloatArray(n) { i -> 0.5f * sin(2 * PI * 440.0 * i / sr).toFloat() }
            val r = FloatArray(n) { i -> 0.5f * sin(2 * PI * 880.0 * i / sr).toFloat() }
            Wav.write(tmp, arrayOf(l, r), sr, 2)

            val read = Wav.read(tmp)
            assertEquals(sr, read.sampleRate)
            assertEquals(2, read.channels)
            assertEquals(n, read.frames)
            assertArrayEquals(l, read.samples[0], 0f)
            assertArrayEquals(r, read.samples[1], 0f)
        } finally { tmp.delete() }
    }

    @Test
    fun `toMono averages stereo channels`() {
        val l = floatArrayOf(0.1f, 0.2f, 0.3f, 0.4f)
        val r = floatArrayOf(0.3f, 0.4f, 0.5f, 0.6f)
        val mono = Wav.Audio(arrayOf(l, r), 48_000, 2).toMono()
        assertArrayEquals(floatArrayOf(0.2f, 0.3f, 0.4f, 0.5f), mono, 1e-6f)
    }

    @Test
    fun `int16 PCM decodes within float32 quantization tolerance`() {
        // Hand-craft a tiny int16 WAV header + a few samples to verify
        // the int16 -> float branch in the reader.
        val tmp = Files.createTempFile("wav-int16", ".wav").toFile()
        try {
            val samples = shortArrayOf(0, 16384, -16384, 32767, -32768)  // 0, +0.5, -0.5, +1, -1
            val dataSize = samples.size * 2
            val header = java.nio.ByteBuffer.allocate(44).order(java.nio.ByteOrder.LITTLE_ENDIAN)
            header.put("RIFF".toByteArray())
            header.putInt(36 + dataSize)
            header.put("WAVE".toByteArray())
            header.put("fmt ".toByteArray())
            header.putInt(16)
            header.putShort(1)                  // PCM
            header.putShort(1)                  // 1 channel
            header.putInt(48_000)
            header.putInt(48_000 * 2)
            header.putShort(2)
            header.putShort(16)
            header.put("data".toByteArray())
            header.putInt(dataSize)
            val body = java.nio.ByteBuffer.allocate(dataSize).order(java.nio.ByteOrder.LITTLE_ENDIAN)
            for (s in samples) body.putShort(s)
            tmp.outputStream().use {
                it.write(header.array())
                it.write(body.array())
            }

            val read = Wav.read(tmp)
            assertEquals(1, read.channels)
            assertEquals(48_000, read.sampleRate)
            assertEquals(samples.size, read.frames)
            // int16/32768 quantization
            val expected = floatArrayOf(0f, 0.5f, -0.5f, 32767f / 32768f, -1f)
            assertArrayEquals(expected, read.samples[0], 1e-4f)
        } finally { tmp.delete() }
    }
}
