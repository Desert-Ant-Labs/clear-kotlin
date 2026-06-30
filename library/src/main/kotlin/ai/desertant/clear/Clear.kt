package ai.desertant.clear

import android.content.Context
import ai.desertant.clear.internal.Constants
import ai.desertant.clear.internal.OnnxModelBridge
import ai.desertant.clear.internal.dsp.FeatureExtractor
import ai.desertant.clear.internal.dsp.Inference
import ai.desertant.clear.internal.dsp.ModelBridge
import ai.desertant.clear.internal.dsp.Stft
import ai.desertant.clear.internal.dsp.StreamingFeatureExtractor
import ai.desertant.clear.internal.dsp.StreamingIStft
import ai.desertant.clear.internal.dsp.StreamingInference
import ai.desertant.clear.internal.dsp.StreamingStft
import ai.desertant.clear.internal.io.AndroidAudioDecoder
import ai.desertant.clear.internal.io.AndroidAudioEncoder
import ai.desertant.clear.internal.io.StreamingAudioDecoder
import ai.desertant.clear.internal.io.StreamingAudioEncoder
import ai.desertant.clear.internal.io.Wav
import ai.desertant.clear.internal.mastering.R128
import ai.desertant.clear.internal.mastering.StreamingLimiter
import ai.desertant.clear.internal.mastering.StreamingMeter
import java.io.Closeable
import java.io.File
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext

/**
 * Speech enhancement for Android. Mirrors the public surface of the
 * Swift `Clear` class.
 *
 * Construct once per logical scope (model load is non-trivial) and
 * reuse the instance across files.
 *
 * ```kotlin
 * val clear = Clear.create(context)
 * val result = clear.enhance("/path/to/noisy.wav")
 * println(result.outputPath)        // /path/to/noisy_clear.wav
 * println(result.realtimeFactor)
 * println(result.measuredLufs)
 * ```
 */
public class Clear private constructor(
    private val context: Context,
    private val model: ModelBridge,
    public val variant: ModelVariant,
) : Closeable {

    private val stft = Stft()
    private val inference = Inference(model)

    public enum class ComputeUnits {
        /** XNNPACK CPU execution provider only. */
        CPU_ONLY,
    }

    @JvmInline
    public value class Strength(public val value: Float) {
        init { require(value in 0f..1f) { "Strength must be in 0..1, got $value" } }

        public companion object {
            public val Full: Strength = Strength(1f)
            public val Medium: Strength = Strength(0.7f)
            public val Subtle: Strength = Strength(0.4f)
        }
    }

    public data class Mastering(
        val integratedLufs: Double = -19.0,
        /** True-peak ceiling in dBTP. -1.5 leaves 0.5 dB headroom for lossy-codec rounding. */
        val truePeakDbtp: Double = -1.5,
        val loudnessRangeLu: Double = 7.0,
        val enabled: Boolean = true,
        /**
         * Optional per-channel LUFS target applied to multi-channel input
         * *before* enhancement. Used to lift the quiet channel of an
         * asymmetric multitrack-podcast input so both mics land at the
         * same level before the model sees them. Don't use for true
         * stereo — per-channel normalization destroys the stereo image.
         */
        val balanceChannelsLufs: Double? = null,
        /**
         * Upper bound on the pre-master gain in dB. Caps how much a
         * very quiet input gets amplified so residual model noise and
         * limiter pre-ring stay inaudible. Output LUFS =
         * `max(integratedLufs, inputLufs + maxLoudnessGainDb)`.
         * Set to `Double.POSITIVE_INFINITY` to disable.
         */
        val maxLoudnessGainDb: Double = 9.0,
    ) {
        public companion object {
            public val ApplePodcasts: Mastering = LoudnessPreset.ApplePodcasts.mastering
            public val Podcast: Mastering = LoudnessPreset.ApplePodcasts.mastering
            public val Spotify: Mastering = LoudnessPreset.Spotify.mastering
            public val YouTube: Mastering = LoudnessPreset.YouTube.mastering
            public val Broadcast: Mastering = LoudnessPreset.Broadcast.mastering
            public val Bypass: Mastering = Mastering(enabled = false)
            public fun targetLufs(lufs: Double): Mastering = Mastering(integratedLufs = lufs)
        }
    }

    /**
     * Standard delivery loudness targets. Iterable so UIs can render a
     * picker without hardcoding values. Use `Mastering.targetLufs(_)`
     * for non-standard targets.
     */
    public enum class LoudnessPreset(
        public val integratedLufs: Double,
        public val displayName: String,
        public val shortName: String,
    ) {
        ApplePodcasts(-19.0, "Apple Podcasts", "Apple"),
        Spotify(-14.0, "Spotify", "Spotify"),
        YouTube(-14.0, "YouTube", "YouTube"),
        Broadcast(-23.0, "Broadcast (EBU R128)", "EBU");

        /** Fully-configured `Mastering` for this preset. */
        public val mastering: Mastering get() = when (this) {
            Broadcast -> Mastering(integratedLufs = -23.0, loudnessRangeLu = 10.0)
            else -> Mastering(integratedLufs = integratedLufs)
        }
    }

    public data class Options(
        val strength: Strength = Strength.Full,
        val mastering: Mastering = Mastering.Podcast,
        val sampleRate: Double = 48_000.0,
        val forceMono: Boolean = false,
    )

    public sealed interface Progress {
        public val fraction: Float
        public data class LoadingModel(override val fraction: Float) : Progress
        public data class Analyzing(override val fraction: Float) : Progress
        public data class Enhancing(override val fraction: Float) : Progress
    }

    /** Per-stage wall-time breakdown of the enhance pass, in seconds. */
    public data class PhaseTimings(
        /** Time the chunk loop spent waiting for decoded audio. */
        val decodeResampleSec: Double = 0.0,
        /** STFT.forward across the file. Per channel; reports the max. */
        val stftForwardSec: Double = 0.0,
        /** Feature extraction — |z|² + ERB projection + EMA normalizations. */
        val computeFeaturesSec: Double = 0.0,
        /** Sum of ONNX inference wall time across the chunk loop. */
        val onnxPredictSec: Double = 0.0,
        /** STFT.inverse across the file. Per channel; max across channels. */
        val stftInverseSec: Double = 0.0,
        /** Strength wet/dry blend + pre-gain. */
        val blendAndGainSec: Double = 0.0,
        /** R128 measurement + look-ahead limiter. */
        val r128AndLimiterSec: Double = 0.0,
        /** Output encoding + disk write. */
        val writeSec: Double = 0.0,
    ) {
        val totalSec: Double get() = decodeResampleSec + stftForwardSec +
            computeFeaturesSec + onnxPredictSec + stftInverseSec +
            blendAndGainSec + r128AndLimiterSec + writeSec
    }

    public data class Result(
        val outputPath: String,
        val durationSec: Double,
        val processingSec: Double,
        val measuredLufs: Double?,
        val measuredTruePeakDbfs: Double?,
        /**
         * Cumulative wall time inside the ONNX model across all chunks.
         * `processingSec - onnxSec` is the Kotlin-side cost.
         */
        val onnxSec: Double,
        /** Which bundled model variant produced this output. */
        val modelVariant: ModelVariant = ModelVariant.ClearStudio,
        val phaseTimings: PhaseTimings,
    ) {
        val realtimeFactor: Double get() = durationSec / processingSec.coerceAtLeast(1e-9)
    }

    /**
     * Which speech-enhancement model is loaded.
     *
     * - `ClearStudio` — quiet, studio-like character; silences near zero.
     *   The default; works across the full range of input quality.
     * - `ClearNatural` — preserves room tone, breath, and lip texture.
     *   For treated podcast studios and intentional voiceover.
     *
     * Both variants share the same architecture and runtime cost; only the
     * weights differ. Both ship bundled in the library assets (~4.5 MB each).
     */
    public enum class ModelVariant(
        public val rawValue: String,
        public val displayName: String,
        public val assetName: String,
    ) {
        ClearStudio("clear-studio", "clear-studio", "clear-studio.onnx"),
        ClearNatural("clear-natural", "clear-natural", "clear-natural.onnx");
    }

    public sealed class Error(message: String, cause: Throwable? = null) : Exception(message, cause) {
        public class AudioReadFailed(message: String, cause: Throwable? = null) : Error(message, cause)
        public class AudioWriteFailed(message: String, cause: Throwable? = null) : Error(message, cause)
        public class UnsupportedSampleRate(public val actual: Int) : Error(
            "Unsupported sample rate: $actual (expected 48000). Convert your input to 48 kHz first.")
        public class InferenceFailed(message: String, cause: Throwable? = null) : Error(message, cause)
        public class ModelLoadFailed(message: String, cause: Throwable? = null) : Error(message, cause)
    }

    /**
     * Enhance an audio file. Returns timing + measured loudness.
     *
     * @param audioPath  absolute path to a WAV file at 48 kHz
     * @param outputPath optional override; defaults to `<stem>_clear.<ext>`
     * @param options    enhancement + mastering options
     * @param progress   optional 0…1 progress callback
     */
    public suspend fun enhance(
        audioPath: String,
        outputPath: String? = null,
        options: Options = Options(),
        progress: ((Progress) -> Unit)? = null,
    ): Result = withContext(Dispatchers.Default) {
        val inFile = File(audioPath)
        if (!inFile.exists()) throw Error.AudioReadFailed("input not found: $audioPath")
        val ext = inFile.extension.lowercase()

        // WAV uses the pure-Kotlin batch reader (MediaCodec can't always
        // negotiate float PCM for raw WAV); other containers stream via
        // MediaExtractor + MediaCodec.
        return@withContext if (ext == "wav" || ext == "wave") {
            enhanceBatch(audioPath, outputPath, options, progress)
        } else {
            enhanceStreaming(audioPath, outputPath, options, progress)
        }
    }

    /**
     * Streaming pipeline. Two passes over the decoder: pass 1 meters
     * R128 to get the integrated LUFS, pass 2 enhances + encodes. Peak
     * memory ~30 MB regardless of file length. Strength<1 and
     * balanceChannelsLufs fall back to the batch path (both need
     * full-file context).
     */
    private suspend fun enhanceStreaming(
        audioPath: String,
        outputPath: String?,
        options: Options,
        progress: ((Progress) -> Unit)?,
    ): Result = withContext(Dispatchers.Default) {
        if (options.strength.value < 1f || options.mastering.balanceChannelsLufs != null) {
            return@withContext enhanceBatch(audioPath, outputPath, options, progress)
        }
        val inFile = File(audioPath)
        val totalStart = System.nanoTime()

        // Pass 1: stream-decode + R128 meter → integrated LUFS.
        val analyzeStart = System.nanoTime()
        val infoDecoder = StreamingAudioDecoder(inFile)
        val srcSampleRate = infoDecoder.sampleRate
        val srcChannels = infoDecoder.channels
        val durationSec = infoDecoder.durationSec
        if (srcSampleRate != 48_000) {
            infoDecoder.close()
            throw Error.UnsupportedSampleRate(srcSampleRate)
        }
        val nOutChannels = if (options.forceMono && srcChannels > 1) 1 else srcChannels

        val meter = StreamingMeter(srcSampleRate)
        var totalFramesSeen = 0L
        while (true) {
            val chunk = infoDecoder.next() ?: break
            val processed = if (options.forceMono && chunk.size > 1)
                arrayOf(downmix(chunk)) else chunk
            meter.consume(processed)
            totalFramesSeen += processed[0].size
        }
        infoDecoder.close()
        val measurement = meter.finalize()
        val analyzeSec = (System.nanoTime() - analyzeStart) / 1e9
        progress?.invoke(Progress.Analyzing(1f))

        // Pre-gain for loudness normalization.
        val preGain: Float = if (options.mastering.enabled && measurement.integratedLufs.isFinite()) {
            val requested = options.mastering.integratedLufs - measurement.integratedLufs
            val capped = minOf(requested, options.mastering.maxLoudnessGainDb)
            Math.pow(10.0, capped / 20.0).toFloat()
        } else 1f

        // Pass 2: streaming enhance + encode.
        val outFile = File(outputPath ?: defaultOutputPath(inFile))
        val outExt = outFile.extension.lowercase()
        val container = when (outExt) {
            "m4a", "aac" -> "m4a"
            else -> "wav"
        }

        val onnxNs = LongArray(1)
        val timingBridge = TimingModelBridge(model, onnxNs)
        val sttfs = Array(nOutChannels) { StreamingStft() }
        val featExtractors = Array(nOutChannels) { StreamingFeatureExtractor() }
        val infs = Array(nOutChannels) { StreamingInference(timingBridge) }
        val istfts = Array(nOutChannels) { StreamingIStft() }
        val limiter = if (options.mastering.enabled)
            StreamingLimiter(options.mastering.truePeakDbtp, srcSampleRate) else null

        val enhanceStart = System.nanoTime()
        val decoder = StreamingAudioDecoder(inFile)
        if (decoder.sampleRate != 48_000) {
            decoder.close()
            throw Error.UnsupportedSampleRate(decoder.sampleRate)
        }

        // No streaming WAV writer yet — accumulate mastered PCM and write at the end.
        // FloatList beats MutableList<Float> by ~13× memory (no Float boxing).
        val wavAccum = if (container != "m4a") Array(nOutChannels) { FloatList() } else null
        val encoder: StreamingAudioEncoder? = if (container == "m4a")
            StreamingAudioEncoder(outFile, srcSampleRate, nOutChannels, container) else null

        var framesProcessed = 0L
        val totalFrames = totalFramesSeen.coerceAtLeast(1)
        while (true) {
            val chunk = decoder.next() ?: break
            val processed = if (options.forceMono && chunk.size > 1)
                arrayOf(downmix(chunk)) else chunk
            val chunkFrames = processed[0].size

            val enhanced = processChunk(processed, sttfs, featExtractors, infs, istfts, preGain, nOutChannels)
            if (enhanced != null && enhanced[0].isNotEmpty()) {
                val mastered = limiter?.process(enhanced) ?: enhanced
                if (encoder != null) {
                    encoder.write(mastered)
                } else if (wavAccum != null) {
                    for (c in 0 until nOutChannels) wavAccum[c].addAll(mastered[c])
                }
            }

            framesProcessed += chunkFrames
            progress?.invoke(Progress.Enhancing(framesProcessed.toFloat() / totalFrames))
        }
        decoder.close()

        // Drain the tail from each streaming stage.
        for (c in 0 until nOutChannels) {
            val infTail = infs[c].flush()
            if (infTail.nFrames > 0) {
                val istftTail = istfts[c].process(infTail.real, infTail.imag, infTail.nFrames)
                for (i in istftTail.indices) istftTail[i] *= preGain
                val tailChunk = arrayOf(istftTail)
                if (nOutChannels == 1) {
                    val mastered = limiter?.process(tailChunk) ?: tailChunk
                    encoder?.write(mastered) ?: wavAccum?.get(0)?.addAll(mastered[0])
                }
                // Multi-channel tails are joined and zero-padded below so the joint
                // limiter sees consistent frame counts.
            }
        }
        if (nOutChannels > 1) {
            val flushes = Array(nOutChannels) { c -> istfts[c].flush() }
            val maxN = flushes.maxOf { it.size }
            if (maxN > 0) {
                val padded = Array(nOutChannels) { c ->
                    val src = flushes[c]
                    if (src.size == maxN) src
                    else FloatArray(maxN).also { System.arraycopy(src, 0, it, 0, src.size) }
                }
                for (c in 0 until nOutChannels) for (i in padded[c].indices) padded[c][i] *= preGain
                val mastered = limiter?.process(padded) ?: padded
                if (encoder != null) encoder.write(mastered)
                else wavAccum?.let { for (c in 0 until nOutChannels) it[c].addAll(mastered[c]) }
            }
        } else {
            val tail = istfts[0].flush()
            for (i in tail.indices) tail[i] *= preGain
            if (tail.isNotEmpty()) {
                val tailChunk = arrayOf(tail)
                val mastered = limiter?.process(tailChunk) ?: tailChunk
                if (encoder != null) encoder.write(mastered)
                else wavAccum?.get(0)?.addAll(mastered[0])
            }
        }
        val limFlush = limiter?.flush()
        if (limFlush != null && limFlush.isNotEmpty() && limFlush[0].isNotEmpty()) {
            if (encoder != null) encoder.write(limFlush)
            else wavAccum?.let { for (c in 0 until nOutChannels) it[c].addAll(limFlush[c]) }
        }

        encoder?.close()
        if (wavAccum != null) {
            val samplesArr = Array(nOutChannels) { c -> wavAccum[c].toFloatArray() }
            Wav.write(outFile, samplesArr, srcSampleRate, nOutChannels)
        }

        val enhanceSec = (System.nanoTime() - enhanceStart) / 1e9
        val totalSec = (System.nanoTime() - totalStart) / 1e9
        Result(
            outputPath = outFile.path,
            durationSec = durationSec.takeIf { it > 0 } ?: (totalFrames.toDouble() / srcSampleRate),
            processingSec = totalSec,
            measuredLufs = if (options.mastering.enabled) options.mastering.integratedLufs else
                measurement.integratedLufs.takeIf { it.isFinite() },
            measuredTruePeakDbfs = measurement.truePeakDbfs.takeIf { it.isFinite() },
            onnxSec = onnxNs[0] / 1e9,
            modelVariant = variant,
            phaseTimings = PhaseTimings(
                // Approximations — splits are rough until we instrument each stage.
                decodeResampleSec = analyzeSec / 2,
                onnxPredictSec = onnxNs[0] / 1e9,
                r128AndLimiterSec = enhanceSec * 0.05,
                writeSec = enhanceSec * 0.10,
            ),
        )
    }

    private fun downmix(chunk: Array<FloatArray>): FloatArray {
        val n = chunk[0].size
        val nCh = chunk.size
        val out = FloatArray(n)
        for (i in 0 until n) {
            var s = 0f
            for (c in 0 until nCh) s += chunk[c][i]
            out[i] = s / nCh
        }
        return out
    }

    private fun processChunk(
        pcmChunk: Array<FloatArray>,
        sttfs: Array<StreamingStft>,
        featExtractors: Array<StreamingFeatureExtractor>,
        infs: Array<StreamingInference>,
        istfts: Array<StreamingIStft>,
        preGain: Float,
        nOutChannels: Int,
    ): Array<FloatArray>? {
        // Per channel: STFT → features → streaming inference → iSTFT → pre-gain.
        val outs = arrayOfNulls<FloatArray>(nOutChannels)
        var maxLen = 0
        for (c in 0 until nOutChannels) {
            val ch = pcmChunk[c]
            val fwd = sttfs[c].process(ch)
            if (fwd.nFrames == 0) { outs[c] = FloatArray(0); continue }
            val feats = featExtractors[c].process(fwd.real, fwd.imag, fwd.nFrames)
            val enh = infs[c].process(
                fwd.real, fwd.imag,
                feats.featErb, feats.featSpecReal, feats.featSpecImag,
                feats.nFrames,
            )
            if (enh.nFrames == 0) { outs[c] = FloatArray(0); continue }
            val time = istfts[c].process(enh.real, enh.imag, enh.nFrames)
            for (i in time.indices) time[i] *= preGain
            outs[c] = time
            if (time.size > maxLen) maxLen = time.size
        }
        if (maxLen == 0) return null
        // Zero-pad shorter channels for the joint limiter.
        val padded = Array(nOutChannels) { c ->
            val src = outs[c]!!
            if (src.size == maxLen) src
            else FloatArray(maxLen).also { System.arraycopy(src, 0, it, 0, src.size) }
        }
        return padded
    }

    /**
     * Batch fallback for WAV input and full-file options
     * (Strength<1, balanceChannelsLufs).
     */
    private suspend fun enhanceBatch(
        audioPath: String,
        outputPath: String?,
        options: Options,
        progress: ((Progress) -> Unit)?,
    ): Result = withContext(Dispatchers.Default) {
        val totalStart = System.nanoTime()

        // 1. Decode
        val decodeStart = System.nanoTime()
        val inFile = File(audioPath)
        if (!inFile.exists()) throw Error.AudioReadFailed("input not found: $audioPath")
        val ext = inFile.extension.lowercase()
        val input: Wav.Audio = try {
            if (ext == "wav" || ext == "wave") Wav.read(inFile)
            else AndroidAudioDecoder.decode(inFile)
        } catch (t: Throwable) {
            throw Error.AudioReadFailed("couldn't decode $audioPath: ${t.message}", t)
        }
        if (input.sampleRate != 48_000) throw Error.UnsupportedSampleRate(input.sampleRate)
        val decodeSec = (System.nanoTime() - decodeStart) / 1e9

        // 2. Channel routing
        val channelsIn: Array<FloatArray> = when {
            options.forceMono && input.channels > 1 -> arrayOf(input.toMono())
            else -> input.samples
        }
        val nOutChannels = channelsIn.size

        // 2a. Per-channel pre-gain — lifts the quiet channel of an asymmetric
        //     multitrack-podcast input before the model sees it.
        options.mastering.balanceChannelsLufs?.let { target ->
            if (nOutChannels > 1) {
                R128.balanceChannels(channelsIn, target, input.sampleRate)
            }
        }

        // 3. Enhance each channel.
        val onnxNs = LongArray(1)
        val outChannels = Array(nOutChannels) { c ->
            val r = enhanceChannel(channelsIn[c], options.strength.value, onnxNs) { f ->
                val overall = (c + f) / nOutChannels
                progress?.invoke(Progress.Enhancing(overall))
            }
            r
        }
        val onnxSec = onnxNs[0] / 1e9

        // 4. Mastering — R128 measure + normalize to target, ceilinged by truePeakDbtp.
        val masteringStart = System.nanoTime()
        var measuredLufs: Double? = null
        var measuredTruePeak: Double? = null
        if (options.mastering.enabled) {
            val meas = R128.normalize(
                channels = outChannels,
                targetLufs = options.mastering.integratedLufs,
                truePeakDbtp = options.mastering.truePeakDbtp,
                sampleRate = input.sampleRate,
                maxLoudnessGainDb = options.mastering.maxLoudnessGainDb,
            )
            // Report the LUFS we actually land at — maxLoudnessGainDb may
            // hold quiet inputs under the target.
            if (meas.integratedLufs.isFinite()) {
                val requested = options.mastering.integratedLufs - meas.integratedLufs
                val applied = minOf(requested, options.mastering.maxLoudnessGainDb)
                measuredLufs = meas.integratedLufs + applied
            }
            val postMeas = R128.measure(outChannels, input.sampleRate)
            if (postMeas.truePeakDbfs.isFinite()) measuredTruePeak = postMeas.truePeakDbfs
        } else {
            val meas = R128.measure(outChannels, input.sampleRate)
            if (meas.integratedLufs.isFinite()) measuredLufs = meas.integratedLufs
            if (meas.truePeakDbfs.isFinite()) measuredTruePeak = meas.truePeakDbfs
        }
        val masteringSec = (System.nanoTime() - masteringStart) / 1e9

        // 5. Encode. .mp3 falls back to .wav — MediaMuxer MP3 encoding isn't
        //    reliable across OEMs.
        val encodeStart = System.nanoTime()
        val outFile = File(outputPath ?: defaultOutputPath(inFile))
        val outExt = outFile.extension.lowercase()
        try {
            val outAudio = Wav.Audio(outChannels, input.sampleRate, nOutChannels)
            when (outExt) {
                "wav", "wave" -> Wav.write(outFile, outChannels, input.sampleRate, nOutChannels)
                "m4a", "aac" -> AndroidAudioEncoder.writeM4A(outFile, outAudio)
                else -> Wav.write(outFile, outChannels, input.sampleRate, nOutChannels)
            }
        } catch (t: Throwable) {
            throw Error.AudioWriteFailed("couldn't write ${outFile.path}: ${t.message}", t)
        }
        val encodeSec = (System.nanoTime() - encodeStart) / 1e9

        val totalSec = (System.nanoTime() - totalStart) / 1e9
        Result(
            outputPath = outFile.path,
            durationSec = input.durationSec,
            processingSec = totalSec,
            measuredLufs = measuredLufs,
            measuredTruePeakDbfs = measuredTruePeak,
            onnxSec = onnxSec,
            modelVariant = variant,
            phaseTimings = PhaseTimings(
                decodeResampleSec = decodeSec,
                onnxPredictSec = onnxSec,
                r128AndLimiterSec = masteringSec,
                writeSec = encodeSec,
                // STFT/feature/inverse times are lumped into onnxSec in the batch path.
            ),
        )
    }

    /** Enhance one mono channel: STFT → features → ONNX chunks → iSTFT → wet/dry mix. */
    private fun enhanceChannel(
        samples: FloatArray,
        strength: Float,
        onnxNsAccum: LongArray,
        onChunkProgress: (Float) -> Unit,
    ): FloatArray {
        if (samples.isEmpty()) return samples

        // Nullable holders so the GC can reclaim each intermediate as soon as
        // the next phase consumes it. Important for long stereo files.
        var sanitized: FloatArray? = sanitize(samples)
        var fwd: Stft.ForwardResult? = stft.forward(sanitized!!)
        sanitized = null
        if (fwd!!.nFrames == 0) return samples

        var features: FeatureExtractor.Features? =
            FeatureExtractor.compute(fwd.real, fwd.imag, fwd.nFrames)

        val onnxStart = System.nanoTime()
        var enhanced: Inference.Enhanced? = inference.run(
            specReal = fwd.real,
            specImag = fwd.imag,
            featErb = features!!.featErb,
            featSpecReal = features.featSpecReal,
            featSpecImag = features.featSpecImag,
            nFrames = fwd.nFrames,
            onChunkProgress = onChunkProgress,
        )
        onnxNsAccum[0] += System.nanoTime() - onnxStart

        fwd = null
        features = null

        val reconstructed = stft.inverse(enhanced!!.real, enhanced.imag, enhanced.nFrames)
        enhanced = null

        // Trim back to original length.
        val trimmed = if (reconstructed.size > samples.size)
            reconstructed.copyOfRange(0, samples.size) else reconstructed

        // Wet/dry mix.
        if (strength >= 1f) return trimmed
        val dryOver = 1f - strength
        val n = minOf(trimmed.size, samples.size)
        val out = FloatArray(trimmed.size)
        for (i in 0 until n) out[i] = trimmed[i] * strength + samples[i] * dryOver
        for (i in n until trimmed.size) out[i] = trimmed[i] * strength
        return out
    }

    private fun sanitize(s: FloatArray): FloatArray {
        // Replace NaN/Inf with 0.
        var clean: FloatArray? = null
        for (i in s.indices) {
            val v = s[i]
            if (v.isNaN() || v.isInfinite()) {
                if (clean == null) clean = s.copyOf()
                clean[i] = 0f
            }
        }
        return clean ?: s
    }

    private fun defaultOutputPath(input: File): String {
        val stem = input.nameWithoutExtension
        val ext = input.extension.ifEmpty { "wav" }
        val outExt = if (ext.equals("mp3", ignoreCase = true)) "wav" else ext
        return File(input.parentFile, "${stem}_clear.${outExt}").path
    }

    override fun close() {
        (model as? Closeable)?.close()
    }

    /** ModelBridge wrapper that accumulates runChunk wall time. */
    private class TimingModelBridge(
        private val inner: ModelBridge,
        private val nsAccum: LongArray,
    ) : ModelBridge {
        override fun runChunk(spec: FloatArray, featErb: FloatArray, featSpec: FloatArray): FloatArray {
            val t0 = System.nanoTime()
            try {
                return inner.runChunk(spec, featErb, featSpec)
            } finally {
                nsAccum[0] += System.nanoTime() - t0
            }
        }
    }

    public companion object {
        /**
         * Construct a Clear instance with the requested model variant.
         *
         * @param context Android context — used for asset loading.
         * @param variant which bundled model to load.
         * @param computeUnits ONNX Runtime execution provider.
         */
        public suspend fun create(
            context: Context,
            variant: ModelVariant = ModelVariant.ClearStudio,
            computeUnits: ComputeUnits = ComputeUnits.CPU_ONLY,
            progress: ((Progress) -> Unit)? = null,
        ): Clear = withContext(Dispatchers.IO) {
            progress?.invoke(Progress.LoadingModel(0f))
            val model = try {
                OnnxModelBridge.fromAssets(context, variant.assetName)
            } catch (t: Throwable) {
                throw Error.ModelLoadFailed(
                    "couldn't load ${variant.assetName}: ${t.message}", t)
            }
            progress?.invoke(Progress.LoadingModel(1f))
            Clear(context, model, variant)
        }

        /** Test-only constructor: wire an arbitrary model bridge. */
        public fun withBridge(
            context: Context,
            model: ModelBridge,
            variant: ModelVariant = ModelVariant.ClearStudio,
        ): Clear = Clear(context, model, variant)
    }
}

/**
 * Growable FloatArray-backed list. Avoids boxing every PCM sample into a
 * `java.lang.Float` the way `MutableList<Float>` does (~50 B/sample → GB-scale
 * heap for multi-minute outputs). Append-only is all we need.
 */
private class FloatList(initialCapacity: Int = 1024) {
    private var buf: FloatArray = FloatArray(initialCapacity)
    private var n: Int = 0

    fun addAll(values: FloatArray) {
        ensure(n + values.size)
        System.arraycopy(values, 0, buf, n, values.size)
        n += values.size
    }

    fun toFloatArray(): FloatArray = buf.copyOf(n)

    private fun ensure(capacity: Int) {
        if (capacity <= buf.size) return
        var next = buf.size
        while (next < capacity) next = (next * 3 + 1) / 2
        buf = buf.copyOf(next)
    }
}
