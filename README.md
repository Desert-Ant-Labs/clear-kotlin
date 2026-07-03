# Clear for Android

Kotlin library that runs the Clear speech-enhancement model on Android.
Mirrors the role and public API of the [`clear-swift`](https://github.com/Desert-Ant-Labs/clear-swift) package on Apple platforms.

Two published artifacts (group `ai.desertant`):
- `clear`: Android library (AAR) with both ONNX model variants bundled
- `clear-dsp`: pure-JVM DSP primitives (transitive)

The library bundles both shipped variants, `clear-studio` (default,
quiet studio-like cleanup) and `clear-natural` (preserves room tone),
selectable via `Clear.ModelVariant`.

39 tests green, including an end-to-end pipeline test that loads the
real `clear-studio.onnx`, processes a WAV through the full STFT → ONNX →
iSTFT → R128 LUFS-normalize chain, and verifies the output lands at the
target loudness within ±0.5 LU.

## Add to your Android app

The library bundles the ONNX model in its assets, so it's a single
dependency. It's distributed through
[JitPack](https://jitpack.io/#Desert-Ant-Labs/clear-kotlin).

### 1. Add the JitPack repository

```kotlin
// settings.gradle.kts
dependencyResolutionManagement {
    repositories {
        google()
        mavenCentral()
        maven(url = "https://jitpack.io")
    }
}
```

### 2. Wire it into your app's `build.gradle.kts`

Replace the tag with the
[latest release](https://github.com/Desert-Ant-Labs/clear-kotlin/releases):

```kotlin
dependencies {
    implementation("com.github.Desert-Ant-Labs.clear-kotlin:clear:0.1.0")
    // com.github.Desert-Ant-Labs.clear-kotlin:clear-dsp is pulled transitively.
    // onnxruntime-android (~10 MB) is also pulled transitively.
}
```

For pure local development you can instead publish to mavenLocal
(`./gradlew :dsp:publishToMavenLocal :library:publishToMavenLocal`,
artifacts land under `~/.m2/repository/ai/desertant/`).

Minimum Android version: **API 24 (Android 7.0)**, which covers ~98% of
devices in 2026.

### 3. Use it

```kotlin
import ai.desertant.clear.Clear
import kotlinx.coroutines.launch

class AudioEditor(private val activity: Activity) {

    // Construct once per logical scope. Reuse across calls.
    private var clear: Clear? = null

    suspend fun ensureLoaded() {
        if (clear == null) clear = Clear.create(activity)
    }

    suspend fun enhance(inputWavPath: String): String {
        ensureLoaded()
        val result = clear!!.enhance(inputWavPath)
        return result.outputPath   // /path/to/<stem>_clear.wav
    }
}

// Caller:
lifecycleScope.launch {
    val editor = AudioEditor(this@MyActivity)
    val outputPath = editor.enhance("/sdcard/Recording.wav")
    Log.d("Clear", "enhanced → $outputPath")
}
```

### Common patterns

**Match a platform's loudness target:**
```kotlin
import ai.desertant.clear.Clear.Mastering

val result = clear.enhance(path,
    options = Clear.Options(mastering = Mastering.Spotify))    // -14 LUFS
```

Available presets: `ApplePodcasts` (-19), `Podcast` (alias),
`Spotify` (-14), `YouTube` (-14), `Broadcast` (-23), `Bypass`,
`targetLufs(custom)`.

**Lighter denoise to preserve room character:**
```kotlin
val result = clear.enhance(path,
    options = Clear.Options(strength = Clear.Strength.Medium))   // 0.7 wet, 0.3 raw
```

**Mono downmix for spoken word:**
```kotlin
val result = clear.enhance(path,
    options = Clear.Options(forceMono = true))
```

**Progress UI:**
```kotlin
val result = clear.enhance(path) { progress ->
    val bar = when (progress) {
        is Clear.Progress.LoadingModel -> 0f
        is Clear.Progress.Analyzing -> progress.fraction * 0.05f
        is Clear.Progress.Enhancing -> 0.05f + progress.fraction * 0.95f
    }
    runOnUiThread { progressBar.progress = (bar * 100).toInt() }
}
```

### Errors to handle

```kotlin
import ai.desertant.clear.Clear.Error

try {
    val result = clear.enhance(path)
} catch (e: Error.AudioReadFailed) {
    // Surface "couldn't read audio"; no retry will help.
} catch (e: Error.UnsupportedSampleRate) {
    // Input must be 48 kHz. v1 doesn't resample.
    // Convert with MediaCodec/MediaExtractor first or use AudioFormat.
} catch (e: Error.InferenceFailed) {
    // Likely device memory pressure; retry once, then surface.
} catch (e: Error.ModelLoadFailed) {
    // Asset missing or corrupted. Treat enhancement as unavailable
    // and ship the original audio.
}
```

For a typical app, the worst acceptable fallback is to ship the
original un-enhanced audio. Don't crash on enhancement failures.

## What v0.1.0 does

- Mono and stereo WAV in → enhanced WAV out
- 48 kHz only (no resampling, so convert your input first)
- Mastering chain: K-weighted LUFS measurement + gain-only loudness
  normalization to target, clipped at the dBTP ceiling
- `Strength` wet/dry blend
- `forceMono` downmix
- Bundled `clear-studio.onnx` + `clear-natural.onnx` (~4.5 MB each), no first-launch download

## What v0.1.0 doesn't do yet

- **Format support beyond WAV.** M4A/MP3/AAC decode via MediaExtractor
  is on the v0.2.0 roadmap. For now, convert your input to WAV
  upstream (Android's `MediaCodec` does this in ~20 LOC).
- **4× polyphase true-peak detection.** Uses sample-peak instead.
  Outputs may overshoot true-peak by up to ~0.5 dB on some content.
- **Look-ahead limiter.** Loud transients are gain-clipped, not
  compressed. Inputs whose dynamic range exceeds the LUFS↔dBTP
  headroom will hit the peak ceiling and get cropped.
- **Microphone / streaming input.** File-based only. Same as iOS v0.1.
- **NPU/NNAPI acceleration.** XNNPACK CPU only.
- **`balanceChannelsLufs`** opt-in pre-gain, exposed in the API but
  not yet applied.

## Performance

Measured on a Nothing Phone (A001T, Snapdragon 7s Gen 3), warm, mono WAV path:

| Stage | Realtime factor |
|---|---|
| Model inference only (clear-studio fp32) | ~47× |
| Full pipeline (decode → STFT → model → iSTFT → master → encode) | ~25× |

A 30-minute episode enhances in ~70-90 s on this class of device; lower-mid SoCs
stay comfortably above realtime.

### Tuning notes (portable, CPU/XNNPACK)

- **`ALL_OPT` graph optimization** (vs `BASIC_OPT`): ~18% faster inference. Applied.
- **Intra-op threads capped at `min(4, cores)`**: matches the performance-core count on
  big.LITTLE SoCs. Going wider spills onto efficiency cores and *halves* throughput.
- **Real FFT** (`realForward`/`realInverse`) instead of a complex FFT on real input:
  ~halves STFT cost, parity-validated against the Swift fixtures.
- **NNAPI / fp16 give no CPU win** here: recurrent ops force CPU fallback, and ORT's
  CPU EP has no native fp16 Conv kernels. fp32 on CPU is the fastest portable
  configuration. A chip-specific NPU EP (e.g. ORT QNN on Qualcomm) is the only way to
  beat it, and is out of scope for an all-devices baseline.

## Building the AAR yourself

If you'd rather pull the AAR directly than via Maven:

```sh
brew install --cask android-commandlinetools   # if not already installed
yes | sdkmanager --licenses
sdkmanager "platforms;android-34" "build-tools;34.0.0" "ndk;26.1.10909125"
export ANDROID_HOME=/opt/homebrew/share/android-commandlinetools

./gradlew :library:assembleRelease
# AAR drops at: library/build/outputs/aar/library-release.aar
```

For just the DSP parity tests (no Android SDK needed):

```sh
brew install openjdk@17 gradle
export JAVA_HOME="/opt/homebrew/opt/openjdk@17/libexec/openjdk.jdk/Contents/Home"

./gradlew :dsp:test
```

## Layout

```
settings.gradle.kts             Gradle root (conditional :library include)
  build.gradle.kts              plugin versions
  gradle.properties             JVM args, AndroidX, Kotlin code style
  dsp/                          PURE JVM module, no Android SDK needed
    build.gradle.kts            kotlin("jvm") + maven-publish
    src/main/kotlin/ai/desertant/clear/internal/
      Constants.kt              DSP geometry constants (Swift-anchored)
      dsp/Stft.kt               STFT forward/inverse (JTransforms backend)
      dsp/ErbFilterbank.kt      32-band ERB projection
      dsp/FeatureExtractor.kt   EMA-normalized feature tensors
      dsp/Inference.kt          Chunked inference loop + ModelBridge
      io/Wav.kt                 PCM int16/24/32 + float32 WAV codec
      mastering/R128.kt         K-weighted LUFS + loudness normalize
    src/test/kotlin/ai/desertant/clear/parity/
                                Stage{1,2,3,5} parity tests + Fixture loader
    src/test/kotlin/ai/desertant/clear/pipeline/
                                End-to-end smoke + ONNX integration tests
    src/test/resources/fixtures/
                                Committed Swift-generated parity fixtures
  library/                      Android library module (needs Android SDK)
    build.gradle.kts            com.android.library + ORT-Android + maven-publish
    src/main/kotlin/ai/desertant/clear/Clear.kt
                                public API (enhance(), options, mastering presets)
    src/main/kotlin/ai/desertant/clear/internal/OnnxModelBridge.kt
                                ORT Android model bridge
    src/main/cpp/               PFFFT JNI scaffold (not yet wired)
    src/main/assets/clear-studio.onnx
    src/main/assets/clear-natural.onnx
                                Bundled models (~4.5 MB each)
```

## Parity fixtures

`dsp/src/test/resources/fixtures/` holds committed numerical-parity
fixtures (synthetic-signal STFT/ERB/feature/ISTFT stages) that pin the
Kotlin DSP to the Swift reference. They're checked in and consumed by the
`:dsp` parity tests; treat a parity-test failure as a real numerical
divergence to investigate.

## See also

- [`clear-swift`](https://github.com/Desert-Ant-Labs/clear-swift): the Swift package this library mirrors.
- [`clear-training`](https://github.com/Desert-Ant-Labs/clear-training): training, evaluation, and the published model card.

## License

[Desert Ant Labs Source-Available License](https://license.desertant.ai/1.0). Free for
most apps; a commercial license is required at scale. Full terms are at the link.
Licensing: <licensing@desertant.ai>.
