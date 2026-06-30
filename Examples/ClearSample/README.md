# ClearSample — Android sample app

A minimal Android app that runs [`clear-kotlin`](../../) end to end: load the
model, enhance a bundled WAV, and a couple of bench activities.

It's a **standalone Gradle build** that consumes the SDK from source via a
composite build (`includeBuild("../..")` in [`settings.gradle.kts`](./settings.gradle.kts)),
the Gradle equivalent of a local `file:../..` dependency. The
`ai.desertant:clear` dependency resolves to the SDK's `:library` module.

## Run

Needs an Android SDK (`ANDROID_HOME` set) and a device/emulator.

```sh
cd Examples/ClearSample
./gradlew installDebug      # build + install
# or open this folder in Android Studio
```

## Depending on a published release instead

To build against a published JitPack release rather than local source, drop
the `includeBuild` block from `settings.gradle.kts`, add the JitPack repo, and
change the dependency to:

```kotlin
implementation("com.github.Desert-Ant-Labs.clear-kotlin:clear:<tag>")
```
