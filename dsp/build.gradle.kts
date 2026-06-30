// Pure-JVM DSP module — platform-neutral float math, no Android deps.
// Split out from :library so unit tests run without the Android SDK.

plugins {
    id("org.jetbrains.kotlin.jvm")
    id("maven-publish")
}

group = "ai.desertant"
version = "0.1.0"

java {
    sourceCompatibility = JavaVersion.VERSION_17
    targetCompatibility = JavaVersion.VERSION_17
}

kotlin {
    jvmToolchain(17)
}

dependencies {
    // JTransforms — pure-Java mixed-radix FFT for the 960-point STFT.
    implementation("com.github.wendykierp:JTransforms:3.1")

    testImplementation("junit:junit:4.13.2")
    testImplementation("org.json:json:20240303")
    // Desktop-JVM ONNX Runtime for smoke tests against the real model.
    testImplementation("com.microsoft.onnxruntime:onnxruntime:1.17.1")
}

tasks.test {
    useJUnit()
    // Headroom for full-file spectrum buffers on long test clips.
    maxHeapSize = "6g"
    testLogging {
        events("passed", "failed", "skipped")
        showStandardStreams = true
    }
}

java {
    withSourcesJar()
}

publishing {
    publications {
        create<MavenPublication>("maven") {
            from(components["java"])
            artifactId = "clear-dsp"
            pom {
                name.set("Clear DSP")
                description.set("Pure-JVM DSP primitives for Clear (STFT, ERB filterbank, " +
                    "feature extraction, chunked inference, WAV I/O). Shared by the " +
                    "Android library and any JVM-side tooling.")
                url.set("https://github.com/Desert-Ant-Labs/clear-kotlin")
                licenses {
                    license {
                        name.set("Desert Ant Labs Source-Available License v1.0")
                        url.set("https://github.com/Desert-Ant-Labs/clear-kotlin/blob/main/LICENSE.md")
                    }
                }
                scm {
                    url.set("https://github.com/Desert-Ant-Labs/clear-kotlin")
                    connection.set("scm:git:https://github.com/Desert-Ant-Labs/clear-kotlin.git")
                }
            }
        }
    }
}

