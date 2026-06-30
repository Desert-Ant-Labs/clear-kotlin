plugins {
    id("com.android.library")
    id("org.jetbrains.kotlin.android")
    id("maven-publish")
}

group = "ai.desertant"
version = "0.1.0"

android {
    namespace = "ai.desertant.clear"
    compileSdk = 34

    publishing {
        singleVariant("release") {
            withSourcesJar()
        }
    }

    defaultConfig {
        minSdk = 24
        consumerProguardFiles("consumer-rules.pro")

        ndk {
            abiFilters += listOf("arm64-v8a")
        }
    }

    buildTypes {
        release {
            isMinifyEnabled = false
        }
    }

    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_17
        targetCompatibility = JavaVersion.VERSION_17
    }

    kotlinOptions {
        jvmTarget = "17"
    }

    testOptions {
        unitTests.isReturnDefaultValues = true
        unitTests.isIncludeAndroidResources = true
    }
}

dependencies {
    api(project(":dsp"))

    // ONNX Runtime Mobile (CPU/XNNPACK). `api` so consumers can construct
    // their own OrtSession without re-declaring the dependency.
    api("com.microsoft.onnxruntime:onnxruntime-android:1.17.1")

    implementation("org.jetbrains.kotlinx:kotlinx-coroutines-android:1.7.3")

    testImplementation("junit:junit:4.13.2")
    testImplementation("org.jetbrains.kotlinx:kotlinx-coroutines-test:1.7.3")
}

afterEvaluate {
    publishing {
        publications {
            create<MavenPublication>("release") {
                from(components["release"])
                artifactId = "clear"
                pom {
                    name.set("Clear for Android")
                    description.set("On-device speech enhancement library. Kotlin + ONNX " +
                        "Runtime Mobile. Mirrors the Swift Clear package's API.")
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
}
