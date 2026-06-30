pluginManagement {
    repositories {
        google {
            content {
                includeGroupByRegex("com\\.android.*")
                includeGroupByRegex("com\\.google.*")
                includeGroupByRegex("androidx.*")
            }
        }
        mavenCentral()
        gradlePluginPortal()
    }
}

dependencyResolutionManagement {
    repositoriesMode.set(RepositoriesMode.FAIL_ON_PROJECT_REPOS)
    repositories {
        google()
        mavenCentral()
    }
}

rootProject.name = "clear"
include(":dsp")

// Conditionally include :library only when the Android SDK is reachable.
// :dsp is pure JVM (numerical-parity tests) and runs fine without it; the
// Android library module needs AGP, which needs the SDK to configure.
// The sample app lives in Examples/ClearSample as its own composite build.
val androidSdkPresent = System.getenv("ANDROID_HOME") != null ||
    System.getenv("ANDROID_SDK_ROOT") != null ||
    file("local.properties").exists()
if (androidSdkPresent) {
    include(":library")
} else {
    println("[clear] Skipping :library — no Android SDK detected. Run the :dsp parity tests without it.")
}
