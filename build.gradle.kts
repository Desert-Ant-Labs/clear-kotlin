// Root project — plugin declarations only. Module configuration lives
// in dsp/build.gradle.kts and library/build.gradle.kts. The sample app
// is a separate composite build under Examples/ClearSample.

plugins {
    id("com.android.library") version "8.3.2" apply false
    id("org.jetbrains.kotlin.android") version "1.9.22" apply false
    id("org.jetbrains.kotlin.jvm") version "1.9.22" apply false
}
