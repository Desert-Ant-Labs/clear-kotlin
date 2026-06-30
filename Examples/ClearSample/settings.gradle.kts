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

rootProject.name = "clear-sample"

// Build the Clear SDK from source in this repo — the Gradle equivalent of a
// local `file:../..` dependency. `ai.desertant:clear` resolves to the SDK's
// :library module. To depend on a published release instead, drop this block
// and add JitPack + `implementation("com.github.Desert-Ant-Labs.clear-kotlin:clear:<tag>")`.
includeBuild("../..") {
    dependencySubstitution {
        substitute(module("ai.desertant:clear")).using(project(":library"))
    }
}
