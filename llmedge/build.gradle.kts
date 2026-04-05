import org.gradle.api.artifacts.repositories.PasswordCredentials

/*
 * Copyright (C) 2024 Shubham Panchal
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

plugins {
    alias(libs.plugins.android.library)
    alias(libs.plugins.kotlin.android)
    id("org.jetbrains.kotlin.plugin.serialization") version "2.0.0"
    id("jacoco")
    id("com.vanniktech.maven.publish") version "0.34.0"
}

val publishingGroup = providers.gradleProperty("GROUP").orElse("io.github.aatricks")
val publishingVersion = providers.gradleProperty("VERSION_NAME").orElse("0.1.0-SNAPSHOT")
val pomArtifactId = providers.gradleProperty("POM_ARTIFACT_ID").orElse(project.name)
val pomName = providers.gradleProperty("POM_NAME").orElse(project.name)
val pomDescription =
    providers.gradleProperty("POM_DESCRIPTION").orElse("Android on-device AI library with bundled native runtimes.")
val pomInceptionYear = providers.gradleProperty("POM_INCEPTION_YEAR").orElse("2024")
val pomUrl = providers.gradleProperty("POM_URL").orElse("https://github.com/Aatricks/llmedge")
val pomLicenseName =
    providers.gradleProperty("POM_LICENSE_NAME").orElse("The Apache License, Version 2.0")
val pomLicenseUrl =
    providers.gradleProperty("POM_LICENSE_URL").orElse("https://www.apache.org/licenses/LICENSE-2.0.txt")
val pomLicenseDist = providers.gradleProperty("POM_LICENSE_DIST").orElse("repo")
val pomScmUrl = providers.gradleProperty("POM_SCM_URL").orElse("https://github.com/Aatricks/llmedge")
val pomScmConnection =
    providers.gradleProperty("POM_SCM_CONNECTION")
        .orElse("scm:git:git://github.com/Aatricks/llmedge.git")
val pomScmDeveloperConnection =
    providers.gradleProperty("POM_SCM_DEV_CONNECTION")
        .orElse("scm:git:ssh://git@github.com/Aatricks/llmedge.git")
val pomDeveloperId = providers.gradleProperty("POM_DEVELOPER_ID").orElse("aatricks")
val pomDeveloperName = providers.gradleProperty("POM_DEVELOPER_NAME").orElse("Aatricks")
val pomDeveloperUrl = providers.gradleProperty("POM_DEVELOPER_URL").orElse("https://github.com/Aatricks")
val remoteRepositoryUrl =
    providers.gradleProperty("llmedge.publish.repoUrl")
        .orElse(providers.environmentVariable("LLMEDGE_PUBLISH_REPO_URL"))
val remoteRepositoryUsername =
    providers.gradleProperty("llmedge.publish.username")
        .orElse(providers.environmentVariable("LLMEDGE_PUBLISH_USERNAME"))
val remoteRepositoryPassword =
    providers.gradleProperty("llmedge.publish.password")
        .orElse(providers.environmentVariable("LLMEDGE_PUBLISH_PASSWORD"))
val githubRepositoryFromEnv =
    providers.environmentVariable("GITHUB_REPOSITORY").map { value -> value.substringAfter('/', "") }
val githubPackagesOwner =
    providers.gradleProperty("llmedge.publish.githubOwner")
        .orElse(providers.environmentVariable("GITHUB_REPOSITORY_OWNER"))
val githubPackagesRepository =
    providers.gradleProperty("llmedge.publish.githubRepository")
        .orElse(githubRepositoryFromEnv)
val githubPackagesUsername =
    providers.gradleProperty("llmedge.publish.githubUsername")
        .orElse(providers.environmentVariable("GITHUB_ACTOR"))
val githubPackagesPassword =
    providers.gradleProperty("llmedge.publish.githubToken")
        .orElse(providers.environmentVariable("GITHUB_TOKEN"))

group = publishingGroup.get()
version = publishingVersion.get()

android {
    // Keeping the Kotlin package as-is to avoid JNI breakage; namespace can be branded separately if desired
    namespace = "io.aatricks.llmedge"
    compileSdk = 35
    ndkVersion = "27.2.12479018"

    val hostIsWindows = System.getProperty("os.name")?.startsWith("Windows", ignoreCase = true) == true
    val defaultVulkanFlag = if (hostIsWindows) "OFF" else "ON"
    val openClFlag =
        providers.gradleProperty("llmedgeAndroidOpencl")
            .orElse(providers.environmentVariable("LLMEDGE_ANDROID_OPENCL"))
            .orElse("OFF")
            .get()
            .uppercase()

    defaultConfig {
        minSdk = 30  // Vulkan 1.2 requires API 30+ (Android 11)
        testInstrumentationRunner = "androidx.test.runner.AndroidJUnitRunner"
        consumerProguardFiles("consumer-rules.pro")
        ndk {
            // The ik_llama.cpp migration is currently validated only on arm64.
            abiFilters += "arm64-v8a"
        }
        externalNativeBuild {
            cmake {
                cppFlags += listOf()
                // allow compiling 16 KB page-aligned shared libraries
                // https://developer.android.com/guide/practices/page-sizes#compile-r27
                arguments += listOf("-DANDROID_SUPPORT_FLEXIBLE_PAGE_SIZES=ON")
                arguments += "-DCMAKE_BUILD_TYPE=Release"
                arguments += "-DSD_VULKAN=$defaultVulkanFlag"
                arguments += "-DGGML_VULKAN=$defaultVulkanFlag"
                arguments += "-DLLMEDGE_ANDROID_OPENCL=$openClFlag"
                arguments += "-DWAN_SUPPORT=ON"

                // (debugging) uncomment the following line to enable debug builds
                // and attach hardware-assisted address sanitizer
                // arguments += "-DCMAKE_BUILD_TYPE=Debug"
                // arguments += listOf("-DANDROID_SANITIZE=hwaddress")
            }
        }
    }

    buildTypes {
        release {
            isMinifyEnabled = false
            proguardFiles(getDefaultProguardFile("proguard-android-optimize.txt"), "proguard-rules.pro")
        }
    }
    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_17
        targetCompatibility = JavaVersion.VERSION_17
    }
    kotlinOptions {
        jvmTarget = "17"
    }
    externalNativeBuild {
        cmake {
            path = file("src/main/cpp/CMakeLists.txt")
            version = "3.22.1"
        }
    }
    packaging {
        jniLibs {
            excludes += "**/libOpenCL.so"
        }
        resources {
            excludes += "/META-INF/{AL2.0,LGPL2.1}"
            // Handle duplicate files from JavaCPP
            pickFirsts += "META-INF/native-image/**"
            pickFirsts += "META-INF/maven/**"
            pickFirsts += "META-INF/INDEX.LIST"
            pickFirsts += "META-INF/LICENSE"
            pickFirsts += "META-INF/LICENSE.txt"
            pickFirsts += "META-INF/NOTICE"
            pickFirsts += "META-INF/NOTICE.txt"
            pickFirsts += "META-INF/LICENSE.md"
            pickFirsts += "META-INF/LICENSE-notice.md"
        }
    }
    testOptions {
        managedDevices {
            devices {
                maybeCreate<com.android.build.api.dsl.ManagedVirtualDevice>("pixel6api33").apply {
                    device = "Pixel 6"
                    apiLevel = 33
                    systemImageSource = "aosp-atd"
                }
            }
        }
    }

    sourceSets {
        getByName("main").java.srcDir(layout.buildDirectory.dir("generated/source/nativeTargetNames/main/kotlin"))
    }
}

dependencies {
    implementation("org.jetbrains.kotlinx:kotlinx-coroutines-core:1.10.1")
    // Sentence Embeddings (on-device) - provides ONNX-based sentence-transformers
    implementation("io.gitlab.shubham0204:sentence-embeddings:v6")
    implementation("androidx.lifecycle:lifecycle-common:2.8.7")

    // Hugging Face Hub client (Ktor + JSON serialization)
    implementation("io.ktor:ktor-client-core:2.3.12")
    implementation("io.ktor:ktor-client-okhttp:2.3.12")
    implementation("io.ktor:ktor-client-content-negotiation:2.3.12")
    implementation("io.ktor:ktor-serialization-kotlinx-json:2.3.12")
    implementation("org.jetbrains.kotlinx:kotlinx-serialization-json:1.7.3")

    // PDF parsing on Android (Apache 2.0)
    implementation("com.tom-roush:pdfbox-android:2.0.27.0")

    // JSON serialization (for simple local embedding index persistence)
    implementation("com.google.code.gson:gson:2.11.0")

    // OCR support - Google ML Kit Text Recognition (Tesseract removed)

    // OCR support - Google ML Kit Text Recognition
    implementation("com.google.mlkit:text-recognition:16.0.0")
    // Image labeling for fast local description
    implementation("com.google.mlkit:image-labeling:17.0.7")
    // Optional: Additional language support for ML Kit
    // implementation("com.google.mlkit:text-recognition-chinese:16.0.0")
    // implementation("com.google.mlkit:text-recognition-japanese:16.0.0")
    // implementation("com.google.mlkit:text-recognition-korean:16.0.0")

    // Coroutines support for ML Kit
    implementation("org.jetbrains.kotlinx:kotlinx-coroutines-play-services:1.10.1")

    testImplementation(libs.junit)
    testImplementation("org.jetbrains.kotlinx:kotlinx-coroutines-test:1.10.1")
    testImplementation("org.jetbrains.kotlin:kotlin-reflect:2.0.0")
    testImplementation("io.mockk:mockk:1.13.12")
    testImplementation("io.ktor:ktor-client-mock:2.3.12")
    testImplementation("com.squareup.okhttp3:mockwebserver:4.12.0")
    testImplementation("org.robolectric:robolectric:4.13")
    testImplementation("androidx.test:core:1.6.1")

    androidTestImplementation("org.jetbrains.kotlinx:kotlinx-coroutines-test:1.10.1")
    androidTestImplementation("androidx.test:runner:1.6.2")
    androidTestImplementation(libs.androidx.junit)
    androidTestImplementation("androidx.test.espresso:espresso-core:3.6.1")
    androidTestImplementation("io.mockk:mockk-android:1.13.12")
}

apply(from = rootProject.file("gradle/llmedge-test-config.gradle"))
apply(from = rootProject.file("gradle/llmedge-jacoco.gradle"))

mavenPublishing {
    coordinates(
        publishingGroup.get(),
        pomArtifactId.get(),
        publishingVersion.get(),
    )
    publishToMavenCentral()
    pom {
        name.set(pomName)
        description.set(pomDescription)
        inceptionYear.set(pomInceptionYear)
        url.set(pomUrl)

        licenses {
            license {
                name.set(pomLicenseName)
                url.set(pomLicenseUrl)
                distribution.set(pomLicenseDist)
            }
        }

        developers {
            developer {
                id.set(pomDeveloperId)
                name.set(pomDeveloperName)
                url.set(pomDeveloperUrl)
            }
        }

        scm {
            url.set(pomScmUrl)
            connection.set(pomScmConnection)
            developerConnection.set(pomScmDeveloperConnection)
        }
    }
}

publishing {
    repositories {
        remoteRepositoryUrl.orNull?.let { configuredUrl ->
            maven {
                name = "ConfiguredRemote"
                url = uri(configuredUrl)
                credentials(PasswordCredentials::class) {
                    username = remoteRepositoryUsername.orNull
                    password = remoteRepositoryPassword.orNull
                }
            }
        }

        val githubOwner = githubPackagesOwner.orNull
        val githubRepository = githubPackagesRepository.orNull
        if (!githubOwner.isNullOrBlank() && !githubRepository.isNullOrBlank()) {
            maven {
                name = "GitHubPackages"
                url = uri("https://maven.pkg.github.com/$githubOwner/$githubRepository")
                credentials(PasswordCredentials::class) {
                    username = githubPackagesUsername.orNull
                    password = githubPackagesPassword.orNull
                }
            }
        }
    }
}

val generateNativeTargetNames by tasks.registering(Exec::class) {
    group = "build setup"
    description = "Generates Kotlin and shell native target metadata from CMake target declarations."
    val kotlinOutputFile =
        layout.buildDirectory.file(
            "generated/source/nativeTargetNames/main/kotlin/io/aatricks/llmedge/core/NativeTargetNames.kt",
        )
    val shellOutputFile =
        layout.buildDirectory.file(
            "generated/source/nativeTargetNames/main/shell/native-targets.sh",
        )
    outputs.files(kotlinOutputFile, shellOutputFile)
    commandLine(
        "bash",
        "${rootProject.projectDir}/scripts/generate_native_target_names.sh",
        kotlinOutputFile.get().asFile.absolutePath,
        shellOutputFile.get().asFile.absolutePath,
    )
}

tasks.named("preBuild") {
    dependsOn(generateNativeTargetNames)
}
