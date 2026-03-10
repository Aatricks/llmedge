import org.gradle.testing.jacoco.plugins.JacocoTaskExtension
import org.gradle.testing.jacoco.tasks.JacocoReport
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
}

android {
    // Keeping the Kotlin package as-is to avoid JNI breakage; namespace can be branded separately if desired
    namespace = "io.aatricks.llmedge"
    compileSdk = 35
    ndkVersion = "27.2.12479018"

    val hostIsWindows = System.getProperty("os.name")?.startsWith("Windows", ignoreCase = true) == true
    val defaultVulkanFlag = if (hostIsWindows) "OFF" else "ON"

    defaultConfig {
        minSdk = 30  // Vulkan 1.2 requires API 30+ (Android 11)
        testInstrumentationRunner = "androidx.test.runner.AndroidJUnitRunner"
        consumerProguardFiles("consumer-rules.pro")
        externalNativeBuild {
            cmake {
                cppFlags += listOf()
                // allow compiling 16 KB page-aligned shared libraries
                // https://developer.android.com/guide/practices/page-sizes#compile-r27
                arguments += listOf("-DANDROID_SUPPORT_FLEXIBLE_PAGE_SIZES=ON")
                arguments += "-DCMAKE_BUILD_TYPE=Release"
                arguments += "-DSD_VULKAN=$defaultVulkanFlag"
                arguments += "-DGGML_VULKAN=$defaultVulkanFlag"
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
        testOptions {
            unitTests.all {
                // Optional: show test stdout/stderr in Gradle output (useful for long-running
                // local E2E tests that download models or run generation).
                val showTestOutput =
                    (System.getenv("LLMEDGE_SHOW_TEST_OUTPUT")?.equals("true", ignoreCase = true) == true) ||
                        (project.findProperty("llmedgeShowTestOutput")?.toString()?.equals("true", ignoreCase = true) == true)
                if (showTestOutput) {
                    it.testLogging {
                        showStandardStreams = true
                        events(
                            org.gradle.api.tasks.testing.logging.TestLogEvent.FAILED,
                            org.gradle.api.tasks.testing.logging.TestLogEvent.SKIPPED,
                            org.gradle.api.tasks.testing.logging.TestLogEvent.PASSED,
                            org.gradle.api.tasks.testing.logging.TestLogEvent.STANDARD_OUT,
                            org.gradle.api.tasks.testing.logging.TestLogEvent.STANDARD_ERROR,
                        )
                    }
                }

                // Check if we're running the E2E test with native library path specified
                val nativeLibPath = System.getenv("LLMEDGE_BUILD_NATIVE_LIB_PATH")
                val skipE2E = System.getenv("LLMEDGE_SKIP_E2E_IN_UNIT")?.equals("true", ignoreCase = true) == true
                
                if (skipE2E) {
                    println("LLMEDGE CI: Skipping E2E tests for task ${it.name}")
                    it.filter.excludeTestsMatching("*LinuxE2ETest")
                    it.filter.excludeTestsMatching("*IntegrationTest")
                    it.filter.excludeTestsMatching("*ImageGenerationTest")
                    it.filter.excludeTestsMatching("*VideoGenerationTest")
                    it.filter.excludeTestsMatching("*StableDiffusionTxt2VidTest")
                    it.filter.excludeTestsMatching("*StableDiffusionTxt2ImgTest")
                    it.filter.excludeTestsMatching("*VideoI2VAndLoraE2ETest")
                    it.filter.excludeTestsMatching("*VideoGenerationSequentialE2ETest")
                    it.filter.excludeTestsMatching("*VideoGenerationHQTest")
                    it.filter.excludeTestsMatching("*VideoGenerationExportTest")
                    it.filter.excludeTestsMatching("*SamplerSchedulerGifTest")
                    it.filter.excludeTestsMatching("*FullPhonePathCatTest")
                    it.filter.excludeTestsMatching("*BarkTTSTest")
                    it.filter.excludeTestsMatching("*WhisperTest")
                    it.filter.excludeTestsMatching("*ReproductionTest")
                    it.filter.excludeTestsMatching("*WanCatGifTask")
                    it.filter.excludeTestsMatching("*LLMEdgeManagerTest")
                    it.filter.excludeTestsMatching("*MemoryManagementTest")
                    it.filter.excludeTestsMatching("*MetadataInferenceTest")
                    it.filter.excludeTestsMatching("*StableDiffusionVideoModelDetectorExtraTest")
                    it.filter.excludeTestsMatching("*StableDiffusionVideoTest")
                    it.filter.excludeTestsMatching("*VideoGenerateParamsTest")
                    it.filter.excludeTestsMatching("*VideoModelDetectorTest")
                    it.filter.excludeTestsMatching("*BitmapConversionTest")
                    it.filter.excludeTestsMatching("*CombinedHeuristicTest")
                }

                val whisperLibPath = System.getenv("LLMEDGE_BUILD_WHISPER_LIB_PATH")
                val barkLibPath = System.getenv("LLMEDGE_BUILD_BARK_LIB_PATH")
                val hasNativeLib = !nativeLibPath.isNullOrBlank() || !whisperLibPath.isNullOrBlank() || !barkLibPath.isNullOrBlank()
                if (!hasNativeLib) {
                    // Disable native load for regular unit tests (no native library available)
                    it.systemProperty("llmedge.disableNativeLoad", "true")
                } else {
                    // For E2E tests with native library, enable native loading and set library path
                    it.systemProperty("llmedge.disableNativeLoad", "false")
                    // Build library path from available native libraries
                    val libraryPaths = mutableListOf<String>()
                    nativeLibPath?.let { libraryPaths.add(File(it).parent) }
                    whisperLibPath?.let { libraryPaths.add(File(it).parent) }
                    barkLibPath?.let { libraryPaths.add(File(it).parent) }
                    // Also include the jni-desktop build directory for dependencies
                    val jniDesktopDir = "${rootProject.projectDir}/scripts/jni-desktop/build/bin"
                    libraryPaths.add(jniDesktopDir)
                    val libraryPath = libraryPaths.joinToString(":")
                    it.jvmArgs("-Djava.library.path=$libraryPath")
                    // Increase memory for video generation tests (needs lots of RAM for model + video frames)
                    it.maxHeapSize = "12g"
                    // Pass environment variables as system properties
                    System.getenv("LLMEDGE_TEST_MODEL_PATH")?.let { path ->
                        it.systemProperty("LLMEDGE_TEST_MODEL_PATH", path)
                    }
                    System.getenv("LLMEDGE_TEST_MODEL_ID")?.let { v ->
                        it.systemProperty("LLMEDGE_TEST_MODEL_ID", v)
                    }
                    System.getenv("LLMEDGE_TEST_MODEL_FILENAME")?.let { v ->
                        it.systemProperty("LLMEDGE_TEST_MODEL_FILENAME", v)
                    }
                    System.getenv("LLMEDGE_TEST_T5_PATH")?.let { path ->
                        it.systemProperty("LLMEDGE_TEST_T5_PATH", path)
                    }
                    System.getenv("LLMEDGE_TEST_VAE_PATH")?.let { path ->
                        it.systemProperty("LLMEDGE_TEST_VAE_PATH", path)
                    }
                    System.getenv("LLMEDGE_TEST_TAESD_PATH")?.let { path ->
                        it.systemProperty("LLMEDGE_TEST_TAESD_PATH", path)
                    }

                    // Optional overrides for explicit WAN component downloads
                    System.getenv("LLMEDGE_TEST_VAE_FILENAME")?.let { v ->
                        it.systemProperty("LLMEDGE_TEST_VAE_FILENAME", v)
                    }
                    System.getenv("LLMEDGE_TEST_T5_MODEL_ID")?.let { v ->
                        it.systemProperty("LLMEDGE_TEST_T5_MODEL_ID", v)
                    }
                    System.getenv("LLMEDGE_TEST_T5_FILENAME")?.let { v ->
                        it.systemProperty("LLMEDGE_TEST_T5_FILENAME", v)
                    }
                    System.getenv("LLMEDGE_TEST_FORCE_DOWNLOAD")?.let { v ->
                        it.systemProperty("LLMEDGE_TEST_FORCE_DOWNLOAD", v)
                    }
                    System.getenv("LLMEDGE_TEST_VIDEO_FRAMES")?.let { v ->
                        it.systemProperty("LLMEDGE_TEST_VIDEO_FRAMES", v)
                    }
                    System.getenv("LLMEDGE_TEST_STEPS")?.let { v ->
                        it.systemProperty("LLMEDGE_TEST_STEPS", v)
                    }
                    System.getenv("LLMEDGE_TEST_WIDTH")?.let { v ->
                        it.systemProperty("LLMEDGE_TEST_WIDTH", v)
                    }
                    System.getenv("LLMEDGE_TEST_HEIGHT")?.let { v ->
                        it.systemProperty("LLMEDGE_TEST_HEIGHT", v)
                    }

                    // Whisper test environment variables
                    System.getenv("LLMEDGE_TEST_WHISPER_MODEL_PATH")?.let { path ->
                        it.systemProperty("LLMEDGE_TEST_WHISPER_MODEL_PATH", path)
                    }
                    System.getenv("LLMEDGE_BUILD_WHISPER_LIB_PATH")?.let { path ->
                        it.systemProperty("LLMEDGE_BUILD_WHISPER_LIB_PATH", path)
                    }
                    System.getenv("LLMEDGE_TEST_AUDIO_PATH")?.let { path ->
                        it.systemProperty("LLMEDGE_TEST_AUDIO_PATH", path)
                    }
                    // Bark TTS test environment variables
                    System.getenv("LLMEDGE_TEST_BARK_MODEL_PATH")?.let { path ->
                        it.systemProperty("LLMEDGE_TEST_BARK_MODEL_PATH", path)
                    }
                    System.getenv("LLMEDGE_BUILD_BARK_LIB_PATH")?.let { path ->
                        it.systemProperty("LLMEDGE_BUILD_BARK_LIB_PATH", path)
                    }
                }
            }
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
    packaging {
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

jacoco {
    toolVersion = "0.8.11"
}

tasks.withType<Test>().configureEach {
    extensions.configure(JacocoTaskExtension::class) {
        isIncludeNoLocationClasses = true
        excludes = listOf("jdk.internal.*")
    }
}

val jacocoTestReport by tasks.registering(JacocoReport::class) {
    dependsOn("testDebugUnitTest")

    reports {
        xml.required.set(true)
        html.required.set(true)

        val reportsDir = layout.buildDirectory.dir("reports/jacoco/jacocoTestReport")
        xml.outputLocation.set(reportsDir.map { it.file("jacocoTestReport.xml") })
        html.outputLocation.set(reportsDir.map { it.dir("html") })
    }

    val fileFilter = listOf(
        "**/R.class",
        "**/R$*.class",
        "**/BuildConfig.*",
        "**/Manifest*.*",
        "**/*Test*.*",
        "**/*$*Companion*",
        // Exclude non-core Android-facing packages from coverage to focus on core library logic
        // These areas involve platform services (ML Kit OCR, Camera, DownloadManager) and
        // external integrations that are not meaningful for unit test coverage targets.
        "**/io/aatricks/llmedge/vision/**",
        "**/io/aatricks/llmedge/vision/ocr/**",
        "**/io/aatricks/llmedge/rag/**",
        "**/io/aatricks/llmedge/huggingface/**",
    )

    val classTrees = listOf(
        layout.buildDirectory.dir("intermediates/javac/debug/classes"),
        layout.buildDirectory.dir("tmp/kotlin-classes/debug")
    ).map { dirProvider ->
        dirProvider.map { dir ->
            fileTree(dir) {
                // Only include core library package and exclude Android/platform-heavy subpackages
                include("io/aatricks/llmedge/**")
                exclude(fileFilter)
            }
        }
    }

    classDirectories.setFrom(classTrees)
    sourceDirectories.setFrom(files("src/main/java", "src/main/kotlin"))
    val coverageData = layout.buildDirectory.asFileTree.matching {
        include("jacoco/testDebugUnitTest.exec")
        include("outputs/unit_test_code_coverage/**/testDebugUnitTest.exec")
        include("outputs/unit_test_code_coverage/**/*.ec")
    }
    executionData.setFrom(coverageData)
}
