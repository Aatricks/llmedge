package io.aatricks.llmedge.core

import android.os.Build
import java.io.File
import java.io.FileNotFoundException

internal object NativeLibraryLoader {
    private val loadedLibraries = mutableSetOf<String>()

    fun isLoadDisabled(): Boolean = java.lang.Boolean.getBoolean("llmedge.disableNativeLoad")

    fun ensureSmolLMLoaded(
        required: Boolean = true,
        onDebug: (String) -> Unit = {},
        onError: (String, Throwable?) -> Unit = { _, _ -> },
    ): String? = loadCandidates(
        component = "SmolLM",
        candidates = smolLmCandidates(),
        required = required,
        onDebug = onDebug,
        onError = onError,
    )

    fun ensureStableDiffusionLoaded(
        required: Boolean = true,
        onDebug: (String) -> Unit = {},
        onError: (String, Throwable?) -> Unit = { _, _ -> },
        verifyBindings: (() -> Boolean)? = null,
    ): String? = loadCandidates(
        component = "StableDiffusion",
        candidates = listOf("sdcpp"),
        required = required,
        onDebug = onDebug,
        onError = onError,
        verifyBindings = verifyBindings,
    )

    fun ensureWhisperLoaded(
        required: Boolean = false,
        onDebug: (String) -> Unit = {},
        onError: (String, Throwable?) -> Unit = { _, _ -> },
    ): String? = loadCandidates(
        component = "Whisper",
        candidates = whisperCandidates(),
        required = required,
        onDebug = onDebug,
        onError = onError,
    )

    fun ensureBarkLoaded(
        required: Boolean = false,
        onDebug: (String) -> Unit = {},
        onError: (String, Throwable?) -> Unit = { _, _ -> },
    ): String? = loadCandidates(
        component = "BarkTTS",
        candidates = listOf("bark_jni"),
        required = required,
        onDebug = onDebug,
        onError = onError,
    )

    fun ensureGgufReaderLoaded(
        required: Boolean = false,
        onDebug: (String) -> Unit = {},
        onError: (String, Throwable?) -> Unit = { _, _ -> },
    ): String? = loadCandidates(
        component = "GGUFReader",
        candidates = listOf("ggufreader"),
        required = required,
        onDebug = onDebug,
        onError = onError,
    )

    @Synchronized
    private fun loadCandidates(
        component: String,
        candidates: List<String>,
        required: Boolean,
        onDebug: (String) -> Unit,
        onError: (String, Throwable?) -> Unit,
        verifyBindings: (() -> Boolean)? = null,
    ): String? {
        if (isLoadDisabled()) {
            onDebug("[$component] Native library load disabled via llmedge.disableNativeLoad=true")
            return null
        }
        var lastError: Throwable? = null
        for (candidate in candidates.distinct()) {
            try {
                loadLibraryOnce(candidate)
                if (verifyBindings != null && !verifyBindings()) {
                    throw NativeBindingException(
                        candidate,
                        "The JNI entry points for $component are present but binding verification failed.",
                    )
                }
                onDebug("Loaded native library '$candidate' for $component")
                return candidate
            } catch (t: Throwable) {
                lastError = t
                onError("Unable to load native library '$candidate' for $component", t)
            }
        }
        if (required) {
            val attempted = candidates.distinct().joinToString(", ")
            throw NativeBindingException(
                candidates.firstOrNull() ?: component.lowercase(),
                "Attempted variants: $attempted",
                lastError,
            )
        }
        return null
    }

    @Synchronized
    private fun loadLibraryOnce(name: String) {
        if (loadedLibraries.contains(name)) {
            return
        }
        try {
            System.loadLibrary(name)
            loadedLibraries += name
        } catch (e: UnsatisfiedLinkError) {
            loadedLibraries -= name
            throw e
        }
    }

    /** Cached CPU features string — read /proc/cpuinfo only once. */
    private val cachedCpuFeatures: String by lazy { readCpuFeaturesFromProc() }

    private fun smolLmCandidates(): List<String> {
        val candidates = mutableListOf<String>()
        val cpuFeatures = cachedCpuFeatures
        val hardware = Build.HARDWARE.orEmpty()
        val supportedAbis = Build.SUPPORTED_ABIS ?: emptyArray()
        val supported32BitAbis = Build.SUPPORTED_32_BIT_ABIS ?: emptyArray()
        val hasFp16 = cpuFeatures.contains("fp16") || cpuFeatures.contains("fphp")
        val hasDotProd = cpuFeatures.contains("dotprod") || cpuFeatures.contains("asimddp")
        val hasSve = cpuFeatures.contains("sve")
        val hasI8mm = cpuFeatures.contains("i8mm")
        val isAtLeastArmV82 =
            cpuFeatures.contains("asimd") && cpuFeatures.contains("crc32") && cpuFeatures.contains("aes")
        val isAtLeastArmV84 = cpuFeatures.contains("dcpop") && cpuFeatures.contains("uscat")
        val isEmulated =
            hardware.contains("goldfish") || hardware.contains("ranchu")

        if (!isEmulated) {
            if (supportedAbis.firstOrNull() == "arm64-v8a") {
                if (isAtLeastArmV84 && hasSve && hasI8mm && hasFp16 && hasDotProd) {
                    candidates += "smollm_v8_4_fp16_dotprod_i8mm_sve"
                }
                if (isAtLeastArmV84 && hasSve && hasFp16 && hasDotProd) {
                    candidates += "smollm_v8_4_fp16_dotprod_sve"
                }
                if (isAtLeastArmV84 && hasI8mm && hasFp16 && hasDotProd) {
                    candidates += "smollm_v8_4_fp16_dotprod_i8mm"
                }
                if (isAtLeastArmV84 && hasFp16 && hasDotProd) {
                    candidates += "smollm_v8_4_fp16_dotprod"
                }
                if (isAtLeastArmV82 && hasFp16 && hasDotProd) {
                    candidates += "smollm_v8_2_fp16_dotprod"
                }
                if (isAtLeastArmV82 && hasFp16) {
                    candidates += "smollm_v8_2_fp16"
                }
                candidates += "smollm_v8"
            } else if (supported32BitAbis.firstOrNull() == "armeabi-v7a") {
                candidates += "smollm_v7a"
            }
        }
        candidates += "smollm"
        return candidates
    }

    private fun whisperCandidates(): List<String> {
        val osName = System.getProperty("os.name")?.lowercase().orEmpty()
        val isDesktopJvm = osName.contains("linux") && !osName.contains("android")
        if (isDesktopJvm) {
            return listOf("whisper_jni", "whisper")
        }
        val isEmulated =
            Build.HARDWARE.contains("goldfish") || Build.HARDWARE.contains("ranchu")
        return if (!isEmulated && Build.SUPPORTED_64_BIT_ABIS.any { it == "arm64-v8a" }) {
            listOf("whisper_arm64", "whisper")
        } else {
            listOf("whisper")
        }
    }

    private fun readCpuFeaturesFromProc(): String {
        val cpuInfo =
            try {
                File("/proc/cpuinfo").readText()
            } catch (_: FileNotFoundException) {
                ""
            }
        return cpuInfo.substringAfter("Features").substringAfter(":").substringBefore("\n").trim()
    }
}
