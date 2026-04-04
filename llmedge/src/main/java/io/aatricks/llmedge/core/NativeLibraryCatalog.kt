package io.aatricks.llmedge.core

import android.os.Build
import java.io.File
import java.io.FileNotFoundException

internal object NativeLibraryCatalog {
    val SMOLLM: String = NativeTargetNames.SMOLLM
    val SMOLLM_V7A: String = NativeTargetNames.SMOLLM_V7A
    val SMOLLM_V8: String = NativeTargetNames.SMOLLM_V8
    val SMOLLM_V8_2_FP16: String = NativeTargetNames.SMOLLM_V8_2_FP16
    val SMOLLM_V8_2_FP16_DOTPROD: String = NativeTargetNames.SMOLLM_V8_2_FP16_DOTPROD
    val SMOLLM_V8_4_FP16_DOTPROD: String = NativeTargetNames.SMOLLM_V8_4_FP16_DOTPROD
    val SMOLLM_V8_4_FP16_DOTPROD_SVE: String = NativeTargetNames.SMOLLM_V8_4_FP16_DOTPROD_SVE
    val SMOLLM_V8_4_FP16_DOTPROD_I8MM: String = NativeTargetNames.SMOLLM_V8_4_FP16_DOTPROD_I8MM
    val SMOLLM_V8_4_FP16_DOTPROD_I8MM_SVE: String = NativeTargetNames.SMOLLM_V8_4_FP16_DOTPROD_I8MM_SVE
    val STABLE_DIFFUSION: String = NativeTargetNames.SDCPP
    const val WHISPER = "whisper"
    const val WHISPER_JNI = "whisper_jni"
    const val WHISPER_ARM64 = "whisper_arm64"
    val BARK: String = NativeTargetNames.BARK_JNI
    val GGUF_READER: String = NativeTargetNames.GGUF_READER

    /** Cached CPU features string — read /proc/cpuinfo only once. */
    private val cachedCpuFeatures: String by lazy { readCpuFeaturesFromProc() }

    fun smolLmCandidates(): List<String> {
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
                    candidates += SMOLLM_V8_4_FP16_DOTPROD_I8MM_SVE
                }
                if (isAtLeastArmV84 && hasSve && hasFp16 && hasDotProd) {
                    candidates += SMOLLM_V8_4_FP16_DOTPROD_SVE
                }
                if (isAtLeastArmV84 && hasI8mm && hasFp16 && hasDotProd) {
                    candidates += SMOLLM_V8_4_FP16_DOTPROD_I8MM
                }
                if (isAtLeastArmV84 && hasFp16 && hasDotProd) {
                    candidates += SMOLLM_V8_4_FP16_DOTPROD
                }
                if (isAtLeastArmV82 && hasFp16 && hasDotProd) {
                    candidates += SMOLLM_V8_2_FP16_DOTPROD
                }
                if (isAtLeastArmV82 && hasFp16) {
                    candidates += SMOLLM_V8_2_FP16
                }
                candidates += SMOLLM_V8
            } else if (supported32BitAbis.firstOrNull() == "armeabi-v7a") {
                candidates += SMOLLM_V7A
            }
        }
        candidates += SMOLLM
        return candidates
    }

    fun whisperCandidates(): List<String> {
        val osName = System.getProperty("os.name")?.lowercase().orEmpty()
        val isDesktopJvm = osName.contains("linux") && !osName.contains("android")
        if (isDesktopJvm) {
            return listOf(WHISPER_JNI, WHISPER)
        }
        val isEmulated =
            Build.HARDWARE.contains("goldfish") || Build.HARDWARE.contains("ranchu")
        return if (!isEmulated && Build.SUPPORTED_64_BIT_ABIS.any { it == "arm64-v8a" }) {
            listOf(WHISPER_JNI, WHISPER_ARM64, WHISPER)
        } else {
            listOf(WHISPER_JNI, WHISPER)
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
