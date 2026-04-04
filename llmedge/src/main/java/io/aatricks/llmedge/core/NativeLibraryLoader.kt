package io.aatricks.llmedge.core

import io.aatricks.llmedge.runtime.RuntimeEnvironmentHolder
import java.io.File

internal object NativeLibraryLoader {
    private const val BUILD_NATIVE_LIB_PATH = "LLMEDGE_BUILD_NATIVE_LIB_PATH"
    private const val BUILD_WHISPER_LIB_PATH = "LLMEDGE_BUILD_WHISPER_LIB_PATH"
    private const val BUILD_BARK_LIB_PATH = "LLMEDGE_BUILD_BARK_LIB_PATH"

    fun isLoadDisabled(): Boolean = java.lang.Boolean.getBoolean("llmedge.disableNativeLoad")

    fun ensureSmolLMLoaded(
        required: Boolean = true,
        onDebug: (String) -> Unit = {},
        onError: (String, Throwable?) -> Unit = { _, _ -> },
    ): String? = loadCandidates(
        component = "SmolLM",
        candidates = NativeLibraryCatalog.smolLmCandidates(),
        exactPathCandidates = listOf(BUILD_NATIVE_LIB_PATH),
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
        candidates = listOf(NativeLibraryCatalog.STABLE_DIFFUSION),
        exactPathCandidates = listOf(BUILD_NATIVE_LIB_PATH),
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
        candidates = NativeLibraryCatalog.whisperCandidates(),
        exactPathCandidates = listOf(BUILD_WHISPER_LIB_PATH, BUILD_NATIVE_LIB_PATH),
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
        candidates = listOf(NativeLibraryCatalog.BARK),
        exactPathCandidates = listOf(BUILD_BARK_LIB_PATH, BUILD_NATIVE_LIB_PATH),
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
        candidates = listOf(NativeLibraryCatalog.GGUF_READER),
        required = required,
        onDebug = onDebug,
        onError = onError,
    )

    @Synchronized
    private fun loadCandidates(
        component: String,
        candidates: List<String>,
        exactPathCandidates: List<String> = emptyList(),
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
        for (propertyName in exactPathCandidates.distinct()) {
            val exactPath = resolveExactLibraryPath(propertyName) ?: continue
            try {
                loadLibraryFileOnce(exactPath)
                if (verifyBindings != null && !verifyBindings()) {
                    throw NativeBindingException(
                        File(exactPath).nameWithoutExtension,
                        "The JNI entry points for $component are present but binding verification failed.",
                    )
                }
                onDebug("Loaded native library '$exactPath' for $component via exact path")
                return exactPath
            } catch (t: Throwable) {
                lastError = t
                onError("Unable to load native library '$exactPath' for $component", t)
            }
        }
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

    private fun resolveExactLibraryPath(propertyName: String): String? {
        val path =
            System.getProperty(propertyName)
                ?: System.getenv(propertyName)
                ?: return null
        val file = File(path)
        return file.takeIf(File::exists)?.absolutePath
    }

    @Synchronized
    private fun loadLibraryOnce(name: String) {
        RuntimeEnvironmentHolder.current().nativeLibraryRegistry.loadOnce(name, System::loadLibrary)
    }

    @Synchronized
    private fun loadLibraryFileOnce(path: String) {
        RuntimeEnvironmentHolder.current().nativeLibraryRegistry.loadOnce("file:$path", System::load)
    }

}
