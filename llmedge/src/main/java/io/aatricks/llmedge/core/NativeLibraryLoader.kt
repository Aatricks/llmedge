package io.aatricks.llmedge.core

import io.aatricks.llmedge.runtime.RuntimeEnvironmentHolder

internal object NativeLibraryLoader {
    fun isLoadDisabled(): Boolean = java.lang.Boolean.getBoolean("llmedge.disableNativeLoad")

    fun ensureSmolLMLoaded(
        required: Boolean = true,
        onDebug: (String) -> Unit = {},
        onError: (String, Throwable?) -> Unit = { _, _ -> },
    ): String? = loadCandidates(
        component = "SmolLM",
        candidates = NativeLibraryCatalog.smolLmCandidates(),
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
        RuntimeEnvironmentHolder.current().nativeLibraryRegistry.loadOnce(name, System::loadLibrary)
    }

}
