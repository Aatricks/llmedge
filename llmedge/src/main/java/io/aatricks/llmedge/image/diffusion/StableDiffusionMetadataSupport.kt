package io.aatricks.llmedge.image.diffusion

import io.aatricks.llmedge.core.AndroidLogAdapter
import io.aatricks.llmedge.model.GgufModelMetadataSupport
import java.io.File
import java.util.Locale

internal object StableDiffusionMetadataSupport {
    private const val LOG_TAG = "StableDiffusion"
    private val metadataCache = mutableMapOf<String, VideoModelMetadata>()
    private val lock = Any()
    private val videoKeywords =
        setOf(
            "wan",
            "hunyuan",
            "video",
            "t2v",
            "i2v",
            "ti2v",
        )
    private val easyCacheKeywords =
        setOf(
            "flux",
            "sd3",
            "wan",
            "qwen-image",
            "qwen_image",
            "qwen image",
            "z-image",
            "z_image",
        )

    fun inferVideoModelMetadata(
        resolvedModelPath: String,
        modelId: String?,
        explicitFilename: String?,
    ): VideoModelMetadata {
        synchronized(lock) {
            metadataCache[resolvedModelPath]?.let { return it }
        }

        AndroidLogAdapter.d(
            LOG_TAG,
            "inferVideoModelMetadata path=$resolvedModelPath exists=${File(resolvedModelPath).exists()}",
        )

        val filename = explicitFilename ?: resolvedModelPath.substringAfterLast('/')
        val lowerName = filename.lowercase(Locale.US)
        val ggufMetadata = GgufModelMetadataSupport.inspect(resolvedModelPath)
        val tags = mutableSetOf<String>()

        AndroidLogAdapter.d(
            LOG_TAG,
            "Inferring video model metadata for $resolvedModelPath using hints/metadata before filename fallback",
        )

        val architecture =
            when {
                !ggufMetadata?.architecture.isNullOrBlank() -> ggufMetadata?.architecture
                !modelId.isNullOrBlank() -> modelId
                lowerName.contains("hunyuan") -> "hunyuan_video"
                lowerName.contains("wan") -> "wan"
                else -> null
            }

        val modelType =
            when {
                containsIgnoreCase(ggufMetadata?.modelName, "ti2v") || lowerName.contains("ti2v") -> "ti2v"
                containsIgnoreCase(ggufMetadata?.modelName, "i2v") || lowerName.contains("i2v") -> "i2v"
                containsIgnoreCase(ggufMetadata?.modelName, "t2v") || lowerName.contains("t2v") -> "t2v"
                else -> null
            }

        val parameterCount =
            when {
                !ggufMetadata?.parameterCount.isNullOrBlank() -> ggufMetadata?.parameterCount
                lowerName.contains("1.3b") || lowerName.contains("1_3b") -> "1.3B"
                lowerName.contains("5b") || lowerName.contains("5_b") -> "5B"
                lowerName.contains("14b") || lowerName.contains("14_b") -> "14B"
                else -> null
            }

        val mobileSupported =
            when (parameterCount) {
                "1.3B", "5B" -> true
                "14B" -> false
                else -> true
            }

        if (containsIgnoreCase(ggufMetadata?.architecture, "wan") || lowerName.contains("wan")) tags += "wan"
        if (
            containsIgnoreCase(ggufMetadata?.modelName, "video") ||
            lowerName.contains("video") ||
            modelType in listOf("t2v", "i2v", "ti2v")
        ) {
            tags += "text-to-video"
        }
        if (containsIgnoreCase(ggufMetadata?.architecture, "hunyuan") || lowerName.contains("hunyuan")) {
            tags += "hunyuan"
        }

        return VideoModelMetadata(
            architecture = architecture,
            modelType = modelType,
            parameterCount = parameterCount,
            mobileSupported = mobileSupported,
            tags = tags,
            filename = filename,
        ).also { metadata ->
            synchronized(lock) {
                metadataCache[resolvedModelPath] = metadata
            }
        }
    }

    fun isVideoModel(metadata: VideoModelMetadata): Boolean {
        val architecture = metadata.architecture.orEmpty().lowercase(Locale.US)
        if (containsKeyword(architecture)) return true

        val modelType = metadata.modelType.orEmpty().lowercase(Locale.US)
        if (containsKeyword(modelType)) return true

        val filename = metadata.filename.orEmpty().lowercase(Locale.US)
        if (containsKeyword(filename)) return true

        return metadata.tags.any { containsKeyword(it.lowercase(Locale.US)) }
    }

    private fun containsKeyword(value: String): Boolean {
        if (value.isEmpty()) return false
        return videoKeywords.any { keyword -> value.contains(keyword) }
    }

    private fun containsIgnoreCase(
        value: String?,
        needle: String,
    ): Boolean = value?.contains(needle, ignoreCase = true) == true

    fun supportsEasyCache(metadata: VideoModelMetadata): Boolean {
        val candidates =
            listOf(
                metadata.architecture,
                metadata.modelType,
                metadata.filename,
            ) + metadata.tags

        return candidates
            .filterNotNull()
            .map { it.lowercase(Locale.US) }
            .any { candidate -> easyCacheKeywords.any(candidate::contains) }
    }
}
