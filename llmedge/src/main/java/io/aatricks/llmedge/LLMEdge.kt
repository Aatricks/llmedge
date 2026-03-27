package io.aatricks.llmedge

import android.content.Context
import io.aatricks.llmedge.core.LLMEdgeScope
import io.aatricks.llmedge.image.ImageClient
import io.aatricks.llmedge.model.DefaultModelResolver
import io.aatricks.llmedge.model.HuggingFaceModelStore
import io.aatricks.llmedge.model.ModelManager
import io.aatricks.llmedge.model.ModelResolver
import io.aatricks.llmedge.rag.RAGClient
import io.aatricks.llmedge.speech.SpeechClient
import io.aatricks.llmedge.text.TextClient
import io.aatricks.llmedge.vision.VisionClient
import io.aatricks.llmedge.vision.VisionPipeline
import kotlinx.coroutines.CoroutineScope

data class VulkanDeviceInfo(
    val deviceCount: Int,
    val totalMemoryMB: Long,
    val freeMemoryMB: Long,
    val deviceIndex: Int = 0,
)

/**
 * High-level facade that groups llmedge's text, speech, image, vision, and RAG clients behind a
 * single lifecycle-aware object.
 *
 * Close this instance when the owning feature or screen is done with inference to release cached
 * native resources deterministically.
 */
class LLMEdge private constructor(
    private val appContext: Context,
    private val edgeScope: LLMEdgeScope,
    val config: LLMEdgeConfig,
    private val resolver: ModelResolver,
) : AutoCloseable {
    val models: ModelManager = ModelManager(appContext, resolver)
    val text: TextClient = TextClient(appContext, edgeScope, config, resolver)
    val speech: SpeechClient = SpeechClient(appContext, edgeScope, config, resolver)
    val image: ImageClient = ImageClient(appContext, edgeScope, config, resolver)
    val vision: VisionClient = VisionClient(appContext, VisionPipeline(appContext, resolver, config), config)
    val rag: RAGClient = RAGClient(appContext, edgeScope, config, resolver)

    override fun close() {
        var failure: Throwable? = null

        fun closeSafely(closeable: AutoCloseable) {
            try {
                closeable.close()
            } catch (t: Throwable) {
                if (failure == null) {
                    failure = t
                } else {
                    failure?.addSuppressed(t)
                }
            }
        }

        closeSafely(rag)
        closeSafely(vision)
        closeSafely(image)
        closeSafely(speech)
        closeSafely(text)
        closeSafely(edgeScope)

        failure?.let { throw IllegalStateException("Failed to close LLMEdge cleanly", it) }
    }

    companion object {
        /**
         * Create a new [LLMEdge] facade.
         *
         * @throws IllegalStateException if one of the managed clients fails during construction or
         * close.
         */
        @JvmStatic
        @JvmOverloads
        fun create(
            context: Context,
            scope: CoroutineScope,
            config: LLMEdgeConfig = LLMEdgeConfig(),
            resolver: ModelResolver = DefaultModelResolver(HuggingFaceModelStore()),
        ): LLMEdge {
            val appContext = context.applicationContext
            val edgeScope = LLMEdgeScope(scope, config.defaultTextThreads.coerceAtLeast(1))
            return LLMEdge(appContext, edgeScope, config, resolver)
        }

        @JvmStatic
        fun isVulkanAvailable(): Boolean {
            val deviceCount = StableDiffusion.getVulkanDeviceCount()
            if (deviceCount <= 0) {
                return false
            }
            val memory = StableDiffusion.getVulkanDeviceMemory(0) ?: return false
            return memory.size >= 2
        }

        @JvmStatic
        fun getVulkanDeviceInfo(): VulkanDeviceInfo? {
            val deviceCount = StableDiffusion.getVulkanDeviceCount()
            if (deviceCount <= 0) {
                return null
            }
            val memory = StableDiffusion.getVulkanDeviceMemory(0) ?: return null
            if (memory.size < 2) {
                return null
            }
            return VulkanDeviceInfo(
                deviceCount = deviceCount,
                freeMemoryMB = memory[0] / (1024 * 1024),
                totalMemoryMB = memory[1] / (1024 * 1024),
                deviceIndex = 0,
            )
        }
    }
}
