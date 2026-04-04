package io.aatricks.llmedge

import android.content.Context
import io.aatricks.llmedge.core.LLMEdgeScope
import io.aatricks.llmedge.image.ImageClient
import io.aatricks.llmedge.image.diffusion.StableDiffusion
import io.aatricks.llmedge.model.BoundModelRepository
import io.aatricks.llmedge.model.DefaultModelRepository
import io.aatricks.llmedge.model.ModelRepository
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
    private val modelRepository: ModelRepository,
) : AutoCloseable {
    private val modelsDelegate = lazy(LazyThreadSafetyMode.SYNCHRONIZED) {
        BoundModelRepository(appContext, modelRepository)
    }
    private val textDelegate = lazy(LazyThreadSafetyMode.SYNCHRONIZED) {
        TextClient(appContext, edgeScope, config, modelRepository)
    }
    private val speechDelegate = lazy(LazyThreadSafetyMode.SYNCHRONIZED) {
        SpeechClient(appContext, edgeScope, config, modelRepository)
    }
    private val imageDelegate = lazy(LazyThreadSafetyMode.SYNCHRONIZED) {
        ImageClient(appContext, edgeScope, config, modelRepository)
    }
    private val visionDelegate = lazy(LazyThreadSafetyMode.SYNCHRONIZED) {
        VisionClient(
            context = appContext,
            pipeline = VisionPipeline(appContext, edgeScope, modelRepository, config),
            config = config,
        )
    }
    private val ragDelegate = lazy(LazyThreadSafetyMode.SYNCHRONIZED) {
        RAGClient(appContext, edgeScope, config, modelRepository)
    }

    val models: BoundModelRepository
        get() = modelsDelegate.value
    val text: TextClient
        get() = textDelegate.value
    val speech: SpeechClient
        get() = speechDelegate.value
    val image: ImageClient
        get() = imageDelegate.value
    val vision: VisionClient
        get() = visionDelegate.value
    val rag: RAGClient
        get() = ragDelegate.value

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

        if (ragDelegate.isInitialized()) closeSafely(ragDelegate.value)
        if (visionDelegate.isInitialized()) closeSafely(visionDelegate.value)
        if (imageDelegate.isInitialized()) closeSafely(imageDelegate.value)
        if (speechDelegate.isInitialized()) closeSafely(speechDelegate.value)
        if (textDelegate.isInitialized()) closeSafely(textDelegate.value)
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
            modelRepository: ModelRepository = DefaultModelRepository(),
        ): LLMEdge {
            val appContext = context.applicationContext
            val edgeScope = LLMEdgeScope(scope, config.text.promptThreads)
            return LLMEdge(appContext, edgeScope, config, modelRepository)
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
        fun isOpenClAvailable(): Boolean = StableDiffusion.isOpenClAvailable()

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
