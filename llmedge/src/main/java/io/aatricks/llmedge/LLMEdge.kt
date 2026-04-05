package io.aatricks.llmedge

import android.content.Context
import io.aatricks.llmedge.core.ClientBootstrap
import io.aatricks.llmedge.core.ClientBootstrapContext
import io.aatricks.llmedge.core.FeatureContext
import io.aatricks.llmedge.core.LLMEdgeScope
import io.aatricks.llmedge.core.runtime.RuntimeCapabilities
import io.aatricks.llmedge.image.ImageClient
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
    private val ownedBootstrap: ClientBootstrapContext? = null,
) : AutoCloseable {
    private val featureContext =
        FeatureContext(
            appContext = appContext,
            edgeScope = edgeScope,
            config = config,
            modelRepository = modelRepository,
        )
    private val modelsDelegate = lazy(LazyThreadSafetyMode.SYNCHRONIZED) {
        BoundModelRepository(appContext, modelRepository)
    }
    private val textDelegate = lazy(LazyThreadSafetyMode.SYNCHRONIZED) {
        TextClient(featureContext)
    }
    private val speechDelegate = lazy(LazyThreadSafetyMode.SYNCHRONIZED) {
        SpeechClient(featureContext)
    }
    private val imageDelegate = lazy(LazyThreadSafetyMode.SYNCHRONIZED) {
        ImageClient(featureContext)
    }
    private val visionDelegate = lazy(LazyThreadSafetyMode.SYNCHRONIZED) {
        VisionClient(
            featureContext = featureContext,
            pipeline = VisionPipeline(featureContext),
        )
    }
    private val ragDelegate = lazy(LazyThreadSafetyMode.SYNCHRONIZED) {
        RAGClient(featureContext)
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
        if (ownedBootstrap != null) {
            closeSafely(ownedBootstrap)
        } else {
            closeSafely(edgeScope)
        }

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
        ): LLMEdge =
            ClientBootstrap.createOwned(context, scope, config.execution.inferenceThreads) { bootstrap ->
                LLMEdge(
                    appContext = bootstrap.appContext,
                    edgeScope = bootstrap.edgeScope,
                    config = config,
                    modelRepository = modelRepository,
                    ownedBootstrap = bootstrap,
                )
            }

        /**
         * Returns backend availability for the text/LLM stack.
         *
         * This probes the SmolLM runtime, not Stable Diffusion.
         */
        @JvmStatic
        fun getTextBackendAvailability(): ComputeBackendAvailability =
            RuntimeCapabilities.textBackendAvailability()

        /**
         * Returns backend availability for the speech-to-text stack.
         *
         * Bark remains CPU-only and is therefore excluded from this GPU probe.
         */
        @JvmStatic
        fun getSpeechBackendAvailability(): ComputeBackendAvailability =
            RuntimeCapabilities.speechBackendAvailability()

        /**
         * Returns backend availability for the image/video diffusion stack.
         */
        @JvmStatic
        fun getImageBackendAvailability(): ComputeBackendAvailability =
            RuntimeCapabilities.imageBackendAvailability()

        /**
         * Returns backend availability for the vision-language stack.
         *
         * Vision rides on the SmolLM runtime and shares its backend capabilities.
         */
        @JvmStatic
        fun getVisionBackendAvailability(): ComputeBackendAvailability =
            RuntimeCapabilities.visionBackendAvailability()

        @Deprecated(
            message = "Ambiguous subsystem name. Prefer getImageBackendAvailability().vulkanAvailable or another per-subsystem capability API.",
            replaceWith = ReplaceWith("getImageBackendAvailability().vulkanAvailable"),
        )
        @JvmStatic
        fun isVulkanAvailable(): Boolean = getImageBackendAvailability().vulkanAvailable

        @Deprecated(
            message = "Ambiguous subsystem name. Prefer getImageBackendAvailability().openClAvailable or another per-subsystem capability API.",
            replaceWith = ReplaceWith("getImageBackendAvailability().openClAvailable"),
        )
        @JvmStatic
        fun isOpenClAvailable(): Boolean = getImageBackendAvailability().openClAvailable

        @Deprecated(
            message = "Ambiguous subsystem name. Prefer getImageBackendAvailability().vulkanDeviceInfo.",
            replaceWith = ReplaceWith("getImageBackendAvailability().vulkanDeviceInfo"),
        )
        @JvmStatic
        fun getVulkanDeviceInfo(): VulkanDeviceInfo? = getImageBackendAvailability().vulkanDeviceInfo
    }
}
