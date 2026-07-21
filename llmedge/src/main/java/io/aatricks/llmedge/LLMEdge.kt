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
import kotlinx.coroutines.launch

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
                // Warm the crash-safe GPU probe so text/vision Vulkan gating (which requires a
                // worker-probe verdict) resolves without the host app calling it explicitly.
                // Skipped under Robolectric, where the worker service can never bind.
                if (android.os.Build.FINGERPRINT != "robolectric") {
                    scope.launch {
                        runCatching {
                            io.aatricks.llmedge.image.ipc.WorkerBackendProber.probe(bootstrap.appContext)
                        }
                    }
                }
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
         * Note: may load the GPU driver; prefer the context overloads.
         */
        @JvmStatic
        fun getTextBackendAvailability(): ComputeBackendAvailability =
            RuntimeCapabilities.textBackendAvailability()
            
        /**
         * Crash-safe variant: Vulkan capability is derived from the isolated-worker probe
         * ([probeImageBackendAvailability]) instead of loading the SmolLM runtime in-process.
         * On drivers below Vulkan 1.2 (e.g. Adreno 619) the in-process check aborts the host.
         */
        @JvmStatic
        fun getTextBackendAvailability(context: Context): ComputeBackendAvailability =
            RuntimeCapabilities.probeDerivedAvailability(context)

        /**
         * Returns backend availability for the speech-to-text stack.
         *
         * Bark remains CPU-only and is therefore excluded from this GPU probe.
         * Note: may load the GPU driver; prefer the context overloads.
         */
        @JvmStatic
        fun getSpeechBackendAvailability(): ComputeBackendAvailability =
            RuntimeCapabilities.speechBackendAvailability()
            
        @JvmStatic
        fun getSpeechBackendAvailability(context: Context): ComputeBackendAvailability {
            io.aatricks.llmedge.image.ipc.WorkerBackendProber.persistedOrNull(context.applicationContext)
            return RuntimeCapabilities.speechBackendAvailability()
        }

        /**
         * Probes backend availability in a safe, isolated worker process. Call this once to populate 
         * the cache before calling getImageBackendAvailability().
         */
        suspend fun probeImageBackendAvailability(context: Context): ComputeBackendAvailability =
            io.aatricks.llmedge.image.ipc.WorkerBackendProber.probe(context.applicationContext)

        /**
         * Returns backend availability for the image/video diffusion stack.
         * 
         * Call probeImageBackendAvailability(context) once to populate; unknown reports unavailable.
         */
        @JvmStatic
        fun getImageBackendAvailability(): ComputeBackendAvailability =
            RuntimeCapabilities.imageBackendAvailability()
            
        @JvmStatic
        fun getImageBackendAvailability(context: Context): ComputeBackendAvailability =
            io.aatricks.llmedge.image.ipc.WorkerBackendProber.cachedOrNull() ?:
            io.aatricks.llmedge.image.ipc.WorkerBackendProber.persistedOrNull(context.applicationContext) ?:
            ComputeBackendAvailability(false, false, null)

        /**
         * Returns backend availability for the vision-language stack.
         *
         * Vision rides on the SmolLM runtime and shares its backend capabilities.
         * Note: may load the GPU driver; prefer the context overloads.
         */
        @JvmStatic
        fun getVisionBackendAvailability(): ComputeBackendAvailability =
            RuntimeCapabilities.visionBackendAvailability()
            
        /**
         * Crash-safe variant: see [getTextBackendAvailability] — vision rides on SmolLM and
         * shares the same in-process abort hazard.
         */
        @JvmStatic
        fun getVisionBackendAvailability(context: Context): ComputeBackendAvailability =
            RuntimeCapabilities.probeDerivedAvailability(context)

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
