package io.aatricks.llmedge.image

import android.app.ActivityManager
import android.content.Context
import io.aatricks.llmedge.LLMEdgeConfig
import io.aatricks.llmedge.core.AndroidLogAdapter
import io.aatricks.llmedge.core.InsufficientMemoryException
import io.aatricks.llmedge.model.ModelRepository
import io.aatricks.llmedge.model.ModelSpec
import kotlin.math.max

/**
 * The conditioning layout required to split an image request into independent native runtimes.
 * This describes tensor semantics, not a named model or device-specific memory threshold.
 */
internal enum class ImageConditioningProfile {
    NONE,
    LLM,
    SD3_CLIP_T5,
    MASKED_T5,
    CHROMA_T5,
}

internal enum class ImageExecutionComponentKind {
    DIFFUSION_MODEL,
    VAE,
    TEXT_ENCODER,
    T5XXL,
    CLIP_L,
    CLIP_G,
}

internal enum class ImageExecutionMode {
    DIRECT,
    SEQUENTIAL,
}

internal data class ImageExecutionRecipe(
    val profile: ImageConditioningProfile,
    val phases: List<List<ImageExecutionComponentKind>>,
) {
    val supportsSequential: Boolean
        get() = profile != ImageConditioningProfile.NONE

    val components: Set<ImageExecutionComponentKind>
        get() = phases.flatten().toSet()

    companion object {
        fun none(): ImageExecutionRecipe =
            ImageExecutionRecipe(
                profile = ImageConditioningProfile.NONE,
                phases = emptyList(),
            )
    }
}

internal data class ImageMemorySnapshot(
    val availableSystemBytes: Long,
    val lowMemoryThresholdBytes: Long,
    val totalSystemBytes: Long,
)

internal data class ImageExecutionDecision(
    val mode: ImageExecutionMode,
    val reason: String,
    val recipe: ImageExecutionRecipe,
)

class UnsupportedStagedExecutionException(message: String) : IllegalStateException(message)

/** Pure decision rules for image direct-versus-sequential execution. */
internal object ImageExecutionPlanner {
    private const val MEBIBYTE = 1024L * 1024L
    private const val MIN_DEVICE_RESERVE_BYTES = 256L * MEBIBYTE
    private const val MIN_WORKSPACE_BYTES = 128L * MEBIBYTE
    // Android Chroma runs have reached roughly twice the staged estimate before the LMK fires.
    private const val CHROMA_SEQUENTIAL_HEADROOM_MULTIPLIER = 2L

    fun recipeFor(params: ImageGenerationRequest): ImageExecutionRecipe {
        val diffusionPhase = buildList {
            add(ImageExecutionComponentKind.DIFFUSION_MODEL)
            if (params.vae != null) add(ImageExecutionComponentKind.VAE)
        }
        return when {
            params.diffusionModelOnly && params.textEncoder != null ->
                ImageExecutionRecipe(
                    profile = ImageConditioningProfile.MASKED_T5,
                    phases = listOf(listOf(ImageExecutionComponentKind.TEXT_ENCODER), diffusionPhase),
                )
            params.splitDiffusionModel && params.t5xxl != null && params.clipL != null && params.clipG != null ->
                ImageExecutionRecipe(
                    profile = ImageConditioningProfile.SD3_CLIP_T5,
                    phases = listOf(
                        listOf(ImageExecutionComponentKind.CLIP_L, ImageExecutionComponentKind.CLIP_G),
                        listOf(ImageExecutionComponentKind.T5XXL),
                        diffusionPhase,
                    ),
                )
            params.splitDiffusionModel && params.t5xxl != null && params.clipL == null && params.clipG == null ->
                ImageExecutionRecipe(
                    profile = ImageConditioningProfile.CHROMA_T5,
                    phases = listOf(listOf(ImageExecutionComponentKind.T5XXL), diffusionPhase),
                )
            params.splitDiffusionModel && params.textEncoder != null ->
                ImageExecutionRecipe(
                    profile = ImageConditioningProfile.LLM,
                    phases = listOf(listOf(ImageExecutionComponentKind.TEXT_ENCODER), diffusionPhase),
                )
            else -> ImageExecutionRecipe.none()
        }
    }

    fun decide(
        sequential: Boolean?,
        recipe: ImageExecutionRecipe,
        componentSizes: Map<ImageExecutionComponentKind, Long>,
        memory: ImageMemorySnapshot,
    ): ImageExecutionDecision {
        if (sequential == false) {
            return ImageExecutionDecision(ImageExecutionMode.DIRECT, "FORCED_DIRECT", recipe)
        }
        if (sequential == true) {
            if (!recipe.supportsSequential) {
                throw UnsupportedStagedExecutionException(
                    "This image request does not expose a splittable conditioning layout.",
                )
            }
            return sequentialDecision(
                reason = "FORCED_SEQUENTIAL",
                recipe = recipe,
                componentSizes = componentSizes,
                memory = memory,
            )
        }
        if (!recipe.supportsSequential) {
            return ImageExecutionDecision(ImageExecutionMode.DIRECT, "AUTO_DIRECT_NONSPLITTABLE", recipe)
        }
        if (recipe.components.any { it !in componentSizes }) {
            return ImageExecutionDecision(ImageExecutionMode.SEQUENTIAL, "AUTO_SEQUENTIAL_ESTIMATE_UNAVAILABLE", recipe)
        }

        val budget = safeBudgetBytes(memory)
        val directEstimate = estimatePeakBytes(recipe.components, componentSizes, memory)
        return if (directEstimate <= budget) {
            ImageExecutionDecision(ImageExecutionMode.DIRECT, "AUTO_DIRECT_FITS", recipe)
        } else {
            sequentialDecision(
                reason = "AUTO_SEQUENTIAL_MEMORY",
                recipe = recipe,
                componentSizes = componentSizes,
                memory = memory,
            )
        }
    }

    fun estimatedDirectPeakBytes(
        recipe: ImageExecutionRecipe,
        componentSizes: Map<ImageExecutionComponentKind, Long>,
        memory: ImageMemorySnapshot,
    ): Long = estimatePeakBytes(recipe.components, componentSizes, memory)

    fun estimatedSequentialPeakBytes(
        recipe: ImageExecutionRecipe,
        componentSizes: Map<ImageExecutionComponentKind, Long>,
        memory: ImageMemorySnapshot,
    ): Long =
        recipe.phases.maxOfOrNull { phase -> estimatePeakBytes(phase, componentSizes, memory) } ?: 0L

    fun requiredSequentialPeakBytes(
        recipe: ImageExecutionRecipe,
        componentSizes: Map<ImageExecutionComponentKind, Long>,
        memory: ImageMemorySnapshot,
    ): Long {
        val estimatedPeak = estimatedSequentialPeakBytes(recipe, componentSizes, memory)
        return if (recipe.profile == ImageConditioningProfile.CHROMA_T5) {
            saturatingMultiply(estimatedPeak, CHROMA_SEQUENTIAL_HEADROOM_MULTIPLIER)
        } else {
            estimatedPeak
        }
    }

    private fun sequentialDecision(
        reason: String,
        recipe: ImageExecutionRecipe,
        componentSizes: Map<ImageExecutionComponentKind, Long>,
        memory: ImageMemorySnapshot,
    ): ImageExecutionDecision {
        if (recipe.components.any { it !in componentSizes }) {
            return ImageExecutionDecision(ImageExecutionMode.SEQUENTIAL, reason, recipe)
        }
        if (memory.availableSystemBytes <= 0L || memory.totalSystemBytes <= 0L) {
            return ImageExecutionDecision(ImageExecutionMode.SEQUENTIAL, reason, recipe)
        }
        val requiredBytes = requiredSequentialPeakBytes(recipe, componentSizes, memory)
        val budget = safeBudgetBytes(memory)
        if (requiredBytes > budget) {
            throw InsufficientMemoryException(
                requiredBytes = requiredBytes,
                availableBytes = budget,
                operation = "sequential image generation",
            )
        }
        return ImageExecutionDecision(ImageExecutionMode.SEQUENTIAL, reason, recipe)
    }

    private fun safeBudgetBytes(memory: ImageMemorySnapshot): Long {
        val reserve = max(MIN_DEVICE_RESERVE_BYTES, memory.totalSystemBytes.coerceAtLeast(0L) / 16L)
        return (memory.availableSystemBytes - memory.lowMemoryThresholdBytes - reserve).coerceAtLeast(0L)
    }

    private fun estimatePeakBytes(
        components: Collection<ImageExecutionComponentKind>,
        componentSizes: Map<ImageExecutionComponentKind, Long>,
        memory: ImageMemorySnapshot,
    ): Long {
        val parameterBytes = components.sumOf { componentSizes.getValue(it).coerceAtLeast(0L) }
        val loadOverhead = parameterBytes / 8L
        val workspace = max(MIN_WORKSPACE_BYTES, memory.totalSystemBytes.coerceAtLeast(0L) / 32L)
        return saturatingAdd(saturatingAdd(parameterBytes, loadOverhead), workspace)
    }

    private fun saturatingAdd(left: Long, right: Long): Long =
        if (Long.MAX_VALUE - left < right) Long.MAX_VALUE else left + right

    private fun saturatingMultiply(value: Long, multiplier: Long): Long =
        if (value > Long.MAX_VALUE / multiplier) Long.MAX_VALUE else value * multiplier
}

internal fun interface ImageExecutionPlanSelector {
    suspend fun decide(
        params: ImageGenerationRequest,
        config: LLMEdgeConfig,
    ): ImageExecutionDecision
}

internal class DefaultImageExecutionPlanSelector(
    private val context: Context,
    private val resolver: ModelRepository,
    private val memorySnapshotProvider: () -> ImageMemorySnapshot = { captureImageMemorySnapshot(context) },
    private val fileSizeProvider: (ModelSpec, java.io.File) -> Long = { _, file -> file.length() },
    private val phaseListener: DiffusionPhaseListener? = null,
) : ImageExecutionPlanSelector {
    override suspend fun decide(
        params: ImageGenerationRequest,
        config: LLMEdgeConfig,
    ): ImageExecutionDecision {
        val recipe = ImageExecutionPlanner.recipeFor(params)
        val memory = memorySnapshotProvider()
        if (params.sequential == false || !recipe.supportsSequential) {
            val decision = ImageExecutionPlanner.decide(
                sequential = params.sequential,
                recipe = recipe,
                componentSizes = emptyMap(),
                memory = memory,
            )
            logDecision(decision, null, memory)
            return decision
        }

        // Sequential planning resolves the real artifacts before estimating their footprint. Keep
        // an isolated worker in the resolving phase while a model download is in flight.
        phaseListener?.onPhase(io.aatricks.llmedge.image.ipc.DiffusionPhases.RESOLVING_MODEL)
        val componentSizes = LinkedHashMap<ImageExecutionComponentKind, Long>()
        for ((component, spec) in componentSpecs(recipe, params, config)) {
            val file = resolver.resolve(context, spec)
            componentSizes[component] = fileSizeProvider(spec, file).coerceAtLeast(0L)
        }
        val decision = ImageExecutionPlanner.decide(params.sequential, recipe, componentSizes, memory)
        logDecision(decision, componentSizes, memory)
        return decision
    }

    private fun logDecision(
        decision: ImageExecutionDecision,
        componentSizes: Map<ImageExecutionComponentKind, Long>?,
        memory: ImageMemorySnapshot,
    ) {
        val estimates =
            componentSizes?.let {
                "directBytes=${ImageExecutionPlanner.estimatedDirectPeakBytes(decision.recipe, it, memory)} " +
                    "sequentialBytes=${ImageExecutionPlanner.estimatedSequentialPeakBytes(decision.recipe, it, memory)} " +
                    "sequentialRequiredBytes=${ImageExecutionPlanner.requiredSequentialPeakBytes(decision.recipe, it, memory)}"
            } ?: "directBytes=not_calculated sequentialBytes=not_calculated"
        AndroidLogAdapter.i(
            "ImageExecutionPlanner",
            "profile=${decision.recipe.profile} mode=${decision.mode} reason=${decision.reason} $estimates " +
                "availableBytes=${memory.availableSystemBytes} thresholdBytes=${memory.lowMemoryThresholdBytes} " +
                "totalBytes=${memory.totalSystemBytes}",
        )
    }

    private fun componentSpecs(
        recipe: ImageExecutionRecipe,
        params: ImageGenerationRequest,
        config: LLMEdgeConfig,
    ): List<Pair<ImageExecutionComponentKind, ModelSpec>> =
        recipe.phases.flatten().distinct().map { component ->
            component to
                when (component) {
                    ImageExecutionComponentKind.DIFFUSION_MODEL -> params.model ?: config.models.image
                    ImageExecutionComponentKind.VAE -> requireNotNull(params.vae)
                    ImageExecutionComponentKind.TEXT_ENCODER -> requireNotNull(params.textEncoder)
                    ImageExecutionComponentKind.T5XXL -> requireNotNull(params.t5xxl)
                    ImageExecutionComponentKind.CLIP_L -> requireNotNull(params.clipL)
                    ImageExecutionComponentKind.CLIP_G -> requireNotNull(params.clipG)
                }
        }

}

private fun captureImageMemorySnapshot(context: Context): ImageMemorySnapshot {
    val activityManager = context.getSystemService(Context.ACTIVITY_SERVICE) as ActivityManager
    val memoryInfo = ActivityManager.MemoryInfo()
    activityManager.getMemoryInfo(memoryInfo)
    return ImageMemorySnapshot(
        availableSystemBytes = memoryInfo.availMem,
        lowMemoryThresholdBytes = memoryInfo.threshold,
        totalSystemBytes = memoryInfo.totalMem,
    )
}
