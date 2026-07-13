package io.aatricks.llmedge.image.ipc

import android.graphics.Bitmap
import android.os.SharedMemory
import io.aatricks.llmedge.image.ImageGenerationRequest
import io.aatricks.llmedge.image.VideoGenerationRequest
import io.aatricks.llmedge.image.diffusion.EasyCacheParams
import io.aatricks.llmedge.image.diffusion.GenerationMetrics
import io.aatricks.llmedge.image.diffusion.ImageRequestMetrics
import io.aatricks.llmedge.image.diffusion.LoraApplyMode
import io.aatricks.llmedge.image.diffusion.SampleMethod
import io.aatricks.llmedge.image.diffusion.Scheduler
import io.aatricks.llmedge.model.ModelArtifactKind
import io.aatricks.llmedge.model.ModelCapability
import io.aatricks.llmedge.model.ModelConversion
import io.aatricks.llmedge.model.ModelHints
import io.aatricks.llmedge.model.ModelSpec
import io.aatricks.llmedge.model.ConversionAdapter
import io.aatricks.llmedge.model.ConversionPrecision
import java.io.File

internal object IpcCodecs {
    private const val TYPE_LOCAL = "local"
    private const val TYPE_HF = "hf"

    fun toIpc(spec: ModelSpec): IpcModelSpec =
        when (spec) {
            is ModelSpec.LocalFile ->
                IpcModelSpec(
                    type = TYPE_LOCAL,
                    path = spec.file.absolutePath,
                    repoId = null,
                    filename = null,
                    revision = null,
                    preferredQuantizations = emptyList(),
                    token = null,
                    forceDownload = false,
                    preferSystemDownloader = true,
                    artifactKind = spec.hints.artifactKind.name,
                    capabilities = spec.hints.capabilities.map(ModelCapability::name),
                    chatTemplate = spec.hints.chatTemplate,
                    conversionPrecision = spec.hints.conversion?.precision?.name,
                    conversionAdapter = spec.hints.conversion?.adapter?.name,
                    conversionTokenizerPre = spec.hints.conversion?.tokenizerPre,
                )
            is ModelSpec.HuggingFace ->
                IpcModelSpec(
                    type = TYPE_HF,
                    path = null,
                    repoId = spec.repoId,
                    filename = spec.filename,
                    revision = spec.revision,
                    preferredQuantizations = spec.preferredQuantizations,
                    token = spec.token,
                    forceDownload = spec.forceDownload,
                    preferSystemDownloader = spec.preferSystemDownloader,
                    artifactKind = spec.hints.artifactKind.name,
                    capabilities = spec.hints.capabilities.map(ModelCapability::name),
                    chatTemplate = spec.hints.chatTemplate,
                    conversionPrecision = spec.hints.conversion?.precision?.name,
                    conversionAdapter = spec.hints.conversion?.adapter?.name,
                    conversionTokenizerPre = spec.hints.conversion?.tokenizerPre,
                )
        }

    fun fromIpc(spec: IpcModelSpec): ModelSpec {
        val hints =
            ModelHints(
                artifactKind = ModelArtifactKind.valueOf(spec.artifactKind),
                capabilities = spec.capabilities.map(ModelCapability::valueOf).toSet(),
                chatTemplate = spec.chatTemplate,
                conversion =
                    spec.conversionPrecision?.let { precision ->
                        ModelConversion(
                            precision = ConversionPrecision.valueOf(precision),
                            adapter = ConversionAdapter.valueOf(spec.conversionAdapter ?: ConversionAdapter.NONE.name),
                            tokenizerPre = spec.conversionTokenizerPre,
                        )
                    },
            )
        return when (spec.type) {
            TYPE_LOCAL -> ModelSpec.LocalFile(File(requireNotNull(spec.path) { "local spec without path" }), hints)
            TYPE_HF ->
                ModelSpec.HuggingFace(
                    repoId = requireNotNull(spec.repoId) { "hf spec without repoId" },
                    filename = spec.filename,
                    revision = spec.revision ?: "main",
                    preferredQuantizations = spec.preferredQuantizations,
                    token = spec.token,
                    forceDownload = spec.forceDownload,
                    preferSystemDownloader = spec.preferSystemDownloader,
                    hints = hints,
                )
            else -> throw IllegalArgumentException("Unknown IpcModelSpec type '${spec.type}'")
        }
    }

    fun toIpc(request: ImageGenerationRequest): IpcImageRequest =
        IpcImageRequest(
            prompt = request.prompt,
            negative = request.negative,
            width = request.width,
            height = request.height,
            steps = request.steps,
            cfgScale = request.cfgScale,
            seed = request.seed,
            flashAttention = request.flashAttention,
            forceSequentialLoad = request.forceSequentialLoad,
            easyCacheEnabled = request.easyCache.enabled,
            easyCacheReuseThreshold = request.easyCache.reuseThreshold,
            easyCacheStartPercent = request.easyCache.startPercent,
            easyCacheEndPercent = request.easyCache.endPercent,
            loraModelDir = request.loraModelDir,
            loraApplyModeId = request.loraApplyMode.id,
            model = request.model?.let(::toIpc),
            vae = request.vae?.let(::toIpc),
            textEncoder = request.textEncoder?.let(::toIpc),
            diffusionModelOnly = request.diffusionModelOnly,
            splitDiffusionModel = request.splitDiffusionModel,
            sequential = request.sequential,
        )

    fun fromIpc(request: IpcImageRequest): ImageGenerationRequest =
        ImageGenerationRequest(
            prompt = request.prompt,
            negative = request.negative,
            width = request.width,
            height = request.height,
            steps = request.steps,
            cfgScale = request.cfgScale,
            seed = request.seed,
            flashAttention = request.flashAttention,
            forceSequentialLoad = request.forceSequentialLoad,
            easyCache =
                EasyCacheParams(
                    enabled = request.easyCacheEnabled,
                    reuseThreshold = request.easyCacheReuseThreshold,
                    startPercent = request.easyCacheStartPercent,
                    endPercent = request.easyCacheEndPercent,
                ),
            loraModelDir = request.loraModelDir,
            loraApplyMode = LoraApplyMode.fromId(request.loraApplyModeId),
            model = request.model?.let(::fromIpc),
            vae = request.vae?.let(::fromIpc),
            textEncoder = request.textEncoder?.let(::fromIpc),
            diffusionModelOnly = request.diffusionModelOnly,
            splitDiffusionModel = request.splitDiffusionModel,
            sequential = request.sequential,
        )

    fun toIpc(request: VideoGenerationRequest): IpcVideoRequest =
        IpcVideoRequest(
            prompt = request.prompt,
            negative = request.negative,
            width = request.width,
            height = request.height,
            videoFrames = request.videoFrames,
            steps = request.steps,
            cfgScale = request.cfgScale,
            seed = request.seed,
            flowShift = request.flowShift,
            flashAttention = request.flashAttention,
            forceSequentialLoad = request.forceSequentialLoad,
            initImage = request.initImage?.let { PixelCodec.encodeBitmap(it, "llmedge_init_image") },
            strength = request.strength,
            sampleMethodId = request.sampleMethod.id,
            schedulerId = request.scheduler.id,
            easyCacheEnabled = request.easyCache.enabled,
            easyCacheReuseThreshold = request.easyCache.reuseThreshold,
            easyCacheStartPercent = request.easyCache.startPercent,
            easyCacheEndPercent = request.easyCache.endPercent,
            loraModelDir = request.loraModelDir,
            loraApplyModeId = request.loraApplyMode.id,
            taehv = request.taehv?.let(::toIpc),
            model = request.model?.let(::toIpc),
            vae = request.vae?.let(::toIpc),
            textEncoder = request.textEncoder?.let(::toIpc),
        )

    fun fromIpc(request: IpcVideoRequest): VideoGenerationRequest =
        VideoGenerationRequest(
            prompt = request.prompt,
            negative = request.negative,
            width = request.width,
            height = request.height,
            videoFrames = request.videoFrames,
            steps = request.steps,
            cfgScale = request.cfgScale,
            seed = request.seed,
            flowShift = request.flowShift,
            flashAttention = request.flashAttention,
            forceSequentialLoad = request.forceSequentialLoad,
            initImage = request.initImage?.let { PixelCodec.decodeBitmap(it) },
            strength = request.strength,
            sampleMethod = SampleMethod.fromId(request.sampleMethodId),
            scheduler = Scheduler.fromId(request.schedulerId),
            easyCache =
                EasyCacheParams(
                    enabled = request.easyCacheEnabled,
                    reuseThreshold = request.easyCacheReuseThreshold,
                    startPercent = request.easyCacheStartPercent,
                    endPercent = request.easyCacheEndPercent,
                ),
            loraModelDir = request.loraModelDir,
            loraApplyMode = LoraApplyMode.fromId(request.loraApplyModeId),
            taehv = request.taehv?.let(::fromIpc),
            model = request.model?.let(::fromIpc),
            vae = request.vae?.let(::fromIpc),
            textEncoder = request.textEncoder?.let(::fromIpc),
        )

    fun toIpc(metrics: GenerationMetrics): IpcGenerationMetrics {
        val request = metrics.imageRequestMetrics
        return IpcGenerationMetrics(
            totalTimeSeconds = metrics.totalTimeSeconds,
            framesPerSecond = metrics.framesPerSecond,
            timePerStep = metrics.timePerStep,
            peakMemoryUsageMb = metrics.peakMemoryUsageMb,
            vulkanEnabled = metrics.vulkanEnabled,
            frameConversionTimeSeconds = metrics.frameConversionTimeSeconds,
            hasRequestMetrics = request != null,
            runtimeAcquireMs = request?.runtimeAcquireMs ?: 0L,
            modelLoadMs = request?.modelLoadMs ?: 0L,
            generateMs = request?.generateMs ?: 0L,
            cacheHit = request?.cacheHit ?: false,
            backend = request?.backend ?: "",
            flashAttentionEnabled = request?.flashAttentionEnabled ?: false,
            easyCacheEnabled = request?.easyCacheEnabled ?: false,
            width = request?.width ?: 0,
            height = request?.height ?: 0,
            steps = request?.steps ?: 0,
        )
    }

    fun fromIpc(metrics: IpcGenerationMetrics): GenerationMetrics {
        val base =
            GenerationMetrics(
                totalTimeSeconds = metrics.totalTimeSeconds,
                framesPerSecond = metrics.framesPerSecond,
                timePerStep = metrics.timePerStep,
                peakMemoryUsageMb = metrics.peakMemoryUsageMb,
                vulkanEnabled = metrics.vulkanEnabled,
                frameConversionTimeSeconds = metrics.frameConversionTimeSeconds,
            )
        if (!metrics.hasRequestMetrics) return base
        return base.withImageRequestMetrics(
            ImageRequestMetrics(
                runtimeAcquireMs = metrics.runtimeAcquireMs,
                modelLoadMs = metrics.modelLoadMs,
                generateMs = metrics.generateMs,
                cacheHit = metrics.cacheHit,
                backend = metrics.backend,
                flashAttentionEnabled = metrics.flashAttentionEnabled,
                easyCacheEnabled = metrics.easyCacheEnabled,
                width = metrics.width,
                height = metrics.height,
                steps = metrics.steps,
            ),
        )
    }
}

/** ARGB_8888 pixels packed into ashmem. One frame per [encodeBitmap]; N frames per [encodeFrames]. */
internal object PixelCodec {
    private const val BYTES_PER_PIXEL = 4

    fun encodeBitmap(bitmap: Bitmap, name: String): IpcFrameBuffer = encodeFrames(listOf(bitmap), name)

    fun decodeBitmap(buffer: IpcFrameBuffer): Bitmap = decodeFrames(buffer).single()

    fun encodeFrames(frames: List<Bitmap>, name: String): IpcFrameBuffer {
        require(frames.isNotEmpty()) { "no frames to encode" }
        val width = frames.first().width
        val height = frames.first().height
        frames.forEach { frame ->
            require(frame.width == width && frame.height == height) { "frames must share dimensions" }
        }
        val frameBytes = width * height * BYTES_PER_PIXEL
        val memory = SharedMemory.create(name, frameBytes * frames.size)
        try {
            val mapped = memory.mapReadWrite()
            try {
                frames.forEach { frame ->
                    val argb = if (frame.config == Bitmap.Config.ARGB_8888) frame else frame.copy(Bitmap.Config.ARGB_8888, false)
                    argb.copyPixelsToBuffer(mapped)
                    if (argb !== frame) argb.recycle()
                }
            } finally {
                SharedMemory.unmap(mapped)
            }
        } catch (error: Throwable) {
            memory.close()
            throw error
        }
        return IpcFrameBuffer(memory = memory, width = width, height = height, frameCount = frames.size)
    }

    fun decodeFrames(buffer: IpcFrameBuffer): List<Bitmap> {
        val frameBytes = buffer.width * buffer.height * BYTES_PER_PIXEL
        val expected = frameBytes * buffer.frameCount
        check(buffer.memory.size >= expected) {
            "frame buffer too small: ${buffer.memory.size} < $expected"
        }
        buffer.memory.use { memory ->
            val mapped = memory.mapReadOnly()
            try {
                return (0 until buffer.frameCount).map { index ->
                    mapped.position(index * frameBytes)
                    mapped.limit(index * frameBytes + frameBytes)
                    val bitmap = Bitmap.createBitmap(buffer.width, buffer.height, Bitmap.Config.ARGB_8888)
                    bitmap.copyPixelsFromBuffer(mapped.slice())
                    bitmap
                }
            } finally {
                SharedMemory.unmap(mapped)
            }
        }
    }
}
