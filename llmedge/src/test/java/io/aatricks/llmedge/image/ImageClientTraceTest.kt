package io.aatricks.llmedge.image

import android.content.Context
import androidx.test.core.app.ApplicationProvider
import io.aatricks.llmedge.LLMEdgeConfig
import io.aatricks.llmedge.core.LLMEdgeScope
import io.aatricks.llmedge.image.diffusion.ImageGenerationPhase
import io.aatricks.llmedge.image.diffusion.PrecomputedCondition
import io.aatricks.llmedge.image.diffusion.SampleMethod
import io.aatricks.llmedge.image.diffusion.Scheduler
import io.aatricks.llmedge.image.diffusion.StableDiffusion
import io.aatricks.llmedge.image.diffusion.VideoProgressCallback
import io.aatricks.llmedge.model.DefaultModelResolver
import io.aatricks.llmedge.model.ModelSpec
import io.mockk.clearAllMocks
import io.mockk.coEvery
import io.mockk.mockkObject
import kotlinx.coroutines.test.runTest
import org.junit.After
import org.junit.Assert.assertEquals
import org.junit.Assert.assertNotNull
import org.junit.Assert.assertTrue
import org.junit.Before
import org.junit.Test
import org.junit.runner.RunWith
import org.robolectric.RobolectricTestRunner
import org.robolectric.annotation.Config

@RunWith(RobolectricTestRunner::class)
@Config(sdk = [34])
class ImageClientTraceTest {
    @Before
    fun setup() {
        System.setProperty("llmedge.disableNativeLoad", "true")
        StableDiffusion.enableNativeBridgeForTests()
        mockkObject(StableDiffusion.Companion)
    }

    @After
    fun teardown() {
        StableDiffusion.resetNativeBridgeForTests()
        System.clearProperty("llmedge.disableNativeLoad")
        try {
            io.mockk.unmockkObject(StableDiffusion.Companion)
        } catch (_: Throwable) {
        }
        clearAllMocks()
    }

    @Test
    fun `image generation records ordered request trace through executor entry`() = runTest {
        val context = ApplicationProvider.getApplicationContext<Context>()
        val modelFile =
            java.io.File.createTempFile("image-trace", ".safetensors", context.filesDir).apply {
                writeBytes(byteArrayOf(0x01))
            }

        StableDiffusion.overrideNativeBridgeForTests {
            object : StableDiffusion.NativeBridge {
                override fun txt2img(
                    handle: Long,
                    prompt: String,
                    negative: String,
                    width: Int,
                    height: Int,
                    steps: Int,
                    cfg: Float,
                    seed: Long,
                    vaeTiling: Boolean,
                    easyCacheEnabled: Boolean,
                    easyCacheReuseThreshold: Float,
                    easyCacheStartPercent: Float,
                    easyCacheEndPercent: Float,
                ): ByteArray = ByteArray(width * height * 3) { 0x33 }

                override fun txt2vid(
                    handle: Long,
                    prompt: String,
                    negative: String,
                    width: Int,
                    height: Int,
                    videoFrames: Int,
                    steps: Int,
                    cfg: Float,
                    seed: Long,
                    sampleMethod: SampleMethod,
                    scheduler: Scheduler,
                    strength: Float,
                    initImage: ByteArray?,
                    initWidth: Int,
                    initHeight: Int,
                    vaceStrength: Float,
                    easyCacheEnabled: Boolean,
                    easyCacheReuseThreshold: Float,
                    easyCacheStartPercent: Float,
                    easyCacheEndPercent: Float,
                ): Array<ByteArray>? = null

                override fun precomputeCondition(
                    handle: Long,
                    prompt: String,
                    negative: String,
                    width: Int,
                    height: Int,
                    clipSkip: Int,
                ): PrecomputedCondition? = null

                override fun setProgressCallback(handle: Long, callback: VideoProgressCallback?) = Unit

                override fun cancelGeneration(handle: Long) = Unit
            }
        }

        coEvery {
            StableDiffusion.loadWithRuntimeBackend(
                any(), any(), any(), any(), any(), any(), any(),
                any(), any(), any(), any(), any(), any(), any(),
                any(), any(), any(), any(), any(), any(), any(),
            )
        } coAnswers {
            val constructor = StableDiffusion::class.java.getDeclaredConstructor(Long::class.javaPrimitiveType)
            constructor.isAccessible = true
            constructor.newInstance(1L)
        }

        val edgeScope = LLMEdgeScope(this, 1)
        val client =
            ImageClient(
                context = context,
                scope = edgeScope,
                config = LLMEdgeConfig(preferPerformanceMode = true),
                resolver = DefaultModelResolver(),
            )

        try {
            val bitmap =
                client.generate(
                    ImageGenerationRequest(
                        prompt = "trace test",
                        width = 128,
                        height = 128,
                        model = ModelSpec.localFile(modelFile),
                    ),
                )

            assertEquals(128, bitmap.width)
            assertEquals(128, bitmap.height)
            assertNotNull(client.getLastGenerationMetrics()?.imageRequestMetrics)

            val phases = client.getLastImageRequestTraceForTests().map { it.phase }
            assertOrderedSubsequence(
                phases,
                listOf(
                    ImageGenerationPhase.REQUESTED,
                    ImageGenerationPhase.RUNTIME_ACQUIRED,
                    ImageGenerationPhase.MODEL_READY,
                    ImageGenerationPhase.TXT2IMG_ENTER,
                    ImageGenerationPhase.EXECUTOR_ENTER,
                    ImageGenerationPhase.WAITING_GENERATION_MUTEX,
                    ImageGenerationPhase.JNI_ARGB_ENTER,
                    ImageGenerationPhase.JNI_RGB_ENTER,
                    ImageGenerationPhase.COMPLETED,
                ),
            )
        } finally {
            client.close()
            edgeScope.close()
        }
    }

    private fun assertOrderedSubsequence(
        actual: List<ImageGenerationPhase>,
        expected: List<ImageGenerationPhase>,
    ) {
        var cursor = 0
        expected.forEach { phase ->
            val next = actual.indexOfFirstFrom(cursor, phase)
            assertTrue("Missing phase $phase in order from $cursor within $actual", next >= cursor)
            cursor = next + 1
        }
    }

    private fun List<ImageGenerationPhase>.indexOfFirstFrom(
        startIndex: Int,
        phase: ImageGenerationPhase,
    ): Int {
        for (index in startIndex until size) {
            if (this[index] == phase) {
                return index
            }
        }
        return -1
    }
}
