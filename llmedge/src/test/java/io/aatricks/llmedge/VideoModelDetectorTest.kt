package io.aatricks.llmedge

import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test
import io.aatricks.llmedge.image.diffusion.StableDiffusion
import io.aatricks.llmedge.image.diffusion.VideoModelMetadata

class VideoModelDetectorTest {

    @Test
    fun `isVideoModel detects wan architecture as video model`() {
        val sd = newStableDiffusion()
        sd.updateModelMetadata(
            VideoModelMetadata(
                architecture = "wan",
                modelType = null,
                parameterCount = "1.3B",
                mobileSupported = true,
                tags = emptySet(),
                filename = "wan-model.gguf"
            )
        )

        assertTrue(sd.isVideoModel())
    }

    @Test
    fun `isVideoModel detects hunyuan architecture as video model`() {
        val sd = newStableDiffusion()
        sd.updateModelMetadata(
            VideoModelMetadata(
                architecture = "hunyuan_video",
                modelType = null,
                parameterCount = "5B",
                mobileSupported = true,
                tags = emptySet(),
                filename = "hunyuan-model.gguf"
            )
        )

        assertTrue(sd.isVideoModel())
    }

    @Test
    fun `isVideoModel detects t2v modelType as video model`() {
        val sd = newStableDiffusion()
        sd.updateModelMetadata(
            VideoModelMetadata(
                architecture = "some-model",
                modelType = "t2v",
                parameterCount = null,
                mobileSupported = true,
                tags = emptySet(),
                filename = "model.gguf"
            )
        )

        assertTrue(sd.isVideoModel())
    }

    @Test
    fun `isVideoModel detects i2v modelType as video model`() {
        val sd = newStableDiffusion()
        sd.updateModelMetadata(
            VideoModelMetadata(
                architecture = "some-model",
                modelType = "i2v",
                parameterCount = null,
                mobileSupported = true,
                tags = emptySet(),
                filename = "model.gguf"
            )
        )

        assertTrue(sd.isVideoModel())
    }

    @Test
    fun `isVideoModel detects ti2v modelType as video model`() {
        val sd = newStableDiffusion()
        sd.updateModelMetadata(
            VideoModelMetadata(
                architecture = "some-model",
                modelType = "ti2v",
                parameterCount = null,
                mobileSupported = true,
                tags = emptySet(),
                filename = "model.gguf"
            )
        )

        assertTrue(sd.isVideoModel())
    }

    @Test
    fun `isVideoModel detects video in filename as video model`() {
        val sd = newStableDiffusion()
        sd.updateModelMetadata(
            VideoModelMetadata(
                architecture = "stable-diffusion-xl",
                modelType = null,
                parameterCount = null,
                mobileSupported = true,
                tags = emptySet(),
                filename = "my_video_model.gguf"
            )
        )

        assertTrue(sd.isVideoModel())
    }

    @Test
    fun `isVideoModel detects text-to-video tag as video model`() {
        val sd = newStableDiffusion()
        sd.updateModelMetadata(
            VideoModelMetadata(
                architecture = "some-model",
                modelType = null,
                parameterCount = null,
                mobileSupported = true,
                tags = setOf("text-to-video"),
                filename = "model.gguf"
            )
        )

        assertTrue(sd.isVideoModel())
    }

    @Test
    fun `isVideoModel detects wan tag as video model`() {
        val sd = newStableDiffusion()
        sd.updateModelMetadata(
            VideoModelMetadata(
                architecture = "some-model",
                modelType = null,
                parameterCount = null,
                mobileSupported = true,
                tags = setOf("wan"),
                filename = "model.gguf"
            )
        )

        assertTrue(sd.isVideoModel())
    }

    @Test
    fun `isVideoModel detects multiple video keywords in combination`() {
        val sd = newStableDiffusion()
        sd.updateModelMetadata(
            VideoModelMetadata(
                architecture = "wan",
                modelType = "t2v",
                parameterCount = "1.3B",
                mobileSupported = true,
                tags = setOf("text-to-video", "wan-model"),
                filename = "wan_video_model.gguf"
            )
        )

        assertTrue(sd.isVideoModel())
    }

    @Test
    fun `isVideoModel returns false for stable diffusion xl without video keywords`() {
        val sd = newStableDiffusion()
        sd.updateModelMetadata(
            VideoModelMetadata(
                architecture = "stable-diffusion-xl",
                modelType = "txt2img",
                parameterCount = "1.3B",
                mobileSupported = true,
                tags = setOf("text-to-image"),
                filename = "sdxl-model.gguf"
            )
        )

        assertFalse(sd.isVideoModel())
    }

    @Test
    fun `isVideoModel returns false for null metadata`() {
        val sd = newStableDiffusion()
        sd.updateModelMetadata(null)

        assertFalse(sd.isVideoModel())
    }

    @Test
    fun `isVideoModel returns false for empty metadata`() {
        val sd = newStableDiffusion()
        sd.updateModelMetadata(
            VideoModelMetadata(
                architecture = null,
                modelType = null,
                parameterCount = null,
                mobileSupported = true,
                tags = emptySet(),
                filename = ""
            )
        )

        assertFalse(sd.isVideoModel())
    }

    @Test
    fun `isVideoModel case insensitive matching works`() {
        val sd = newStableDiffusion()
        sd.updateModelMetadata(
            VideoModelMetadata(
                architecture = "WAN",
                modelType = "T2V",
                parameterCount = null,
                mobileSupported = true,
                tags = setOf("VIDEO"),
                filename = "MODEL.GGUF"
            )
        )

        assertTrue(sd.isVideoModel())
    }

    private fun newStableDiffusion(): StableDiffusion {
        val constructor = StableDiffusion::class.java.getDeclaredConstructor(Long::class.javaPrimitiveType)
        constructor.isAccessible = true
        return constructor.newInstance(1L)
    }
}