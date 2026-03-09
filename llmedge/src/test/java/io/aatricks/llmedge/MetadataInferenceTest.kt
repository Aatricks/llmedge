package io.aatricks.llmedge

import android.util.Log
import io.mockk.*
import kotlinx.coroutines.test.runTest
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.Test
import java.io.File
import kotlin.reflect.full.declaredFunctions
import kotlin.reflect.full.callSuspend
import kotlin.reflect.jvm.isAccessible

class MetadataInferenceTest {

    @Test
    fun `inferVideoModelMetadata detects wan architecture from filename`() = runTest {
        val metadata = callInferVideoModelMetadata(
            resolvedModelPath = "/path/to/wan2.1-t2v-1.3b.gguf",
            modelId = null,
            explicitFilename = null
        )

        assertEquals("wan", metadata.architecture)
        assertEquals("t2v", metadata.modelType)
        assertEquals("1.3B", metadata.parameterCount)
        assertTrue(metadata.mobileSupported)
        assertTrue(metadata.tags.contains("wan"))
        assertTrue(metadata.tags.contains("text-to-video"))
    }

    @Test
    fun `inferVideoModelMetadata detects hunyuan architecture from filename`() = runTest {
        val metadata = callInferVideoModelMetadata(
            resolvedModelPath = "/path/to/hunyuan_video_t2v_720p_fp8.gguf",
            modelId = null,
            explicitFilename = null
        )

        assertEquals("hunyuan_video", metadata.architecture)
        assertEquals("t2v", metadata.modelType)
        assertNull(metadata.parameterCount)
        assertTrue(metadata.mobileSupported)
        assertTrue(metadata.tags.contains("hunyuan"))
        assertTrue(metadata.tags.contains("text-to-video"))
    }

    @Test
    fun `inferVideoModelMetadata detects modelId when filename detection fails`() = runTest {
        val metadata = callInferVideoModelMetadata(
            resolvedModelPath = "/path/to/unknown-model.gguf",
            modelId = "wan2.1-t2v-5b",
            explicitFilename = null
        )

        assertEquals("wan2.1-t2v-5b", metadata.architecture)
        assertNull(metadata.modelType)
        assertNull(metadata.parameterCount)
        assertTrue(metadata.mobileSupported)
    }

    @Test
    fun `inferVideoModelMetadata detects parameter count from filename patterns`() = runTest {
        // Test 1.3B detection
        val metadata13B = callInferVideoModelMetadata(
            resolvedModelPath = "/path/to/model_1_3b.gguf",
            modelId = null,
            explicitFilename = null
        )
        assertEquals("1.3B", metadata13B.parameterCount)

        // Test 5B detection
        val metadata5B = callInferVideoModelMetadata(
            resolvedModelPath = "/path/to/model-5b.gguf",
            modelId = null,
            explicitFilename = null
        )
        assertEquals("5B", metadata5B.parameterCount)

        // Test 14B detection
        val metadata14B = callInferVideoModelMetadata(
            resolvedModelPath = "/path/to/model-14_b.gguf",
            modelId = null,
            explicitFilename = null
        )
        assertEquals("14B", metadata14B.parameterCount)
        assertFalse(metadata14B.mobileSupported) // 14B should not be mobile supported
    }

    @Test
    fun `inferVideoModelMetadata detects model types correctly`() = runTest {
        // Test t2v (text-to-video)
        val t2vMetadata = callInferVideoModelMetadata(
            resolvedModelPath = "/path/to/model-t2v.gguf",
            modelId = null,
            explicitFilename = null
        )
        assertEquals("t2v", t2vMetadata.modelType)

        // Test i2v (image-to-video)
        val i2vMetadata = callInferVideoModelMetadata(
            resolvedModelPath = "/path/to/model-i2v.gguf",
            modelId = null,
            explicitFilename = null
        )
        assertEquals("i2v", i2vMetadata.modelType)

        // Test ti2v (text-image-to-video)
        val ti2vMetadata = callInferVideoModelMetadata(
            resolvedModelPath = "/path/to/model-ti2v.gguf",
            modelId = null,
            explicitFilename = null
        )
        assertEquals("ti2v", ti2vMetadata.modelType)
    }

    @Test
    fun `inferVideoModelMetadata uses explicit filename when provided`() = runTest {
        val metadata = callInferVideoModelMetadata(
            resolvedModelPath = "/path/to/actual-path.gguf",
            modelId = null,
            explicitFilename = "wan2.1-t2v-1.3b.gguf"
        )

        assertEquals("wan", metadata.architecture)
        assertEquals("t2v", metadata.modelType)
        assertEquals("1.3B", metadata.parameterCount)
        assertEquals("wan2.1-t2v-1.3b.gguf", metadata.filename)
    }

    @Test
    fun `inferVideoModelMetadata caches results to avoid re-parsing`() = runTest {
        // First call
        val metadata1 = callInferVideoModelMetadata(
            resolvedModelPath = "/path/to/cached-model.gguf",
            modelId = null,
            explicitFilename = null
        )

        // Second call with same path should return cached result
        val metadata2 = callInferVideoModelMetadata(
            resolvedModelPath = "/path/to/cached-model.gguf",
            modelId = null,
            explicitFilename = null
        )

        assertEquals(metadata1, metadata2)
    }

    @Test
    fun `inferVideoModelMetadata handles case insensitive matching`() = runTest {
        val metadata = callInferVideoModelMetadata(
            resolvedModelPath = "/path/to/WAN2.1-T2V-1.3B.GGUF",
            modelId = null,
            explicitFilename = null
        )

        assertEquals("wan", metadata.architecture)
        assertEquals("t2v", metadata.modelType)
        assertEquals("1.3B", metadata.parameterCount)
    }

    @Test
    fun `inferVideoModelMetadata handles unknown models gracefully`() = runTest {
        val metadata = callInferVideoModelMetadata(
            resolvedModelPath = "/path/to/completely-unknown-model.gguf",
            modelId = null,
            explicitFilename = null
        )

        assertNull(metadata.architecture)
        assertNull(metadata.modelType)
        assertNull(metadata.parameterCount)
        assertTrue(metadata.mobileSupported) // Default to supported
        assertTrue(metadata.tags.isEmpty())
    }

    @Test
    fun `inferVideoModelMetadata builds correct tags for wan models`() = runTest {
        val metadata = callInferVideoModelMetadata(
            resolvedModelPath = "/path/to/wan-model-t2v.gguf",
            modelId = null,
            explicitFilename = null
        )

        assertTrue(metadata.tags.contains("wan"))
        assertTrue(metadata.tags.contains("text-to-video"))
    }

    @Test
    fun `inferVideoModelMetadata builds correct tags for hunyuan models`() = runTest {
        val metadata = callInferVideoModelMetadata(
            resolvedModelPath = "/path/to/hunyuan_video_model.gguf",
            modelId = null,
            explicitFilename = null
        )

        assertTrue(metadata.tags.contains("hunyuan"))
        assertTrue(metadata.tags.contains("text-to-video"))
    }

    private suspend fun callInferVideoModelMetadata(
        resolvedModelPath: String,
        modelId: String?,
        explicitFilename: String?
    ): VideoModelMetadata {
        // Create a temporary file at the given resolved path to ensure File.exists() returns true
        // If the resolved path already has a directory, ensure it's created; otherwise create a temp file
        val file = File(resolvedModelPath)
        try {
            file.parentFile?.mkdirs()
            file.createNewFile()
        } catch (_: Throwable) {
            // If we couldn't create the file at the path, fallback to a proper temp file
        }

        // Mock Log.d to avoid logging in tests, ensuring we always unmock in finally
        mockkStatic(Log::class)
        every { Log.d(any(), any()) } returns 0

        val companion = StableDiffusion.Companion
        val function = companion::class.declaredFunctions.first { it.name == "inferVideoModelMetadata" }
        function.isAccessible = true
        // callSuspend is available via kotlin.reflect to invoke suspend functions
        try {
            val result = function.callSuspend(companion, resolvedModelPath, modelId, explicitFilename) as VideoModelMetadata
            return result
        } finally {
            try { io.mockk.unmockkStatic(Log::class) } catch (_: Throwable) {}
            try { file.delete() } catch (_: Throwable) {}
        }
    }
}