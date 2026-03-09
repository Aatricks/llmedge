package io.aatricks.llmedge.text

import android.content.Context
import androidx.test.core.app.ApplicationProvider
import io.aatricks.llmedge.GGUFReader
import io.aatricks.llmedge.LLMEdgeConfig
import io.aatricks.llmedge.SmolLM
import io.aatricks.llmedge.core.LLMEdgeScope
import io.aatricks.llmedge.model.ModelResolver
import io.aatricks.llmedge.model.ModelSpec
import java.io.File
import kotlinx.coroutines.test.advanceUntilIdle
import kotlinx.coroutines.test.runTest
import org.junit.After
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Before
import org.junit.Test
import org.junit.runner.RunWith
import org.robolectric.RobolectricTestRunner
import org.robolectric.annotation.Config

@RunWith(RobolectricTestRunner::class)
@Config(sdk = [34])
class TextClientTest {
    private fun createTempGgufFile(dir: File): File =
        File.createTempFile("llmedge-text", ".gguf", dir).apply {
            writeBytes(
                byteArrayOf(
                    'G'.code.toByte(),
                    'G'.code.toByte(),
                    'U'.code.toByte(),
                    'F'.code.toByte(),
                    0x00,
                ),
            )
        }

    @Before
    fun setUp() {
        System.setProperty("llmedge.disableNativeLoad", "true")
        GGUFReader.overrideNativeBridgeForTests {
            object : GGUFReader.NativeBridge {
                override fun getGGUFContextNativeHandle(modelPath: String): Long = 1L

                override fun getContextSize(nativeHandle: Long): Long = 2048L

                override fun getChatTemplate(nativeHandle: Long): String =
                    "{% for message in messages %}{{ message.content }}{% endfor %}"

                override fun getArchitecture(nativeHandle: Long): String = "llama"

                override fun getParameterCount(nativeHandle: Long): String = "135M"

                override fun getModelName(nativeHandle: Long): String = "Test GGUF"

                override fun releaseGGUFContext(nativeHandle: Long) = Unit
            }
        }
    }

    @After
    fun tearDown() {
        SmolLM.resetNativeBridgeForTests()
        GGUFReader.resetNativeBridgeForTests()
        System.clearProperty("llmedge.disableNativeLoad")
    }

    @Test
    fun `prepare and repeated generate reuse cached runtime`() = runTest {
        val context = ApplicationProvider.getApplicationContext<Context>()
        val modelFile = createTempGgufFile(context.cacheDir)
        val modelSpec = ModelSpec.localFile(modelFile)
        val resolver =
            object : ModelResolver {
                override suspend fun resolve(
                    context: Context,
                    spec: ModelSpec,
                    onProgress: ((io.aatricks.llmedge.core.ProgressEvent.Downloading) -> Unit)?,
                ): File =
                    when (spec) {
                        is ModelSpec.LocalFile -> spec.file
                        else -> error("Unexpected spec: $spec")
                    }
            }

        var loadCalls = 0
        var startCalls = 0
        var clearKvCacheCalls = 0
        var closeCalls = 0

        SmolLM.overrideNativeBridgeForTests {
            object : SmolLM.NativeBridge {
                private var queue = ArrayDeque<String>()

                override fun loadModel(
                    instance: SmolLM,
                    modelPath: String,
                    minP: Float,
                    temperature: Float,
                    storeChats: Boolean,
                    contextSize: Long,
                    chatTemplate: String,
                    nThreads: Int,
                    useMmap: Boolean,
                    useMlock: Boolean,
                    useVulkan: Boolean,
                    useFlashAttn: Boolean,
                ): Long {
                    loadCalls++
                    return 1L
                }

                override fun setReasoningOptions(
                    instance: SmolLM,
                    modelPtr: Long,
                    disableThinking: Boolean,
                    reasoningBudget: Int,
                ) = Unit

                override fun addChatMessage(
                    instance: SmolLM,
                    modelPtr: Long,
                    message: String,
                    role: String,
                ) = Unit

                override fun getResponseGenerationSpeed(instance: SmolLM, modelPtr: Long): Float = 42f

                override fun getResponseGeneratedTokenCount(instance: SmolLM, modelPtr: Long): Long = 1L

                override fun getResponseGenerationDurationMicros(instance: SmolLM, modelPtr: Long): Long = 1000L

                override fun getContextSizeUsed(instance: SmolLM, modelPtr: Long): Int = 64

                override fun getNativeModelPtr(instance: SmolLM, modelPtr: Long): Long = 0L

                override fun nativeDecodePreparedEmbeddings(
                    instance: SmolLM,
                    modelPtr: Long,
                    embdPath: String,
                    metaPath: String,
                    nBatch: Int,
                ): Boolean = true

                override fun close(instance: SmolLM, modelPtr: Long) {
                    closeCalls++
                }

                override fun startCompletion(instance: SmolLM, modelPtr: Long, prompt: String) {
                    startCalls++
                    queue = ArrayDeque(listOf("ok", "[EOG]"))
                }

                override fun completionLoop(instance: SmolLM, modelPtr: Long): String = queue.removeFirst()

                override fun completionLoopBatch(instance: SmolLM, modelPtr: Long, maxTokens: Int): String =
                    completionLoop(instance, modelPtr)

                override fun stopCompletion(instance: SmolLM, modelPtr: Long) = Unit

                override fun clearKvCache(instance: SmolLM, modelPtr: Long) {
                    clearKvCacheCalls++
                }
            }
        }

        val edgeScope = LLMEdgeScope(this, 1)
        val client =
            TextClient(
                context = context,
                scope = edgeScope,
                config = LLMEdgeConfig(textCacheSize = 2, textCacheMemoryMb = 64),
                modelResolver = resolver,
            )

        try {
            client.prepare(model = modelSpec)
            assertEquals(1, loadCalls)

            assertEquals("ok", client.generate(prompt = "hello", model = modelSpec))
            assertEquals("ok", client.generate(prompt = "again", model = modelSpec))

            assertEquals(1, loadCalls)
            assertEquals(2, startCalls)
            assertTrue(clearKvCacheCalls >= 4)

            client.close()
            advanceUntilIdle()
            assertEquals(1, closeCalls)
        } finally {
            edgeScope.close()
        }
    }

    @Test
    fun `different cache keys evict prior text runtime`() = runTest {
        val context = ApplicationProvider.getApplicationContext<Context>()
        val modelFile = createTempGgufFile(context.cacheDir)
        val modelSpec = ModelSpec.localFile(modelFile)
        val resolver =
            object : ModelResolver {
                override suspend fun resolve(
                    context: Context,
                    spec: ModelSpec,
                    onProgress: ((io.aatricks.llmedge.core.ProgressEvent.Downloading) -> Unit)?,
                ): File =
                    when (spec) {
                        is ModelSpec.LocalFile -> spec.file
                        else -> error("Unexpected spec: $spec")
                    }
            }

        var closeCalls = 0

        SmolLM.overrideNativeBridgeForTests {
            object : SmolLM.NativeBridge {
                private var queue = ArrayDeque<String>()

                override fun loadModel(
                    instance: SmolLM,
                    modelPath: String,
                    minP: Float,
                    temperature: Float,
                    storeChats: Boolean,
                    contextSize: Long,
                    chatTemplate: String,
                    nThreads: Int,
                    useMmap: Boolean,
                    useMlock: Boolean,
                    useVulkan: Boolean,
                    useFlashAttn: Boolean,
                ): Long = 1L

                override fun setReasoningOptions(
                    instance: SmolLM,
                    modelPtr: Long,
                    disableThinking: Boolean,
                    reasoningBudget: Int,
                ) = Unit

                override fun addChatMessage(instance: SmolLM, modelPtr: Long, message: String, role: String) = Unit

                override fun getResponseGenerationSpeed(instance: SmolLM, modelPtr: Long): Float = 1f

                override fun getResponseGeneratedTokenCount(instance: SmolLM, modelPtr: Long): Long = 1L

                override fun getResponseGenerationDurationMicros(instance: SmolLM, modelPtr: Long): Long = 1L

                override fun getContextSizeUsed(instance: SmolLM, modelPtr: Long): Int = 1

                override fun getNativeModelPtr(instance: SmolLM, modelPtr: Long): Long = 0L

                override fun nativeDecodePreparedEmbeddings(
                    instance: SmolLM,
                    modelPtr: Long,
                    embdPath: String,
                    metaPath: String,
                    nBatch: Int,
                ): Boolean = true

                override fun close(instance: SmolLM, modelPtr: Long) {
                    closeCalls++
                }

                override fun startCompletion(instance: SmolLM, modelPtr: Long, prompt: String) {
                    queue = ArrayDeque(listOf("x", "[EOG]"))
                }

                override fun completionLoop(instance: SmolLM, modelPtr: Long): String = queue.removeFirst()

                override fun completionLoopBatch(instance: SmolLM, modelPtr: Long, maxTokens: Int): String =
                    completionLoop(instance, modelPtr)

                override fun stopCompletion(instance: SmolLM, modelPtr: Long) = Unit

                override fun clearKvCache(instance: SmolLM, modelPtr: Long) = Unit
            }
        }

        val edgeScope = LLMEdgeScope(this, 1)
        val client =
            TextClient(
                context = context,
                scope = edgeScope,
                config = LLMEdgeConfig(textCacheSize = 1, textCacheMemoryMb = 64),
                modelResolver = resolver,
            )

        try {
            client.generate(
                prompt = "first",
                model = modelSpec,
                options = TextModelOptions(contextSize = 1024),
            )
            client.generate(
                prompt = "second",
                model = modelSpec,
                options = TextModelOptions(contextSize = 2048),
            )

            advanceUntilIdle()
            assertEquals(1, closeCalls)

            client.close()
            advanceUntilIdle()
            assertEquals(2, closeCalls)
        } finally {
            edgeScope.close()
        }
    }
}