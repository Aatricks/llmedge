package io.aatricks.llmedge.text

import android.content.Context
import androidx.test.core.app.ApplicationProvider
import io.aatricks.llmedge.runtime.GGUFReader
import io.aatricks.llmedge.LLMEdgeConfig
import io.aatricks.llmedge.text.runtime.SmolLM
import io.aatricks.llmedge.core.LLMEdgeScope
import io.aatricks.llmedge.model.ModelResolver
import io.aatricks.llmedge.model.ModelSpec
import java.io.File
import kotlinx.coroutines.flow.toList
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
        var completionLoopCalls = 0
        val completionLoopBatchArgs = mutableListOf<Int>()

        SmolLM.overrideNativeBridgeForTests {
            object : SmolLM.NativeBridge {
                private var queue = ArrayDeque<String>()

                private fun nextPiece(): String = queue.removeFirst()

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

                override fun completionLoop(instance: SmolLM, modelPtr: Long): String {
                    completionLoopCalls++
                    return nextPiece()
                }

                override fun completionLoopBatch(instance: SmolLM, modelPtr: Long, maxTokens: Int): String {
                    completionLoopBatchArgs += maxTokens
                    return nextPiece()
                }

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
                config = LLMEdgeConfig(textCacheSize = 2, textCacheMemoryMb = 64, defaultTextBatchSize = 6),
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
            assertEquals(0, completionLoopCalls)
            assertEquals(listOf(6, 6, 6, 6), completionLoopBatchArgs)

            client.close()
            advanceUntilIdle()
            assertEquals(1, closeCalls)
        } finally {
            edgeScope.close()
        }
    }

    @Test
    fun `explicit single token batch size keeps single token loop`() = runTest {
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

        var completionLoopCalls = 0
        var completionLoopBatchCalls = 0

        SmolLM.overrideNativeBridgeForTests {
            object : SmolLM.NativeBridge {
                private var queue = ArrayDeque<String>()

                private fun nextPiece(): String = queue.removeFirst()

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

                override fun close(instance: SmolLM, modelPtr: Long) = Unit

                override fun startCompletion(instance: SmolLM, modelPtr: Long, prompt: String) {
                    queue = ArrayDeque(listOf("x", "[EOG]"))
                }

                override fun completionLoop(instance: SmolLM, modelPtr: Long): String {
                    completionLoopCalls++
                    return nextPiece()
                }

                override fun completionLoopBatch(instance: SmolLM, modelPtr: Long, maxTokens: Int): String {
                    completionLoopBatchCalls++
                    return nextPiece()
                }

                override fun stopCompletion(instance: SmolLM, modelPtr: Long) = Unit

                override fun clearKvCache(instance: SmolLM, modelPtr: Long) = Unit
            }
        }

        val edgeScope = LLMEdgeScope(this, 1)
        val client =
            TextClient(
                context = context,
                scope = edgeScope,
                config = LLMEdgeConfig(textCacheSize = 1, textCacheMemoryMb = 64, defaultTextBatchSize = 6),
                modelResolver = resolver,
            )

        try {
            assertEquals("x", client.generate(prompt = "hello", model = modelSpec, batchSize = 1))
            assertTrue(completionLoopCalls > 0)
            assertEquals(0, completionLoopBatchCalls)
        } finally {
            client.close()
            edgeScope.close()
        }
    }

    @Test
    fun `generate clears native messages before reapplying system prompt`() = runTest {
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

        val operations = mutableListOf<String>()

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

                override fun clearMessages(instance: SmolLM, modelPtr: Long) {
                    operations += "clear"
                }

                override fun setReasoningOptions(
                    instance: SmolLM,
                    modelPtr: Long,
                    disableThinking: Boolean,
                    reasoningBudget: Int,
                ) = Unit

                override fun addChatMessage(instance: SmolLM, modelPtr: Long, message: String, role: String) {
                    operations += "$role:$message"
                }

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
                override fun close(instance: SmolLM, modelPtr: Long) = Unit
                override fun startCompletion(instance: SmolLM, modelPtr: Long, prompt: String) {
                    queue = ArrayDeque(listOf("ok", "[EOG]"))
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
            client.generate(prompt = "hello", model = modelSpec, systemPrompt = "Keep it brief")
            client.generate(prompt = "again", model = modelSpec, systemPrompt = "Keep it brief")

            assertEquals(
                listOf(
                    "clear",
                    "system:Keep it brief",
                    "clear",
                    "system:Keep it brief",
                ),
                operations,
            )
        } finally {
            client.close()
            edgeScope.close()
        }
    }

    @Test
    fun `stream uses configured batched native path`() = runTest {
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

        var completionLoopCalls = 0
        val completionLoopBatchArgs = mutableListOf<Int>()

        SmolLM.overrideNativeBridgeForTests {
            object : SmolLM.NativeBridge {
                private var queue = ArrayDeque<String>()

                private fun nextPiece(): String = queue.removeFirst()

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

                override fun close(instance: SmolLM, modelPtr: Long) = Unit

                override fun startCompletion(instance: SmolLM, modelPtr: Long, prompt: String) {
                    queue = ArrayDeque(listOf("chunk", "[EOG]"))
                }

                override fun completionLoop(instance: SmolLM, modelPtr: Long): String {
                    completionLoopCalls++
                    return nextPiece()
                }

                override fun completionLoopBatch(instance: SmolLM, modelPtr: Long, maxTokens: Int): String {
                    completionLoopBatchArgs += maxTokens
                    return nextPiece()
                }

                override fun stopCompletion(instance: SmolLM, modelPtr: Long) = Unit

                override fun clearKvCache(instance: SmolLM, modelPtr: Long) = Unit
            }
        }

        val edgeScope = LLMEdgeScope(this, 1)
        val client =
            TextClient(
                context = context,
                scope = edgeScope,
                config = LLMEdgeConfig(defaultTextStreamBatchSize = 3),
                modelResolver = resolver,
            )

        try {
            val events = client.stream(prompt = "hello", model = modelSpec).toList()
            assertEquals(0, completionLoopCalls)
            assertEquals(listOf(3, 3), completionLoopBatchArgs)
            assertTrue(events.filterIsInstance<TextStreamEvent.Chunk>().map { it.value }.contains("chunk"))
        } finally {
            client.close()
            edgeScope.close()
        }
    }

    @Test
    fun `stream request batch size overrides configured default`() = runTest {
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

        val completionLoopBatchArgs = mutableListOf<Int>()

        SmolLM.overrideNativeBridgeForTests {
            object : SmolLM.NativeBridge {
                private var queue = ArrayDeque<String>()

                private fun nextPiece(): String = queue.removeFirst()

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
                override fun close(instance: SmolLM, modelPtr: Long) = Unit
                override fun startCompletion(instance: SmolLM, modelPtr: Long, prompt: String) {
                    queue = ArrayDeque(listOf("chunk", "[EOG]"))
                }
                override fun completionLoop(instance: SmolLM, modelPtr: Long): String = nextPiece()
                override fun completionLoopBatch(instance: SmolLM, modelPtr: Long, maxTokens: Int): String {
                    completionLoopBatchArgs += maxTokens
                    return nextPiece()
                }
                override fun stopCompletion(instance: SmolLM, modelPtr: Long) = Unit
                override fun clearKvCache(instance: SmolLM, modelPtr: Long) = Unit
            }
        }

        val edgeScope = LLMEdgeScope(this, 1)
        val client =
            TextClient(
                context = context,
                scope = edgeScope,
                config = LLMEdgeConfig(defaultTextStreamBatchSize = 4),
                modelResolver = resolver,
            )

        try {
            val events =
                client.stream(
                    request = TextGenerationRequest(
                        prompt = "hello",
                        model = modelSpec,
                        batchSize = 2,
                    ),
                ).toList()

            assertEquals(listOf(2, 2), completionLoopBatchArgs)
            assertTrue(events.filterIsInstance<TextStreamEvent.Chunk>().map { it.value }.contains("chunk"))
        } finally {
            client.close()
            edgeScope.close()
        }
    }

    @Test
    fun `chat session does not duplicate system prompt in rendered user prompt`() = runTest {
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

        var startedPrompt: String? = null
        val systemMessages = mutableListOf<String>()

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

                override fun addChatMessage(instance: SmolLM, modelPtr: Long, message: String, role: String) {
                    if (role == "system") {
                        systemMessages += message
                    }
                }

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
                override fun close(instance: SmolLM, modelPtr: Long) = Unit
                override fun startCompletion(instance: SmolLM, modelPtr: Long, prompt: String) {
                    startedPrompt = prompt
                    queue = ArrayDeque(listOf("reply", "[EOG]"))
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
            val reply = client.session(model = modelSpec, systemPrompt = "Be concise").reply("Hi")

            assertEquals("reply", reply)
            assertEquals(listOf("Be concise"), systemMessages)
            assertTrue(startedPrompt?.contains("User: Hi") == true)
            assertTrue(startedPrompt?.contains("System: Be concise") == false)
        } finally {
            client.close()
            edgeScope.close()
        }
    }

    @Test
    fun `text client forwards separate prompt and generation thread counts`() = runTest {
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

        val configuredThreads = mutableListOf<Pair<Int, Int>>()

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

                override fun configureThreading(
                    instance: SmolLM,
                    modelPtr: Long,
                    generationThreads: Int,
                    promptThreads: Int,
                ) {
                    configuredThreads += generationThreads to promptThreads
                }

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
                override fun getEstimatedNativeMemoryBytes(instance: SmolLM, modelPtr: Long): Long = 512L
                override fun nativeDecodePreparedEmbeddings(
                    instance: SmolLM,
                    modelPtr: Long,
                    embdPath: String,
                    metaPath: String,
                    nBatch: Int,
                ): Boolean = true
                override fun close(instance: SmolLM, modelPtr: Long) = Unit
                override fun startCompletion(instance: SmolLM, modelPtr: Long, prompt: String) {
                    queue = ArrayDeque(listOf("ok", "[EOG]"))
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
                config = LLMEdgeConfig(defaultTextThreads = 6, defaultTextGenerationThreads = 2),
                modelResolver = resolver,
            )

        try {
            client.generate(prompt = "hello", model = modelSpec)
            client.generate(
                prompt = "custom",
                model = modelSpec,
                options = TextModelOptions(numThreads = 5, generationThreads = 3),
            )

            assertEquals(listOf(2 to 6, 3 to 5), configuredThreads)
        } finally {
            client.close()
            edgeScope.close()
        }
    }

    @Test
    fun `generate retries decode failure with cpu safe runtime`() = runTest {
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

        val loadConfigs = mutableListOf<Pair<Boolean, Boolean>>()
        var closeCalls = 0

        SmolLM.overrideNativeBridgeForTests {
            object : SmolLM.NativeBridge {
                private var queue = ArrayDeque<String>()
                private var acceleratedRuntime = false

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
                    acceleratedRuntime = useVulkan || useFlashAttn
                    loadConfigs += useVulkan to useFlashAttn
                    return 1L
                }

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
                override fun getEstimatedNativeMemoryBytes(instance: SmolLM, modelPtr: Long): Long = 512L
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
                    queue = ArrayDeque(listOf("ok", "[EOG]"))
                }

                override fun completionLoop(instance: SmolLM, modelPtr: Long): String {
                    if (acceleratedRuntime) {
                        throw IllegalStateException("llama_decode() failed")
                    }
                    return queue.removeFirst()
                }

                override fun completionLoopBatch(instance: SmolLM, modelPtr: Long, maxTokens: Int): String {
                    if (acceleratedRuntime) {
                        throw IllegalStateException("llama_decode() failed")
                    }
                    return queue.removeFirst()
                }

                override fun stopCompletion(instance: SmolLM, modelPtr: Long) = Unit
                override fun clearKvCache(instance: SmolLM, modelPtr: Long) = Unit
            }
        }

        val edgeScope = LLMEdgeScope(this, 1)
        val client =
            TextClient(
                context = context,
                scope = edgeScope,
                config = LLMEdgeConfig(textUseVulkan = true, defaultUseFlashAttention = true),
                modelResolver = resolver,
            )

        try {
            val response = client.generate(prompt = "hello", model = modelSpec)

            assertEquals("ok", response)
            assertEquals(listOf(true to true, false to false), loadConfigs)
            assertEquals(1, closeCalls)
        } finally {
            client.close()
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