package io.aatricks.llmedge

import java.io.File
import kotlinx.coroutines.test.runTest
import org.junit.After
import org.junit.Assert.assertEquals
import org.junit.Assert.assertNotNull
import org.junit.Before
import org.junit.Test
import io.aatricks.llmedge.runtime.GGUFReader
import io.aatricks.llmedge.text.runtime.SmolLM

class SmolLMLoadTest {
    private fun createTempGgufFile(): File =
        File.createTempFile("llmedge-smollm", ".gguf").apply {
            writeBytes(byteArrayOf('G'.code.toByte(), 'G'.code.toByte(), 'U'.code.toByte(), 'F'.code.toByte(), 0x00))
            deleteOnExit()
        }

    @Before
    fun setUp() {
        System.setProperty("llmedge.disableNativeLoad", "true")
        // Stub GGUFReader native bridge so SmolLM.load() can execute without native libs
        GGUFReader.overrideNativeBridgeForTests { _ ->
            object : GGUFReader.NativeBridge {
                override fun getGGUFContextNativeHandle(modelPath: String): Long = 42L
                override fun getContextSize(nativeHandle: Long): Long = 4096L
                override fun getChatTemplate(nativeHandle: Long): String = "<|im_start|>system {{content}}<|im_end|>"
                override fun getArchitecture(nativeHandle: Long): String = "llama"
                override fun getParameterCount(nativeHandle: Long): String = "7B"
                override fun getModelName(nativeHandle: Long): String = "TestModel"
                override fun releaseGGUFContext(nativeHandle: Long) {}
            }
        }
    }

    @After
    fun tearDown() {
        GGUFReader.resetNativeBridgeForTests()
        SmolLM.resetNativeBridgeForTests()
        System.clearProperty("llmedge.disableNativeLoad")
    }

    @Test
    fun `load resolves context and template and applies reasoning options`() = runTest {
        var capturedCtx: Long = -1
        var capturedTemplate: String? = null
        val configuredThreads = mutableListOf<Pair<Int, Int>>()
        val setReasoningArgs = mutableListOf<Pair<Boolean, Int>>()

        SmolLM.overrideNativeBridgeForTests { _ ->
            object : SmolLM.NativeBridge {
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
                    kvCacheTypeK: Int,
                    kvCacheTypeV: Int,
                    nGpuLayers: Int,
                ): Long {
                    capturedCtx = contextSize
                    capturedTemplate = chatTemplate
                    return 123L
                }

                override fun configureThreading(
                    instance: SmolLM,
                    modelPtr: Long,
                    generationThreads: Int,
                    promptThreads: Int,
                ) {
                    configuredThreads += generationThreads to promptThreads
                }

                override fun setReasoningOptions(instance: SmolLM, modelPtr: Long, disableThinking: Boolean, reasoningBudget: Int) {
                    setReasoningArgs.add(disableThinking to reasoningBudget)
                }

                override fun addChatMessage(instance: SmolLM, modelPtr: Long, message: String, role: String) {}
                override fun getResponseGenerationSpeed(instance: SmolLM, modelPtr: Long): Float = 0f
                override fun getResponseGeneratedTokenCount(instance: SmolLM, modelPtr: Long): Long = 0L
                override fun getResponseGenerationDurationMicros(instance: SmolLM, modelPtr: Long): Long = 0L
                override fun getContextSizeUsed(instance: SmolLM, modelPtr: Long): Int = 0
                override fun getNativeModelPtr(instance: SmolLM, modelPtr: Long): Long = 99L
                override fun nativeDecodePreparedEmbeddings(instance: SmolLM, modelPtr: Long, embdPath: String, metaPath: String, nBatch: Int): Boolean = true
                override fun close(instance: SmolLM, modelPtr: Long) {}
                override fun startCompletion(instance: SmolLM, modelPtr: Long, prompt: String) {}
                override fun completionLoop(instance: SmolLM, modelPtr: Long): String = "[EOG]"
                override fun completionLoopBatch(instance: SmolLM, modelPtr: Long, maxTokens: Int): String = "[EOG]"
                override fun stopCompletion(instance: SmolLM, modelPtr: Long) {}
                override fun clearKvCache(instance: SmolLM, modelPtr: Long) {}
            }
        }

        val smol = SmolLM()
        val modelFile = createTempGgufFile()
        val params = SmolLM.InferenceParams(
            contextSize = null,
            chatTemplate = null,
            numThreads = 6,
            generationThreads = 2,
            thinkingMode = SmolLM.ThinkingMode.DEFAULT,
            reasoningBudget = null,
        )

        smol.load(modelFile.absolutePath, params)

        // Verify resolved metadata from GGUFReader was used
        assertEquals(4096L, capturedCtx)
        assertEquals("<|im_start|>system {{content}}<|im_end|>", capturedTemplate)
        // Verify reasoning options applied at least once
        assertNotNull(smol.getThinkingMode()) // ensure instance is usable
        assertEquals(listOf(2 to 6), configuredThreads)
        // We can't assert exact calls, but at least one call should have been recorded
        assertEquals(true, setReasoningArgs.isNotEmpty())
    }
}
