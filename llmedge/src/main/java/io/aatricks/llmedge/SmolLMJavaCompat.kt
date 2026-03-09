package io.aatricks.llmedge

import android.content.Context
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.asCoroutineDispatcher
import kotlinx.coroutines.launch
import kotlinx.coroutines.runBlocking
import kotlinx.coroutines.withContext
import kotlinx.coroutines.CoroutineDispatcher
import kotlinx.coroutines.flow.collect
import java.util.concurrent.Executors

/**
 * Small Java-friendly compatibility helpers for `SmolLM`.
 *
 * Purpose: expose blocking, callback and builder-style APIs so Java callers don't
 * need to work directly with Kotlin coroutines, Flows or default args.
 */
object SmolLMJavaCompat {
    // background dispatcher for async helpers; daemon threads so they don't block process exit
    private val bgDispatcher: CoroutineDispatcher = Executors.newCachedThreadPool { r ->
        Thread(r, "SmolLM-JavaCompat").apply { isDaemon = true }
    }.asCoroutineDispatcher()

    /** Blocking load wrapper. Call this from a background thread or another safe context. */
    @JvmStatic
    fun loadBlocking(smol: SmolLM, modelPath: String, params: SmolLM.InferenceParams? = null) {
        runBlocking {
            if (params == null) smol.load(modelPath) else smol.load(modelPath, params)
        }
    }

    /** Blocking helper that downloads from HF (uses Context) and loads the model. */
    @JvmStatic
    fun loadFromHfBlocking(
        smol: SmolLM,
        context: Context,
        modelId: String,
        revision: String = "main",
        token: String? = null,
        params: SmolLM.InferenceParams? = null
    ) {
        runBlocking {
            // library's loadFromHuggingFace already calls load(), so just call it directly
            smol.loadFromHuggingFace(
                context = context,
                modelId = modelId,
                revision = revision,
                token = token,
                params = params ?: SmolLM.InferenceParams()
            )
        }
    }

    /** Asynchronous load with a Java callback. Runs on a background thread. */
    @JvmStatic
    fun loadAsync(smol: SmolLM, modelPath: String, params: SmolLM.InferenceParams?, callback: LoadCallback) {
        CoroutineScope(bgDispatcher).launch {
            try {
                if (params == null) smol.load(modelPath) else smol.load(modelPath, params)
                callback.onSuccess()
            } catch (t: Throwable) {
                callback.onError(t)
            }
        }
    }

    /** Callback interface for asynchronous loads. */
    interface LoadCallback {
        fun onSuccess()
        fun onError(t: Throwable)
    }

    /** Java-friendly builder for SmolLM.InferenceParams. */
    class InferenceParamsBuilder {
        private var minP: Float = 0.1f
        private var temperature: Float = 0.8f
        private var storeChats: Boolean = true
        private var contextSize: Long? = null
        private var chatTemplate: String? = null
        private var numThreads: Int = 4
        private var generationThreads: Int? = null
        private var useMmap: Boolean = true
        private var useMlock: Boolean = false
        private var thinkingMode: SmolLM.ThinkingMode = SmolLM.ThinkingMode.DEFAULT
        private var reasoningBudget: Int? = null

        fun setMinP(v: Float) = apply { minP = v }
        fun setTemperature(v: Float) = apply { temperature = v }
        @Deprecated(
            message = "Prefer edge.text.session(...) or leave storeChats disabled in new code; native storeChats is kept for targeted compatibility flows.",
            replaceWith = ReplaceWith("this"),
        )
        fun setStoreChats(v: Boolean) = apply { storeChats = v }
        fun setContextSize(v: Long?) = apply { contextSize = v }
        fun setChatTemplate(v: String?) = apply { chatTemplate = v }
        fun setNumThreads(v: Int) = apply { numThreads = v }
        fun setGenerationThreads(v: Int?) = apply { generationThreads = v }
        fun setUseMmap(v: Boolean) = apply { useMmap = v }
        fun setUseMlock(v: Boolean) = apply { useMlock = v }
        fun setThinkingMode(v: SmolLM.ThinkingMode) = apply { thinkingMode = v }
        fun setReasoningBudget(v: Int?) = apply { reasoningBudget = v }

        @JvmOverloads
        fun build(): SmolLM.InferenceParams = SmolLM.InferenceParams(
            minP = minP,
            temperature = temperature,
            storeChats = storeChats,
            contextSize = contextSize,
            chatTemplate = chatTemplate,
            numThreads = numThreads,
            generationThreads = generationThreads,
            useMmap = useMmap,
            useMlock = useMlock,
            thinkingMode = thinkingMode,
            reasoningBudget = reasoningBudget
        )
    }

    /** Listener for streaming responses (consumed from Java). */
    interface StreamListener {
        fun onChunk(chunk: String)
        fun onComplete()
        fun onError(t: Throwable)
    }

    /** Collects the coroutine Flow returned by getResponseAsFlow and forwards pieces to the listener. */
    @JvmStatic
    fun streamResponse(smol: SmolLM, query: String, listener: StreamListener) {
        CoroutineScope(bgDispatcher).launch {
            try {
                val flow = smol.getResponseAsFlow(query)
                flow.collect { piece ->
                    // SmolLM.getResponseAsFlow yields pieces and uses [EOG] marker; forward everything except marker
                    if (piece == "[EOG]") return@collect
                    listener.onChunk(piece)
                }
                listener.onComplete()
            } catch (t: Throwable) {
                listener.onError(t)
            }
        }
    }

    /** Java-friendly helper to export model state bytes (KV cache + full state). */
    @JvmStatic
    fun getStateBytes(smol: SmolLM): ByteArray? = smol.getStateBytes()

    /** Java-friendly helper to import model state bytes (KV cache + full state). */
    @JvmStatic
    fun setStateBytes(smol: SmolLM, state: ByteArray): Boolean = smol.setStateBytes(state)

    /** Java-friendly helper to export sequence state bytes (per-sequence KV cache). */
    @JvmStatic
    fun getSequenceStateBytes(smol: SmolLM, seqId: Int): ByteArray? = smol.getSequenceStateBytes(seqId)

    /** Java-friendly helper to import sequence state bytes (per-sequence KV cache). */
    @JvmStatic
    fun setSequenceStateBytes(smol: SmolLM, seqId: Int, state: ByteArray): Boolean = smol.setSequenceStateBytes(seqId, state)

    /** Java-friendly helper to clear KV cache in the current model context. */
    @JvmStatic
    fun clearKvCache(smol: SmolLM) = smol.clearKvCache()

    /** Shutdown helper for internal dispatcher (optional). */
    @JvmStatic
    fun shutdownCompatDispatcher() {
        try {
            (bgDispatcher as? java.io.Closeable)?.close()
        } catch (_: Throwable) {
        }
    }
}
