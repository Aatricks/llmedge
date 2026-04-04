package io.aatricks.llmedge.rag

import android.content.Context
import android.net.Uri
import io.aatricks.llmedge.LLMEdgeConfig
import io.aatricks.llmedge.core.LLMEdgeScope
import io.aatricks.llmedge.model.DefaultModelRepository
import io.aatricks.llmedge.model.ModelRepository
import io.aatricks.llmedge.model.ModelSpec
import io.aatricks.llmedge.text.TextModelOptions
import io.aatricks.llmedge.text.ManagedTextModel
import io.aatricks.llmedge.text.createTextRuntimePool
import io.aatricks.llmedge.text.runtime.SmolLM
import kotlinx.coroutines.CoroutineScope

class RAGSession internal constructor(
    val engine: RAGEngine,
    private val runtime: ManagedTextModel,
) : AutoCloseable {
    suspend fun init() {
        engine.init()
    }

    suspend fun indexPdf(uri: Uri): Int = engine.indexPdf(uri)

    suspend fun ask(question: String, topK: Int = 5): String = engine.ask(question, topK)

    suspend fun contextFor(question: String, topK: Int = 5): String = engine.contextFor(question, topK)

    suspend fun retrieve(question: String, topK: Int = 5) = engine.retrieve(question, topK)

    suspend fun retrievalPreview(question: String, topK: Int = 5): String =
        engine.retrievalPreview(question, topK)

    fun getLastGenerationMetrics(): SmolLM.GenerationMetrics = runtime.model.getLastGenerationMetrics()

    override fun close() {
        runtime.close()
    }
}

class RAGClient internal constructor(
    private val context: Context,
    private val scope: LLMEdgeScope,
    private val config: LLMEdgeConfig,
    private val resolver: ModelRepository,
    private val ownedScope: LLMEdgeScope? = null,
) : AutoCloseable {
    companion object {
        @JvmStatic
        @JvmOverloads
        fun create(
            context: Context,
            scope: CoroutineScope,
            config: LLMEdgeConfig = LLMEdgeConfig(),
            modelRepository: ModelRepository = DefaultModelRepository(),
        ): RAGClient {
            val appContext = context.applicationContext
            val edgeScope = LLMEdgeScope(scope, config.text.promptThreads)
            return RAGClient(appContext, edgeScope, config, modelRepository, ownedScope = edgeScope)
        }
    }

    private val runtimePool = createTextRuntimePool(context, scope, config, resolver)

    /**
     * Create a new retrieval-augmented generation session backed by a dedicated [SmolLM] instance.
     *
     * Call [RAGSession.close] when the session is no longer needed.
     *
     * @throws io.aatricks.llmedge.core.LLMEdgeException when model resolution or loading fails.
     */
    suspend fun createSession(
        model: ModelSpec = config.models.text,
        embeddingConfig: EmbeddingConfig = EmbeddingConfig(),
        splitter: TextSplitter = TextSplitter(),
        options: TextModelOptions = TextModelOptions(),
    ): RAGSession {
        val runtime = runtimePool.loadDetached(model, options)
        val session =
            RAGSession(
                engine = RAGEngine(context, runtime.model, splitter = splitter, embeddingConfig = embeddingConfig),
                runtime = runtime,
            )
        return scope.resources.register(session)
    }

    override fun close() {
        try {
            runtimePool.close()
        } finally {
            ownedScope?.close()
        }
    }
}
