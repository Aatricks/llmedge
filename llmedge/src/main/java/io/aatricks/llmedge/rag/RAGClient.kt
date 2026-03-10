package io.aatricks.llmedge.rag

import android.content.Context
import android.net.Uri
import io.aatricks.llmedge.LLMEdgeConfig
import io.aatricks.llmedge.text.runtime.SmolLM
import io.aatricks.llmedge.core.LLMEdgeScope
import io.aatricks.llmedge.model.ModelResolver
import io.aatricks.llmedge.model.ModelSpec
import io.aatricks.llmedge.text.TextModelOptions
import io.aatricks.llmedge.text.toInferenceParams

class RAGSession internal constructor(
    val engine: RAGEngine,
    private val model: SmolLM,
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

    fun getLastGenerationMetrics(): SmolLM.GenerationMetrics = model.getLastGenerationMetrics()

    override fun close() {
        model.close()
    }
}

class RAGClient internal constructor(
    private val context: Context,
    private val scope: LLMEdgeScope,
    private val config: LLMEdgeConfig,
    private val resolver: ModelResolver,
) : AutoCloseable {
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
        val file = resolver.resolve(context, model)
        val smol = SmolLM(useVulkan = options.useVulkan ?: config.textUseVulkan)
        smol.load(file.absolutePath, options.toInferenceParams(config))
        val session =
            RAGSession(
                engine = RAGEngine(context, smol, splitter = splitter, embeddingConfig = embeddingConfig),
                model = smol,
            )
        return scope.resources.register(session)
    }

    override fun close() = Unit
}
