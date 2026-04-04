package io.aatricks.llmedge.core

import android.content.Context
import io.aatricks.llmedge.LLMEdgeConfig
import io.aatricks.llmedge.model.ModelRepository
import kotlinx.coroutines.CoroutineScope

internal data class FeatureContext(
    val appContext: Context,
    val edgeScope: LLMEdgeScope,
    val config: LLMEdgeConfig,
    val modelRepository: ModelRepository,
)

internal inline fun <T> createOwnedFeature(
    context: Context,
    scope: CoroutineScope,
    config: LLMEdgeConfig,
    modelRepository: ModelRepository,
    build: (FeatureContext, ClientBootstrapContext) -> T,
): T =
    createOwnedClient(context, scope, config) { bootstrap ->
        build(
            FeatureContext(
                appContext = bootstrap.appContext,
                edgeScope = bootstrap.edgeScope,
                config = config,
                modelRepository = modelRepository,
            ),
            bootstrap,
        )
    }

abstract class OwnedFeatureClient internal constructor(
    featureContext: FeatureContext,
    ownedBootstrap: ClientBootstrapContext?,
) : OwnedClient(ownedBootstrap) {
    private val dependencies = featureContext

    protected val appContext: Context
        get() = dependencies.appContext

    protected val edgeScope: LLMEdgeScope
        get() = dependencies.edgeScope

    protected val config: LLMEdgeConfig
        get() = dependencies.config

    protected val modelRepository: ModelRepository
        get() = dependencies.modelRepository
}
