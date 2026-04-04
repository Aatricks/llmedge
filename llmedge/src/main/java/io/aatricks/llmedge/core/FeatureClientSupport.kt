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

internal fun featureContextForTesting(
    context: Context,
    scope: LLMEdgeScope,
    config: LLMEdgeConfig,
    modelRepository: ModelRepository,
): FeatureContext =
    FeatureContext(
        appContext = context,
        edgeScope = scope,
        config = config,
        modelRepository = modelRepository,
    )

internal inline fun <T> createFeatureForTesting(
    context: Context,
    scope: LLMEdgeScope,
    config: LLMEdgeConfig,
    modelRepository: ModelRepository,
    ownedBootstrap: ClientBootstrapContext? = null,
    build: (FeatureContext, ClientBootstrapContext?) -> T,
): T =
    build(
        featureContextForTesting(
            context = context,
            scope = scope,
            config = config,
            modelRepository = modelRepository,
        ),
        ownedBootstrap,
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

internal inline fun <T> createOwnedFeatureClient(
    context: Context,
    scope: CoroutineScope,
    config: LLMEdgeConfig,
    modelRepository: ModelRepository,
    build: (FeatureContext, ClientBootstrapContext?) -> T,
): T =
    createOwnedFeature(context, scope, config, modelRepository) { featureContext, bootstrap ->
        build(featureContext, bootstrap)
    }

internal inline fun <T> createFeatureClientForTesting(
    context: Context,
    scope: LLMEdgeScope,
    config: LLMEdgeConfig,
    modelRepository: ModelRepository,
    ownedBootstrap: ClientBootstrapContext? = null,
    build: (FeatureContext, ClientBootstrapContext?) -> T,
): T =
    createFeatureForTesting(context, scope, config, modelRepository, ownedBootstrap, build)

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
