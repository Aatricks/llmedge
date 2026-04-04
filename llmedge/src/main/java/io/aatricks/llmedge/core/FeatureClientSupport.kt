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

internal class FeatureClientFactory<T, TDependencies>(
    private val build: (FeatureContext, ClientBootstrapContext?, TDependencies) -> T,
) {
    fun create(
        context: Context,
        scope: CoroutineScope,
        config: LLMEdgeConfig,
        modelRepository: ModelRepository,
        dependencies: TDependencies,
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
                dependencies,
            )
        }

    fun forTesting(
        context: Context,
        scope: LLMEdgeScope,
        config: LLMEdgeConfig,
        modelRepository: ModelRepository,
        dependencies: TDependencies,
        ownedBootstrap: ClientBootstrapContext? = null,
    ): T =
        build(
            featureContextForTesting(
                context = context,
                scope = scope,
                config = config,
                modelRepository = modelRepository,
            ),
            ownedBootstrap,
            dependencies,
        )
}

internal fun <T> featureClientFactory(
    build: (FeatureContext, ClientBootstrapContext?) -> T,
): FeatureClientFactory<T, Unit> =
    FeatureClientFactory { featureContext, bootstrap, _ ->
        build(featureContext, bootstrap)
    }

internal fun <T, TDependencies> featureClientFactory(
    build: (FeatureContext, ClientBootstrapContext?, TDependencies) -> T,
): FeatureClientFactory<T, TDependencies> = FeatureClientFactory(build)

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
