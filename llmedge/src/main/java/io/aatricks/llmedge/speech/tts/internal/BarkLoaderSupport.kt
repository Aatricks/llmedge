package io.aatricks.llmedge.speech.tts.internal

import android.content.Context
import io.aatricks.llmedge.core.ModelLoadException
import io.aatricks.llmedge.model.ModelFileValidator
import io.aatricks.llmedge.huggingface.HuggingFaceHub
import io.aatricks.llmedge.speech.tts.BarkTTS
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext

internal object BarkLoaderSupport {
    fun checkBindings(check: () -> Boolean): Boolean =
        try {
            check()
        } catch (_: UnsatisfiedLinkError) {
            false
        }

    fun loadFromPath(
        modelPath: String,
        create: (absolutePath: String, seed: Int, temperature: Float, fineTemperature: Float, verbosity: Int) -> Long,
        seed: Int,
        temperature: Float,
        fineTemperature: Float,
        verbosity: Int,
    ): BarkTTS {
        val validatedModel = ModelFileValidator.requireReadableFile(modelPath, "Bark model")
        val handle =
            create(
                validatedModel.absolutePath,
                seed,
                temperature,
                fineTemperature,
                verbosity,
            )

        if (handle == 0L) {
            throw ModelLoadException(
                validatedModel.absolutePath,
                "The native Bark loader returned an invalid handle.",
            )
        }

        return BarkTTS.createFromHandleForRuntime(handle)
    }

    suspend fun loadWithContext(
        context: Context,
        modelPath: String,
        loadFromPath: suspend (String, Int, Float, Float, Int) -> BarkTTS,
        seed: Int,
        temperature: Float,
        fineTemperature: Float,
        verbosity: Int,
    ): BarkTTS =
        withContext(Dispatchers.IO) {
            val actualPath =
                ModelFileValidator.resolveReadableFile(
                    context,
                    modelPath,
                    "Bark model",
                ).absolutePath

            loadFromPath(actualPath, seed, temperature, fineTemperature, verbosity)
        }

    suspend fun loadFromHuggingFace(
        context: Context,
        modelId: String,
        filename: String,
        seed: Int,
        temperature: Float,
        fineTemperature: Float,
        verbosity: Int,
        token: String?,
        loadFromPath: suspend (String, Int, Float, Float, Int) -> BarkTTS,
    ): BarkTTS =
        withContext(Dispatchers.IO) {
            val result =
                HuggingFaceHub.ensureModelOnDisk(
                    context = context,
                    modelId = modelId,
                    filename = filename,
                    token = token,
                )

            loadFromPath(result.file.absolutePath, seed, temperature, fineTemperature, verbosity)
        }
}
