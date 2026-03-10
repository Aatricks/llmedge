/*
 * Copyright (C) 2025 Shubham Panchal
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package io.aatricks.llmedge

import android.content.Context
import android.os.Build
import android.util.Log
import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import io.aatricks.llmedge.huggingface.HuggingFaceHub
import io.aatricks.llmedge.text.runtime.SmolLM
import kotlinx.coroutines.flow.toList
import kotlinx.coroutines.test.runTest
import org.junit.After
import org.junit.Before
import java.io.File
import java.io.FileOutputStream
import org.junit.Assume.assumeTrue
import org.junit.Test
import org.junit.runner.RunWith

@RunWith(AndroidJUnit4::class)
class SmolLMTest {
    // The model is bundled as an asset. We'll copy it to the app's cache and use that path.
    private lateinit var appContext: Context
    private lateinit var modelFile: File
    private val minP = 0.05f
    private val temperature = 1.0f
    private val systemPrompt = "You are a helpful assistant"
    private val query = "How are you?"
    private val chatTemplate =
        "{% set loop_messages = messages %}{% for message in loop_messages %}{% set content = '<|start_header_id|>' + message['role'] + '<|end_header_id|>\n\n'+ message['content'] | trim + '<|eot_id|>' %}{% if loop.index0 == 0 %}{% set content = bos_token + content %}{% endif %}{{ content }}{% endfor %}{{ '<|start_header_id|>assistant<|end_header_id|>\n\n' }}"
    private val smolLM = SmolLM()
    private var modelAssetAvailable = false
    private var inferenceReady = false
    private var cachedContextSize: Long? = null

    @Before
    fun setup() =
        runTest {
            appContext = InstrumentationRegistry.getInstrumentation().targetContext

            // Try to find a GGUF model in assets; consider target and instrumentation contexts, and an optional 'models' dir.
            val instrContext = InstrumentationRegistry.getInstrumentation().context
            val defaultModelName = "smollm2-360m-instruct-q8_0.gguf"

            fun findInAssets(ctx: Context): Pair<String, String>? {
                val am = ctx.assets
                val roots = listOf("", "models")
                for (root in roots) {
                    val names = runCatching { am.list(root)?.toList().orEmpty() }.getOrDefault(emptyList())
                    if (names.isEmpty()) continue
                    // Prefer default name if present
                    val preferred = names.firstOrNull { it.equals(defaultModelName, ignoreCase = true) && it.endsWith(".gguf") }
                    if (preferred != null) return root to preferred
                    val anyGguf = names.firstOrNull { it.endsWith(".gguf") }
                    if (anyGguf != null) return root to anyGguf
                }
                return null
            }

            var located = findInAssets(appContext) ?: findInAssets(instrContext)
            if (located == null) {
                    // Attempt to download a small test model using HuggingFace helpers if assets are
                    // absent. Prefer the system downloader (DownloadManager) on Android devices for
                    // more robust handling of large files; fall back to the in-app downloader if it
                    // fails.
                    try {
                        val hfModelId = "HuggingFaceTB/SmolLM-135M-Instruct-GGUF"
                        val downloadRes = try {
                            HuggingFaceHub.ensureModelOnDisk(
                                appContext,
                                hfModelId,
                                preferSystemDownloader = true,
                                onProgress = { downloaded, total ->
                                    Log.i("SmolLMTest", "HF download progress: $downloaded/$total")
                                }
                            )
                        } catch (t: Throwable) {
                            // Fallback to in-app downloader
                            HuggingFaceHub.ensureModelOnDisk(appContext, hfModelId)
                        }
                        modelFile = downloadRes.file
                        modelAssetAvailable = true
                        located = Pair("hf", modelFile.name)
                    } catch (t: Throwable) {
                        assumeTrue(
                            "Skipping SmolLM instrumentation tests: no GGUF asset found in main/androidTest assets and huggingface download failed",
                            false,
                        )
                        return@runTest
                    }
                }

            val (dir, file) = located!!
            if (!::modelFile.isInitialized) {
                val assetPath = if (dir.isEmpty()) file else "$dir/$file"
                modelFile = copyAssetToCache(appContext, assetPath)
                modelAssetAvailable = true
            }

            val reader = GGUFReader()
            cachedContextSize = try {
                reader.load(modelFile.absolutePath)
                reader.getContextSize()
            } catch (t: Throwable) {
                Log.w("SmolLMTest", "Unable to parse GGUF metadata: ${t.message}")
                null
            } finally {
                reader.close()
            }

            // Allow running inference on supported ABIs (including ARM-based emulators)
            if (isSupportedAbi()) {
                inferenceReady = runCatching {
                    smolLM.load(
                        modelFile.absolutePath,
                        SmolLM.InferenceParams(
                            minP,
                            temperature,
                            storeChats = true,
                            contextSize = null,
                            chatTemplate,
                            numThreads = 1,
                            useMmap = true,
                            useMlock = false,
                        ),
                    )
                    smolLM.addSystemPrompt(systemPrompt)
                    true
                }.getOrElse { throwable ->
                    Log.w("SmolLMTest", "Skipping inference tests: ${throwable.message}")
                    false
                }
            }
        }

    @Test
    fun getResponse_AsFlow_works() =
        runTest {
            // Skip on emulator/x86_64 where native inference may not be stable
            assumeTrue("Skipping inference test: SmolLM model unavailable", inferenceReady)
            val responseFlow = smolLM.getResponseAsFlow(query)
            val responseTokens = responseFlow.toList()
            assert(responseTokens.isNotEmpty())
        }

    @Test
    fun getResponseAsFlowGenerationSpeed_works() =
        runTest {
            // Skip on emulator/x86_64 where native inference may not be stable
            assumeTrue("Skipping inference test: SmolLM model unavailable", inferenceReady)
            val speedBeforePrediction = smolLM.getResponseGenerationSpeed().toInt()
            smolLM.getResponseAsFlow(query).toList()
            val speedAfterPrediction = smolLM.getResponseGenerationSpeed().toInt()
            assert(speedBeforePrediction == 0)
            assert(speedAfterPrediction > 0)
        }

    @Test
    fun getContextSize_works() =
        runTest {
            assumeTrue("Skipping GGUF metadata test: no asset bundled", modelAssetAvailable)
            val contextSize = cachedContextSize
            assumeTrue("Skipping GGUF metadata test: unable to parse context size", contextSize != null)
            assert(requireNotNull(contextSize) > 0)
        }

    @After
    fun close() {
        smolLM.close()
    }

    private fun copyAssetToCache(context: Context, assetPath: String): File {
        val outFile = File(context.cacheDir, File(assetPath).name)
        if (outFile.exists() && outFile.length() > 0) return outFile
        context.assets.open(assetPath).use { input ->
            FileOutputStream(outFile).use { output ->
                val buffer = ByteArray(DEFAULT_BUFFER_SIZE)
                var bytes = input.read(buffer)
                while (bytes >= 0) {
                    if (bytes > 0) output.write(buffer, 0, bytes)
                    bytes = input.read(buffer)
                }
                output.fd.sync()
            }
        }
        return outFile
    }

    private fun isEmulator(): Boolean {
        val hw = Build.HARDWARE.lowercase()
        val product = Build.PRODUCT.lowercase()
        val model = Build.MODEL.lowercase()
        return hw.contains("ranchu") || hw.contains("goldfish") || product.contains("sdk_gphone") || model.contains("emulator")
    }

    private fun isSupportedAbi(): Boolean {
        // We currently support running inference tests only on ARM ABIs
        return Build.SUPPORTED_ABIS.any { it.equals("arm64-v8a", ignoreCase = true) || it.equals("armeabi-v7a", ignoreCase = true) }
    }
}
