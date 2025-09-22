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
import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import kotlinx.coroutines.flow.toList
import kotlinx.coroutines.test.runTest
import org.junit.After
import org.junit.Before
import org.junit.Test
import org.junit.runner.RunWith
import org.junit.Assume.assumeTrue
import android.os.Build
import java.io.File
import java.io.FileOutputStream

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

            val located = findInAssets(appContext) ?: findInAssets(instrContext)
                ?: error("No GGUF model found in assets. Place a .gguf model in llmedge/src/main/assets or llmedge/src/androidTest/assets (optionally under 'models').")

            val (dir, file) = located
            val assetPath = if (dir.isEmpty()) file else "$dir/$file"
            modelFile = copyAssetToCache(appContext, assetPath)

            // Avoid loading and running heavy native inference on emulators / unsupported ABIs
            if (!isEmulator() && isSupportedAbi()) {
                smolLM.load(
                    modelFile.absolutePath,
                    SmolLM.InferenceParams(
                        minP,
                        temperature,
                        storeChats = true,
                        contextSize = 0,
                        chatTemplate,
                        numThreads = 1, // keep conservative for test stability
                        useMmap = true,
                        useMlock = false,
                    ),
                )
                smolLM.addSystemPrompt(systemPrompt)
            }
        }

    @Test
    fun getResponse_AsFlow_works() =
        runTest {
            // Skip on emulator/x86_64 where native inference may not be stable
            assumeTrue("Skipping inference test on emulator/unsupported ABI", !isEmulator() && isSupportedAbi())
            val responseFlow = smolLM.getResponseAsFlow(query)
            val responseTokens = responseFlow.toList()
            assert(responseTokens.isNotEmpty())
        }

    @Test
    fun getResponseAsFlowGenerationSpeed_works() =
        runTest {
            // Skip on emulator/x86_64 where native inference may not be stable
            assumeTrue("Skipping inference test on emulator/unsupported ABI", !isEmulator() && isSupportedAbi())
            val speedBeforePrediction = smolLM.getResponseGenerationSpeed().toInt()
            smolLM.getResponseAsFlow(query).toList()
            val speedAfterPrediction = smolLM.getResponseGenerationSpeed().toInt()
            assert(speedBeforePrediction == 0)
            assert(speedAfterPrediction > 0)
        }

    @Test
    fun getContextSize_works() =
        runTest {
            val ggufReader = GGUFReader()
            ggufReader.load(modelFile.absolutePath)
            val contextSize = ggufReader.getContextSize()
            assert(contextSize != null && contextSize > 0)
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
