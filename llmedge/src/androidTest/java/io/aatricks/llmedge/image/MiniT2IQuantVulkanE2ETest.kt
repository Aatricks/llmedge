/*
 * Copyright (C) 2026 Aatricks
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

package io.aatricks.llmedge.image

import android.graphics.Bitmap
import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import io.aatricks.llmedge.huggingface.HuggingFaceHub
import io.aatricks.llmedge.image.diffusion.GenerateParams
import io.aatricks.llmedge.image.diffusion.StableDiffusion
import io.aatricks.llmedge.image.diffusion.StableDiffusionComponentPaths
import io.aatricks.llmedge.model.ModelSpec
import kotlinx.coroutines.runBlocking
import org.junit.Assert.assertNotNull
import org.junit.Assert.assertTrue
import org.junit.Assume.assumeTrue
import org.junit.Test
import org.junit.runner.RunWith
import java.io.File
import java.io.FileOutputStream

@RunWith(AndroidJUnit4::class)
class MiniT2IQuantVulkanE2ETest {

    @Test
    fun testMiniT2IQuantVulkanE2E() = runBlocking {
        val enabled = InstrumentationRegistry.getArguments().getString("llmedge.minit2iQuantE2E") == "1"
        assumeTrue("Skipping MiniT2I quant Vulkan E2E test; enable with -e llmedge.minit2iQuantE2E 1", enabled)

        val targetContext = InstrumentationRegistry.getInstrumentation().targetContext

        // Resolve assets via MiniT2I.largeImageRequest
        val request = MiniT2I.largeImageRequest(
            prompt = "a red apple on a table, detailed",
            width = 256,
            height = 256,
            steps = 1
        )

        val textEncoderSpec = request.textEncoder as ModelSpec.HuggingFace
        val diffusionModelSpec = request.model as ModelSpec.HuggingFace

        // Download/resume files from HuggingFace
        val textEncoderResult = HuggingFaceHub.ensureModelOnDisk(
            context = targetContext,
            modelId = textEncoderSpec.repoId,
            filename = textEncoderSpec.filename,
            preferSystemDownloader = false
        )

        val diffusionResult = HuggingFaceHub.ensureRepoFileOnDisk(
            context = targetContext,
            modelId = diffusionModelSpec.repoId,
            revision = diffusionModelSpec.revision,
            filename = diffusionModelSpec.filename,
            preferSystemDownloader = false
        )

        val textEncoderFile = textEncoderResult.file
        val diffusionModelFile = diffusionResult.file

        assertTrue("Text encoder file should exist", textEncoderFile.exists() && textEncoderFile.length() > 0)
        assertTrue("Diffusion model file should exist", diffusionModelFile.exists() && diffusionModelFile.length() > 0)

        // Load model with Vulkan, Q8_0 weights and tensor rules
        val sd = StableDiffusion.load(
            context = targetContext,
            diffusionModelPath = diffusionModelFile.absolutePath,
            vaePath = null,
            t5xxlPath = textEncoderFile.absolutePath,
            nThreads = 4,
            forceVulkan = true,
            componentPaths = StableDiffusionComponentPaths(
                weightType = "q8_0",
                tensorTypeRules = ".*mask_token.*=f16"
            )
        )

        val bmp = try {
            sd.txt2img(
                GenerateParams(
                    prompt = request.prompt,
                    width = request.width,
                    height = request.height,
                    steps = request.steps,
                    cfgScale = request.cfgScale,
                    seed = 42L
                )
            )
        } finally {
            sd.close()
        }

        assertNotNull("Generated bitmap should not be null", bmp)
        assertTrue("Generated bitmap should not be empty", bmp.width > 0 && bmp.height > 0)

        // Write output to filesDir as minit2i-quant-e2e.png
        val outFile = File(targetContext.filesDir, "minit2i-quant-e2e.png")
        FileOutputStream(outFile).use { out ->
            bmp.compress(Bitmap.CompressFormat.PNG, 100, out)
        }
        assertTrue("Output image should exist and be non-empty", outFile.exists() && outFile.length() > 0)
    }
}
