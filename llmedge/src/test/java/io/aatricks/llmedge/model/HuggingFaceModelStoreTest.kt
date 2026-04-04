package io.aatricks.llmedge.model

import android.content.Context
import androidx.test.core.app.ApplicationProvider
import io.aatricks.llmedge.huggingface.HuggingFaceHub
import io.mockk.clearAllMocks
import io.mockk.coEvery
import io.mockk.coVerify
import io.mockk.mockkObject
import io.mockk.unmockkObject
import java.io.File
import kotlinx.coroutines.test.runTest
import org.junit.After
import org.junit.Assert.assertEquals
import org.junit.Before
import org.junit.Test
import org.junit.runner.RunWith
import org.robolectric.RobolectricTestRunner
import org.robolectric.annotation.Config

@RunWith(RobolectricTestRunner::class)
@Config(sdk = [34])
class DefaultModelRepositoryHuggingFaceTest {
    private lateinit var context: Context

    @Before
    fun setup() {
        context = ApplicationProvider.getApplicationContext()
        mockkObject(HuggingFaceHub)
    }

    @After
    fun tearDown() {
        clearAllMocks()
        unmockkObject(HuggingFaceHub)
    }

    @Test
    fun `resolve uses model endpoint for gguf files`() = runTest {
        val file = File.createTempFile("hf-store", ".gguf").apply { writeBytes(byteArrayOf(1)) }
        val result =
            HuggingFaceHub.ModelDownloadResult(
                requestedModelId = "repo/model",
                requestedRevision = "main",
                modelId = "repo/model",
                revision = "main",
                file = file,
                fileInfo = HuggingFaceHub.ModelFileMetadata(file.name, file.length(), null),
                fromCache = false,
                aliasApplied = false,
            )
        coEvery {
            HuggingFaceHub.ensureModelOnDisk(
                context = any(),
                modelId = any(),
                revision = any(),
                preferredQuantizations = any(),
                filename = any(),
                token = any(),
                forceDownload = any(),
                preferSystemDownloader = any(),
                onProgress = any(),
            )
        } returns result

        val repository = DefaultModelRepository()
        val resolved =
            repository.resolve(
                context,
                ModelSpec.HuggingFace(repoId = "repo/model", filename = "model.gguf"),
            )

        assertEquals(file.absolutePath, resolved.absolutePath)
        coVerify(exactly = 1) { HuggingFaceHub.ensureModelOnDisk(context = any(), modelId = any(), revision = any(), preferredQuantizations = any(), filename = any(), token = any(), forceDownload = any(), preferSystemDownloader = any(), onProgress = any()) }
        coVerify(exactly = 0) { HuggingFaceHub.ensureRepoFileOnDisk(context = any(), modelId = any(), revision = any(), filename = any(), allowedExtensions = any(), token = any(), forceDownload = any(), preferSystemDownloader = any(), onProgress = any()) }
    }

    @Test
    fun `resolve uses repo file endpoint for non gguf files`() = runTest {
        val file = File.createTempFile("hf-store", ".bin").apply { writeBytes(byteArrayOf(1)) }
        val result =
            HuggingFaceHub.ModelDownloadResult(
                requestedModelId = "repo/model",
                requestedRevision = "main",
                modelId = "repo/model",
                revision = "main",
                file = file,
                fileInfo = HuggingFaceHub.ModelFileMetadata(file.name, file.length(), null),
                fromCache = false,
                aliasApplied = false,
            )
        coEvery {
            HuggingFaceHub.ensureRepoFileOnDisk(
                context = any(),
                modelId = any(),
                revision = any(),
                filename = any(),
                allowedExtensions = any(),
                token = any(),
                forceDownload = any(),
                preferSystemDownloader = any(),
                onProgress = any(),
            )
        } returns result

        val repository = DefaultModelRepository()
        val resolved =
            repository.resolve(
                context,
                ModelSpec.HuggingFace(repoId = "repo/model", filename = "model.bin"),
            )

        assertEquals(file.absolutePath, resolved.absolutePath)
        coVerify(exactly = 0) { HuggingFaceHub.ensureModelOnDisk(context = any(), modelId = any(), revision = any(), preferredQuantizations = any(), filename = any(), token = any(), forceDownload = any(), preferSystemDownloader = any(), onProgress = any()) }
        coVerify(exactly = 1) { HuggingFaceHub.ensureRepoFileOnDisk(context = any(), modelId = any(), revision = any(), filename = any(), allowedExtensions = any(), token = any(), forceDownload = any(), preferSystemDownloader = any(), onProgress = any()) }
    }
}
