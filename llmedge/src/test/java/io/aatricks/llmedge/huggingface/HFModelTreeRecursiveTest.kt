package io.aatricks.llmedge.huggingface

import io.ktor.client.HttpClient
import io.ktor.client.engine.mock.MockEngine
import io.ktor.client.engine.mock.respond
import io.ktor.client.plugins.contentnegotiation.ContentNegotiation
import io.ktor.http.HttpHeaders
import io.ktor.http.headersOf
import io.ktor.serialization.kotlinx.json.json
import kotlinx.coroutines.runBlocking
import kotlinx.serialization.json.Json
import org.junit.Assert.assertEquals
import org.junit.Assert.assertNotNull
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.Test

/**
 * Regression test for nested repo-file resolution (e.g. MiniT2I's
 * "minit2i-b-16/transformer/diffusion_pytorch_model.safetensors").
 *
 * The HF tree API is non-recursive by default: the subdirectory shows up as a "directory" entry
 * and the nested file never becomes a selection candidate, so [HFFileSelectionSupport.selectRepoFile]
 * returned null and the app failed with "No file found for '<repo>' matching <nested path>".
 * Exact repo files must therefore be listed recursively.
 */
class HFModelTreeRecursiveTest {
    private val nestedPath = "minit2i-b-16/transformer/diffusion_pytorch_model.safetensors"

    // What HF returns WITHOUT ?recursive=true: the nested file is hidden behind a directory entry.
    private val nonRecursiveBody = """
        [
          {"type":"directory","path":"minit2i-b-16"},
          {"type":"file","size":123,"path":"README.md"}
        ]
    """.trimIndent()

    // What HF returns WITH ?recursive=true: the nested file is present.
    private val recursiveBody = """
        [
          {"type":"directory","path":"minit2i-b-16"},
          {"type":"file","size":123,"path":"README.md"},
          {"type":"file","size":1032534472,"path":"$nestedPath"}
        ]
    """.trimIndent()

    private fun treeReturning(
        body: String,
        capturedUrls: MutableList<String>,
    ): HFModelTree {
        val engine =
            MockEngine { request ->
                capturedUrls += request.url.toString()
                respond(
                    content = body,
                    headers = headersOf(HttpHeaders.ContentType, "application/json"),
                )
            }
        val client =
            HttpClient(engine) {
                install(ContentNegotiation) { json(Json { ignoreUnknownKeys = true }) }
            }
        return HFModelTree(client)
    }

    @Test
    fun `recursive listing requests recursive=true and exposes nested files`() =
        runBlocking {
            val urls = mutableListOf<String>()
            val files =
                treeReturning(recursiveBody, urls)
                    .getModelFileTree("MiniT2I/MiniT2I", "main", recursive = true)

            assertTrue(
                "recursive listing must request ?recursive=true, was ${urls.first()}",
                urls.first().contains("recursive=true"),
            )
            val selected = HFFileSelectionSupport.selectRepoFile(files, nestedPath, listOf(".safetensors"))
            assertNotNull("nested repo file must be selectable from a recursive listing", selected)
            assertEquals(nestedPath, selected!!.path)
        }

    @Test
    fun `non-recursive listing hides nested files (reproduces the bug)`() =
        runBlocking {
            val urls = mutableListOf<String>()
            val files =
                treeReturning(nonRecursiveBody, urls)
                    .getModelFileTree("MiniT2I/MiniT2I", "main", recursive = false)

            assertTrue(
                "default listing must not request recursion, was ${urls.first()}",
                !urls.first().contains("recursive=true"),
            )
            val selected = HFFileSelectionSupport.selectRepoFile(files, nestedPath, listOf(".safetensors"))
            assertNull("nested repo file is invisible without a recursive listing", selected)
        }
}
