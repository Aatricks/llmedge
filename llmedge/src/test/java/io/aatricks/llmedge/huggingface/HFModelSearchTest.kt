package io.aatricks.llmedge.huggingface

import io.ktor.client.HttpClient
import io.ktor.client.plugins.contentnegotiation.ContentNegotiation
import io.ktor.serialization.kotlinx.json.json
import io.ktor.client.engine.mock.MockEngine
import io.ktor.client.engine.mock.respond
import io.ktor.http.HttpHeaders
import io.ktor.http.HttpStatusCode
import io.ktor.http.headersOf
import kotlinx.coroutines.runBlocking
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test
import java.util.concurrent.atomic.AtomicInteger

class HFModelSearchTest {
    @Test
    fun `searchModels honors Link next pagination`() = runBlocking {
        val callIndex = AtomicInteger(0)
        val nextUrl = "https://huggingface.co/api/models?search=abc&page=2"
        val engine = MockEngine { request ->
            val idx = callIndex.getAndIncrement()
            if (idx == 0) {
                val headers = headersOf(
                    HttpHeaders.Link to listOf("<$nextUrl>; rel=\"next\""),
                    HttpHeaders.ContentType to listOf("application/json"),
                )
                respond(content = "[]".toByteArray(), status = HttpStatusCode.OK, headers = headers)
            } else if (idx == 1) {
                // Second call should use the nextUrl provided
                if (request.url.toString() != nextUrl) {
                    throw AssertionError("Expected next url to be used for pagination, got ${'$'}{request.url}")
                }
                respond(content = "[]".toByteArray(), status = HttpStatusCode.OK, headers = headersOf(HttpHeaders.ContentType to listOf("application/json")))
            } else {
                respond("[]", HttpStatusCode.OK)
            }
        }
        val client = HttpClient(engine) {
            install(ContentNegotiation) { json() }
        }
        val search = HFModelSearch(client)

        val first = search.searchModels(query = "abc")
        assertEquals(0, first.size)
        val second = search.searchModels(query = "abc")
        assertEquals(0, second.size)
    }

    @Test
    fun `searchModels throws IllegalStateException on non-2xx status`() = runBlocking {
        val engine = MockEngine { request ->
            respond(content = "Error body".toByteArray(), status = HttpStatusCode.InternalServerError)
        }
        val client = HttpClient(engine) {
            install(ContentNegotiation) { json() }
        }
        val search = HFModelSearch(client)
        try {
            search.searchModels(query = "abc")
            org.junit.Assert.fail("Expected searchModels to throw IllegalStateException")
        } catch (e: IllegalStateException) {
            assertTrue("Expected status code in message", e.message?.contains("500") == true)
        }
    }

    @Test
    fun `searchModels resets pagination when query or filters change`() = runBlocking {
        val callUrls = mutableListOf<String>()
        val nextUrl = "https://huggingface.co/api/models?search=abc&page=2"
        val engine = MockEngine { request ->
            callUrls.add(request.url.toString())
            val headers = headersOf(
                HttpHeaders.Link to listOf("<$nextUrl>; rel=\"next\""),
                HttpHeaders.ContentType to listOf("application/json"),
            )
            respond(content = "[]".toByteArray(), status = HttpStatusCode.OK, headers = headers)
        }
        val client = HttpClient(engine) {
            install(ContentNegotiation) { json() }
        }
        val search = HFModelSearch(client)

        // First call with query "abc"
        search.searchModels(query = "abc")
        // Second call with DIFFERENT query "xyz" -> pagination should reset and request listModelsEndpoint
        search.searchModels(query = "xyz")

        assertEquals(2, callUrls.size)
        // Check that the second call did not use nextUrl because the query changed
        assertTrue("Second URL should not be the next page URL", callUrls[1] != nextUrl)
        assertTrue("Second URL should query xyz", callUrls[1].contains("search=xyz"))
    }
}
