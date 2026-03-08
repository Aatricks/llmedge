package io.aatricks.llmedge

import android.content.Context
import androidx.test.core.app.ApplicationProvider
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.SupervisorJob
import org.junit.Assert.assertNotNull
import org.junit.Test
import org.junit.runner.RunWith
import org.robolectric.RobolectricTestRunner
import org.robolectric.annotation.Config

@RunWith(RobolectricTestRunner::class)
@Config(sdk = [34])
class LLMEdgeTest {
    @Test
    fun `create exposes domain clients`() {
        val context = ApplicationProvider.getApplicationContext<Context>()
        val edge = LLMEdge.create(context, CoroutineScope(SupervisorJob()))

        try {
            assertNotNull(edge.models)
            assertNotNull(edge.text)
            assertNotNull(edge.speech)
            assertNotNull(edge.image)
            assertNotNull(edge.vision)
            assertNotNull(edge.rag)
        } finally {
            edge.close()
        }
    }
}
