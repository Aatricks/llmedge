package io.aatricks.llmedge.text

import android.content.Context
import androidx.test.core.app.ApplicationProvider
import io.aatricks.llmedge.LLMEdge
import io.aatricks.llmedge.LLMEdgeConfig
import io.aatricks.llmedge.model.ModelSpec
import java.io.File
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.SupervisorJob
import kotlinx.coroutines.runBlocking
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Assume
import org.junit.Test
import org.junit.runner.RunWith
import org.robolectric.RobolectricTestRunner
import org.robolectric.annotation.Config

@RunWith(RobolectricTestRunner::class)
@Config(sdk = [34])
class ChatSessionLinuxE2ETest {

    private val MODEL_PATH_ENV = "LLMEDGE_TEST_TEXT_MODEL_PATH"
    private val LIB_PATH_ENV = "LLMEDGE_BUILD_NATIVE_LIB_PATH"

    @Test
    fun `llmedge text session runs multi turn chat on linux`() = runBlocking {
        val modelPath = System.getenv(MODEL_PATH_ENV) ?: System.getProperty(MODEL_PATH_ENV)
        println("[ChatSessionLinuxE2ETest] modelPath=$modelPath")
        Assume.assumeTrue("No text test model specified in $MODEL_PATH_ENV", !modelPath.isNullOrBlank())

        val libPath =
            System.getenv(LIB_PATH_ENV)
                ?: System.getProperty(LIB_PATH_ENV)
                ?: "${System.getProperty("user.dir")}/llmedge/build/native/linux-x86_64/libsmollm.so"
        Assume.assumeTrue("Native library not found at $libPath", File(libPath).exists())
        Assume.assumeTrue(
            "Native loading is disabled",
            System.getProperty("llmedge.disableNativeLoad") != "true"
        )

        val context = ApplicationProvider.getApplicationContext<Context>()
        val edge =
            LLMEdge.create(
                context = context,
                scope = CoroutineScope(SupervisorJob()),
                config = LLMEdgeConfig(textUseVulkan = false),
            )

        try {
            val session =
                edge.text.session(
                    model = ModelSpec.localFile(modelPath!!),
                    memory = ConversationWindow(maxTurns = 6, stripThinkTags = true),
                    systemPrompt = "You are a concise assistant.",
                    options = TextModelOptions(contextSize = 2048, temperature = 0.7f, useVulkan = false),
                )

            session.prepare()
            val first = session.reply("Say hello in one sentence.", maxTokens = 48)
            val second = session.reply("What did I just ask you to do?", maxTokens = 48)

            println("[ChatSessionLinuxE2ETest] first=$first")
            println("[ChatSessionLinuxE2ETest] second=$second")

            assertTrue("First response should not be empty", first.isNotBlank())
            assertTrue("Second response should not be empty", second.isNotBlank())
            assertEquals("Expected two user/assistant turns in history", 4, session.historySnapshot().size)
        } finally {
            edge.close()
        }
    }
}
