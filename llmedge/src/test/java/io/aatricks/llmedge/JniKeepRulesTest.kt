package io.aatricks.llmedge

import java.io.File
import org.junit.Assert.assertTrue
import org.junit.Test

/**
 * The native layer resolves some Java members by name, which R8 in a consumer app
 * will rename or strip unless `consumer-rules.pro` keeps them. Nothing about that
 * coupling is visible from Kotlin, so this derives the required keeps from the JNI
 * sources: adding a new by-name lookup without a matching rule fails here rather
 * than in a minified release build.
 */
class JniKeepRulesTest {
    @Test
    fun `every class resolved by name from JNI is kept`() {
        val rules = consumerRules()
        val missing =
            nativeSources()
                .flatMap { findClassLookups(it.readText()) }
                .distinct()
                .filterNot { rules.contains(it) }

        assertTrue("consumer-rules.pro is missing keeps for $missing", missing.isEmpty())
    }

    @Test
    fun `every member resolved by name from JNI is kept`() {
        val rules = normalise(consumerRules())
        val missing =
            nativeSources()
                .flatMap { memberLookups(it.readText()) }
                .distinct()
                .filterNot { normalise(it) in rules }

        assertTrue("consumer-rules.pro is missing keeps for $missing", missing.isEmpty())
    }

    @Test
    fun `the jni sources are actually being scanned`() {
        val sources = nativeSources()
        assertTrue("no JNI sources found", sources.size > 5)
        assertTrue(
            "expected the Whisper segment lookup to be discovered",
            sources.any { "Whisper\$TranscriptionSegment" in it.readText() },
        )
    }

    private fun consumerRules(): String = moduleDir().resolve("consumer-rules.pro").readText()

    private fun nativeSources(): List<File> =
        moduleDir()
            .resolve("src/main/cpp")
            .walkTopDown()
            .filter { it.isFile && (it.extension == "cpp" || it.extension == "h") }
            .toList()

    private fun moduleDir(): File =
        generateSequence(File("").absoluteFile) { it.parentFile }
            .firstNotNullOfOrNull { candidate ->
                when {
                    candidate.resolve("consumer-rules.pro").isFile -> candidate
                    candidate.resolve("llmedge/consumer-rules.pro").isFile -> candidate.resolve("llmedge")
                    else -> null
                }
            } ?: error("Unable to locate the llmedge module directory")

    /** Class names passed to `FindClass`, excluding the JDK types JNI always resolves. */
    private fun findClassLookups(source: String): List<String> =
        FIND_CLASS.findAll(source)
            .map { it.groupValues[1] }
            .filter { it.startsWith("io/aatricks/") }
            .map { it.replace('/', '.') }
            .toList()

    /**
     * Declarations for members looked up by name. `GetMethodID` is only reported for
     * classes this library owns; the generic callback helper always resolves one of
     * our own callback interfaces, so its lookups always count.
     */
    private fun memberLookups(source: String): List<String> {
        val ownedClassVariables =
            FIND_CLASS_ASSIGNMENT.findAll(source)
                .filter { it.groupValues[2].startsWith("io/aatricks/") }
                .map { it.groupValues[1] }
                .toSet()

        val fromCallbackHelper =
            CALLBACK_HELPER.findAll(source).mapNotNull { declarationAt(source, it.range.last) }

        val fromGetMethodId =
            GET_METHOD_ID.findAll(source)
                .filter { it.groupValues[1] in ownedClassVariables }
                .mapNotNull { declarationAt(source, it.range.last) }

        return (fromCallbackHelper + fromGetMethodId).toList()
    }

    /** Reads the `"name", "signature"` pair that follows an opening call parenthesis. */
    private fun declarationAt(source: String, callStart: Int): String? {
        val window = source.substring(callStart, minOf(source.length, callStart + CALL_WINDOW))
        val literals = STRING_LITERAL.findAll(window).map { it.groupValues[1] }.toList()
        val signatureIndex = literals.indexOfFirst { it.startsWith("(") && it.contains(")") }
        if (signatureIndex < 1) return null
        val name = literals[signatureIndex - 1]
        val signature = literals[signatureIndex]
        val parameters = javaParameters(signature) ?: return null
        return "$name($parameters)"
    }

    /** `(IIIIF)V` -> `int, int, int, int, float`. Returns null on anything unparsed. */
    private fun javaParameters(signature: String): String? {
        val parameters = signature.substringAfter('(', "").substringBefore(')')
        if (!signature.startsWith("(") || !signature.contains(")")) return null
        val types = mutableListOf<String>()
        var index = 0
        while (index < parameters.length) {
            var arrayDepth = 0
            while (index < parameters.length && parameters[index] == '[') {
                arrayDepth++
                index++
            }
            if (index >= parameters.length) return null
            val type =
                when (val token = parameters[index]) {
                    'L' -> {
                        val end = parameters.indexOf(';', index)
                        if (end < 0) return null
                        val name = parameters.substring(index + 1, end).replace('/', '.')
                        index = end
                        name
                    }
                    else -> PRIMITIVES[token] ?: return null
                }
            index++
            types += type + "[]".repeat(arrayDepth)
        }
        return types.joinToString(", ")
    }

    private fun normalise(text: String): String = text.replace(Regex("\\s+"), "")

    private companion object {
        const val CALL_WINDOW = 400
        val FIND_CLASS = Regex("""FindClass\(\s*"([^"]+)"""")
        val FIND_CLASS_ASSIGNMENT = Regex("""jclass\s+(\w+)\s*=\s*env->FindClass\(\s*"([^"]+)"""")
        val GET_METHOD_ID = Regex("""GetMethodID\(\s*(\w+)\s*,""")
        val CALLBACK_HELPER = Regex("""llmedge_get_callback_method\(""")
        val STRING_LITERAL = Regex(""""([^"]*)"""")
        val PRIMITIVES =
            mapOf(
                'I' to "int",
                'J' to "long",
                'F' to "float",
                'D' to "double",
                'Z' to "boolean",
                'B' to "byte",
                'C' to "char",
                'S' to "short",
            )
    }
}
