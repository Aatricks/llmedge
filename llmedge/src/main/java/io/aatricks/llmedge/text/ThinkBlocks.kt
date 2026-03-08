package io.aatricks.llmedge.text

private val THINK_BLOCK_REGEX =
    Regex(
        pattern = "<think>.*?</think>",
        options = setOf(RegexOption.IGNORE_CASE, RegexOption.DOT_MATCHES_ALL),
    )

private val DANGLING_THINK_BLOCK_REGEX =
    Regex(
        pattern = "<think>.*$",
        options = setOf(RegexOption.IGNORE_CASE, RegexOption.DOT_MATCHES_ALL),
    )

internal fun String.stripThinkBlocks(): String {
    val withoutClosedBlocks = THINK_BLOCK_REGEX.replace(this, "")
    return DANGLING_THINK_BLOCK_REGEX.replace(withoutClosedBlocks, "").trim()
}
