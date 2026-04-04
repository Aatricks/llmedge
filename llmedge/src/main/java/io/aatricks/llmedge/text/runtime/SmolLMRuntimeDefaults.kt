package io.aatricks.llmedge.text.runtime

internal object SmolLMRuntimeDefaults {
    const val DEFAULT_CONTEXT_SIZE_CAP: Long = 8_192L
    const val MIN_CONTEXT_SIZE: Long = 1_024L
    const val DEFAULT_REASONING_BUDGET: Int = -1

    val ggufFileTypeNames =
        mapOf(
            138 to "IQ2_K",
            139 to "IQ3_K",
            140 to "IQ4_K",
            141 to "IQ5_K",
            142 to "IQ6_K",
            149 to "Q8_KV",
        )

    val ggufTensorTypeNames =
        mapOf(
            137 to "IQ2_K",
            138 to "IQ3_K",
            139 to "IQ4_K",
            140 to "IQ5_K",
            141 to "IQ6_K",
            151 to "Q8_KV",
        )

    object DefaultInferenceParams {
        const val contextSize: Long = 1024L
        const val chatTemplate: String =
            "{% for message in messages %}{% if loop.first and messages[0]['role'] != 'system' %}{{ '<|im_start|>system You are a helpful AI assistant named SmolLM, trained by Hugging Face<|im_end|> ' }}{% endif %}{{'<|im_start|>' + message['role'] + ' ' + message['content'] + '<|im_end|>' + ' '}}{% endfor %}{% if add_generation_prompt %}{{ '<|im_start|>assistant ' }}{% endif %}"
    }
}
