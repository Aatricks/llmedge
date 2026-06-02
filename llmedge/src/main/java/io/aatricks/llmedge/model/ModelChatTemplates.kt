package io.aatricks.llmedge.model

/**
 * Chat templates carried by built-in model presets whose GGUF metadata does not ship a usable template.
 */
internal object ModelChatTemplates {
    /**
     * Canonical chat template for Microsoft BitNet b1.58 2B4T, copied verbatim from the
     * `chat_template` field of `microsoft/bitnet-b1.58-2B-4T` `tokenizer_config.json`.
     *
     * Supplied explicitly because the ik_llama-compatible IQ2_BN GGUF builds do not carry the correct
     * template in metadata, so relying on the embedded one yields malformed prompts.
     */
    const val BITNET: String =
        "{% set loop_messages = messages %}" +
            "{% for message in loop_messages %}" +
            "{% set content = message['role'] | capitalize + ': '+ message['content'] | trim + '<|eot_id|>' %}" +
            "{{ content }}" +
            "{% endfor %}" +
            "{% if add_generation_prompt %}{{ 'Assistant: ' }}{% endif %}"
}
