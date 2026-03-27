#include "chat.h"
#include "llama.h"

#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

void test_llmedge_llama_compat();
void test_gguf_reader_metadata();

namespace {

void require(bool condition, const std::string & message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

void test_jinja_template_renders_and_legacy_rejects() {
    const std::string template_src =
        "{% for message in messages %}[{{ message.role }}] {{ message.content }} {% endfor %}"
        "{% if add_generation_prompt %}[assistant]{% endif %}";

    auto templates = common_chat_templates_init(/* model = */ nullptr, template_src);

    common_chat_templates_inputs inputs;
    inputs.use_jinja = true;
    inputs.add_generation_prompt = true;

    common_chat_msg user_message;
    user_message.role = "user";
    user_message.content = "hello";
    inputs.messages.push_back(user_message);

    const auto rendered = common_chat_templates_apply(templates.get(), inputs).prompt;
    require(rendered == "[user] hello [assistant]", "Jinja renderer did not format the template as expected");

    const llama_chat_message legacy_messages[] = {
        { "user", "hello" },
    };
    const int legacy_result =
        llama_chat_apply_template(template_src.c_str(), legacy_messages, 1, true, nullptr, 0);
    require(legacy_result < 0, "Legacy llama_chat_apply_template unexpectedly accepted a loop-based Jinja template");
}

void test_legacy_template_still_formats() {
    const std::string template_src =
        "{%- for message in messages -%}\n"
        "  {{- '<|im_start|>' + message.role + '\\n' + message.content + '<|im_end|>\\n' -}}\n"
        "{%- endfor -%}\n"
        "{%- if add_generation_prompt -%}\n"
        "  {{- '<|im_start|>assistant\\n' -}}\n"
        "{%- endif -%}";

    const llama_chat_message messages[] = {
        { "user", "hello" },
    };

    const int required_length =
        llama_chat_apply_template(template_src.c_str(), messages, 1, true, nullptr, 0);
    require(required_length > 0, "Legacy llama_chat_apply_template failed on a supported chatml template");

    std::vector<char> buffer(required_length);
    const int rendered_length =
        llama_chat_apply_template(template_src.c_str(), messages, 1, true, buffer.data(), required_length);
    require(rendered_length == required_length, "Legacy renderer returned an unexpected output length");

    const std::string rendered(buffer.data(), rendered_length);
    require(
        rendered == "<|im_start|>user\nhello<|im_end|>\n<|im_start|>assistant\n",
        "Legacy renderer output changed unexpectedly");
}

void test_malformed_jinja_throws_while_legacy_template_remains_available() {
    bool threw = false;
    try {
        (void) common_chat_templates_init(/* model = */ nullptr, "{% for message in messages %}");
    } catch (const std::exception &) {
        threw = true;
    }
    require(threw, "Malformed Jinja template should fail compilation");

    test_legacy_template_still_formats();
}

}  // namespace

int main() {
    try {
        test_jinja_template_renders_and_legacy_rejects();
        test_legacy_template_still_formats();
        test_malformed_jinja_throws_while_legacy_template_remains_available();
        test_llmedge_llama_compat();
        test_gguf_reader_metadata();
        std::cout << "smollm_chat_template_tests passed" << std::endl;
        return EXIT_SUCCESS;
    } catch (const std::exception & error) {
        std::cerr << "smollm_chat_template_tests failed: " << error.what() << std::endl;
        return EXIT_FAILURE;
    }
}
