#include "LLMInference.h"
#include "llm_backend_support.h"
#ifdef __ANDROID__
#include <android/log.h>
#include <sched.h>
#define TAG "[SmolLMAndroid-Cpp]"
#define LOGi(...) __android_log_print(ANDROID_LOG_INFO, TAG, __VA_ARGS__)
#define LOGe(...) __android_log_print(ANDROID_LOG_ERROR, TAG, __VA_ARGS__)
#else
#include <cstdio>
#define TAG "[SmolLM-Cpp]"
#define LOGi(...) fprintf(stdout, "%s ", TAG); fprintf(stdout, __VA_ARGS__); fprintf(stdout, "\n")
#define LOGe(...) fprintf(stderr, "%s ", TAG); fprintf(stderr, __VA_ARGS__); fprintf(stderr, "\n")
#endif
#include <algorithm>
#include <numeric>
#include <cstring>
#include <iostream>
#include <limits>
#include <memory>

void
LLMInference::loadModel(const char *model_path, float minP, float temperature, bool storeChats, long contextSize,
                        const char *chatTemplate, int nThreads, bool useMmap, bool useMlock, int backendId,
                        bool useFlashAttn, int kvCacheTypeKCode, int kvCacheTypeVCode, int nGpuLayers, int nUbatch) {
    const RequestedBackend requestedBackend = static_cast<RequestedBackend>(backendId);
    LOGi("loading model with"
         "\n\tmodel_path = %s"
         "\n\tminP = %f"
         "\n\ttemperature = %f"
         "\n\tstoreChats = %d"
         "\n\tcontextSize = %li"
         "\n\tchatTemplate = %s"
         "\n\tnThreads = %d"
         "\n\tuseMmap = %d"
         "\n\tuseMlock = %d"
         "\n\tbackend = %s"
         "\n\tuseFlashAttn = %d",
         model_path, minP, temperature, storeChats, contextSize, chatTemplate, nThreads, useMmap, useMlock,
         backend_name(requestedBackend), useFlashAttn);

#if !defined(GGML_USE_VULKAN)
    if (requestedBackend == RequestedBackend::VULKAN) {
        throw std::runtime_error("The requested llmedge text runtime was built without Vulkan support");
    }
#endif
#if !defined(GGML_USE_OPENCL)
    if (requestedBackend == RequestedBackend::OPENCL) {
        throw std::runtime_error("The requested llmedge text runtime was built without OpenCL support");
    }
#endif

    // create an instance of llama_model
    llama_model_params model_params = llama_model_default_params();
    model_params.use_mmap = useMmap;
    model_params.use_mlock = useMlock;
    std::string requested_device_name;
    if (is_gpu_backend(requestedBackend)) {
        requested_device_name = find_backend_registry_name(requestedBackend, 0);
        if (requested_device_name.empty()) {
            throw std::runtime_error(std::string("Requested backend not available: ") + backend_name(requestedBackend));
        }

        LOGi("Using %s backend entry: %s",
             backend_name(requestedBackend),
             requested_device_name.c_str());

        model_params.devices = requested_device_name.c_str();
        model_params.split_mode = LLAMA_SPLIT_MODE_NONE;
        model_params.n_gpu_layers = nGpuLayers > 0 ? nGpuLayers : -1;
    }
    _model = llama_model_load_from_file(model_path, model_params);
    if (!_model) {
        LOGe("failed to load model from %s", model_path);
        throw std::runtime_error(std::string("loadModel() failed on ") + backend_name(requestedBackend));
    }

    // create an instance of llama_context
    llama_context_params ctx_params = llama_context_default_params();
    const long safeContext = std::clamp(contextSize, 1L, static_cast<long>(std::numeric_limits<uint32_t>::max()));
    if (safeContext != contextSize) {
        LOGi("contextSize %ld adjusted to %ld to fit llama context limits", contextSize, safeContext);
    }
    ctx_params.n_ctx = static_cast<uint32_t>(safeContext);
    // n_batch must cover the full context because startCompletion() submits the whole
    // prompt as a single llama_batch. This is cheap in the vendored fork (verified in
    // llama.cpp:3693ff): llama_decode splits work into n_ubatch micro-batches
    // internally, and the logits buffer is sized by actual outputs
    // (llama_output_reserve), not n_batch — only output_ids scales with n_batch,
    // at 4 bytes per token.
    ctx_params.n_batch = static_cast<uint32_t>(safeContext);
    // Micro-batch size drives prefill throughput vs. compute-buffer memory.
    // Default stays at 128 (cache-friendly on small ARM cores) but callers can
    // raise it via InferenceParams.nUbatch for faster prompt processing.
    ctx_params.n_ubatch = nUbatch > 0
        ? std::min(static_cast<uint32_t>(nUbatch), ctx_params.n_batch)
        : std::min(ctx_params.n_batch, 128u);
    ctx_params.n_threads = nThreads;
    ctx_params.n_threads_batch = nThreads;
    ctx_params.flash_attn = useFlashAttn;
#if !defined(__ANDROID__) && defined(__x86_64__)
    // Observed on local Linux x86_64 with the ik_llama.cpp fork: Q8_KV decode aborts inside
    // the fused up/gate IQK kernel path. Keeping the host JNI build on the non-fused code path
    // restores functional inference for desktop validation without changing Android behavior.
    ctx_params.fused_up_gate = false;
#endif
    // Map llmedge-owned KV cache type codes to the backend-specific ggml_type.
    ctx_params.type_k = llmedge_resolve_kv_cache_type(kvCacheTypeKCode, "kvCacheTypeK");
    ctx_params.type_v = llmedge_resolve_kv_cache_type(kvCacheTypeVCode, "kvCacheTypeV");
    if (is_gpu_backend(requestedBackend) && model_params.n_gpu_layers != 0) {
        ctx_params.offload_kqv = true;
    }
    _ctx = llama_init_from_model(_model, ctx_params);
    if (!_ctx && is_gpu_backend(requestedBackend) && ctx_params.offload_kqv) {
        LOGi("Context creation with KQV offload failed, retrying without offload");
        ctx_params.offload_kqv = false;
        _ctx = llama_init_from_model(_model, ctx_params);
    }
    if (!_ctx) {
        LOGe("llama_new_context_with_model() returned null)");
        if (is_gpu_backend(requestedBackend) && ctx_params.offload_kqv) {
            throw std::runtime_error(std::string("llama_new_context_with_model() returned null with the requested ") +
                                     backend_name(requestedBackend) + "/KQV offload configuration");
        }
        throw std::runtime_error(std::string("llama_new_context_with_model() returned null on ") +
                                 backend_name(requestedBackend));
    }

    common_params_sampling sampler_params;
    sampler_params.top_k = 40;
    sampler_params.top_p = 1.0f;
    sampler_params.tfs_z = 1.0f;
    sampler_params.typical_p = 1.0f;
    sampler_params.temp = temperature;
    sampler_params.min_p = minP > 0.0f ? minP : 0.0f;
    sampler_params.penalty_repeat = 1.0f;
    sampler_params.penalty_freq = 0.0f;
    sampler_params.penalty_present = 0.0f;
    sampler_params.dry_multiplier = 0.0f;
    sampler_params.samplers_sequence = {
        llama_sampler_type::TOP_K,
        llama_sampler_type::MIN_P,
        llama_sampler_type::TEMPERATURE,
        llama_sampler_type::DIST,
    };
    _sampler = common_sampler_init(_model, sampler_params);
    if (_sampler == nullptr) {
        throw std::runtime_error("common_sampler_init() returned null");
    }

    _formattedMessages = std::vector<char>(llama_n_ctx(_ctx));
    // Messages hold strdup'd strings; a bare clear() on reload would leak them.
    for (llama_chat_message &message : _messages) {
        free(const_cast<char *>(message.role));
        free(const_cast<char *>(message.content));
    }
    _messages.clear();

    // Invalidate any existing system prompt KV snapshot
    _systemPromptKVSnapshot.clear();
    _cachedSystemPromptHash = 0;
    _systemPromptTokenCount = 0;
    _chatTemplates.reset();
    _chatFormatterMode = ChatFormatterMode::LEGACY;
    _prevFormatterMode = ChatFormatterMode::LEGACY;
    if (chatTemplate != nullptr && chatTemplate[0] != '\0') {
        _chatTemplateSrc = chatTemplate;
    } else {
        const char* modelTemplate = llama_model_chat_template(_model, nullptr);
        _chatTemplateSrc = modelTemplate ? modelTemplate : "";
    }

    try {
        _chatTemplates = common_chat_templates_init(_model, _chatTemplateSrc);
        _chatFormatterMode = ChatFormatterMode::JINJA;
        LOGi("Initialized Jinja chat templates");
    } catch (const std::exception& error) {
        LOGe("Failed to initialize Jinja chat templates, falling back to legacy formatting: %s", error.what());
        _chatTemplates.reset();
    }
    this->_storeChats = storeChats;
    _disableThinking = false;
    _reasoningBudget = -1;
}

void
LLMInference::addChatMessage(const char *message, const char *role) {
    char* roleCopy = strdup(role);
    char* msgCopy = strdup(message);
    if (!roleCopy || !msgCopy) {
        free(roleCopy);
        free(msgCopy);
        throw std::runtime_error("strdup() failed in addChatMessage (out of memory)");
    }
    _messages.push_back({roleCopy, msgCopy});
}

std::vector<common_chat_msg>
LLMInference::buildCommonChatMessages(size_t messageCount) const {
    const size_t boundedCount = std::min(messageCount, _messages.size());
    std::vector<common_chat_msg> messages;
    messages.reserve(boundedCount);
    for (size_t i = 0; i < boundedCount; ++i) {
        common_chat_msg message;
        message.role = _messages[i].role ? _messages[i].role : "";
        message.content = _messages[i].content ? _messages[i].content : "";
        messages.push_back(std::move(message));
    }
    return messages;
}

LLMInference::FormattedPrompt
LLMInference::formatChatMessagesJinja(size_t messageCount, bool addGenerationPrompt) const {
    if (!_chatTemplates) {
        throw std::runtime_error("Jinja chat templates are unavailable");
    }

    common_chat_templates_inputs inputs;
    inputs.messages = buildCommonChatMessages(messageCount);
    inputs.add_generation_prompt = addGenerationPrompt;
    inputs.use_jinja = true;

    auto params = common_chat_templates_apply(_chatTemplates.get(), inputs);
    if (params.prompt.size() > static_cast<size_t>(std::numeric_limits<int>::max())) {
        throw std::runtime_error("formatted Jinja prompt exceeds int range");
    }

    FormattedPrompt result;
    result.prompt = std::move(params.prompt);
    result.renderedLength = static_cast<int>(result.prompt.size());
    result.modeUsed = ChatFormatterMode::JINJA;
    return result;
}

LLMInference::FormattedPrompt
LLMInference::formatChatMessagesLegacy(size_t messageCount, bool addGenerationPrompt) const {
    const size_t boundedCount = std::min(messageCount, _messages.size());
    const char* templateSrc = _chatTemplateSrc.empty() ? nullptr : _chatTemplateSrc.c_str();
    int renderedLength =
        llama_chat_apply_template(templateSrc, _messages.data(), boundedCount, addGenerationPrompt, nullptr, 0);
    if (renderedLength < 0) {
        throw std::runtime_error("legacy llama_chat_apply_template() failed");
    }

    std::vector<char> buffer(std::max(renderedLength, 1));
    renderedLength =
        llama_chat_apply_template(templateSrc, _messages.data(), boundedCount, addGenerationPrompt, buffer.data(),
                                  static_cast<int32_t>(buffer.size()));
    if (renderedLength < 0) {
        throw std::runtime_error("legacy llama_chat_apply_template() failed on second pass");
    }
    if (renderedLength > static_cast<int>(buffer.size())) {
        throw std::runtime_error("legacy llama_chat_apply_template() exceeded output buffer");
    }

    FormattedPrompt result;
    result.prompt.assign(buffer.data(), renderedLength);
    result.renderedLength = renderedLength;
    result.modeUsed = ChatFormatterMode::LEGACY;
    return result;
}

void
LLMInference::downgradeChatFormatter(const char* reason) {
    if (_chatFormatterMode == ChatFormatterMode::LEGACY) {
        return;
    }
    _chatFormatterMode = ChatFormatterMode::LEGACY;
    LOGe("Disabling Jinja chat formatting for this session and falling back to legacy mode: %s", reason);
}

LLMInference::FormattedPrompt
LLMInference::formatChatMessages(size_t messageCount, bool addGenerationPrompt) {
    if (_chatFormatterMode == ChatFormatterMode::JINJA && _chatTemplates) {
        try {
            return formatChatMessagesJinja(messageCount, addGenerationPrompt);
        } catch (const std::exception& error) {
            downgradeChatFormatter(error.what());
        }
    }

    return formatChatMessagesLegacy(messageCount, addGenerationPrompt);
}

float
LLMInference::getResponseGenerationTime() const {
    return getResponseTokensPerSecond();
}

float
LLMInference::getResponseTokensPerSecond() const {
    if (_responseGenerationTime <= 0 || _responseNumTokens <= 0) {
        return 0.f;
    }
    return (_responseNumTokens * 1e6f) / static_cast<float>(_responseGenerationTime);
}

long
LLMInference::getResponseTokenCount() const {
    return _responseNumTokens;
}

int64_t
LLMInference::getResponseGenerationTimeMicros() const {
    return _responseGenerationTime;
}

uint64_t
LLMInference::getEstimatedMemoryBytes() const {
    const uint64_t modelBytes = _model ? llama_model_size(_model) : 0ULL;
    return modelBytes + getStateMemoryBytes();
}

uint64_t
LLMInference::getStateMemoryBytes() const {
    return _ctx ? static_cast<uint64_t>(llama_state_get_size(_ctx)) : 0ULL;
}

int
LLMInference::getContextSizeUsed() const {
    return _nCtxUsed;
}

bool
LLMInference::isContextLimitReached() const {
    return _contextLimitReached;
}

void
LLMInference::startCompletion(const char *query) {
    if (!_storeChats) {
        for (auto it = _messages.begin(); it != _messages.end();) {
            if (std::strcmp(it->role, "system") != 0) {
                free(const_cast<char *>(it->role));
                free(const_cast<char *>(it->content));
                it = _messages.erase(it);
            } else {
                ++it;
            }
        }
        _prevLen = 0;
        _formattedMessages.assign(llama_n_ctx(_ctx), 0);
    }
    _responseGenerationTime = 0;
    _responseNumTokens = 0;
    _response.clear();
    _response.reserve(2048);
    _cacheResponseTokens.clear();
    _cacheResponseTokens.reserve(32);
    std::string finalQuery = query ? std::string(query) : std::string();
    const bool suppressThinking = _disableThinking || _reasoningBudget == 0;
    const bool looksStructuredPrompt =
        finalQuery.rfind("<|", 0) == 0 ||
        finalQuery.find("<|system|>") != std::string::npos ||
        finalQuery.find("<|user|>") != std::string::npos ||
        finalQuery.find("<|assistant|>") != std::string::npos ||
        finalQuery.find("<|im_start|>") != std::string::npos;
    if (suppressThinking && !looksStructuredPrompt && finalQuery.find("/no_think") == std::string::npos) {
        if (!finalQuery.empty()) {
            finalQuery.insert(0, "/no_think\n");
        } else {
            finalQuery = "/no_think";
        }
    }
    addChatMessage(finalQuery.c_str(), "user");
    const auto renderedPrompt = formatChatMessages(_messages.size(), true);
    const int newLen = renderedPrompt.renderedLength;
    _formattedMessages.assign(renderedPrompt.prompt.begin(), renderedPrompt.prompt.end());
    if (_prevLen > 0 && renderedPrompt.modeUsed != _prevFormatterMode) {
        LOGi("Chat formatter changed between turns; resetting cached prompt prefix");
        _prevLen = 0;
    }

    // --- System prompt KV cache snapshotting ---
    bool restoredFromSnapshot = false;
    int snapshotNPast = 0;
    _eogReached = false;
    _contextLimitReached = false;

    if (_prevLen == 0) {
        size_t systemMsgCount = 0;
        std::string systemContent;
        bool systemIsPrefix = true;
        for (size_t i = 0; i < _messages.size(); ++i) {
            if (std::strcmp(_messages[i].role, "system") == 0) {
                // The snapshot slices the rendered prompt at format(first N messages),
                // which only equals the rendered system prompt when every system
                // message sits at the front. A system message added after a user
                // message (possible via the direct addSystemPrompt API) would cache
                // the wrong prefix under the system content's hash.
                if (i != systemMsgCount) {
                    systemIsPrefix = false;
                }
                systemContent += _messages[i].content;
                systemMsgCount++;
            }
        }

        if (systemMsgCount > 0 && systemIsPrefix) {
            size_t currentHash = std::hash<std::string>{}(systemContent);

            int sysTemplateLen = 0;
            try {
                const auto systemPrompt = formatChatMessages(systemMsgCount, false);
                if (systemPrompt.modeUsed == renderedPrompt.modeUsed) {
                    sysTemplateLen = systemPrompt.renderedLength;
                } else {
                    LOGi("Skipping system prompt KV snapshot because formatter mode changed during rendering");
                }
            } catch (const std::exception& error) {
                LOGe("Failed to format system prompt for KV snapshotting, skipping snapshot: %s", error.what());
            }

            if (currentHash == _cachedSystemPromptHash && !_systemPromptKVSnapshot.empty()) {
                // Fast path: this instance decodes only on sequence 0 and never removes
                // positions below the system prefix between turns, so when the cache
                // still holds at least that many positions the prefix is ours — trim
                // back to it instead of memcpy'ing the serialized state (multi-MB per
                // turn) back in. Anything that emptied the cache (e.g. an external
                // clearKvCache) fails the position check and takes the full restore.
                const llama_pos cachePosMax = llmedge_kv_cache_seq_pos_max(_ctx, 0);
                if (_systemPromptTokenCount > 0 && cachePosMax + 1 >= _systemPromptTokenCount) {
                    LOGi("Trimming KV cache to system prompt prefix (hash=%zu, tokens=%d)",
                         currentHash, _systemPromptTokenCount);
                    llmedge_kv_cache_seq_rm(_ctx, 0, _systemPromptTokenCount, -1);
                    _prevLen = sysTemplateLen;
                    restoredFromSnapshot = true;
                    snapshotNPast = _systemPromptTokenCount;
                } else {
                    // Restore KV state from cached snapshot
                    LOGi("Restoring system prompt KV snapshot (hash=%zu, tokens=%d)",
                         currentHash, _systemPromptTokenCount);
                    llmedge_kv_cache_seq_rm(_ctx, -1, -1, -1);
                    size_t nset = llmedge_state_seq_set_data(
                        _ctx, _systemPromptKVSnapshot.data(), _systemPromptKVSnapshot.size(), 0);
                    if (nset == 0) {
                        LOGe("Failed to restore system prompt KV snapshot, processing normally");
                        _systemPromptKVSnapshot.clear();
                        _cachedSystemPromptHash = 0;
                        _systemPromptTokenCount = 0;
                    } else {
                        _prevLen = sysTemplateLen;
                        restoredFromSnapshot = true;
                        snapshotNPast = _systemPromptTokenCount;
                    }
                }
            } else if (sysTemplateLen > 0) {
                // Decode system prompt and create a new snapshot
                LOGi("Creating system prompt KV snapshot (hash=%zu)", currentHash);
                llmedge_kv_cache_seq_rm(_ctx, -1, -1, -1);

                std::string sysPrompt(_formattedMessages.begin(),
                                      _formattedMessages.begin() + sysTemplateLen);
                std::vector<llama_token> sysTokens = common_tokenize(_model, sysPrompt, true, true);

                if (!sysTokens.empty()) {
                    llama_batch sysBatch = {};
                    sysBatch.token = sysTokens.data();
                    sysBatch.n_tokens = static_cast<int32_t>(sysTokens.size());
                    std::vector<llama_pos> sysPos(sysTokens.size());
                    for (size_t i = 0; i < sysTokens.size(); i++)
                        sysPos[i] = static_cast<llama_pos>(i);
                    sysBatch.pos = sysPos.data();

                    if (llama_decode(_ctx, sysBatch) < 0) {
                        LOGe("Failed to decode system prompt for snapshot");
                    } else {
                        size_t stateSize = llmedge_state_seq_get_size(_ctx, 0);
                        _systemPromptKVSnapshot.resize(stateSize);
                        size_t ncopy = llmedge_state_seq_get_data(
                            _ctx, _systemPromptKVSnapshot.data(),
                            _systemPromptKVSnapshot.size(), 0);
                        if (ncopy == stateSize) {
                            _cachedSystemPromptHash = currentHash;
                            _systemPromptTokenCount = static_cast<int>(sysTokens.size());
                            _prevLen = sysTemplateLen;
                            restoredFromSnapshot = true;
                            snapshotNPast = _systemPromptTokenCount;
                            LOGi("System prompt KV snapshot saved (%zu bytes, %d tokens)",
                                 stateSize, _systemPromptTokenCount);
                        } else {
                            LOGe("System prompt KV snapshot copy failed");
                            _systemPromptKVSnapshot.clear();
                        }
                    }
                }
            }
        }
    }
    // --- End system prompt KV cache snapshotting ---

    std::string prompt(_formattedMessages.begin() + _prevLen, _formattedMessages.begin() + newLen);
    // Only add special tokens (like BOS) if we are at the start of the context
    bool add_special = (_prevLen == 0); 
    _promptTokens = common_tokenize(_model, prompt, add_special, true);
    if (_promptTokens.empty()) {
        LOGe("tokenize() returned no tokens for prompt; aborting completion");
        throw std::runtime_error("empty prompt tokenization");
    }
    if (_promptTokens.size() > static_cast<size_t>(std::numeric_limits<int32_t>::max())) {
        LOGe("prompt token count %zu exceeds int32 range", _promptTokens.size());
        throw std::runtime_error("prompt too long for llama_batch");
    }

    if (_batch == nullptr) {
        _batch = new llama_batch();
    }
    std::memset(_batch, 0, sizeof(llama_batch));
    _batch->token = _promptTokens.data();
    _batch->n_tokens = static_cast<int32_t>(_promptTokens.size());

    // Fix KV cache reuse
    int n_past = 0;
    if (restoredFromSnapshot) {
         n_past = snapshotNPast;
    } else if (_preservePreparedKvForNextCompletion) {
         int max_seq_pos = llmedge_kv_cache_seq_pos_max(_ctx, 0);
         if (max_seq_pos >= 0) {
             n_past = max_seq_pos + 1;
         }
         LOGi("Preserving prepared KV cache for completion (n_past=%d)", n_past);
    } else if (_storeChats && _prevLen > 0) {
         int max_seq_pos = llmedge_kv_cache_seq_pos_max(_ctx, 0);
         if (max_seq_pos >= 0) {
             n_past = max_seq_pos + 1;
         }
    } else {
         n_past = 0;
         llmedge_kv_cache_seq_rm(_ctx, -1, -1, -1);
    }

        _preservePreparedKvForNextCompletion = false;
    
    _nPast = n_past;
    LOGi("startCompletion: n_past=%d, n_tokens=%d, prevLen=%d", n_past, _batch->n_tokens, _prevLen);

    _batchPos.resize(std::max(_promptTokens.size(), static_cast<size_t>(1)));
    std::iota(_batchPos.begin(), _batchPos.begin() + _promptTokens.size(),
              static_cast<llama_pos>(_nPast));
    _nPast += static_cast<int>(_promptTokens.size());
    _batch->pos = _batchPos.data();
}

// taken from:
// https://github.com/ggerganov/llama.cpp/blob/master/examples/llama.android/llama/src/main/cpp/llama-android.cpp#L38
bool
LLMInference::_isValidUtf8(const char *response) {
    if (!response) {
        return true;
    }
    const unsigned char *bytes = (const unsigned char *) response;
    int num;
    while (*bytes != 0x00) {
        if ((*bytes & 0x80) == 0x00) {
            // U+0000 to U+007F
            num = 1;
        } else if ((*bytes & 0xE0) == 0xC0) {
            // U+0080 to U+07FF
            num = 2;
        } else if ((*bytes & 0xF0) == 0xE0) {
            // U+0800 to U+FFFF
            num = 3;
        } else if ((*bytes & 0xF8) == 0xF0) {
            // U+10000 to U+10FFFF
            num = 4;
        } else {
            return false;
        }

        bytes += 1;
        for (int i = 1; i < num; ++i) {
            if ((*bytes & 0xC0) != 0x80) {
                return false;
            }
            bytes += 1;
        }
    }
    return true;
}

std::string
LLMInference::completionLoop() {
    if (_eogReached) {
        return "[EOG]";
    }
    if (_batch == nullptr || _batch->n_tokens <= 0) {
        LOGe("completionLoop invoked with empty llama_batch");
        throw std::runtime_error("llama batch missing tokens");
    }
    // check if the length of the inputs to the model
    // have exceeded the context size of the model
    uint32_t contextSize = llama_n_ctx(_ctx);
    _nCtxUsed = llmedge_kv_cache_seq_pos_max(_ctx, 0) + 1;
    if (_nCtxUsed + _batch->n_tokens > contextSize) {
        if (_responseNumTokens <= 0) {
            // Nothing generated yet: the prompt alone exceeds the context window,
            // so surface an error rather than silently returning nothing.
            throw std::runtime_error("context size reached");
        }
        // Mid-generation overflow: stop gracefully so the caller keeps the partial
        // response instead of losing it to an exception. Queryable via
        // isContextLimitReached().
        LOGe("context window full after %ld generated tokens; ending completion", _responseNumTokens);
        _contextLimitReached = true;
        _eogReached = true;
        if (_storeChats) {
            addChatMessage(_response.c_str(), "assistant");
        }
        _response.clear();
        _cacheResponseTokens.clear();
        return "[EOG]";
    }

    auto start = ggml_time_us();
    // run the model
    if (llama_decode(_ctx, *_batch) < 0) {
        throw std::runtime_error("llama_decode() failed");
    }

    // sample a token and check if it is an EOG (end of generation token)
    // convert the integer token to its corresponding word-piece
    _currToken = common_sampler_sample(_sampler, _ctx, -1);
    common_sampler_accept(_sampler, _ctx, _currToken, true);
    if (llama_vocab_is_eog(llama_model_get_vocab(_model), _currToken)) {
        _eogReached = true;
        if (_storeChats) {
            addChatMessage(_response.c_str(), "assistant");
        }
        _response.clear();
        _cacheResponseTokens.clear();
        return "[EOG]";
    }
    std::string piece = common_token_to_piece(_ctx, _currToken, true);
    auto end = ggml_time_us();
    _responseGenerationTime += (end - start);
    _responseNumTokens += 1;
    _cacheResponseTokens += piece;

    // re-init the batch with the newly predicted token
    // key, value pairs of all previous tokens have been cached
    // in the KV cache
    _batch->token = &_currToken;
    _batch->n_tokens = 1;
    
    // Set position for the next token using cached n_past
    _batchPos[0] = _nPast;
    _nPast++;
    _batch->pos = _batchPos.data();

    _batch->seq_id = nullptr;
    _batch->n_seq_id = nullptr;
    _batch->logits = nullptr;

    if (_isValidUtf8(_cacheResponseTokens.c_str())) {
        _response += _cacheResponseTokens;
        std::string valid_utf8_piece = std::move(_cacheResponseTokens);
        _cacheResponseTokens.clear();
        return valid_utf8_piece;
    }

    return "";
}

std::string
LLMInference::completionLoopBatch(int maxTokens) {
    std::string result;
    result.reserve(maxTokens * 4); // pre-allocate ~4 bytes per token average
    for (int i = 0; i < maxTokens; i++) {
        std::string piece = completionLoop();
        if (piece == "[EOG]") {
            if (result.empty()) return "[EOG]";
            break;
        }
        if (!piece.empty()) {
            result += piece;
        }
    }
    return result;
}

void
LLMInference::stopCompletion() {
    if (_storeChats) {
        const auto renderedPrompt = formatChatMessages(_messages.size(), false);
        _prevLen = renderedPrompt.renderedLength;
        _prevFormatterMode = renderedPrompt.modeUsed;
    } else {
        _prevLen = 0;
        _prevFormatterMode = _chatFormatterMode;
        _nPast = 0;
        _preservePreparedKvForNextCompletion = false;
    }
    _response.clear();
    _cacheResponseTokens.clear();
}

void
LLMInference::clearMessages() {
    for (llama_chat_message &message: _messages) {
        free(const_cast<char *>(message.role));
        free(const_cast<char *>(message.content));
    }
    _messages.clear();
    _prevLen = 0;
    _prevFormatterMode = _chatFormatterMode;
    _nPast = 0;
    _nCtxUsed = 0;
    _response.clear();
    _cacheResponseTokens.clear();
    _promptTokens.clear();
    _preservePreparedKvForNextCompletion = false;
    _eogReached = false;
    _contextLimitReached = false;
    if (_batch) {
        std::memset(_batch, 0, sizeof(llama_batch));
    }
    if (_ctx) {
        _formattedMessages.assign(llama_n_ctx(_ctx), 0);
    } else {
        _formattedMessages.clear();
    }
}

void
LLMInference::markPreparedKvForNextCompletion() {
    _preservePreparedKvForNextCompletion = true;
}

void
LLMInference::setReasoningOptions(bool disableThinking, int reasoningBudget) {
    const bool requestedNoThink = disableThinking || reasoningBudget == 0;
    _disableThinking = requestedNoThink;
    _reasoningBudget = reasoningBudget;
    LOGi("Reasoning controls: disableThinking=%d, reasoningBudget=%d", _disableThinking, _reasoningBudget);
}

void
LLMInference::configureThreading(int generationThreads, int promptThreads) {
    if (!_ctx) {
        return;
    }
    const int effectiveGenerationThreads = std::max(1, generationThreads);
    const int effectivePromptThreads = std::max(1, promptThreads);
    llama_set_n_threads(_ctx, effectiveGenerationThreads, effectivePromptThreads);
    LOGi("Configured llama threads: generation=%d, prompt_batch=%d",
         effectiveGenerationThreads, effectivePromptThreads);
}

void
LLMInference::setThreadAffinity(uint64_t coreMask) {
    _coreMask = coreMask;
    if (coreMask == 0) return;

#ifdef __ANDROID__
    cpu_set_t cpuset;
    CPU_ZERO(&cpuset);
    for (int i = 0; i < 64; i++) {
        if (coreMask & (1ULL << i)) {
            CPU_SET(i, &cpuset);
        }
    }
    // Pin calling thread to P-cores
    sched_setaffinity(0, sizeof(cpuset), &cpuset);

    // Also pin OpenMP worker threads so they don't drift to E-cores.
    // Setting affinity on the master thread before the first parallel region
    // causes workers (via fork-join) to inherit the mask on most runtimes.
    // For static-openmp linked builds, explicitly pinning via the parallel
    // region is the most reliable approach.
#if defined(_OPENMP)
    #pragma omp parallel
    {
        sched_setaffinity(0, sizeof(cpuset), &cpuset);
    }
#endif
#endif
}

LLMInference::~LLMInference() {
    // free memory held by the message text in messages
    // (as we had used strdup() to create a malloc'ed copy)
    for (llama_chat_message &message: _messages) {
        free(const_cast<char *>(message.role));
        free(const_cast<char *>(message.content));
    }
    llama_free(_ctx);
    llama_free_model(_model);
    delete _batch;
    // common_sampler_free() dereferences its argument without a null check, so a
    // load that failed before sampler init must not reach it with nullptr.
    if (_sampler) {
        common_sampler_free(_sampler);
    }
}

// Safe accessors used by JNI/native glue. Return internal pointers; caller must not free.
llama_model* LLMInference::getModel() {
    return _model;
}

llama_context* LLMInference::getContext() {
    return _ctx;
}
