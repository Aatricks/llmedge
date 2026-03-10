#include "LLMInference.h"
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

void
LLMInference::loadModel(const char *model_path, float minP, float temperature, bool storeChats, long contextSize,
                        const char *chatTemplate, int nThreads, bool useMmap, bool useMlock, bool useVulkan,
                        bool useFlashAttn, int kvCacheTypeK, int kvCacheTypeV, int nGpuLayers) {
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
         "\n\tuseVulkan = %d"
         "\n\tuseFlashAttn = %d",
         model_path, minP, temperature, storeChats, contextSize, chatTemplate, nThreads, useMmap, useMlock, useVulkan, useFlashAttn);

    // load dynamic backends
    ggml_backend_load_all();

    // create an instance of llama_model
    llama_model_params model_params = llama_model_default_params();
    model_params.use_mmap = useMmap;
    model_params.use_mlock = useMlock;
    if (useVulkan) {
        model_params.n_gpu_layers = nGpuLayers > 0 ? nGpuLayers : 99;
    }
    _model = llama_model_load_from_file(model_path, model_params);
    if (!_model && useVulkan) {
        // Vulkan init may have failed — fall back to CPU-only
        LOGi("Vulkan model load failed, retrying with CPU-only (n_gpu_layers=0)");
        model_params.n_gpu_layers = 0;
        _model = llama_model_load_from_file(model_path, model_params);
    }
    if (!_model) {
        LOGe("failed to load model from %s", model_path);
        throw std::runtime_error("loadModel() failed");
    }

    // create an instance of llama_context
    llama_context_params ctx_params = llama_context_default_params();
    const long safeContext = std::clamp(contextSize, 1L, static_cast<long>(std::numeric_limits<uint32_t>::max()));
    if (safeContext != contextSize) {
        LOGi("contextSize %ld adjusted to %ld to fit llama context limits", contextSize, safeContext);
    }
    ctx_params.n_ctx = static_cast<uint32_t>(safeContext);
    ctx_params.n_batch = static_cast<uint32_t>(safeContext);
    // Smaller micro-batches improve cache locality on ARM
    ctx_params.n_ubatch = std::min(ctx_params.n_batch, 128u);
    ctx_params.n_threads = nThreads;
    ctx_params.n_threads_batch = nThreads;
    ctx_params.no_perf = true;
    // Flash attention: let llama.cpp auto-detect the best mode
    ctx_params.flash_attn_type = useFlashAttn ? LLAMA_FLASH_ATTN_TYPE_AUTO : LLAMA_FLASH_ATTN_TYPE_DISABLED;
    // KV cache type: defaults to F16 (~30-40% memory savings) but can be overridden
    // to Q8_0/Q4_0 for further savings on memory-constrained devices.
    // -1 means use the default (F16); positive values are ggml_type enum values.
    ctx_params.type_k = kvCacheTypeK >= 0 ? static_cast<ggml_type>(kvCacheTypeK) : GGML_TYPE_F16;
    ctx_params.type_v = kvCacheTypeV >= 0 ? static_cast<ggml_type>(kvCacheTypeV) : GGML_TYPE_F16;
    if (useVulkan && model_params.n_gpu_layers > 0) {
        ctx_params.offload_kqv = true;
    }
    _ctx = llama_init_from_model(_model, ctx_params);
    if (!_ctx && useVulkan && ctx_params.offload_kqv) {
        LOGi("Context creation with KQV offload failed, retrying without offload");
        ctx_params.offload_kqv = false;
        _ctx = llama_init_from_model(_model, ctx_params);
    }
    if (!_ctx) {
        LOGe("llama_new_context_with_model() returned null)");
        throw std::runtime_error("llama_new_context_with_model() returned null");
    }

    // create an instance of llama_sampler
    llama_sampler_chain_params sampler_params = llama_sampler_chain_default_params();
    sampler_params.no_perf = true;
    _sampler = llama_sampler_chain_init(sampler_params);
    // Expanded sampling chain: top-k → min-p → temperature → dist
    llama_sampler_chain_add(_sampler, llama_sampler_init_top_k(40));
    if (minP > 0.0f) {
        llama_sampler_chain_add(_sampler, llama_sampler_init_min_p(minP, 1));
    }
    llama_sampler_chain_add(_sampler, llama_sampler_init_temp(temperature));
    llama_sampler_chain_add(_sampler, llama_sampler_init_dist(LLAMA_DEFAULT_SEED));

    _formattedMessages = std::vector<char>(llama_n_ctx(_ctx));
    _messages.clear();

    // Invalidate any existing system prompt KV snapshot
    _systemPromptKVSnapshot.clear();
    _cachedSystemPromptHash = 0;
    _systemPromptTokenCount = 0;

    if (chatTemplate == nullptr) {
        _chatTemplate = llama_model_chat_template(_model, nullptr);
    } else {
        _chatTemplate = strdup(chatTemplate);
        if (!_chatTemplate) {
            throw std::runtime_error("strdup() failed for chatTemplate (out of memory)");
        }
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
    // apply the chat-template
    int newLen = llama_chat_apply_template(_chatTemplate, _messages.data(), _messages.size(), true,
                                           _formattedMessages.data(), _formattedMessages.size());
    if (newLen > (int) _formattedMessages.size()) {
        // resize the output buffer `_formattedMessages`
        // and re-apply the chat template
        _formattedMessages.resize(newLen);
        newLen = llama_chat_apply_template(_chatTemplate, _messages.data(), _messages.size(), true,
                                           _formattedMessages.data(), _formattedMessages.size());
    }
    if (newLen < 0) {
        throw std::runtime_error("llama_chat_apply_template() in LLMInference::startCompletion() failed");
    }

    // --- System prompt KV cache snapshotting ---
    bool restoredFromSnapshot = false;
    int snapshotNPast = 0;
    _eogReached = false;

    if (_prevLen == 0) {
        size_t systemMsgCount = 0;
        std::string systemContent;
        for (const auto& msg : _messages) {
            if (std::strcmp(msg.role, "system") == 0) {
                systemContent += msg.content;
                systemMsgCount++;
            }
        }

        if (systemMsgCount > 0) {
            size_t currentHash = std::hash<std::string>{}(systemContent);

            int sysTemplateLen = llama_chat_apply_template(
                _chatTemplate, _messages.data(), systemMsgCount, false, nullptr, 0);
            if (sysTemplateLen < 0) sysTemplateLen = 0;

            if (currentHash == _cachedSystemPromptHash && !_systemPromptKVSnapshot.empty()) {
                // Restore KV state from cached snapshot
                LOGi("Restoring system prompt KV snapshot (hash=%zu, tokens=%d)",
                     currentHash, _systemPromptTokenCount);
                llama_memory_seq_rm(llama_get_memory(_ctx), -1, -1, -1);
                size_t nset = llama_state_seq_set_data(
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
            } else if (sysTemplateLen > 0) {
                // Decode system prompt and create a new snapshot
                LOGi("Creating system prompt KV snapshot (hash=%zu)", currentHash);
                llama_memory_seq_rm(llama_get_memory(_ctx), -1, -1, -1);

                std::string sysPrompt(_formattedMessages.begin(),
                                      _formattedMessages.begin() + sysTemplateLen);
                std::vector<llama_token> sysTokens = common_tokenize(
                    llama_model_get_vocab(_model), sysPrompt, true, true);

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
                        size_t stateSize = llama_state_seq_get_size(_ctx, 0);
                        _systemPromptKVSnapshot.resize(stateSize);
                        size_t ncopy = llama_state_seq_get_data(
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
    _promptTokens = common_tokenize(llama_model_get_vocab(_model), prompt, add_special, true);
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
         int max_seq_pos = llama_memory_seq_pos_max(llama_get_memory(_ctx), 0);
         if (max_seq_pos >= 0) {
             n_past = max_seq_pos + 1;
         }
         LOGi("Preserving prepared KV cache for completion (n_past=%d)", n_past);
    } else if (_storeChats && _prevLen > 0) {
         int max_seq_pos = llama_memory_seq_pos_max(llama_get_memory(_ctx), 0);
         if (max_seq_pos >= 0) {
             n_past = max_seq_pos + 1;
         }
    } else {
         n_past = 0;
         llama_memory_seq_rm(llama_get_memory(_ctx), -1, -1, -1);
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
    _nCtxUsed = llama_memory_seq_pos_max(llama_get_memory(_ctx), 0) + 1;
    if (_nCtxUsed + _batch->n_tokens > contextSize) {
        throw std::runtime_error("context size reached");
    }

    auto start = ggml_time_us();
    // run the model
    if (llama_decode(_ctx, *_batch) < 0) {
        throw std::runtime_error("llama_decode() failed");
    }

    // sample a token and check if it is an EOG (end of generation token)
    // convert the integer token to its corresponding word-piece
    _currToken = llama_sampler_sample(_sampler, _ctx, -1);
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
        _prevLen = llama_chat_apply_template(_chatTemplate, _messages.data(), _messages.size(), false, nullptr, 0);
        if (_prevLen < 0) {
            throw std::runtime_error("llama_chat_apply_template() in LLMInference::stopCompletion() failed");
        }
    } else {
        _prevLen = 0;
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
    _nPast = 0;
    _nCtxUsed = 0;
    _response.clear();
    _cacheResponseTokens.clear();
    _promptTokens.clear();
    _preservePreparedKvForNextCompletion = false;
    _eogReached = false;
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
    llama_model_free(_model);
    delete _batch;
    llama_sampler_free(_sampler);
}

// Safe accessors used by JNI/native glue. Return internal pointers; caller must not free.
llama_model* LLMInference::getModel() {
    return _model;
}

llama_context* LLMInference::getContext() {
    return _ctx;
}
