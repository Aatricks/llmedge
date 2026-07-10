#pragma once
#include "llmedge_llama_compat.h"
#include "llama.h"
#include "chat.h"
#include "common.h"
#include "sampling.h"
#include <atomic>
#include <string>
#include <vector>
#include <functional>

class LLMInference {
    enum class ChatFormatterMode {
        LEGACY,
        JINJA,
    };

    struct FormattedPrompt {
        std::string prompt;
        int         renderedLength = 0;
        ChatFormatterMode modeUsed = ChatFormatterMode::LEGACY;
    };

    // llama.cpp-specific types
    llama_context* _ctx = nullptr;
    llama_model*   _model = nullptr;
    common_sampler* _sampler = nullptr;
    llama_token    _currToken = 0;
    llama_batch*   _batch = nullptr;

    // container to store user/assistant messages in the chat
    std::vector<llama_chat_message> _messages;
    // stores the string generated after applying
    // the chat-template to all messages in `_messages`
    std::vector<char> _formattedMessages;
    // stores the tokens for the last query
    // appended to `_messages`
    std::vector<llama_token> _promptTokens;
    std::vector<llama_pos>   _batchPos;
    int                      _nPast = 0;
    int                      _prevLen = 0;
    std::string              _chatTemplateSrc;
    common_chat_templates_ptr _chatTemplates;
    ChatFormatterMode        _chatFormatterMode = ChatFormatterMode::LEGACY;
    ChatFormatterMode        _prevFormatterMode = ChatFormatterMode::LEGACY;

    // stores the complete response for the given query
    std::string _response;
    std::string _cacheResponseTokens;
    // whether to cache previous messages in `_messages`
    bool _storeChats = true;
    bool _disableThinking = false;
    int  _reasoningBudget = -1;

    // CPU thread affinity mask for big.LITTLE pinning (0 = no affinity)
    uint64_t _coreMask = 0;

    // response generation metrics
    int64_t _responseGenerationTime = 0;
    long    _responseNumTokens      = 0;

    // length of context window consumed during the conversation
    int _nCtxUsed = 0;

    // System prompt KV cache snapshot
    std::vector<uint8_t> _systemPromptKVSnapshot;
    size_t _cachedSystemPromptHash = 0;
    int _systemPromptTokenCount = 0;

    // One-shot flag used by the multimodal projector path to preserve pre-decoded
    // image embeddings in the KV cache for the next completion request.
    bool _preservePreparedKvForNextCompletion = false;

    // Set when an EOG (end-of-generation) token is received so that
    // subsequent calls to completionLoop() return "[EOG]" immediately
    // instead of attempting another llama_decode with stale batch data.
    bool _eogReached = false;

    // Set by stopCompletion() (possibly from another thread, between native
    // calls) so the next completionLoop() ends the stream instead of decoding.
    // Cleared by startCompletion().
    std::atomic<bool> _stopRequested{false};

    // Set when a completion ended because the context window filled up
    // mid-generation (the partial response was still returned to the caller).
    bool _contextLimitReached = false;

    bool _isValidUtf8(const char* response);
    FormattedPrompt formatChatMessages(size_t messageCount, bool addGenerationPrompt);
    FormattedPrompt formatChatMessagesJinja(size_t messageCount, bool addGenerationPrompt) const;
    FormattedPrompt formatChatMessagesLegacy(size_t messageCount, bool addGenerationPrompt) const;
    std::vector<common_chat_msg> buildCommonChatMessages(size_t messageCount) const;
    void downgradeChatFormatter(const char* reason);

  public:
    void loadModel(const char* modelPath, float minP, float temperature, bool storeChats, long contextSize,
                   const char* chatTemplate, int nThreads, bool useMmap, bool useMlock, int backendId,
                   bool useFlashAttn = true, int kvCacheTypeKCode = -1, int kvCacheTypeVCode = -1,
                   int nGpuLayers = 99, int nUbatch = 0);

    void addChatMessage(const char* message, const char* role);

    float getResponseGenerationTime() const;

    float getResponseTokensPerSecond() const;

    long getResponseTokenCount() const;

    int64_t getResponseGenerationTimeMicros() const;

    uint64_t getEstimatedMemoryBytes() const;

    uint64_t getStateMemoryBytes() const;

    int getContextSizeUsed() const;

    bool isContextLimitReached() const;

    void startCompletion(const char* query);

    std::string completionLoop();

    std::string completionLoopBatch(int maxTokens);

    void stopCompletion();

    void clearMessages();

    void markPreparedKvForNextCompletion();

    void setReasoningOptions(bool disableThinking, int reasoningBudget);

    void configureThreading(int generationThreads, int promptThreads);

    void setThreadAffinity(uint64_t coreMask);

    ~LLMInference();

    // Expose internal model/context for JNI integrations (safe accessor)
    // These return the raw pointers managed by this instance. Caller must not free them.
    llama_model* getModel();
    llama_context* getContext();
};
