#pragma once

#include <jni.h>

#include <cstdint>
#include <cstring>
#include <mutex>
#include <string>

// Single process-wide mutex for setenv/unsetenv/getenv around backend selection.
// POSIX environment mutation is not thread-safe against concurrent getenv, and the
// whisper and stable-diffusion loaders both tweak env vars — they must share one lock.
inline std::mutex& llmedge_process_env_mutex() {
    static std::mutex env_mutex;
    return env_mutex;
}

inline void llmedge_throw_java_exception(JNIEnv* env, const char* class_name, const char* message) {
    if (!env) {
        return;
    }
    jclass ex_class = env->FindClass(class_name);
    if (!ex_class) {
        return;
    }
    env->ThrowNew(ex_class, message);
}

inline void llmedge_clear_global_ref(JNIEnv* env, jobject& ref) {
    if (ref && env) {
        env->DeleteGlobalRef(ref);
    }
    ref = nullptr;
}

inline jmethodID llmedge_get_callback_method(
        JNIEnv* env,
        jobject callback,
        const char* name,
        const char* signature,
        const char* error_class,
        const char* error_message) {
    if (!env || !callback) {
        return nullptr;
    }
    jclass callback_class = env->GetObjectClass(callback);
    if (!callback_class) {
        llmedge_throw_java_exception(env, error_class, error_message);
        return nullptr;
    }
    jmethodID method_id = env->GetMethodID(callback_class, name, signature);
    env->DeleteLocalRef(callback_class);
    if (!method_id) {
        llmedge_throw_java_exception(env, error_class, error_message);
    }
    return method_id;
}

// Build a java.lang.String from standard UTF-8 bytes. NewStringUTF requires
// *Modified* UTF-8 and misbehaves (CheckJNI abort / corruption) on 4-byte
// sequences such as emoji, so untrusted/model-produced text must go through
// `new String(byte[], "UTF-8")` instead.
inline jstring llmedge_new_string_utf8(JNIEnv* env, const char* utf8) {
    if (!env) {
        return nullptr;
    }
    if (!utf8) {
        utf8 = "";
    }
    const size_t len = std::strlen(utf8);
    jbyteArray bytes = env->NewByteArray(static_cast<jsize>(len));
    if (!bytes) {
        return nullptr;
    }
    if (len > 0) {
        env->SetByteArrayRegion(bytes, 0, static_cast<jsize>(len),
                                reinterpret_cast<const jbyte*>(utf8));
    }
    jclass string_class = env->FindClass("java/lang/String");
    if (!string_class) {
        env->DeleteLocalRef(bytes);
        return nullptr;
    }
    jmethodID ctor = env->GetMethodID(string_class, "<init>", "([BLjava/lang/String;)V");
    jstring charset = env->NewStringUTF("UTF-8");
    jstring result = nullptr;
    if (ctor && charset) {
        result = static_cast<jstring>(env->NewObject(string_class, ctor, bytes, charset));
    }
    if (charset) env->DeleteLocalRef(charset);
    env->DeleteLocalRef(string_class);
    env->DeleteLocalRef(bytes);
    return result;
}

// Convert a java.lang.String to *standard* UTF-8. GetStringUTFChars yields
// Modified UTF-8 (supplementary characters such as emoji become 6-byte
// surrogate sequences, U+0000 becomes 0xC0 0x80), which native tokenizers
// reject as invalid bytes — so go through UTF-16 and encode real UTF-8.
// Returns false only if the UTF-16 chars could not be obtained (a Java
// exception is then pending). A null jstring converts to "" and returns true.
inline bool llmedge_jstring_to_utf8(JNIEnv* env, jstring value, std::string* out) {
    out->clear();
    if (!value) {
        return true;
    }
    const jsize len = env->GetStringLength(value);
    const jchar* chars = env->GetStringChars(value, nullptr);
    if (!chars) {
        return false;
    }
    out->reserve(static_cast<size_t>(len) * 3);
    for (jsize i = 0; i < len; ++i) {
        uint32_t cp = chars[i];
        if (cp >= 0xD800 && cp <= 0xDBFF && i + 1 < len) {
            const uint32_t low = chars[i + 1];
            if (low >= 0xDC00 && low <= 0xDFFF) {
                cp = 0x10000 + ((cp - 0xD800) << 10) + (low - 0xDC00);
                ++i;
            }
        }
        if (cp >= 0xD800 && cp <= 0xDFFF) {
            cp = 0xFFFD;  // unpaired surrogate -> replacement character
        }
        if (cp < 0x80) {
            out->push_back(static_cast<char>(cp));
        } else if (cp < 0x800) {
            out->push_back(static_cast<char>(0xC0 | (cp >> 6)));
            out->push_back(static_cast<char>(0x80 | (cp & 0x3F)));
        } else if (cp < 0x10000) {
            out->push_back(static_cast<char>(0xE0 | (cp >> 12)));
            out->push_back(static_cast<char>(0x80 | ((cp >> 6) & 0x3F)));
            out->push_back(static_cast<char>(0x80 | (cp & 0x3F)));
        } else {
            out->push_back(static_cast<char>(0xF0 | (cp >> 18)));
            out->push_back(static_cast<char>(0x80 | ((cp >> 12) & 0x3F)));
            out->push_back(static_cast<char>(0x80 | ((cp >> 6) & 0x3F)));
            out->push_back(static_cast<char>(0x80 | (cp & 0x3F)));
        }
    }
    env->ReleaseStringChars(value, chars);
    return true;
}

// Convenience overload for call sites that treat conversion failure as "".
inline std::string llmedge_jstring_to_utf8(JNIEnv* env, jstring value) {
    std::string out;
    llmedge_jstring_to_utf8(env, value, &out);
    return out;
}

inline jobject llmedge_new_global_ref_or_throw(JNIEnv* env, jobject target, const char* oom_message) {
    if (!env || !target) {
        return nullptr;
    }
    jobject ref = env->NewGlobalRef(target);
    if (!ref) {
        llmedge_throw_java_exception(env, "java/lang/OutOfMemoryError", oom_message);
    }
    return ref;
}
