#pragma once

#include <jni.h>

#include <cstring>
#include <mutex>

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
