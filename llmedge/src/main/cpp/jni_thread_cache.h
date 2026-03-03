/**
 * Thread-local JNIEnv cache using pthread TLS.
 *
 * Each native thread attaches to the JVM once on first callback and is
 * automatically detached when the thread exits (via the pthread_key
 * destructor).  This avoids the overhead of AttachCurrentThread /
 * DetachCurrentThread on every callback invocation.
 *
 * Usage:
 *   1. Call jni_thread_cache_init(jvm) once (e.g. in JNI_OnLoad or when
 *      the JavaVM* is first obtained).
 *   2. In callbacks, call jni_thread_cache_get_env() instead of the
 *      manual Attach/Detach dance.
 *
 * The helpers live in an unnamed namespace so each translation unit gets
 * its own copy — safe for separate .so files that are never linked together.
 */

#pragma once

#include <jni.h>
#include <pthread.h>

namespace {

static JavaVM* g_jvm = nullptr;
static pthread_key_t g_jni_env_key;
static pthread_once_t g_key_once = PTHREAD_ONCE_INIT;

static void jni_thread_cache_detach(void* env) {
    if (g_jvm && env) {
        g_jvm->DetachCurrentThread();
    }
}

static void jni_thread_cache_make_key() {
    pthread_key_create(&g_jni_env_key, jni_thread_cache_detach);
}

// Call once to store the JavaVM pointer used for attaching threads.
static void jni_thread_cache_init(JavaVM* jvm) {
    g_jvm = jvm;
    pthread_once(&g_key_once, jni_thread_cache_make_key);
}

// Get JNIEnv for the current thread, attaching if needed.
// The thread will be auto-detached when it exits.
static JNIEnv* jni_thread_cache_get_env() {
    pthread_once(&g_key_once, jni_thread_cache_make_key);
    auto* env = static_cast<JNIEnv*>(pthread_getspecific(g_jni_env_key));
    if (env) return env;

    if (!g_jvm) return nullptr;

    // Check if already attached (e.g. the JNI calling thread).
    jint status = g_jvm->GetEnv(reinterpret_cast<void**>(&env), JNI_VERSION_1_6);
    if (status == JNI_OK) {
        // Already attached — cache but do NOT register for auto-detach
        // because this may be a JVM-owned thread (main / binder).
        return env;
    }

    if (status == JNI_EDETACHED) {
#if defined(__ANDROID__)
        if (g_jvm->AttachCurrentThread(&env, nullptr) != JNI_OK) {
#else
        if (g_jvm->AttachCurrentThread(reinterpret_cast<void**>(&env), nullptr) != JNI_OK) {
#endif
            return nullptr;
        }
        pthread_setspecific(g_jni_env_key, env);
        return env;
    }

    return nullptr;
}

} // unnamed namespace
