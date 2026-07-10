#include "jni_thread_cache.h"
#include <pthread.h>

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

void jni_thread_cache_init(JavaVM* jvm) {
    g_jvm = jvm;
    pthread_once(&g_key_once, jni_thread_cache_make_key);
}

JNIEnv* jni_thread_cache_get_env() {
    pthread_once(&g_key_once, jni_thread_cache_make_key);
    auto* env = static_cast<JNIEnv*>(pthread_getspecific(g_jni_env_key));
    if (env) return env;

    if (!g_jvm) return nullptr;

    jint status = g_jvm->GetEnv(reinterpret_cast<void**>(&env), JNI_VERSION_1_6);
    if (status == JNI_OK) {
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
