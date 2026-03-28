#pragma once

#include <jni.h>

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
