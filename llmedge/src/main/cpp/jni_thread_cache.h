/*
 * Shared JNI thread-attach cache.
 *
 * Native callbacks (progress, segment, log) fire on internal worker threads
 * that are not attached to the JVM. Attaching/detaching per callback is
 * expensive and error-prone; this helper caches one JNIEnv per thread and
 * auto-detaches on thread exit.
 *
 * Usage:
 *   1. Call jni_thread_cache_init(jvm) once per .so (e.g. in JNI_OnLoad or when
 *      the JavaVM* is first obtained).
 *   2. In callbacks, call jni_thread_cache_get_env() instead of the
 *      manual Attach/Detach dance.
 *
 * The state lives in jni_thread_cache.cpp, compiled once into each shared
 * library, so every translation unit of a given .so sees the same JavaVM*.
 * (An earlier unnamed-namespace version gave each .cpp its own copy, which
 * silently disabled callbacks initialized in a different translation unit.)
 */

#pragma once

#include <jni.h>

void jni_thread_cache_init(JavaVM* jvm);
JNIEnv* jni_thread_cache_get_env();
