#include "sd_jni_internal.h"
#include "jni_utils.h"

#include <jni.h>

#include <atomic>
#include <iostream>
#include <fstream>
#include <vector>
#include <limits>

static void save_frame_as_ppm(const uint8_t* data, int width, int height, int frame_index) {
    std::string filename = "/tmp/cat_frame_" + std::to_string(frame_index) + ".ppm";
    std::ofstream file(filename, std::ios::binary);
    if (!file) {
        std::cerr << "Failed to open " << filename << " for writing" << std::endl;
        return;
    }
    
    // PPM header
    file << "P6\n" << width << " " << height << "\n255\n";
    
    // Write RGB data (assuming 3 channels)
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            int idx = (y * width + x) * 3;
            file.write(reinterpret_cast<const char*>(&data[idx]), 3);
        }
    }
    
    file.close();
    std::cout << "Saved frame " << frame_index << " to " << filename << std::endl;
}

#ifndef SD_TEST_JAVA_CLASS_DIR
#define SD_TEST_JAVA_CLASS_DIR "."
#endif

extern "C" {
JNIEXPORT jlong JNICALL Java_io_aatricks_llmedge_image_diffusion_StableDiffusion_nativeCreate(
    JNIEnv* env, jclass clazz, jstring jModelPath, jstring jVaePath, jstring jT5xxlPath,
    jstring jTaesdPath, jstring jDiffusionModelPath, jstring jLlmPath,
    jstring jClipLPath, jstring jClipGPath, jstring jClipVisionPath, jstring jLlmVisionPath,
    jstring jHighNoiseDiffusionModelPath, jstring jEmbeddingsConnectorsPath,
    jstring jAudioVaePath, jstring jControlNetPath, jstring jPhotoMakerPath,
    jint nThreads, jboolean enableOpenCl, jboolean useVulkan, jboolean offloadToCpu,
    jboolean keepClipOnCpu, jboolean keepVaeOnCpu, jboolean flashAttn, jboolean jvaeDecodeOnly,
    jfloat flowShift, jstring jLoraModelDir, jint jLoraApplyMode, jboolean jMiniT2iConditionerOnly,
    jstring jWeightType, jstring jTensorTypeRules);
JNIEXPORT void JNICALL Java_io_aatricks_llmedge_image_diffusion_StableDiffusion_nativeDestroy(JNIEnv* env, jobject thiz, jlong handlePtr);
JNIEXPORT jobjectArray JNICALL Java_io_aatricks_llmedge_image_diffusion_StableDiffusion_nativeTxt2Vid(
    JNIEnv* env, jobject thiz, jlong handlePtr,
    jstring jPrompt, jstring jNegative,
    jint width, jint height,
    jint videoFrames, jint steps, jfloat cfg, jlong seed,
    jint jSampleMethod, jint jScheduler, jfloat jStrength,
    jbyteArray jInitImage, jint initWidth, jint initHeight,
    jfloat jVaceStrength,
    jboolean jEasyCacheEnabled, jfloat jEasyCacheReuseThreshold, jfloat jEasyCacheStartPercent, jfloat jEasyCacheEndPercent);
JNIEXPORT void JNICALL Java_io_aatricks_llmedge_image_diffusion_StableDiffusion_nativeSetProgressCallback(
        JNIEnv* env, jobject thiz, jlong handlePtr, jobject progressCallback);
JNIEXPORT void JNICALL Java_io_aatricks_llmedge_image_diffusion_StableDiffusion_nativeCancelGeneration(
        JNIEnv* env, jobject thiz, jlong handlePtr);
}


static std::atomic<int> g_frameBufferFrees{0};
static std::atomic<int> g_frameArrayFrees{0};

void sd_jni_notify_frame_buffer_freed(const void* ptr) {
    if (ptr) {
        g_frameBufferFrees.fetch_add(1);
    }
}

void sd_jni_notify_frame_array_freed(const void* ptr) {
    if (ptr) {
        g_frameArrayFrees.fetch_add(1);
    }
}

static void reset_free_counters() {
    g_frameBufferFrees.store(0);
    g_frameArrayFrees.store(0);
}

class JvmHolder {
public:
    JvmHolder() {
        JavaVMInitArgs vm_args{};
        vm_args.version = JNI_VERSION_1_8;

        std::string classPathOption = std::string("-Djava.class.path=") + SD_TEST_JAVA_CLASS_DIR;
        options_.push_back(classPathOption);

        std::vector<JavaVMOption> vmOptions;
        for (auto& opt : options_) {
            JavaVMOption option{};
            option.optionString = opt.data();
            vmOptions.push_back(option);
        }
        vm_args.nOptions = static_cast<jint>(vmOptions.size());
        vm_args.options = vmOptions.data();
        vm_args.ignoreUnrecognized = JNI_TRUE;

        JNIEnv* envLocal = nullptr;
        if (JNI_CreateJavaVM(&vm_, reinterpret_cast<void**>(&envLocal), &vm_args) != JNI_OK) {
            throw std::runtime_error("Failed to create JVM for native tests");
        }
        env_ = envLocal;
    }

    ~JvmHolder() {
        if (vm_) {
            vm_->DestroyJavaVM();
        }
    }

    JNIEnv* env() const { return env_; }
    JavaVM* vm() const { return vm_; }

private:
    JavaVM* vm_ = nullptr;
    JNIEnv* env_ = nullptr;
    std::vector<std::string> options_;
};

static bool test_nativeTxt2Vid_memory(JNIEnv* env) {
    reset_free_counters();
    jstring modelPath = env->NewStringUTF("stub-model.gguf");
        jlong handle = Java_io_aatricks_llmedge_image_diffusion_StableDiffusion_nativeCreate(
            env, nullptr, modelPath, nullptr, nullptr, nullptr, nullptr, nullptr,
            nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, 4,
            JNI_FALSE, JNI_FALSE, JNI_FALSE, JNI_FALSE, JNI_FALSE, JNI_FALSE, JNI_FALSE,
            std::numeric_limits<float>::infinity(), nullptr, 0, JNI_FALSE, nullptr, nullptr);
    env->DeleteLocalRef(modelPath);
    if (handle == 0) {
        std::cerr << "nativeCreate returned null handle" << std::endl;
        return false;
    }

    jstring prompt = env->NewStringUTF("a cat playing with yarn");
        jobjectArray frames = Java_io_aatricks_llmedge_image_diffusion_StableDiffusion_nativeTxt2Vid(
            env, nullptr, handle, prompt, nullptr,
            256, 256, 4, 12, 7.5f, 42L,
            0, 0, 0.8f,
            nullptr, 0, 0,
            0.0f,
            JNI_FALSE, 0.2f, 0.15f, 0.95f);
    env->DeleteLocalRef(prompt);

    bool success = true;
    if (!frames) {
        std::cerr << "nativeTxt2Vid returned null" << std::endl;
        success = false;
    } else {
        const jsize frameCount = env->GetArrayLength(frames);
        if (frameCount != 4) {
            std::cerr << "Expected 4 frames, got " << frameCount << std::endl;
            success = false;
        }
        const jsize expectedBytes = 256 * 256 * 3;
        for (jsize i = 0; i < frameCount; ++i) {
            auto frameBytes = static_cast<jbyteArray>(env->GetObjectArrayElement(frames, i));
            if (!frameBytes) {
                std::cerr << "Frame " << i << " is null" << std::endl;
                success = false;
                continue;
            }
            jsize length = env->GetArrayLength(frameBytes);
            if (length != expectedBytes) {
                std::cerr << "Frame " << i << " expected " << expectedBytes
                          << " bytes, got " << length << std::endl;
                success = false;
            }
            env->DeleteLocalRef(frameBytes);
        }
        
        // Save frames as PPM files for video creation
        for (jsize i = 0; i < frameCount; ++i) {
            auto frameBytes = static_cast<jbyteArray>(env->GetObjectArrayElement(frames, i));
            if (frameBytes) {
                jsize length = env->GetArrayLength(frameBytes);
                std::vector<uint8_t> frameData(static_cast<size_t>(length));
                env->GetByteArrayRegion(frameBytes, 0, length, reinterpret_cast<jbyte*>(frameData.data()));
                save_frame_as_ppm(frameData.data(), 256, 256, static_cast<int>(i));
                env->DeleteLocalRef(frameBytes);
            }
        }
    }

    Java_io_aatricks_llmedge_image_diffusion_StableDiffusion_nativeDestroy(env, nullptr, handle);

    if (g_frameBufferFrees.load() != 4) {
        std::cerr << "Expected 4 frame buffer frees, got " << g_frameBufferFrees.load() << std::endl;
        success = false;
    }
    if (g_frameArrayFrees.load() != 1) {
        std::cerr << "Expected 1 frame array free, got " << g_frameArrayFrees.load() << std::endl;
        success = false;
    }
    return success;
}

static bool test_progressive_loading_e2e(JNIEnv* env) {
    std::cout << "=== Testing Progressive Loading E2E ===" << std::endl;
    
    // Step 1: Load T5-only context
    std::cout << "Step 1: Loading T5-only context..." << std::endl;
    jstring t5Path = env->NewStringUTF("stub-t5.gguf");
        jlong t5Handle = Java_io_aatricks_llmedge_image_diffusion_StableDiffusion_nativeCreate(
        env, nullptr, t5Path, nullptr, nullptr, nullptr, nullptr, nullptr,
        nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, 4,
        JNI_FALSE, JNI_FALSE, JNI_FALSE, JNI_FALSE, JNI_FALSE, JNI_FALSE, JNI_FALSE,
        std::numeric_limits<float>::infinity(), nullptr, 0, JNI_FALSE, nullptr, nullptr);
    env->DeleteLocalRef(t5Path);
    
    if (t5Handle == 0) {
        std::cerr << "Failed to create T5-only context" << std::endl;
        return false;
    }
    std::cout << "T5-only context created successfully" << std::endl;
    
    // Step 2: Precompute conditions using T5
    std::cout << "Step 2: Precomputing conditions..." << std::endl;
    jstring prompt = env->NewStringUTF("1girl");
    jstring negPrompt = env->NewStringUTF("");
    
    // Call the precompute function (we'll need to add this to the JNI interface)
    // For now, simulate by calling txt2vid which should internally use precompute
        jobjectArray frames = Java_io_aatricks_llmedge_image_diffusion_StableDiffusion_nativeTxt2Vid(
            env, nullptr, t5Handle, prompt, negPrompt,
            256, 256, 13, 12, 7.5f, 42L,
            0, 0, 0.8f,
            nullptr, 0, 0,
            0.0f,
            JNI_FALSE, 0.2f, 0.15f, 0.95f);
    
    if (!frames) {
        std::cerr << "Precompute conditions failed" << std::endl;
        Java_io_aatricks_llmedge_image_diffusion_StableDiffusion_nativeDestroy(env, nullptr, t5Handle);
        env->DeleteLocalRef(prompt);
        env->DeleteLocalRef(negPrompt);
        return false;
    }
    std::cout << "Conditions precomputed successfully" << std::endl;
    
    // Step 3: Free T5 context
    std::cout << "Step 3: Freeing T5 context..." << std::endl;
    Java_io_aatricks_llmedge_image_diffusion_StableDiffusion_nativeDestroy(env, nullptr, t5Handle);
    std::cout << "T5 context freed" << std::endl;
    
    // Step 4: Load diffusion model
    std::cout << "Step 4: Loading diffusion model..." << std::endl;
    jstring modelPath = env->NewStringUTF("stub-diffusion.safetensors");
    jstring vaePath = env->NewStringUTF("stub-vae.safetensors");
        jlong diffusionHandle = Java_io_aatricks_llmedge_image_diffusion_StableDiffusion_nativeCreate(
            env, nullptr, modelPath, vaePath, nullptr, nullptr, nullptr, nullptr,
            nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, 4,
            JNI_FALSE, JNI_FALSE, JNI_FALSE, JNI_FALSE, JNI_FALSE, JNI_FALSE, JNI_FALSE,
            std::numeric_limits<float>::infinity(), nullptr, 0, JNI_FALSE, nullptr, nullptr);
    env->DeleteLocalRef(modelPath);
    env->DeleteLocalRef(vaePath);
    
    if (diffusionHandle == 0) {
        std::cerr << "Failed to create diffusion context" << std::endl;
        env->DeleteLocalRef(prompt);
        env->DeleteLocalRef(negPrompt);
        return false;
    }
    std::cout << "Diffusion model loaded successfully" << std::endl;
    
    // Step 5: Generate video with precomputed conditions
    std::cout << "Step 5: Generating video..." << std::endl;
    // This would use the new nativeTxt2VidWithPrecomputedCondition function
    // For now, test with regular txt2vid
    frames = Java_io_aatricks_llmedge_image_diffusion_StableDiffusion_nativeTxt2Vid(
            env, nullptr, diffusionHandle, prompt, negPrompt,
            256, 256, 13, 12, 7.5f, 42L,
            0, 0, 0.8f, // sample method, scheduler, strength
            nullptr, 0, 0,
            0.0f, // vace strength
            JNI_FALSE, 0.2f, 0.15f, 0.95f); // easycache params
    
    env->DeleteLocalRef(prompt);
    env->DeleteLocalRef(negPrompt);
    
    if (!frames) {
        std::cerr << "Video generation failed" << std::endl;
        Java_io_aatricks_llmedge_image_diffusion_StableDiffusion_nativeDestroy(env, nullptr, diffusionHandle);
        return false;
    }
    
    const jsize frameCount = env->GetArrayLength(frames);
    std::cout << "Video generation successful! Generated " << frameCount << " frames" << std::endl;
    
    // Cleanup
    Java_io_aatricks_llmedge_image_diffusion_StableDiffusion_nativeDestroy(env, nullptr, diffusionHandle);
    
    return frameCount > 0;
}

static bool test_progress_callback_bridge(JNIEnv* env) {
    jstring modelPath = env->NewStringUTF("stub-model.gguf");
        jlong handlePtr = Java_io_aatricks_llmedge_image_diffusion_StableDiffusion_nativeCreate(
            env, nullptr, modelPath, nullptr, nullptr, nullptr, nullptr, nullptr,
            nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, 2,
            JNI_FALSE, JNI_FALSE, JNI_FALSE, JNI_FALSE, JNI_FALSE, JNI_FALSE, JNI_FALSE,
            std::numeric_limits<float>::infinity(), nullptr, 0, JNI_FALSE, nullptr, nullptr);
    env->DeleteLocalRef(modelPath);
    if (handlePtr == 0) {
        std::cerr << "Failed to create handle for progress test" << std::endl;
        return false;
    }

    jclass callbackClass = env->FindClass("io/aatricks/llmedge/NativeTestProgressCallback");
    if (!callbackClass) {
        std::cerr << "Unable to locate NativeTestProgressCallback" << std::endl;
        Java_io_aatricks_llmedge_image_diffusion_StableDiffusion_nativeDestroy(env, nullptr, handlePtr);
        return false;
    }
    jmethodID ctor = env->GetMethodID(callbackClass, "<init>", "()V");
    jobject callbackInstance = env->NewObject(callbackClass, ctor);
    if (!callbackInstance) {
        std::cerr << "Failed to instantiate progress callback" << std::endl;
        Java_io_aatricks_llmedge_image_diffusion_StableDiffusion_nativeDestroy(env, nullptr, handlePtr);
        return false;
    }

    Java_io_aatricks_llmedge_image_diffusion_StableDiffusion_nativeSetProgressCallback(env, nullptr, handlePtr, callbackInstance);

    auto* handle = reinterpret_cast<SdHandle*>(handlePtr);
    handle->totalFrames = 4;
    handle->stepsPerFrame = 3;
    handle->totalSteps = handle->totalFrames * handle->stepsPerFrame;
    handle->currentFrame = 0;

    try {
        sd_video_progress_wrapper(5, handle->totalSteps, 1.5f, handle);
    } catch (const std::exception& ex) {
        std::cerr << "Progress wrapper threw unexpectedly: " << ex.what() << std::endl;
        Java_io_aatricks_llmedge_image_diffusion_StableDiffusion_nativeDestroy(env, nullptr, handlePtr);
        return false;
    }

    jfieldID callCountField = env->GetFieldID(callbackClass, "callCount", "I");
    jfieldID lastFrameField = env->GetFieldID(callbackClass, "lastFrame", "I");
    jint callCount = env->GetIntField(callbackInstance, callCountField);
    jint reportedFrame = env->GetIntField(callbackInstance, lastFrameField);
    if (callCount == 0 || reportedFrame != 1) {
        std::cerr << "Progress callback did not update fields as expected" << std::endl;
        Java_io_aatricks_llmedge_image_diffusion_StableDiffusion_nativeDestroy(env, nullptr, handlePtr);
        return false;
    }

    handle->cancellationRequested.store(true);
    bool cancellationRaised = false;
    try {
        sd_video_progress_wrapper(6, handle->totalSteps, 2.0f, handle);
    } catch (const std::runtime_error&) {
        cancellationRaised = true;
    }
    if (!cancellationRaised) {
        std::cerr << "Expected cancellation exception" << std::endl;
        Java_io_aatricks_llmedge_image_diffusion_StableDiffusion_nativeDestroy(env, nullptr, handlePtr);
        return false;
    }

    Java_io_aatricks_llmedge_image_diffusion_StableDiffusion_nativeCancelGeneration(env, nullptr, handlePtr);
    Java_io_aatricks_llmedge_image_diffusion_StableDiffusion_nativeDestroy(env, nullptr, handlePtr);
    env->DeleteLocalRef(callbackInstance);
    return true;
}

// Verifies llmedge_jstring_to_utf8 produces *standard* UTF-8 for supplementary
// characters (emoji), where GetStringUTFChars would produce Modified UTF-8
// (a 6-byte surrogate encoding native tokenizers reject).
static bool test_jstring_utf8_conversion(JNIEnv* env) {
    // "hi <U+1F600> <U+00E9>" — U+1F600 needs a surrogate pair in UTF-16.
    const jchar utf16[] = {'h', 'i', ' ', 0xD83D, 0xDE00, ' ', 0x00E9};
    jstring value = env->NewString(utf16, sizeof(utf16) / sizeof(jchar));
    if (!value) {
        std::cerr << "NewString failed" << std::endl;
        return false;
    }

    const std::string expected = "hi \xF0\x9F\x98\x80 \xC3\xA9";  // real UTF-8
    std::string actual;
    if (!llmedge_jstring_to_utf8(env, value, &actual)) {
        std::cerr << "llmedge_jstring_to_utf8 reported failure" << std::endl;
        env->DeleteLocalRef(value);
        return false;
    }
    if (actual != expected) {
        std::cerr << "UTF-8 conversion mismatch: got " << actual.size()
                  << " bytes, expected " << expected.size() << std::endl;
        env->DeleteLocalRef(value);
        return false;
    }

    // Modified UTF-8 from GetStringUTFChars must differ on the emoji —
    // this documents why the byte path exists.
    const char* modified = env->GetStringUTFChars(value, nullptr);
    const bool modifiedDiffers = modified && expected != modified;
    if (modified) {
        env->ReleaseStringUTFChars(value, modified);
    }
    env->DeleteLocalRef(value);
    if (!modifiedDiffers) {
        std::cerr << "Expected GetStringUTFChars to yield Modified UTF-8 (differs from standard)" << std::endl;
        return false;
    }

    // Null string converts to empty.
    std::string fromNull = "sentinel";
    if (!llmedge_jstring_to_utf8(env, nullptr, &fromNull) || !fromNull.empty()) {
        std::cerr << "null jstring should convert to empty string" << std::endl;
        return false;
    }
    return true;
}

int main() {
    try {
        JvmHolder jvm;
        JNIEnv* env = jvm.env();
        bool txt2vidResult = test_nativeTxt2Vid_memory(env);
        bool progressResult = test_progress_callback_bridge(env);
        bool progressiveResult = test_progressive_loading_e2e(env);
        bool utf8Result = test_jstring_utf8_conversion(env);
        if (!txt2vidResult || !progressResult || !progressiveResult || !utf8Result) {
            std::cerr << "video_jni_tests FAILED" << std::endl;
            return 1;
        }
        std::cout << "video_jni_tests PASSED" << std::endl;
        return 0;
    } catch (const std::exception& ex) {
        std::cerr << "Exception during tests: " << ex.what() << std::endl;
        return 1;
    }
}
