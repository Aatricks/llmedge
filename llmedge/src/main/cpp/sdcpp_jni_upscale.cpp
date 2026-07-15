#include "sdcpp_jni_shared.h"
#include "jni_utils.h"

extern "C" JNIEXPORT jintArray JNICALL Java_io_aatricks_llmedge_image_diffusion_StableDiffusion_nativeUpscale(
        JNIEnv* env, jclass clazz, jstring esrganPath, jint nThreads, jint tileSize, jstring backend,
        jintArray argbPixels, jint width, jint height, jint factor, jintArray outDims) {
    (void)clazz;
    std::string pathStr = "";
    if (esrganPath) {
        pathStr = llmedge_jstring_to_utf8(env, esrganPath);
    }
    std::string backendStr = "cpu";
    if (backend) {
        backendStr = llmedge_jstring_to_utf8(env, backend);
    }

    const int pixelCount = width * height;
    jint* pixels = env->GetIntArrayElements(argbPixels, nullptr);
    if (!pixels) {
        throwJavaException(env, "java/lang/IllegalArgumentException", "Failed to get input ARGB pixels");
        return nullptr;
    }

    uint8_t* rgbData = static_cast<uint8_t*>(malloc(pixelCount * 3));
    if (!rgbData) {
        env->ReleaseIntArrayElements(argbPixels, pixels, JNI_ABORT);
        throwJavaException(env, "java/lang/OutOfMemoryError", "Failed to allocate memory for upscaler input");
        return nullptr;
    }

    for (int i = 0; i < pixelCount; ++i) {
        jint val = pixels[i];
        rgbData[3 * i + 0] = static_cast<uint8_t>((val >> 16) & 0xFF);
        rgbData[3 * i + 1] = static_cast<uint8_t>((val >> 8) & 0xFF);
        rgbData[3 * i + 2] = static_cast<uint8_t>(val & 0xFF);
    }
    env->ReleaseIntArrayElements(argbPixels, pixels, JNI_ABORT);

    upscaler_ctx_t* ctx = new_upscaler_ctx(pathStr.c_str(), false, nThreads, tileSize, backendStr.c_str(), backendStr.c_str());
    if (!ctx) {
        free(rgbData);
        throwJavaException(env, "java/lang/RuntimeException", "Failed to create upscaler context");
        return nullptr;
    }

    sd_image_t input;
    input.width = static_cast<uint32_t>(width);
    input.height = static_cast<uint32_t>(height);
    input.channel = 3;
    input.data = rgbData;

    int effectiveFactor = (factor == 0) ? get_upscale_factor(ctx) : factor;

    sd_image_t* images_out = nullptr;
    int num_images_out = 0;
    bool success = upscale(ctx, input, static_cast<uint32_t>(effectiveFactor), &images_out, &num_images_out);

    free(rgbData);

    if (!success || num_images_out < 1) {
        if (images_out) {
            free_sd_images(images_out, num_images_out);
        }
        free_upscaler_ctx(ctx);
        throwJavaException(env, "java/lang/RuntimeException", "Upscale operation failed");
        return nullptr;
    }

    jintArray result = rgb_to_argb_int_array(env, images_out[0].data, images_out[0].width, images_out[0].height, images_out[0].channel);
    if (!result) {
        free_sd_images(images_out, num_images_out);
        free_upscaler_ctx(ctx);
        if (!env->ExceptionCheck()) {
            throwJavaException(env, "java/lang/OutOfMemoryError", "Failed to allocate ARGB destination array");
        }
        return nullptr;
    }

    if (outDims) {
        jsize outDimsLen = env->GetArrayLength(outDims);
        if (outDimsLen >= 2) {
            jint dims[2] = { static_cast<jint>(images_out[0].width), static_cast<jint>(images_out[0].height) };
            env->SetIntArrayRegion(outDims, 0, 2, dims);
        }
    }

    free_sd_images(images_out, num_images_out);
    free_upscaler_ctx(ctx);

    return result;
}
