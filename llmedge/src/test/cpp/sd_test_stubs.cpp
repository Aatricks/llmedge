#include "sdcpp_jni_shared.h"
#include "model.h"

#include <cstdlib>
#include <cstring>
#include <map>
#include <string>
#include <vector>

extern "C" {

static sd_log_cb_t g_log_cb = nullptr;
static void* g_log_user_data = nullptr;
static sd_progress_cb_t g_progress_cb = nullptr;
static void* g_progress_user_data = nullptr;

void sd_set_log_callback(sd_log_cb_t cb, void* data) {
    g_log_cb = cb;
    g_log_user_data = data;
    if (g_log_cb) {
        g_log_cb(SD_LOG_INFO, "sd_set_log_callback invoked", g_log_user_data);
    }
}

void sd_set_progress_callback(sd_progress_cb_t cb, void* data) {
    g_progress_cb = cb;
    g_progress_user_data = data;
}

void sd_set_preview_callback(sd_preview_cb_t, enum preview_t, int, bool, bool, void*) {
    // Not used in tests.
}

int32_t sd_get_num_physical_cores() {
    return 4;
}

const char* sd_get_system_info() {
    return "sd_test_stubs";
}

const char* sd_type_name(enum sd_type_t) { return "stub"; }
enum sd_type_t str_to_sd_type(const char*) { return SD_TYPE_F16; }
const char* sd_rng_type_name(enum rng_type_t) { return "stub"; }
enum rng_type_t str_to_rng_type(const char*) { return STD_DEFAULT_RNG; }
const char* sd_sample_method_name(enum sample_method_t) { return "stub"; }
enum sample_method_t str_to_sample_method(const char*) { return EULER_SAMPLE_METHOD; }
const char* sd_scheduler_name(enum scheduler_t) { return "stub"; }
enum scheduler_t str_to_scheduler(const char*) { return DISCRETE_SCHEDULER; }
const char* sd_prediction_name(enum prediction_t) { return "stub"; }
enum prediction_t str_to_prediction(const char*) { return EPS_PRED; }
const char* sd_preview_name(enum preview_t) { return "stub"; }
enum preview_t str_to_preview(const char*) { return PREVIEW_NONE; }
const char* sd_lora_apply_mode_name(enum lora_apply_mode_t) { return "stub"; }
enum lora_apply_mode_t str_to_lora_apply_mode(const char*) { return LORA_APPLY_AUTO; }

void sd_ctx_params_init(sd_ctx_params_t* params) {
    std::memset(params, 0, sizeof(sd_ctx_params_t));
}

char* sd_ctx_params_to_str(const sd_ctx_params_t*) { return nullptr; }

typedef struct sd_ctx_t {
    int dummy;
} sd_ctx_t_impl;

sd_ctx_t* new_sd_ctx(const sd_ctx_params_t*) {
    return reinterpret_cast<sd_ctx_t*>(new sd_ctx_t_impl{1});
}

void free_sd_ctx(sd_ctx_t* ctx) {
    delete reinterpret_cast<sd_ctx_t_impl*>(ctx);
}

enum sample_method_t sd_get_default_sample_method(const sd_ctx_t*) {
    return EULER_SAMPLE_METHOD;
}

enum scheduler_t sd_get_default_scheduler(const sd_ctx_t*, enum sample_method_t) {
    return DISCRETE_SCHEDULER;
}

void sd_sample_params_init(sd_sample_params_t* params) {
    std::memset(params, 0, sizeof(sd_sample_params_t));
    params->sample_steps = 0;
}

char* sd_sample_params_to_str(const sd_sample_params_t*) { return nullptr; }

void sd_img_gen_params_init(sd_img_gen_params_t* params) {
    std::memset(params, 0, sizeof(sd_img_gen_params_t));
    params->sample_params.sample_steps = 0;
}

char* sd_img_gen_params_to_str(const sd_img_gen_params_t*) { return nullptr; }

void sd_vid_gen_params_init(sd_vid_gen_params_t* params) {
    std::memset(params, 0, sizeof(sd_vid_gen_params_t));
}

char* sd_vid_gen_params_to_str(const sd_vid_gen_params_t*) { return nullptr; }

static void fill_image(sd_image_t& image, int width, int height, int channel, uint8_t seed) {
    image.width = static_cast<uint32_t>(width);
    image.height = static_cast<uint32_t>(height);
    image.channel = static_cast<uint32_t>(channel);
    const size_t byteCount = static_cast<size_t>(width) * height * channel;
    image.data = static_cast<uint8_t*>(std::malloc(byteCount));
    for (size_t i = 0; i < byteCount; ++i) {
        image.data[i] = static_cast<uint8_t>(seed + (i % 253));
    }
}

bool generate_image(sd_ctx_t*, const sd_img_gen_params_t* params,
                    sd_image_t** images_out, int* num_images_out) {
    if (!images_out || !num_images_out) return false;
    auto* images = static_cast<sd_image_t*>(std::malloc(sizeof(sd_image_t)));
    fill_image(images[0], params->width > 0 ? params->width : 256,
               params->height > 0 ? params->height : 256,
               3, 42);
    *images_out = images;
    *num_images_out = 1;
    return true;
}

bool generate_video(sd_ctx_t*, const sd_vid_gen_params_t* params,
                    sd_image_t** frames_out, int* num_frames_out, sd_audio_t** audio_out) {
    const int frames = params->video_frames > 0 ? params->video_frames : 4;
    *num_frames_out = frames;
    auto* images = static_cast<sd_image_t*>(std::malloc(sizeof(sd_image_t) * frames));
    const int steps = params->sample_params.sample_steps > 0 ? params->sample_params.sample_steps : 10;

    for (int i = 0; i < frames; ++i) {
        fill_image(images[i], params->width > 0 ? params->width : 256,
                   params->height > 0 ? params->height : 256,
                   3, static_cast<uint8_t>(i));
        if (g_progress_cb) {
            const int frameBase = i * steps;
            for (int s = 0; s < steps; ++s) {
                g_progress_cb(frameBase + s, frames * steps, 0.1f * (frameBase + s), g_progress_user_data);
            }
        }
    }
    *frames_out = images;
    if (audio_out) {
        *audio_out = nullptr;
    }
    return true;
}

void free_sd_audio(sd_audio_t*) {}

upscaler_ctx_t* new_upscaler_ctx(const char* esrgan_path, bool direct, int n_threads, int tile_size, const char* backend, const char* params_backend) {
    (void)esrgan_path; (void)direct; (void)n_threads; (void)tile_size; (void)backend; (void)params_backend;
    return reinterpret_cast<upscaler_ctx_t*>(1);
}
void free_upscaler_ctx(upscaler_ctx_t*) {}
int get_upscale_factor(upscaler_ctx_t*) { return 4; }
bool upscale(upscaler_ctx_t* upscaler_ctx, sd_image_t input, uint32_t upscale_factor, sd_image_t** images_out, int* num_images_out) {
    (void)upscaler_ctx; (void)upscale_factor;
    if (g_progress_cb) {
        for (int i = 1; i <= 4; ++i) {
            g_progress_cb(i, 4, 0.5f, g_progress_user_data);
        }
    }
    if (!images_out || !num_images_out) return false;
    sd_image_t* out = (sd_image_t*)calloc(1, sizeof(sd_image_t));
    if (!out) return false;
    out->width = input.width * 4;
    out->height = input.height * 4;
    out->channel = 3;
    out->data = (uint8_t*)calloc(out->width * out->height * out->channel, sizeof(uint8_t));
    if (!out->data) {
        free(out);
        return false;
    }
    *images_out = out;
    *num_images_out = 1;
    return true;
}
void free_sd_images(sd_image_t* result_images, int num_images) {
    if (!result_images) return;
    for (int i = 0; i < num_images; ++i) {
        if (result_images[i].data) {
            free(result_images[i].data);
        }
    }
    free(result_images);
}

bool convert(const char*, const char*, const char*, enum sd_type_t, const char*, bool) { return true; }

bool preprocess_canny(sd_image_t, float, float, float, float, bool) { return true; }

}  // extern "C"

ModelLoader::ModelLoader() = default;
bool ModelLoader::init_from_file(const std::string&, const std::string&) { return true; }
int64_t ModelLoader::get_params_mem_size(ggml_backend_t, ggml_type) { return 1024 * 1024; }
std::map<ggml_type, uint32_t> ModelLoader::get_wtype_stat() { return {}; }
std::map<ggml_type, uint32_t> ModelLoader::get_conditioner_wtype_stat() { return {}; }
std::map<ggml_type, uint32_t> ModelLoader::get_diffusion_model_wtype_stat() { return {}; }
std::map<ggml_type, uint32_t> ModelLoader::get_vae_wtype_stat() { return {}; }
SDVersion ModelLoader::get_sd_version() { return VERSION_SD1; }
void ModelLoader::set_wtype_override(ggml_type, std::string) {}

extern "C" {

// Mock ggml_backend_free
void ggml_backend_free(ggml_backend_t) {}

sd_condition_raw_t* sd_precompute_condition(sd_ctx_t* sd_ctx,
                                            const char* prompt,
                                            int clip_skip,
                                            int width,
                                            int height,
                                            bool is_video) {
    (void)sd_ctx;
    (void)prompt;
    (void)clip_skip;
    (void)width;
    (void)height;
    (void)is_video;

    sd_condition_raw_t* cond = (sd_condition_raw_t*)calloc(1, sizeof(sd_condition_raw_t));
    if (!cond) return nullptr;

    // c_crossattn: 2D tensor shaped as [4, 4] with deterministic values
    cond->c_crossattn.ndims = 2;
    cond->c_crossattn.ne[0] = 4;
    cond->c_crossattn.ne[1] = 4;
    cond->c_crossattn.ne[2] = 0;
    cond->c_crossattn.ne[3] = 0;
    size_t crossCount = (size_t)cond->c_crossattn.ne[0] * cond->c_crossattn.ne[1];
    cond->c_crossattn.data = (float*)malloc(sizeof(float) * crossCount);
    if (cond->c_crossattn.data) {
        for (size_t i = 0; i < crossCount; ++i) {
            cond->c_crossattn.data[i] = 0.05f * (float)(i + 1);
        }
    }

    // c_vector: 1D tensor shaped as [1]
    cond->c_vector.ndims = 1;
    cond->c_vector.ne[0] = 1;
    cond->c_vector.ne[1] = 0;
    cond->c_vector.ne[2] = 0;
    cond->c_vector.ne[3] = 0;
    cond->c_vector.data = (float*)malloc(sizeof(float) * 1);
    if (cond->c_vector.data) {
        cond->c_vector.data[0] = 1.0f;  // dummy scalar
    }

    // c_concat: not used in basic tests; set to zero-sized entry
    cond->c_concat.ndims = 0;
    cond->c_concat.ne[0] = 0;
    cond->c_concat.ne[1] = 0;
    cond->c_concat.ne[2] = 0;
    cond->c_concat.ne[3] = 0;
    cond->c_concat.data = nullptr;

    return cond;
}

void sd_free_condition(sd_condition_raw_t* cond) {
    if (!cond) return;

    if (cond->c_crossattn.data) {
        free(cond->c_crossattn.data);
        cond->c_crossattn.data = nullptr;
    }
    if (cond->c_vector.data) {
        free(cond->c_vector.data);
        cond->c_vector.data = nullptr;
    }
    if (cond->c_concat.data) {
        free(cond->c_concat.data);
        cond->c_concat.data = nullptr;
    }
    free(cond);
}

// sd_generate_video_with_precomputed_condition is provided by sdcpp_jni.cpp
// (compatibility shim) and forwards to the generate_video stub above.

sd_image_t* sd_generate_image_with_precomputed_condition(sd_ctx_t* sd_ctx,
                                                         const sd_img_gen_params_t* sd_img_gen_params,
                                                         const sd_condition_raw_t* cond,
                                                         const sd_condition_raw_t* uncond) {
    (void)cond;
    (void)uncond;
    sd_image_t* images = nullptr;
    int num_images = 0;
    if (!generate_image(sd_ctx, sd_img_gen_params, &images, &num_images)) {
        return nullptr;
    }
    return images;
}

}  // extern "C"
