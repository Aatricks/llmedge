#include <cstdlib>
#include "sdcpp_jni_shared.h"

// ---------------------------------------------------------------------------
// Compatibility shims for extension APIs that llmedge relies on but which may
// be absent in older upstream snapshots.
// ---------------------------------------------------------------------------
extern "C" SD_API sd_condition_raw_t* sd_precompute_condition(sd_ctx_t*,
                                                              const char*,
                                                              int,
                                                              int,
                                                              int,
                                                              bool) {
    return nullptr;
}

extern "C" SD_API void sd_free_condition(sd_condition_raw_t* cond) {
    if (!cond) return;
    if (cond->c_crossattn.data) free(cond->c_crossattn.data);
    if (cond->c_vector.data) free(cond->c_vector.data);
    if (cond->c_concat.data) free(cond->c_concat.data);
    free(cond);
}

extern "C" SD_API sd_image_t* sd_generate_image_with_precomputed_condition(
        sd_ctx_t* sd_ctx,
        const sd_img_gen_params_t* params,
        const sd_condition_raw_t*,
        const sd_condition_raw_t*) {
    return generate_image(sd_ctx, params);
}

extern "C" SD_API sd_image_t* sd_generate_video_with_precomputed_condition(
        sd_ctx_t* sd_ctx,
        const sd_vid_gen_params_t* params,
        const sd_condition_raw_t*,
        const sd_condition_raw_t*,
        int* num_frames_out) {
    return generate_video(sd_ctx, params, num_frames_out);
}
