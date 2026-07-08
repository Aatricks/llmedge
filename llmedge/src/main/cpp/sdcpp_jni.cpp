#include <cstdlib>
#include "sdcpp_jni_shared.h"

// ---------------------------------------------------------------------------
// Compatibility shims for extension APIs that llmedge relies on but which may
// be absent in older upstream snapshots.
//
// sd_precompute_condition / sd_free_condition / sd_generate_image_with_precomputed_condition are
// now implemented for real in stable-diffusion.cpp (llmedge Lever 1), so their shims are gone.
// Only the video precompute path remains a passthrough shim (not yet implemented natively).
// ---------------------------------------------------------------------------
extern "C" SD_API sd_image_t* sd_generate_video_with_precomputed_condition(
        sd_ctx_t* sd_ctx,
        const sd_vid_gen_params_t* params,
        const sd_condition_raw_t*,
        const sd_condition_raw_t*,
        int* num_frames_out) {
    sd_image_t* frames = nullptr;
    generate_video(sd_ctx, params, &frames, num_frames_out, nullptr);
    return frames;
}
