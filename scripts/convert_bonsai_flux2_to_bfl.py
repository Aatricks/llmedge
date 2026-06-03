#!/usr/bin/env python3
"""Convert a PrismML Bonsai Image (FLUX.2 Klein 4B) diffusers transformer safetensors
into the BFL ("black-forest-labs") tensor naming/layout that stable-diffusion.cpp expects.

Bonsai ships its DiT in the `Flux2KleinPipeline` diffusers convention
(`single_transformer_blocks.*`, fused `to_qkv_mlp_proj`, `ff.linear_in/out`, shared
`*_stream_modulation`). sdcpp's FLUX.2 loader wants the BFL convention
(`double_blocks.*`/`single_blocks.*`, `img_attn.qkv`, `img_mlp.0/2`, `*_stream_modulation.lin`).

The mapping is almost all pure renames; the only structural op is concatenating the three
separate double-block attention projections (`to_q,to_k,to_v` and `add_{q,k,v}_proj`) into the
single fused `*_attn.qkv` tensor. Because safetensors stores tensors row-major (C order) with the
output dim outermost, that concatenation along dim 0 is exactly a concatenation of the raw byte
buffers — so this runs with no numpy/safetensors dependency and streams tensor-by-tensor.

After running this, quantize with stable-diffusion.cpp:
    sd -M convert -m <out.safetensors> --type tq1_0 -o bonsai-flux2-klein-tq1_0.gguf

Usage:
    python3 convert_bonsai_flux2_to_bfl.py <bonsai_diffusion_pytorch_model.safetensors> <out.safetensors>
"""
import json
import struct
import sys

DOUBLE_BLOCKS = 5
SINGLE_BLOCKS = 20


def read_header(f):
    f.seek(0)
    n = struct.unpack("<Q", f.read(8))[0]
    hdr = json.loads(f.read(n).decode("utf-8"))
    data_start = 8 + n
    return hdr, data_start


def build_plan(src):
    """Return ordered list of (out_name, dtype, out_shape, [src_names]) — multiple src => concat dim0."""
    plan = []

    def direct(out_name, src_name):
        meta = src[src_name]
        plan.append((out_name, meta["dtype"], meta["shape"], [src_name]))

    def concat(out_name, src_names):
        metas = [src[s] for s in src_names]
        dtype = metas[0]["dtype"]
        rest = metas[0]["shape"][1:]
        out_dim0 = sum(m["shape"][0] for m in metas)
        plan.append((out_name, dtype, [out_dim0] + rest, list(src_names)))

    # --- top-level / embedders ---
    direct("img_in.weight", "x_embedder.weight")
    direct("txt_in.weight", "context_embedder.weight")
    direct("time_in.in_layer.weight", "time_guidance_embed.timestep_embedder.linear_1.weight")
    direct("time_in.out_layer.weight", "time_guidance_embed.timestep_embedder.linear_2.weight")
    direct("final_layer.adaLN_modulation.1.weight", "norm_out.linear.weight")
    direct("final_layer.linear.weight", "proj_out.weight")
    direct("single_stream_modulation.lin.weight", "single_stream_modulation.linear.weight")
    direct("double_stream_modulation_img.lin.weight", "double_stream_modulation_img.linear.weight")
    direct("double_stream_modulation_txt.lin.weight", "double_stream_modulation_txt.linear.weight")

    # --- double-stream blocks ---
    for i in range(DOUBLE_BLOCKS):
        s = f"transformer_blocks.{i}."
        d = f"double_blocks.{i}."
        # image stream
        concat(d + "img_attn.qkv.weight", [s + "attn.to_q.weight", s + "attn.to_k.weight", s + "attn.to_v.weight"])
        direct(d + "img_attn.proj.weight", s + "attn.to_out.0.weight")
        direct(d + "img_attn.norm.query_norm.scale", s + "attn.norm_q.weight")
        direct(d + "img_attn.norm.key_norm.scale", s + "attn.norm_k.weight")
        direct(d + "img_mlp.0.weight", s + "ff.linear_in.weight")
        direct(d + "img_mlp.2.weight", s + "ff.linear_out.weight")
        # text stream
        concat(d + "txt_attn.qkv.weight", [s + "attn.add_q_proj.weight", s + "attn.add_k_proj.weight", s + "attn.add_v_proj.weight"])
        direct(d + "txt_attn.proj.weight", s + "attn.to_add_out.weight")
        direct(d + "txt_attn.norm.query_norm.scale", s + "attn.norm_added_q.weight")
        direct(d + "txt_attn.norm.key_norm.scale", s + "attn.norm_added_k.weight")
        direct(d + "txt_mlp.0.weight", s + "ff_context.linear_in.weight")
        direct(d + "txt_mlp.2.weight", s + "ff_context.linear_out.weight")

    # --- single-stream blocks ---
    for i in range(SINGLE_BLOCKS):
        s = f"single_transformer_blocks.{i}."
        d = f"single_blocks.{i}."
        direct(d + "linear1.weight", s + "attn.to_qkv_mlp_proj.weight")
        direct(d + "linear2.weight", s + "attn.to_out.weight")
        direct(d + "norm.query_norm.scale", s + "attn.norm_q.weight")
        direct(d + "norm.key_norm.scale", s + "attn.norm_k.weight")

    return plan


def main():
    if len(sys.argv) != 3:
        print(__doc__)
        sys.exit(1)
    in_path, out_path = sys.argv[1], sys.argv[2]

    with open(in_path, "rb") as f:
        src, data_start = read_header(f)
        src = {k: v for k, v in src.items() if k != "__metadata__"}
        plan = build_plan(src)

        # Validate every source tensor is consumed exactly once.
        consumed = [s for (_o, _d, _sh, ss) in plan for s in ss]
        missing = set(src) - set(consumed)
        if missing:
            print(f"ERROR: {len(missing)} source tensors unmapped, e.g. {sorted(missing)[:8]}", file=sys.stderr)
            sys.exit(2)
        print(f"source tensors: {len(src)} | output tensors: {len(plan)} | all source consumed")

        # Build output header with sequential offsets.
        out_hdr = {}
        offset = 0
        for out_name, dtype, shape, src_names in plan:
            nbytes = sum(
                (src[s]["data_offsets"][1] - src[s]["data_offsets"][0]) for s in src_names
            )
            out_hdr[out_name] = {"dtype": dtype, "shape": shape, "data_offsets": [offset, offset + nbytes]}
            offset += nbytes
        hdr_bytes = json.dumps(out_hdr, separators=(",", ":")).encode("utf-8")
        pad = (8 - (len(hdr_bytes) % 8)) % 8
        hdr_bytes += b" " * pad

        with open(out_path, "wb") as out:
            out.write(struct.pack("<Q", len(hdr_bytes)))
            out.write(hdr_bytes)
            for out_name, dtype, shape, src_names in plan:
                for s in src_names:
                    a, b = src[s]["data_offsets"]
                    f.seek(data_start + a)
                    remaining = b - a
                    while remaining > 0:
                        chunk = f.read(min(remaining, 8 << 20))
                        if not chunk:
                            raise IOError(f"short read on {s}")
                        out.write(chunk)
                        remaining -= len(chunk)
        print(f"wrote {out_path} ({offset} tensor bytes)")


if __name__ == "__main__":
    main()
