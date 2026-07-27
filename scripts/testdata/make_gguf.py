"""Writes a minimal but spec-valid GGUF v3 file with real tensor data.

Used to verify llmedge_gguf_get_tensor_name_prefixes against the actual ggml parser,
which validates tensor offsets and data size — a hand-written header alone is rejected.
"""
import struct
import sys

ALIGNMENT = 32
GGML_TYPE_F32 = 0


def u32(v):
    return struct.pack("<I", v)


def u64(v):
    return struct.pack("<Q", v)


def gstr(s):
    b = s.encode("utf-8")
    return u64(len(b)) + b


def kv_str(key, value):
    return gstr(key) + u32(8) + gstr(value)


def kv_u32(key, value):
    return gstr(key) + u32(4) + u32(value)


def build(path, architecture, tensor_names):
    kvs = kv_str("general.architecture", architecture) + kv_u32("general.alignment", ALIGNMENT)
    kv_count = 2

    elements = 4
    tensor_bytes = elements * 4  # F32
    padded = (tensor_bytes + ALIGNMENT - 1) // ALIGNMENT * ALIGNMENT

    infos = b""
    for index, name in enumerate(tensor_names):
        infos += gstr(name)
        infos += u32(1)                 # n_dims
        infos += u64(elements)          # dims[0]
        infos += u32(GGML_TYPE_F32)     # type
        infos += u64(index * padded)    # offset into the tensor-data section

    header = b"GGUF" + u32(3) + u64(len(tensor_names)) + u64(kv_count) + kvs + infos
    pad = (-len(header)) % ALIGNMENT
    data = b"".join(struct.pack("<f", float(i)) * elements + b"\x00" * (padded - tensor_bytes)
                    for i in range(len(tensor_names)))

    with open(path, "wb") as f:
        f.write(header + b"\x00" * pad + data)
    print(f"wrote {path}: {len(tensor_names)} tensors, arch={architecture}")


if __name__ == "__main__":
    out = sys.argv[1]
    kind = sys.argv[2]
    if kind == "bundle":
        build(out, "sd3", [
            "model.diffusion_model.joint_blocks.0.weight",
            "text_encoders.clip_l.transformer.weight",
            "first_stage_model.decoder.conv_in.weight",
        ])
    else:
        build(out, "sd3", [
            "joint_blocks.0.x_block.attn.qkv.weight",
            "pos_embed",
            "x_embedder.proj.weight",
        ])
