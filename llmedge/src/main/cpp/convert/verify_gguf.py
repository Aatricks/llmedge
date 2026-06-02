"""Verify a GGUF written by GgufWriter using the canonical gguf-py reader.

    python verify_gguf.py /tmp/test_write.gguf <repo_root>
"""
import sys
import numpy as np

path = sys.argv[1]
repo_root = sys.argv[2] if len(sys.argv) > 2 else "."
sys.path.insert(0, f"{repo_root}/llama.cpp/gguf-py")

from gguf import GGUFReader  # noqa: E402

r = GGUFReader(path)
failures = 0


def check(cond, msg):
    global failures
    if not cond:
        print("FAIL:", msg)
        failures += 1


keys = set(r.fields.keys())
for k in [
    "general.architecture", "llama.block_count", "llama.context_length",
    "llama.attention.layer_norm_rms_epsilon", "test.flag", "test.arr_u32",
    "tokenizer.ggml.tokens",
]:
    check(k in keys, f"missing KV {k}")

tensors = {t.name: t for t in r.tensors}
check("token_embd.weight" in tensors, "missing tensor token_embd.weight")
check("output_norm.weight" in tensors, "missing tensor output_norm.weight")

if "token_embd.weight" in tensors:
    t = tensors["token_embd.weight"]
    check(int(np.prod(t.shape)) == 6, f"token_embd element count {list(t.shape)}")
    check(np.allclose(np.array(t.data).flatten(), [1, 2, 3, 4, 5, 6]),
          f"token_embd data {np.array(t.data).flatten()}")
if "output_norm.weight" in tensors:
    t = tensors["output_norm.weight"]
    check(np.allclose(np.array(t.data).flatten(), [0.5, 0.25]),
          f"output_norm data {np.array(t.data).flatten()}")

if failures == 0:
    print("OK: gguf-py read the GgufWriter output; KVs + tensors verified")
    sys.exit(0)
print(f"{failures} check(s) failed")
sys.exit(1)
