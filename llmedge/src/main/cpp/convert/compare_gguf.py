"""Tensor-by-tensor oracle: compare two GGUFs (mine vs a reference) as fp32.

    python compare_gguf.py <mine.gguf> <ref.gguf> <repo_root>

Verifies every tensor in <mine> exists in <ref> with the same shape and (dtype-agnostic) values.
This is the ground-truth check for the converter: a wrong Q/K permutation or name map shows up as a
shape mismatch or large value diff, not as plausible-looking garbage.
"""
import sys
import numpy as np

mine_path, ref_path = sys.argv[1], sys.argv[2]
repo_root = sys.argv[3] if len(sys.argv) > 3 else "."
sys.path.insert(0, f"{repo_root}/llama.cpp/gguf-py")
from gguf import GGUFReader  # noqa: E402


def tensors(path):
    r = GGUFReader(path)
    return {t.name: t for t in r.tensors}


mine, ref = tensors(mine_path), tensors(ref_path)
print(f"mine: {len(mine)} tensors, ref: {len(ref)} tensors")

failures = 0
checked = 0
for name, mt in sorted(mine.items()):
    if name not in ref:
        print(f"FAIL: {name} not in reference")
        failures += 1
        continue
    rt = ref[name]
    ms = [int(x) for x in mt.shape]
    rs = [int(x) for x in rt.shape]
    if ms != rs:
        print(f"FAIL: {name} shape {ms} != ref {rs}")
        failures += 1
        continue
    a = np.array(mt.data, dtype=np.float32).flatten()
    b = np.array(rt.data, dtype=np.float32).flatten()
    if a.shape != b.shape:
        print(f"FAIL: {name} element count {a.shape} != {b.shape}")
        failures += 1
        continue
    if not np.allclose(a, b, rtol=2e-3, atol=2e-3):
        diff = np.max(np.abs(a - b))
        nbad = int(np.sum(np.abs(a - b) > 2e-3))
        print(f"FAIL: {name} value mismatch (max|d|={diff:.4g}, {nbad}/{a.size} off)")
        failures += 1
        continue
    checked += 1

missing_in_mine = sorted(set(ref) - set(mine))
# Reference may legitimately have nothing extra for a tied-embedding Llama; report if it does.
if missing_in_mine:
    print(f"NOTE: {len(missing_in_mine)} tensor(s) in ref but not mine: {missing_in_mine[:6]}")

if failures == 0:
    print(f"OK: {checked} tensors match the upstream reference (values + shapes)")
    sys.exit(0)
print(f"{failures} tensor(s) failed")
sys.exit(1)
