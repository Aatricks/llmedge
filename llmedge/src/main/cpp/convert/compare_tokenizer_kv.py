"""KV oracle: compare the tokenizer.* metadata of two GGUFs (mine vs a reference).

    python compare_tokenizer_kv.py <mine.gguf> <ref.gguf> <repo_root>

Checks every `tokenizer.*` key in the reference exists in mine with an identical value (strings,
scalars, and arrays element-by-element). Also flags tokenizer.* keys present in mine but not ref.
This is the ground-truth check for Layer 4 (tokenizer baking): a wrong token_type, a missing merge,
or a shifted special-token id shows up as a concrete diff, not as plausible-looking output.
"""
import sys

import numpy as np

mine_path, ref_path = sys.argv[1], sys.argv[2]
repo_root = sys.argv[3] if len(sys.argv) > 3 else "."
sys.path.insert(0, f"{repo_root}/llama.cpp/gguf-py")
from gguf import GGUFReader  # noqa: E402
from gguf.constants import GGUFValueType  # noqa: E402


def field_value(f):
    """Return a comparable Python value for a GGUF field."""
    t = f.types
    if t and t[-1] == GGUFValueType.STRING:
        if len(t) > 1 and t[0] == GGUFValueType.ARRAY:
            return [str(bytes(f.parts[d]), "utf-8") for d in f.data]
        return str(bytes(f.parts[f.data[0]]), "utf-8")
    if t and t[0] == GGUFValueType.ARRAY:
        return [int(f.parts[d][0]) if isinstance(f.parts[d][0], np.integer) else f.parts[d][0].item()
                for d in f.data]
    v = f.parts[f.data[0]][0]
    return v.item() if hasattr(v, "item") else v


def tok_fields(path):
    r = GGUFReader(path)
    return {k: field_value(f) for k, f in r.fields.items() if k.startswith("tokenizer.")}


mine, ref = tok_fields(mine_path), tok_fields(ref_path)
print(f"mine: {len(mine)} tokenizer KVs, ref: {len(ref)} tokenizer KVs")

failures = 0
for key, rv in sorted(ref.items()):
    if key not in mine:
        print(f"FAIL: {key} missing in mine")
        failures += 1
        continue
    mv = mine[key]
    if isinstance(rv, list):
        if len(mv) != len(rv):
            print(f"FAIL: {key} length {len(mv)} != ref {len(rv)}")
            failures += 1
            continue
        nbad = sum(1 for a, b in zip(mv, rv) if a != b)
        if nbad:
            first = next((i for i, (a, b) in enumerate(zip(mv, rv)) if a != b))
            print(f"FAIL: {key} {nbad}/{len(rv)} elems differ (first at [{first}]: {mv[first]!r} != {rv[first]!r})")
            failures += 1
        else:
            print(f"  ok: {key} (array, {len(rv)} elems)")
    else:
        if mv != rv:
            print(f"FAIL: {key} {mv!r} != ref {rv!r}")
            failures += 1
        else:
            print(f"  ok: {key} = {mv!r}")

extra = sorted(set(mine) - set(ref))
if extra:
    print(f"NOTE: {len(extra)} tokenizer KV(s) in mine but not ref: {extra}")

if failures == 0:
    print(f"OK: all {len(ref)} tokenizer KVs match the reference")
    sys.exit(0)
print(f"{failures} tokenizer KV(s) failed")
sys.exit(1)
