#!/usr/bin/env python3
"""Host-side safetensors -> GGUF converter for llmedge (Track B / Phase B1).

One converter, a precision parameter:
  * ``--precision f16``     -> "direct" (no precision loss), produced straight by the upstream converter
  * ``--precision q8_0``    -> light, near-lossless quantization (upstream converter)
  * ``--precision q4_k_m``  -> "lossy", needs a built ``llama-quantize`` (pass ``--quantize-bin``)
  * ``--precision iq2_bn``  -> ik_llama ternary quant, needs ``llama-quantize`` (ternary models only)

Generic HF architectures inherit the coverage of the in-repo ``llama.cpp/convert_hf_to_gguf.py``.
Bonsai needs ``--adapter bonsai-qlinear`` first: it folds the per-output ``.scales`` into the weights
(see bonsai_fold.py) and rewrites the config to stock ``LlamaForCausalLM`` so the upstream converter
accepts it.

Examples
--------
    # Stock Llama-family model -> F16 GGUF
    python convert.py --source unsloth/Qwen3-0.6B --precision f16 --out qwen3-0.6b-f16.gguf

    # Bonsai 0.5B -> Q8_0 GGUF (fold QLinear scales first)
    python convert.py --source deepgrove/Bonsai --adapter bonsai-qlinear \
        --precision q8_0 --out bonsai-0.5b-q8_0.gguf

Requirements: see requirements.txt (torch, safetensors, numpy, huggingface_hub, sentencepiece).
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
UPSTREAM_CONVERTER = REPO_ROOT / "llama.cpp" / "convert_hf_to_gguf.py"

# precision -> upstream --outtype it can emit directly (others go through llama-quantize from f16)
DIRECT_OUTTYPE = {"f16": "f16", "q8_0": "q8_0", "bf16": "bf16"}
QUANTIZE_ONLY = {"q4_k_m": "Q4_K_M", "q5_k_m": "Q5_K_M", "iq2_bn": "IQ2_BN", "iq2_bn_r4": "IQ2_BN_R4"}


def log(msg: str) -> None:
    print(f"[convert] {msg}", flush=True)


def resolve_source(source: str) -> Path:
    """Return a local dir for ``source`` (a path, or a HF repo id to snapshot-download)."""
    p = Path(source)
    if p.is_dir():
        return p
    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        sys.exit("huggingface_hub not installed and --source is not a local dir. pip install -r requirements.txt")
    log(f"downloading {source} from Hugging Face ...")
    return Path(snapshot_download(repo_id=source, allow_patterns=[
        "*.safetensors", "*.json", "*.model", "tokenizer*", "*.txt",
    ]))


def _load_all_safetensors(src: Path):
    from safetensors.torch import load_file
    index = src / "model.safetensors.index.json"
    files = []
    if index.exists():
        weight_map = json.loads(index.read_text())["weight_map"]
        files = sorted({src / shard for shard in weight_map.values()})
    else:
        single = src / "model.safetensors"
        if not single.exists():
            cand = sorted(src.glob("*.safetensors"))
            if not cand:
                sys.exit(f"no .safetensors found in {src}")
            files = cand
        else:
            files = [single]
    tensors = {}
    for f in files:
        log(f"loading {f.name}")
        tensors.update(load_file(str(f)))
    return tensors


def apply_bonsai_adapter(src: Path, work: Path) -> Path:
    """Fold Bonsai QLinear scales and rewrite config into a stock-Llama dir under ``work``."""
    import torch
    from safetensors.torch import save_file

    from bonsai_fold import fold_scales, num_folded, rewrite_config

    out_dir = work / "bonsai_stock_llama"
    out_dir.mkdir(parents=True, exist_ok=True)

    tensors = _load_all_safetensors(src)
    log(f"folding {num_folded(tensors)} .scales tensors into their weights")
    # Upcast to float32 for an exact fold, then let the converter requantize.
    tensors = {k: v.to(torch.float32) for k, v in tensors.items()}
    folded = fold_scales(tensors)
    save_file(folded, str(out_dir / "model.safetensors"), metadata={"format": "pt"})

    cfg = json.loads((src / "config.json").read_text())
    (out_dir / "config.json").write_text(json.dumps(rewrite_config(cfg), indent=2))

    # Copy tokenizer / aux files the converter needs (but not the original weights/config/custom code).
    for f in src.iterdir():
        if f.suffix == ".safetensors" or f.name in {"config.json"} or f.name.endswith(".py"):
            continue
        if f.is_file():
            shutil.copy2(f, out_dir / f.name)
    log(f"stock-Llama checkpoint written to {out_dir}")
    return out_dir


def run_converter(model_dir: Path, out_gguf: Path, outtype: str) -> None:
    if not UPSTREAM_CONVERTER.exists():
        sys.exit(f"upstream converter not found at {UPSTREAM_CONVERTER} "
                 "(run: git submodule update --init llama.cpp)")
    cmd = [sys.executable, str(UPSTREAM_CONVERTER), str(model_dir),
           "--outfile", str(out_gguf), "--outtype", outtype]
    log("running: " + " ".join(cmd))
    subprocess.run(cmd, check=True)


def quantize(in_gguf: Path, out_gguf: Path, qtype: str, quantize_bin: str | None) -> None:
    if not quantize_bin:
        sys.exit(f"--precision needs llama-quantize for {qtype}; pass --quantize-bin /path/to/llama-quantize "
                 "(build it from the ik_llama submodule with cmake -- a host build, no Android NDK needed)")
    cmd = [quantize_bin, str(in_gguf), str(out_gguf), qtype]
    log("running: " + " ".join(cmd))
    subprocess.run(cmd, check=True)


def main() -> None:
    ap = argparse.ArgumentParser(description="safetensors -> GGUF for llmedge")
    ap.add_argument("--source", required=True, help="HF repo id or local model dir")
    ap.add_argument("--out", required=True, help="output .gguf path")
    ap.add_argument("--precision", default="f16",
                    choices=sorted(set(DIRECT_OUTTYPE) | set(QUANTIZE_ONLY)))
    ap.add_argument("--adapter", default="none", choices=["none", "bonsai-qlinear"])
    ap.add_argument("--quantize-bin", default=os.environ.get("LLAMA_QUANTIZE"),
                    help="path to a built llama-quantize (needed for k-quants / iq2_bn)")
    args = ap.parse_args()

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix="llmedge-convert-") as tmp:
        work = Path(tmp)
        model_dir = resolve_source(args.source)
        if args.adapter == "bonsai-qlinear":
            model_dir = apply_bonsai_adapter(model_dir, work)

        if args.precision in DIRECT_OUTTYPE:
            run_converter(model_dir, out, DIRECT_OUTTYPE[args.precision])
        else:
            f16 = work / "model-f16.gguf"
            run_converter(model_dir, f16, "f16")
            quantize(f16, out, QUANTIZE_ONLY[args.precision], args.quantize_bin)

    log(f"done -> {out}  ({out.stat().st_size / 1e6:.0f} MB)" if out.exists() else "done")


if __name__ == "__main__":
    main()
