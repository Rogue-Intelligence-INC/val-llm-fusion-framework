"""Bake one exported Valhalla structure into a new checkpoint. Run with the venv python.

The upstream bridge (tools/valhalla_model_bridge/bake_valhalla_model.py)
imports tier_a_tf_front / apply_and_compare, which are not present in this
checkout. This runner implements the legacy bake that bridge describes:
project the Triad patch_vector to hidden size and add strength * vector to
the embedding rows of the anchor and corpus tokens. The base model is
read-only; a new checkpoint directory is written.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

ANCHOR = "Valhalla model decompose anchor: LM layer quads adapted by Fate into new weights."


def _project(vec: list[float], dim: int):
    import torch

    src = torch.tensor(vec, dtype=torch.float32)
    if src.numel() == dim:
        out = src
    elif src.numel() > dim:
        step = src.numel() / dim
        out = src[[min(int(i * step), src.numel() - 1) for i in range(dim)]]
    else:
        out = torch.zeros(dim)
        out[: src.numel()] = src
    norm = out.norm()
    return out / norm if norm > 0 else out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", required=True)
    parser.add_argument("--export", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--strength", type=float, default=0.08)
    parser.add_argument("--corpus", type=Path)
    args = parser.parse_args()

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    export_doc = json.loads(args.export.read_text(encoding="utf-8"))
    patch = export_doc["report"]["model_patch"]
    patch_vector = patch["patch_vector"]
    corpus = [ANCHOR]
    if args.corpus and args.corpus.is_file():
        corpus += [line.strip() for line in args.corpus.read_text(encoding="utf-8").splitlines() if line.strip()][:40]

    tokenizer = AutoTokenizer.from_pretrained(args.base)
    model = AutoModelForCausalLM.from_pretrained(args.base, torch_dtype=torch.float32)
    model.eval()
    vec = _project(patch_vector, model.config.hidden_size)
    emb = model.model.embed_tokens.weight
    token_ids: list[int] = []
    for line in corpus:
        token_ids.extend(tokenizer.encode(line, add_special_tokens=False))
    unique = sorted(set(token_ids))
    before = emb.data[unique].clone() if unique else None
    with torch.no_grad():
        for tid in unique:
            emb.data[tid] += args.strength * vec.to(emb.device)
    delta = float((emb.data[unique] - before).norm()) if unique else 0.0

    args.out.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(args.out)
    tokenizer.save_pretrained(args.out)
    manifest = {
        "manifest_version": "valhalla-bake-1.0",
        "product": "valhalla-server-fused-bake",
        "runtime_patch": False,
        "base_model_id": args.base,
        "bake_mode": "legacy-embed",
        "strength": args.strength,
        "weight_delta_norm": delta,
        "patch_dim": len(patch_vector),
        "tile_count": patch.get("tile_count"),
        "stem_cell_count": patch.get("stem_cell_count"),
    }
    (args.out / "valhalla_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps({"weight_delta_norm": delta, "tokens_patched": len(unique)}))


if __name__ == "__main__":
    main()
