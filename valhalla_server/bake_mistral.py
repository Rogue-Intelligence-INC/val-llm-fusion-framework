"""Tier-A style bake for a sharded 24B checkpoint (Mistral3 layout).

Legacy bake at safetensors level (same recipe as bake_runner, which needs a
single-file AutoModel load and therefore cannot carry a 48GB VL model):
project the Triad patch_vector to the model hidden size and add
strength * vector to the embedding rows of the anchor+corpus tokens.
Only the shard holding embed_tokens is rewritten; other shards are
hardlinked. Base directory is read-only.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path

ANCHOR = "Valhalla model decompose anchor: LM layer quads adapted by Fate into new weights."
EMBED_KEYS = (
    "model.language_model.embed_tokens.weight",
    "model.embed_tokens.weight",
    "language_model.model.embed_tokens.weight",
    "language_model.embed_tokens.weight",
)


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
    from safetensors import safe_open
    from safetensors.torch import load_file, save_file
    from transformers import AutoTokenizer

    base = Path(args.base)
    export_doc = json.loads(args.export.read_text(encoding="utf-8"))
    patch_vector = export_doc["report"]["model_patch"]["patch_vector"]

    corpus = [ANCHOR]
    if args.corpus and args.corpus.is_file():
        corpus += [line.strip() for line in
                   args.corpus.read_text(encoding="utf-8").splitlines() if line.strip()][:40]

    # Locate the embedding shard + hidden size.
    embed_shard, embed_key, hidden = None, None, None
    for shard in sorted(base.glob("*.safetensors")):
        with safe_open(str(shard), framework="pt") as f:
            for key in EMBED_KEYS:
                if key in f.keys():
                    embed_shard, embed_key = shard, key
                    hidden = f.get_slice(key).get_shape()[1]
                    break
        if embed_shard:
            break
    if not embed_shard:
        raise RuntimeError("找不到 embed_tokens 权重。")

    tokenizer = AutoTokenizer.from_pretrained(str(base))
    token_ids: list[int] = []
    for line in corpus:
        token_ids.extend(tokenizer.encode(line, add_special_tokens=False))
    unique = sorted(set(token_ids))

    vec = _project(patch_vector, int(hidden))
    args.out.mkdir(parents=True, exist_ok=True)
    delta = 0.0
    for shard in sorted(base.glob("*.safetensors")):
        target = args.out / shard.name
        if shard != embed_shard:
            try:
                os.link(shard, target)
            except OSError:
                shutil.copy(shard, target)
            continue
        tensors = load_file(str(shard))
        emb = tensors[embed_key]
        before = emb[unique].clone()
        with torch.no_grad():
            emb[unique] = emb[unique] + (args.strength * vec).to(emb.dtype)
        delta = float((emb[unique].float() - before.float()).norm())
        tensors[embed_key] = emb
        save_file(tensors, str(target), metadata={"format": "pt"})
        print(f"patched {embed_key} in {shard.name}: rows={len(unique)} delta={delta:.4f}", flush=True)

    for name in ("config.json", "generation_config.json", "model.safetensors.index.json",
                 "tokenizer.json", "tokenizer_config.json", "special_tokens_map.json",
                 "chat_template.json", "tekken.json", "preprocessor_config.json",
                 "processor_config.json", "params.json"):
        src = base / name
        if src.is_file():
            shutil.copy(src, args.out / name)

    manifest = {
        "manifest_version": "valhalla-bake-1.0",
        "product": "valhalla-server-tier-a-bake-24b",
        "runtime_patch": False,
        "base_model_id": str(base),
        "bake_mode": "legacy-embed-safetensors",
        "strength": args.strength,
        "weight_delta_norm": delta,
        "patch_dim": len(patch_vector),
        "tokens_patched": len(unique),
        "hidden_size": int(hidden),
    }
    (args.out / "valhalla_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps({"weight_delta_norm": delta, "tokens_patched": len(unique)}))


if __name__ == "__main__":
    main()
