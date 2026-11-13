"""Server-side model decompose: LM layers -> MODEL_QUAD shards.

Same allocation as tools/model_decompose/shred_qwen_weights.py in the
Valhalla repo: layer signatures from LayerNorm + attention row means,
layers split into four quads by depth, Q0->Hub, Q1->Tile, Q2->Stem,
Q3->Hub. Ports use the dl_front projection (seeds and cap 512).

Run with the Valhalla venv python:
  python -m valhalla_server.shred --model <hf dir> --out plan.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

EMBED_DIM = 384
CAP = 512
NUM_QUADS = 4
BODY_FOR_QUAD = ("hub", "tile", "stem", "hub")
SEED_HUB = 0x4855425F464E54
SEED_TILE = 0x54494C455F464E54
SEED_STEM = 0x5354454D5F464E54
MASK = (1 << 64) - 1


def _normalize(vec: list[float]) -> list[float]:
    norm = sum(v * v for v in vec) ** 0.5 or 1e-9
    return [v / norm for v in vec]


def _pad(vec: list[float], dim: int = EMBED_DIM) -> list[float]:
    if len(vec) >= dim:
        return list(vec[:dim])
    return list(vec) + [0.0] * (dim - len(vec))


def _mix(a: list[float], b: list[float]) -> list[float]:
    pa, pb = _pad(a), _pad(b)
    return _normalize([(x + y) / 2.0 for x, y in zip(pa, pb)])


def _project(embedding: list[float], seed: int, cap: int = CAP) -> list[float]:
    if not embedding:
        return []
    out = [0.0] * cap
    h = seed
    for i in range(cap):
        h = (h * 6364136223846793005 + i) & MASK
        j = h % len(embedding)
        w = ((h >> 17) / float(0xFFFFFFFF)) * 2.0 - 1.0
        out[i] = embedding[j] * w
    return _normalize(out)


def _ports(quad_means: list[list[float]], encoder: str) -> dict:
    flat = _normalize([sum(_pad(q)[i] for q in quad_means) / NUM_QUADS for i in range(EMBED_DIM)])
    return {
        "hub": _project(_mix(quad_means[0], quad_means[1]), SEED_HUB),
        "tile": _project(_mix(quad_means[1], quad_means[2]), SEED_TILE),
        "stem": _project(_mix(quad_means[2], quad_means[3]), SEED_STEM),
        "embedding": flat,
        "embed_dim": EMBED_DIM,
        "encoder": encoder,
        "modality_label": "model_weight",
    }


def _detect_prefix(keys) -> str | None:
    for prefix in ("model.language_model.layers.", "model.layers.",
                   "language_model.model.layers.", "language_model.layers."):
        if any(k.startswith(prefix) for k in keys):
            return prefix
    return None


def _load_needed(model_dir: Path) -> tuple[dict, str]:
    """Stream shards; keep only LN vectors and attention row-means.

    Works for single-file and sharded checkpoints and for Mistral3-style
    nested language-model keys, without materializing the whole model.
    """
    import torch
    from safetensors import safe_open

    shards = sorted(model_dir.glob("*.safetensors"))
    if not shards:
        raise RuntimeError("模型目录里没有 safetensors。")
    prefix = None
    weights: dict = {}
    layer_parts: dict[int, list] = {}
    for shard in shards:
        with safe_open(str(shard), framework="pt") as f:
            keys = list(f.keys())
            if prefix is None:
                prefix = _detect_prefix(keys)
            if prefix is None:
                continue
            for key in keys:
                if not key.startswith(prefix):
                    continue
                rest = key[len(prefix):]
                idx_s, _, tail = rest.partition(".")
                if not idx_s.isdigit():
                    continue
                idx = int(idx_s)
                if tail in ("input_layernorm.weight", "post_attention_layernorm.weight"):
                    weights[key] = f.get_tensor(key)
                elif tail in ("self_attn.q_proj.weight", "self_attn.k_proj.weight",
                              "self_attn.v_proj.weight", "self_attn.o_proj.weight"):
                    weights[key] = f.get_tensor(key).float().mean(dim=0)
    if prefix is None:
        raise RuntimeError("模型里找不到 decoder 层权重。")
    return weights, prefix


def _layer_signature(weights: dict, prefix: str, index: int) -> list[float]:
    base = f"{prefix}{index}"
    parts = []
    for name in ("input_layernorm", "post_attention_layernorm"):
        tensor = weights.get(f"{base}.{name}.weight")
        if tensor is not None:
            parts.append(tensor.float().reshape(-1))
    attn = f"{base}.self_attn"
    for name in ("q_proj", "k_proj", "v_proj", "o_proj"):
        tensor = weights.get(f"{attn}.{name}.weight")
        if tensor is not None:
            parts.append(tensor.float().reshape(-1))  # already row-mean
    if not parts:
        return [0.0] * EMBED_DIM
    acc = parts[0].clone()
    for part in parts[1:]:
        n = min(acc.numel(), part.numel())
        acc[:n] = acc[:n] + part[:n]
    acc = acc / len(parts)
    return _normalize(_pad([float(v) for v in acc.tolist()]))


def build_plan(model_dir: Path) -> dict:
    weights, prefix = _load_needed(model_dir)
    indices = sorted({int(k[len(prefix):].split(".")[0]) for k in weights
                      if k.startswith(prefix)})
    if not indices:
        raise RuntimeError("模型里找不到 decoder 层权重。")
    sigs = [_layer_signature(weights, prefix, i) for i in indices]
    size = max(1, len(sigs) // NUM_QUADS)
    quads: list[list[float]] = []
    for qi in range(NUM_QUADS):
        chunk = sigs[qi * size:(qi + 1) * size] or sigs[-1:]
        quads.append([sum(col) / len(chunk) for col in zip(*chunk)])
    name = model_dir.name
    encoder = f"model/{name}"
    whole = _ports(quads, encoder)
    shards = []
    for qi in range(NUM_QUADS):
        ports = _ports(quads, f"{encoder}/quad_{qi}")
        shards.append({
            "label": f"MODEL_QUAD_{qi}",
            "body": BODY_FOR_QUAD[qi],
            "phase": "aggregate" if qi == NUM_QUADS - 1 else "decompose",
            "layer_range": [qi * size, min((qi + 1) * size, len(sigs))],
            "ports": ports,
        })
    shards.append({"label": "MODEL_FATE_AGGREGATE", "body": "hub", "phase": "aggregate", "ports": whole})
    shards.append({"label": "MODEL_RECOMPOSE", "body": "triad", "phase": "recompose", "ports": whole})
    return {
        "protocol": "valhalla-model-decompose-v1",
        "base_model": str(model_dir),
        "layer_count": len(sigs),
        "quad_dims": [len(q) for q in quads],
        "whole_ports": whole,
        "shards": shards,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    plan = build_plan(args.model)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(plan, indent=2), encoding="utf-8")
    print(f"model decompose: layers={plan['layer_count']} shards={len(plan['shards'])} -> {args.out}")


if __name__ == "__main__":
    main()
