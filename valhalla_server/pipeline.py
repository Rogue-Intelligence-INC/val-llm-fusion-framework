"""Decompose -> Fate adapt -> free fusion -> bake -> encrypt -> publish.

Allocation follows the Valhalla repos, not model roles:
- decompose cuts one LM by layer depth into MODEL_QUAD shards
  (Q0 Hub, Q1 Tile, Q2 Stem, Q3 Hub) and Fate adapts them;
- fusion feeds text and audio ports of the same samples into Hub, Tile
  and Stem freely (valhalla-free-multimodal-fusion-v2 incubate);
- bake writes the fused structure into a new checkpoint;
- the checkpoint weights and the engines stay VALENC1 at rest.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import threading
from datetime import datetime, timezone
from pathlib import Path

from valhalla_server import runtime
from valhalla_server.config import BASE_MODEL, BUILDS, JOBS, REGISTRY, VAULT, VENV_PY
from valhalla_server.valenc import encrypt_file, load_key

DEFAULT_SAMPLES = [
    {"id": "s1", "text": "ValhallaBase primary protocol is valhalla-base-v1."},
    {"id": "s2", "text": "Valhalla retrieves first, fuses structure, then writes the sentence."},
    {"id": "s3", "text": "Hub Tile Stem run freely with Fate adaptation and no preset routing."},
    {"id": "s4", "text": "A baked checkpoint is a new model; the base model stays unchanged."},
]


def _hash_embed(text: str, dim: int = 384) -> list[float]:
    h = 0x544558545F464E54
    for byte in text.encode("utf-8", errors="replace"):
        h = ((h * 6364136223846793005) + byte) & ((1 << 64) - 1)
    out = []
    for i in range(dim):
        h = ((h * 6364136223846793005) + i) & ((1 << 64) - 1)
        out.append(((h >> 33) / float(2**32 - 1)) * 2.0 - 1.0)
    norm = sum(v * v for v in out) ** 0.5 or 1e-9
    return [v / norm for v in out]


def _run(cmd: list[str], *, stdin: str | None = None, env: dict | None = None, timeout: int = 1800) -> str:
    done = subprocess.run(
        cmd,
        input=stdin,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=timeout,
        env={**os.environ, **(env or {})},
        check=False,
    )
    if done.returncode != 0:
        detail = (done.stderr or done.stdout or "命令失败。").strip()
        raise RuntimeError(detail[-600:])
    return done.stdout


def _last_json(text: str) -> dict:
    start = text.find("{")
    if start < 0:
        raise RuntimeError("子进程没有返回 JSON。")
    body = json.loads(text[start:])
    if not isinstance(body, dict):
        raise RuntimeError("子进程返回的不是 JSON 对象。")
    return body


def _registry() -> dict:
    if REGISTRY.is_file():
        try:
            return json.loads(REGISTRY.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return {}
    return {}


def _save_registry(models: dict) -> None:
    REGISTRY.parent.mkdir(parents=True, exist_ok=True)
    REGISTRY.write_text(json.dumps(models, ensure_ascii=False, indent=2), encoding="utf-8")


def list_models() -> list[dict]:
    out = []
    for model_id, row in _registry().items():
        out.append({
            "id": model_id,
            "object": "model",
            "base_model": row.get("base_model"),
            "endpoint": f"/v1/models/{model_id}/chat/completions",
            "encrypted": True,
            "format": "valenc1",
            "created": row.get("created"),
        })
    return out


def model_row(model_id: str) -> dict:
    row = _registry().get(model_id)
    if not row:
        raise KeyError(model_id)
    return row


def _job_path(job_id: str) -> Path:
    return JOBS / f"{job_id}.json"


def job_status(job_id: str) -> dict:
    path = _job_path(job_id)
    if not path.is_file():
        raise KeyError(job_id)
    return json.loads(path.read_text(encoding="utf-8"))


def _set_job(job_id: str, **fields) -> dict:
    path = _job_path(job_id)
    row = {}
    if path.is_file():
        row = json.loads(path.read_text(encoding="utf-8"))
    row.update(fields)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(row, ensure_ascii=False, indent=2), encoding="utf-8")
    return row


def run_pipeline(job_id: str, model_id: str, base_model: str, strength: float, samples: list[dict]) -> None:
    work = JOBS / job_id
    work.mkdir(parents=True, exist_ok=True)
    try:
        _set_job(job_id, status="running", stage="provision", model_id=model_id)
        runtime.provision()

        _set_job(job_id, stage="decompose")
        plan_path = work / "model_shards.json"
        _run([str(VENV_PY), "-m", "valhalla_server.shred", "--model", base_model, "--out", str(plan_path)], timeout=900)

        _set_job(job_id, stage="model_fate")
        staging = work / "staging"
        fate_out = _run(
            [str(runtime.engine("valhalla_model_fate.exe")), "--shards", str(plan_path),
             "--manifest-out", str(staging), "--base-model-id", base_model,
             "--bake-mode", "full", "--strength", str(strength)],
            timeout=900,
        )
        fate = _last_json(fate_out)
        outcome = fate.get("model_fate") or {}

        _set_job(job_id, stage="fusion")
        fused_samples = [
            {"id": str(s.get("id") or f"s{i}"), "text": str(s.get("text", "")).strip(),
             "audio_features": _hash_embed(str(s.get("text", ""))),
             "image_features": _hash_embed(str(s.get("text", "")) + "#image")}
            for i, s in enumerate(samples) if str(s.get("text", "")).strip()
        ]
        if not fused_samples:
            raise RuntimeError("samples 里没有可用文本。")
        fusion_out = _run(
            [str(runtime.engine("valhalla_trimodal_incubate.exe")), "--manifest-out", str(work / "fusion_staging")],
            stdin=json.dumps({"samples": fused_samples, "base_model_id": base_model, "strength": strength, "bake_mode": "full"}),
            env={"VALHALLA_FREE_MULTIMODAL": "1"},
            timeout=900,
        )
        fusion = _last_json(fusion_out)
        export_doc = fusion.get("export_doc")
        if not isinstance(export_doc, dict):
            raise RuntimeError("融合没有导出结构。")
        export_path = work / "fusion_export.json"
        export_path.write_text(json.dumps(export_doc, ensure_ascii=False), encoding="utf-8")
        corpus_path = work / "corpus.txt"
        corpus_path.write_text("\n".join(s["text"] for s in fused_samples), encoding="utf-8")

        _set_job(job_id, stage="bake")
        build = BUILDS / model_id
        if build.exists():
            shutil.rmtree(build)
        bake_out = _run(
            [str(VENV_PY), "-m", "valhalla_server.bake_runner", "--base", base_model,
             "--export", str(export_path), "--out", str(build), "--strength", str(strength),
             "--corpus", str(corpus_path)],
            timeout=1800,
        )
        bake_metrics = _last_json(bake_out[bake_out.rfind("{"):]) if "{" in bake_out else {}

        manifest_path = build / "valhalla_manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.is_file() else {}
        persistent = fusion.get("persistent_fusion") or {}
        manifest.update({
            "product": "valhalla-unified-core-v1",
            "protocol": "valhalla-unified-core-v1",
            "runtime_patch": False,
            "model_fate": {
                "protocol": "valhalla-model-decompose-v1",
                "base_model": base_model,
                "adaptation_rounds": outcome.get("adaptation_rounds"),
                "stopped_reason": outcome.get("stopped_reason"),
                "decompose_score": outcome.get("decompose_score"),
                "aggregate_score": outcome.get("aggregate_score"),
                "model_closure_score": outcome.get("model_closure_score"),
            },
            "free_multimodal": {
                "protocol": persistent.get("protocol"),
                "samples_fed": persistent.get("samples_fed"),
                "rounds_per_sample": persistent.get("rounds_per_sample"),
                "closure_rounds": persistent.get("closure_rounds"),
                "aggregate_score": persistent.get("aggregate_score"),
                "layout": "non_isolated_single_session",
                "tri_modal": persistent.get("tri_modal"),
            },
            "bake": {"strength": strength, "weight_delta_norm": bake_metrics.get("weight_delta_norm")},
            "structure_in_weights": True,
        })
        manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

        # finalize_hf_repo pattern: model card + LFS attributes next to the weights.
        tag = Path(base_model).name
        hf_name = f"{tag}-Unified-Core-v1-test-s{strength:.2f}"
        (build / "README.md").write_text(
            f"# {hf_name}\n\nValhalla Unified Core v1 bake (non-isolated tri-modal fusion, "
            f"strength {strength}). Base: {base_model}. Weights in this deployment are "
            "stored VALENC1-encrypted; see valhalla_manifest.json.\n",
            encoding="utf-8",
        )
        (build / ".gitattributes").write_text("*.safetensors filter=lfs diff=lfs merge=lfs -text\n", encoding="utf-8")

        _set_job(job_id, stage="encrypt")
        vault = VAULT / model_id
        if vault.exists():
            shutil.rmtree(vault)
        vault.mkdir(parents=True, exist_ok=True)
        key = load_key()
        weights = build / "model.safetensors"
        if not weights.is_file():
            raise RuntimeError("烘焙没有生成 model.safetensors。")
        encrypt_file(weights, vault / "model.safetensors.enc", key)
        digest = hashlib.sha256((vault / "model.safetensors.enc").read_bytes()).hexdigest()
        for name in ("config.json", "generation_config.json", "tokenizer.json", "tokenizer_config.json", "vocab.json", "merges.txt", "special_tokens_map.json", "added_tokens.json", "chat_template.jinja", "valhalla_manifest.json", "README.md", ".gitattributes"):
            src = build / name
            if src.is_file():
                shutil.copy(src, vault / name)
        weights.unlink()

        models = _registry()
        models[model_id] = {
            "base_model": base_model,
            "hf_name": hf_name,
            "protocol": "valhalla-unified-core-v1",
            "strength": strength,
            "created": datetime.now(timezone.utc).isoformat(),
            "endpoint": f"/v1/models/{model_id}/chat/completions",
            "vault_sha256": digest,
            "manifest": {
                "model_fate": manifest["model_fate"],
                "free_multimodal": manifest["free_multimodal"],
                "bake": manifest["bake"],
            },
        }
        _save_registry(models)
        _set_job(job_id, status="done", stage="published", endpoint=f"/v1/models/{model_id}/chat/completions")
    except Exception as exc:  # noqa: BLE001 - job status carries the error
        _set_job(job_id, status="failed", error=str(exc)[-600:])


def start_pipeline(model_id: str, base_model: str, strength: float, samples: list[dict]) -> dict:
    job_id = f"job_{datetime.now(timezone.utc).strftime('%Y%m%d%H%M%S')}_{abs(hash(model_id)) % 100000:05d}"
    _set_job(job_id, status="queued", stage="queued", model_id=model_id, created=datetime.now(timezone.utc).isoformat())
    thread = threading.Thread(target=run_pipeline, args=(job_id, model_id, base_model or BASE_MODEL, strength, samples or DEFAULT_SAMPLES), daemon=True)
    thread.start()
    return {"job_id": job_id, "status": "queued", "status_url": f"/v1/jobs/{job_id}"}
