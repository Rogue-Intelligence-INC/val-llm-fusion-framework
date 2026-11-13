"""Automated Valhalla model processing (programmatized pipeline).

One job runs the whole chain that used to be manual:
  validate -> decompose (shred) -> model_fate adapt -> trimodal fusion
  -> bake (Tier-A legacy embed; sharded-aware) -> GGUF convert -> quantize
  -> deploy decode head (llama-server) -> verify -> register decode head.

Small single-file checkpoints still go through pipeline.start_pipeline;
modelops targets LM folders that must become llama.cpp decode heads
(e.g. Mistral-24B). Bake uses generic samples only, never caller test data,
unless the caller explicitly passes samples.
"""

from __future__ import annotations

import hashlib
import json
import threading
from datetime import datetime, timezone
from pathlib import Path

from valhalla_server import head_server, mistral_head, runtime
from valhalla_server.config import DATA, DECODE_HEADS, JOBS, LLAMA_SRC, VENV_PY
from valhalla_server.pipeline import DEFAULT_SAMPLES, _hash_embed, _last_json, _run, _set_job


# ------------------------------------------------------------ registry
def _heads() -> dict:
    if DECODE_HEADS.is_file():
        try:
            return json.loads(DECODE_HEADS.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return {}
    return {}


def _save_heads(heads: dict) -> None:
    DECODE_HEADS.parent.mkdir(parents=True, exist_ok=True)
    DECODE_HEADS.write_text(json.dumps(heads, ensure_ascii=False, indent=2), encoding="utf-8")


def list_decode_heads() -> list[dict]:
    out = []
    for head_id, row in _heads().items():
        out.append({"id": head_id, "object": "decode_head", **row,
                    "endpoint": mistral_head.URL})
    return out


def deploy_decode_head(head_id: str) -> dict:
    row = _heads().get(head_id)
    if not row:
        raise KeyError(head_id)
    ok = head_server.start(str(row["gguf"]))
    return {"id": head_id, "deployed": ok, **head_server.status()}


# ------------------------------------------------------------ job
def _is_sharded(base: Path) -> bool:
    shards = sorted(base.glob("*.safetensors"))
    if len(shards) != 1:
        return True
    from safetensors import safe_open

    with safe_open(str(shards[0]), framework="pt") as f:
        keys = list(f.keys())
    return not any(k == "model.embed_tokens.weight" for k in keys)


def run_modelops(job_id: str, head_id: str, base_model: str, strength: float,
                 quant: str, samples: list[dict], deploy: bool) -> None:
    work = JOBS / job_id
    work.mkdir(parents=True, exist_ok=True)
    base = Path(base_model)
    try:
        _set_job(job_id, status="running", stage="validate", model_id=head_id, kind="modelops")
        if not base.is_dir() or not list(base.glob("*.safetensors")):
            raise RuntimeError("base_model 不是含 safetensors 的模型目录。")
        runtime.provision()

        _set_job(job_id, stage="decompose")
        plan_path = work / "model_shards.json"
        _run([str(VENV_PY), "-m", "valhalla_server.shred", "--model", str(base),
              "--out", str(plan_path)], timeout=1800)
        plan = json.loads(plan_path.read_text(encoding="utf-8"))

        _set_job(job_id, stage="model_fate")
        fate = _last_json(_run(
            [str(runtime.engine("valhalla_model_fate.exe")), "--shards", str(plan_path),
             "--manifest-out", str(work / "staging"), "--base-model-id", str(base),
             "--bake-mode", "full", "--strength", str(strength)], timeout=900))
        outcome = fate.get("model_fate") or fate

        _set_job(job_id, stage="fusion")
        fused = [{"id": str(s.get("id") or f"s{i}"), "text": str(s.get("text", "")).strip(),
                  "audio_features": _hash_embed(str(s.get("text", ""))),
                  "image_features": _hash_embed(str(s.get("text", "")) + "#image")}
                 for i, s in enumerate(samples) if str(s.get("text", "")).strip()]
        if not fused:
            raise RuntimeError("samples 里没有可用文本。")
        fusion = _last_json(_run(
            [str(runtime.engine("valhalla_trimodal_incubate.exe")),
             "--manifest-out", str(work / "fusion_staging")],
            stdin=json.dumps({"samples": fused, "base_model_id": str(base),
                              "strength": strength, "bake_mode": "full"}),
            env={"VALHALLA_FREE_MULTIMODAL": "1"}, timeout=900))
        export_doc = fusion.get("export_doc")
        if not isinstance(export_doc, dict):
            raise RuntimeError("融合没有导出结构。")
        export_path = work / "fusion_export.json"
        export_path.write_text(json.dumps(export_doc, ensure_ascii=False), encoding="utf-8")
        corpus_path = work / "corpus.txt"
        corpus_path.write_text("\n".join(s["text"] for s in fused), encoding="utf-8")

        _set_job(job_id, stage="bake")
        tag = f"{base.name}-Valhalla-s{strength:.2f}"
        out_dir = base.parent / tag
        module = "valhalla_server.bake_mistral" if _is_sharded(base) else "valhalla_server.bake_runner"
        bake_out = _run([str(VENV_PY), "-m", module, "--base", str(base),
                         "--export", str(export_path), "--out", str(out_dir),
                         "--strength", str(strength), "--corpus", str(corpus_path)],
                        timeout=1800)
        bake = _last_json(bake_out[bake_out.rfind("{"):])
        manifest = json.loads((out_dir / "valhalla_manifest.json").read_text(encoding="utf-8"))

        _set_job(job_id, stage="convert")
        f16 = Path(f"{out_dir}-F16.gguf")
        gguf = Path(f"{out_dir}-{quant}.gguf")
        convert = LLAMA_SRC / "convert_hf_to_gguf.py"
        _run([str(VENV_PY), str(convert), str(out_dir), "--outfile", str(f16),
              "--outtype", "f16"], timeout=1800)

        _set_job(job_id, stage="quantize")
        from valhalla_server.config import LLAMA_BIN

        _run([str(LLAMA_BIN / "llama-quantize.exe"), str(f16), str(gguf), quant], timeout=1800)
        f16.unlink(missing_ok=True)
        digest = hashlib.sha256(gguf.read_bytes()).hexdigest()

        verify: dict = {}
        if deploy:
            _set_job(job_id, stage="deploy")
            if not head_server.start(str(gguf)):
                raise RuntimeError("解码头服务启动失败。")
            _set_job(job_id, stage="verify")
            grounded = mistral_head.decode(
                "What is Project Aurora's internal codename?",
                ["Project Aurora's internal codename is AUR-7."])
            trap = mistral_head.decode(
                "What is the wingspan of Project Aurora's drone?",
                ["Project Aurora's internal codename is AUR-7."])
            verify = {
                "grounded_ok": bool(grounded and "AUR-7" in grounded),
                "trap_abstained": bool(trap and "know" in trap.lower()),
                "grounded_answer": grounded, "trap_answer": trap,
            }
            if not verify["grounded_ok"]:
                raise RuntimeError(f"部署后验证失败：{verify}")

        heads = _heads()
        heads[head_id] = {
            "base_model": str(base), "strength": strength, "quant": quant,
            "gguf": str(gguf), "gguf_sha256": digest,
            "hf_dir": str(out_dir),
            "weight_delta_norm": bake.get("weight_delta_norm"),
            "layers": plan.get("layer_count"),
            "model_fate": {
                "adaptation_rounds": outcome.get("adaptation_rounds"),
                "stopped_reason": outcome.get("stopped_reason"),
                "decompose_score": outcome.get("decompose_score"),
                "model_closure_score": outcome.get("model_closure_score"),
            },
            "verify": verify,
            "created": datetime.now(timezone.utc).isoformat(),
        }
        _save_heads(heads)
        _set_job(job_id, status="done", stage="registered", head_id=head_id,
                 gguf=str(gguf), verify=verify)
    except Exception as exc:  # noqa: BLE001 - job status carries the error
        _set_job(job_id, status="failed", error=str(exc)[-600:])


def start_modelops(head_id: str, base_model: str, strength: float = 0.08,
                   quant: str = "Q4_K_M", samples: list[dict] | None = None,
                   deploy: bool = True) -> dict:
    job_id = f"job_{datetime.now(timezone.utc).strftime('%Y%m%d%H%M%S')}_{abs(hash(head_id)) % 100000:05d}"
    _set_job(job_id, status="queued", stage="queued", model_id=head_id, kind="modelops",
             created=datetime.now(timezone.utc).isoformat())
    thread = threading.Thread(
        target=run_modelops,
        args=(job_id, head_id, base_model, strength, quant,
              samples or DEFAULT_SAMPLES, deploy),
        daemon=True)
    thread.start()
    return {"job_id": job_id, "status": "queued", "status_url": f"/v1/jobs/{job_id}"}
