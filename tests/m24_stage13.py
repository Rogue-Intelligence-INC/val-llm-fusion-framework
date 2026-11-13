"""Tier-A stage 1-3 for Mistral-24B: shred -> model_fate -> trimodal fusion.

Uses generic Valhalla samples only (never the benchmark facts) so the bake
cannot leak test answers into the weights.
"""
import json
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

from valhalla_server import runtime  # noqa: E402
from valhalla_server.config import VENV_PY  # noqa: E402
from valhalla_server.pipeline import DEFAULT_SAMPLES, _hash_embed, _last_json  # noqa: E402

BASE = r"C:\Ai\models\Mistral-Small-3.1-24B-Instruct-2503"
WORK = HERE / "mistral24"
WORK.mkdir(exist_ok=True)


def run(cmd, stdin=None, env=None, timeout=1800):
    import os
    done = subprocess.run(cmd, input=stdin, capture_output=True, text=True,
                          encoding="utf-8", errors="replace", timeout=timeout,
                          env={**os.environ, **(env or {})})
    if done.returncode != 0:
        raise RuntimeError((done.stderr or done.stdout)[-600:])
    return done.stdout


def main():
    plan = WORK / "model_shards.json"
    if not plan.exists():
        print(run([str(VENV_PY), "-m", "valhalla_server.shred", "--model", BASE,
                   "--out", str(plan)], timeout=1800), flush=True)
    doc = json.loads(plan.read_text(encoding="utf-8"))
    print(f"plan layers={doc['layer_count']}", flush=True)

    fate_out = run([str(runtime.engine("valhalla_model_fate.exe")),
                    "--shards", str(plan), "--manifest-out", str(WORK / "staging"),
                    "--base-model-id", BASE, "--bake-mode", "full", "--strength", "0.08"],
                   timeout=900)
    fate = _last_json(fate_out)
    print("model_fate:", json.dumps(fate.get("model_fate") or fate, ensure_ascii=False)[:400], flush=True)

    fused = [{"id": str(s.get("id") or f"s{i}"), "text": str(s["text"]).strip(),
              "audio_features": _hash_embed(str(s["text"])),
              "image_features": _hash_embed(str(s["text"]) + "#image")}
             for i, s in enumerate(DEFAULT_SAMPLES)]
    fusion_out = run([str(runtime.engine("valhalla_trimodal_incubate.exe")),
                      "--manifest-out", str(WORK / "fusion_staging")],
                     stdin=json.dumps({"samples": fused, "base_model_id": BASE,
                                       "strength": 0.08, "bake_mode": "full"}),
                     env={"VALHALLA_FREE_MULTIMODAL": "1"}, timeout=900)
    fusion = _last_json(fusion_out)
    export_doc = fusion.get("export_doc")
    if not isinstance(export_doc, dict):
        raise RuntimeError("fusion produced no export_doc")
    (WORK / "fusion_export.json").write_text(json.dumps(export_doc, ensure_ascii=False), encoding="utf-8")
    (WORK / "corpus.txt").write_text("\n".join(s["text"] for s in fused), encoding="utf-8")
    print("fusion persistent:", json.dumps(fusion.get("persistent_fusion") or {}, ensure_ascii=False)[:300], flush=True)
    print("STAGE13_DONE", flush=True)


if __name__ == "__main__":
    main()
