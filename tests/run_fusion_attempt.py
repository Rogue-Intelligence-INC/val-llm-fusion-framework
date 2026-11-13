"""Trimodal fusion attempt with REAL media.

Arms (valhalla_trimodal_incubate, protocol valhalla-free-multimodal-fusion-v3):
  F-text: samples carry text only (zero media features)
  F-tri : same texts + real TTS audio features + real SD-Turbo image features
Then a ValhallaBase session ingests the samples as context_items
(text+audio+image in ONE non-isolated session) and answers:
  - plain text questions about the facts
  - a question carrying only image_features ("which project is this about?")
  - a question carrying only audio_features
Mistral arms cannot consume feature vectors at all (text-only API here), so
the cross-modal probes are Valhalla-only by construction.
"""

from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))

from run_stress_points import infer_long  # noqa: E402

OUT = HERE / "results_unique_points"
MEDIA = HERE / "media"
DIM = 384


def incubate(samples: list[dict], tag: str) -> dict:
    from valhalla_server import runtime

    import os

    done = subprocess.run(
        [str(runtime.engine("valhalla_trimodal_incubate.exe")),
         "--manifest-out", str(HERE / f"fusion_{tag}")],
        input=json.dumps({"samples": samples, "base_model_id": "mistral-fusion-probe",
                          "strength": 0.08, "bake_mode": "full"}),
        capture_output=True, text=True, encoding="utf-8", errors="replace",
        timeout=900, env={**os.environ, "VALHALLA_FREE_MULTIMODAL": "1"})
    if done.returncode != 0:
        raise RuntimeError((done.stderr or done.stdout)[-500:])
    body = json.loads(done.stdout[done.stdout.find("{"):])
    return body


def main():
    feats = json.loads((MEDIA / "features.json").read_text(encoding="utf-8"))
    ids = sorted(feats)
    zeros = [0.0] * DIM
    text_samples = [{"id": i, "text": feats[i]["text"],
                     "audio_features": zeros, "image_features": zeros} for i in ids]
    tri_samples = [{"id": i, "text": feats[i]["text"],
                    "audio_features": feats[i]["audio_features"],
                    "image_features": feats[i]["image_features"]} for i in ids]

    results = {}
    for tag, samples in (("text", text_samples), ("tri", tri_samples)):
        body = incubate(samples, tag)
        keep = {k: body.get(k) for k in ("persistent_fusion", "export_doc") if k in body}
        pf = body.get("persistent_fusion") or {}
        results[tag] = {
            "aggregate_score": pf.get("aggregate_score"),
            "closure_rounds": pf.get("closure_rounds"),
            "protocol": pf.get("protocol"),
            "report_keys": sorted((pf.get("report") or {}).keys()),
            "tile_stem": {k: (pf.get("report") or {}).get(k)
                          for k in ("tile_count", "stem_cell_count", "hub_routed")},
        }
        print(f"[fusion-{tag}] aggregate={pf.get('aggregate_score')} "
              f"tile_stem={results[tag]['tile_stem']}", flush=True)

    # ---- session QA with multimodal context items (non-isolated) ----
    from valhalla_server import internal
    from valhalla_server.inference import _materialize

    folder = str(_materialize("unified-core-v1"))
    if not internal.start(folder):
        raise RuntimeError("runtime failed")
    config = internal.chat_config(folder)
    items = [{"text": feats[i]["text"],
              "audio_features": feats[i]["audio_features"],
              "image_features": feats[i]["image_features"]} for i in ids]
    session = None

    def ask(qid, text, context=None, qfeats=None, first=False):
        nonlocal session
        q = {"id": qid, "text": text, "score_type": "open"}
        if qfeats:
            q.update(qfeats)
        payload = {"questions": [q], "config": config, "context": []}
        if first:
            payload["context_items"] = items
        if session:
            payload["session"] = session
        body = infer_long(payload, timeout=900)
        session = body.get("session") or session
        return (body.get("rows") or [{}])[0]

    probes = []
    r = ask("mm1", "Who leads Project Aurora?", first=True)
    probes.append({"id": "mm1", "kind": "text_q", "answer": r.get("answer"),
                   "idx": r.get("retrieval_corpus_idx")})
    print(f"[mm1 text] {str(r.get('answer'))[:120]}", flush=True)
    r = ask("mm2", "Who leads Project Borealis?")
    probes.append({"id": "mm2", "kind": "text_q", "answer": r.get("answer"),
                   "idx": r.get("retrieval_corpus_idx")})
    print(f"[mm2 text] {str(r.get('answer'))[:120]}", flush=True)
    r = ask("mm3", "Which project is this image about? Answer with the project name.",
            qfeats={"image_features": feats["s1"]["image_features"]})
    probes.append({"id": "mm3", "kind": "image_q", "answer": r.get("answer"),
                   "idx": r.get("retrieval_corpus_idx")})
    print(f"[mm3 image-only] {str(r.get('answer'))[:160]}", flush=True)
    r = ask("mm4", "Which project is this audio about? Answer with the project name.",
            qfeats={"audio_features": feats["s2"]["audio_features"]})
    probes.append({"id": "mm4", "kind": "audio_q", "answer": r.get("answer"),
                   "idx": r.get("retrieval_corpus_idx")})
    print(f"[mm4 audio-only] {str(r.get('answer'))[:160]}", flush=True)

    stamp = time.strftime("%Y%m%d_%H%M%S")
    (OUT / f"fusion_{stamp}.json").write_text(
        json.dumps({"incubate": results, "probes": probes},
                   ensure_ascii=False, indent=2), encoding="utf-8")
    print("FUSION_DONE", flush=True)


if __name__ == "__main__":
    main()
