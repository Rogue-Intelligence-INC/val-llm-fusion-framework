"""Probe: what does the ENGINE itself do with a feature-anchored
question (no hybrid)? Prints native answer, fog, core fusion row."""
import base64
import json
import sys
import urllib.request
from pathlib import Path

sys.path.insert(0, r"C:\Ai\rogue-products\valhalla-server\tests")
sys.path.insert(0, r"C:\Ai\rogue-products\valhalla-server")
from run_extreme_stress import CONFIG, post  # noqa: E402
from valhalla_server import media  # noqa: E402

GW = "http://127.0.0.1:8780"
ENGINE = "http://127.0.0.1:8787/v1/infer"
MEDIA = Path(r"C:\Ai\rogue-products\valhalla-server\tests\media")
MODEL = "unified-core-v1"


def gw(path, payload):
    return post(GW + path, payload)


def main():
    sess = None
    # reuse the session the battery built via gateway ask path is not
    # exposed; rebuild a small one directly on the engine instead:
    items = []
    sys.path.insert(0, r"C:\Ai\rogue-products\valhalla-server\tests")
    from run_multimodal_fixed import ITEMS, DISTRACTORS  # noqa: E402

    ctx = []
    for it in ITEMS:
        ctx.append({"text": it["text"],
                    "audio_features": media.encode_file(MEDIA / it["wav"], "audio"),
                    "image_features": media.encode_file(MEDIA / it["png"], "image")})
    for d in DISTRACTORS:
        ctx.append({"text": d})
    body = post(ENGINE, {"questions": [], "fate_warmup_only": True,
                         "context_items": ctx, "config": CONFIG})
    sess = body.get("session")
    print("ingested lines:", len((sess or {}).get("corpus") or []))

    for tag, q in (
        ("img_q", {"id": "q6", "text": "Which project is this image about?", "score_type": "open",
                   "image_features": media.encode_file(MEDIA / "s2.png", "image")}),
        ("aud_q", {"id": "q7", "text": "Which project is this audio about?", "score_type": "open",
                   "audio_features": media.encode_file(MEDIA / "s3.wav", "audio")}),
        ("img_q_kw", {"id": "q6b", "text": "Which project is this image about? Project Borealis or Project Aurora?", "score_type": "open",
                      "image_features": media.encode_file(MEDIA / "s2.png", "image")}),
    ):
        body = post(ENGINE, {"questions": [q], "config": CONFIG, "context": [], "session": sess})
        sess = body.get("session") or sess
        row = (body.get("rows") or [{}])[0]
        print(f"\n[{tag}] keys={sorted(body.keys())}")
        print(f"  answer={str(row.get('answer'))[:120]!r}")
        print(f"  fog={json.dumps(row.get('fog_decision'))} idx={row.get('retrieval_corpus_idx')} path={str(row.get('decode_path'))[:100]}")
        for k in ("core_fusion", "fusion", "universal"):
            if k in row:
                print(f"  {k}={json.dumps(row[k])[:200]}")
        print(f"  row_keys={sorted(row.keys())}")


if __name__ == "__main__":
    main()
