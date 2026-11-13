"""Probe: what does Fate (incl. FOG) actually decide per question?"""
import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))

from run_unique_points import FACTS_A, FACTS_B, UPDATES  # noqa: E402
from run_stress_points import infer_long  # noqa: E402


def main():
    from valhalla_server import internal
    from valhalla_server.inference import _materialize

    folder = str(_materialize("unified-core-v1"))
    if not internal.start(folder):
        raise RuntimeError("runtime failed")
    config = internal.chat_config(folder)
    print("gateway chat_config fate keys:",
          {k: v for k, v in config.items() if "fate" in k or "proactive" in k}, flush=True)
    corpus = FACTS_A + FACTS_B + UPDATES
    qs = [
        ("g1", "Who leads Project Aurora?"),
        ("t1", "What is the wingspan of Project Aurora's drone?"),
        ("t3", "Who is the CEO of Helix Cells?"),
        ("t6", "What year was Project Aurora founded?"),
    ]
    session = None
    for i, (qid, text) in enumerate(qs):
        payload = {"questions": [{"id": qid, "text": text, "score_type": "open"}],
                   "config": config, "context": corpus if i == 0 else []}
        if session:
            payload["session"] = session
        t0 = time.time()
        body = infer_long(payload)
        session = body.get("session") or session
        r = (body.get("rows") or [{}])[0]
        keep = {k: r.get(k) for k in ("answer", "decode_path", "fog_decision",
                                      "fog_evidence", "top_score", "score")
                if k in r}
        print(f"[{qid}] {time.time()-t0:.1f}s {json.dumps(keep, ensure_ascii=False)[:400]}", flush=True)
    print("row keys:", sorted((body.get('rows') or [{}])[0].keys()), flush=True)


if __name__ == "__main__":
    main()
