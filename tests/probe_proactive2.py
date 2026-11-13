"""Proactive probe #2: heavier warmup, dump the full idle response body."""
import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))

from run_stress_points import infer_long  # noqa: E402
from run_unique_points import FACTS_A, FACTS_B, UPDATES  # noqa: E402


def main():
    from valhalla_server import internal
    from valhalla_server.inference import _materialize

    folder = str(_materialize("unified-core-v1"))
    if not internal.start(folder):
        raise RuntimeError("runtime failed")
    config = internal.chat_config(folder)
    corpus = FACTS_A + FACTS_B + UPDATES
    session = None

    def call(payload):
        nonlocal session
        if session:
            payload["session"] = session
        body = infer_long(payload, timeout=900)
        session = body.get("session") or session
        return body

    qs = ["Who leads Project Aurora?", "Who leads Project Borealis?",
          "What is Project Aurora's budget?", "What is Project Borealis's budget?",
          "Where will Project Aurora hold its first flight test?",
          "Near which islands will Project Borealis be deployed?",
          "What is Project Aurora's internal codename?",
          "What is Project Borealis's internal codename?",
          "When is Project Aurora's first flight test scheduled?",
          "When is Project Borealis's first sea trial scheduled?"]
    for i, q in enumerate(qs):
        body = call({"questions": [{"id": f"w{i}", "text": q, "score_type": "open"}],
                     "config": config, "context": corpus if i == 0 else []})
        r = (body.get("rows") or [{}])[0]
        print(f"[warm {i}] {str(r.get('answer'))[:80]}", flush=True)

    for attempt in range(4):
        t0 = time.time()
        body = call({"questions": [], "proactive_idle": True, "config": config, "context": []})
        print(f"[idle {attempt}] {time.time()-t0:.1f}s keys={sorted(body.keys())}", flush=True)
        print(json.dumps(body, ensure_ascii=False)[:1500], flush=True)


if __name__ == "__main__":
    main()
