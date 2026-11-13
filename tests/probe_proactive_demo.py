"""Demonstrate the proactive EMIT machinery with the signal gate lowered to
one QA turn (32). NOT the production default (64) - labelled demo only."""
import json
import sys
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
    config["proactive_min_fate_signals"] = 32  # DEMO ONLY
    corpus = FACTS_A + FACTS_B + UPDATES
    session = None

    def call(payload):
        nonlocal session
        if session:
            payload["session"] = session
        body = infer_long(payload, timeout=900)
        session = body.get("session") or session
        return body

    call({"questions": [{"id": "w0", "text": "Who leads Project Aurora?", "score_type": "open"}],
          "config": config, "context": corpus})
    call({"questions": [{"id": "w1", "text": "Does the context say anything about Mara Voss?", "score_type": "open"}],
          "config": config, "context": []})
    print("warm done (gate=32 demo)", flush=True)
    for attempt in range(3):
        body = call({"questions": [], "proactive_idle": True, "config": config, "context": []})
        print(f"--- idle {attempt} ---", flush=True)
        print("proactive:", json.dumps(body.get("proactive"), ensure_ascii=False)[:1600], flush=True)
        print("rows:", json.dumps(body.get("rows"), ensure_ascii=False)[:1200], flush=True)


if __name__ == "__main__":
    main()
