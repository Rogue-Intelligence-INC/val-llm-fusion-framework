"""Print ONLY the proactive + telemetry sections of the idle response."""
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
          "What is Project Borealis's internal codename?"]
    for i, q in enumerate(qs):
        call({"questions": [{"id": f"w{i}", "text": q, "score_type": "open"}],
              "config": config, "context": corpus if i == 0 else []})
    print("warm done", flush=True)
    for attempt in range(3):
        body = call({"questions": [], "proactive_idle": True, "config": config, "context": []})
        print(f"--- idle {attempt} ---", flush=True)
        print("proactive:", json.dumps(body.get("proactive"), ensure_ascii=False)[:1200], flush=True)
        print("rows:", json.dumps(body.get("rows"), ensure_ascii=False)[:800], flush=True)
        print("fate_telemetry:", json.dumps(body.get("fate_telemetry"), ensure_ascii=False)[:600], flush=True)


if __name__ == "__main__":
    main()
