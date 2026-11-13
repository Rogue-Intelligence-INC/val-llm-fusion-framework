"""Proactive demo v2: warmup questions WITH reference answers (so Fate QA
feedback can build hub_prefs spread) + demo signal gate 32."""
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))

from run_stress_points import infer_long  # noqa: E402
from run_unique_points import FACTS_A, FACTS_B, UPDATES  # noqa: E402

WARM = [
    ("Who leads Project Aurora?", "Dr. Mara Voss", ["Mara Voss"]),
    ("Who leads Project Borealis?", "Dr. Ilya Petrov", ["Ilya Petrov"]),
    ("What is Project Aurora's internal codename?", "AUR-7", ["AUR-7"]),
    ("What is Project Borealis's internal codename?", "BOR-3", ["BOR-3"]),
    ("Where will Project Aurora hold its first flight test?", "Reykjavik", ["Reykjavik"]),
    ("Near which islands will Project Borealis be deployed?", "Faroe Islands", ["Faroe"]),
]


def main():
    from valhalla_server import internal
    from valhalla_server.inference import _materialize

    folder = str(_materialize("unified-core-v1"))
    if not internal.start(folder):
        raise RuntimeError("runtime failed")
    config = internal.chat_config(folder)
    config["proactive_min_fate_signals"] = 32  # DEMO ONLY (production 64)
    corpus = FACTS_A + FACTS_B + UPDATES
    session = None

    def call(payload):
        nonlocal session
        if session:
            payload["session"] = session
        body = infer_long(payload, timeout=900)
        session = body.get("session") or session
        return body

    for i, (q, ref, kws) in enumerate(WARM):
        body = call({"questions": [{"id": f"w{i}", "text": q, "score_type": "open",
                                    "reference_answer": ref, "keywords": kws}],
                     "config": config, "context": corpus if i == 0 else []})
        tel = body.get("fate_telemetry") or {}
        print(f"[warm {i}] spread={tel.get('hub_prefs_spread')}", flush=True)
    for attempt in range(3):
        body = call({"questions": [], "proactive_idle": True, "config": config, "context": []})
        print(f"--- idle {attempt} ---", flush=True)
        print("proactive:", json.dumps(body.get("proactive"), ensure_ascii=False)[:1600], flush=True)
        print("rows:", json.dumps(body.get("rows"), ensure_ascii=False)[:1200], flush=True)


if __name__ == "__main__":
    main()
