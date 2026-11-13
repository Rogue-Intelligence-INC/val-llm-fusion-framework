"""Proactive probe #4: feed Fate signals via incremental ingest, then idle."""
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))

from run_stress_points import infer_long  # noqa: E402
from run_unique_points import FACTS_A, FACTS_B, UPDATES  # noqa: E402

EXTRA = [
    "Project Aurora's second flight test will be held in Akureyri.",
    "Project Borealis will add a second sensor array in the summer.",
    "Project Aurora's drone carries a multispectral camera.",
    "Project Borealis's sensor array reports data every hour.",
]


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

    def ask(i, q, context=None):
        return call({"questions": [{"id": f"q{i}", "text": q, "score_type": "open"}],
                     "config": config, "context": context or []})

    ask(0, "Who leads Project Aurora?", corpus)
    ask(1, "Who leads Project Borealis?")
    # Incremental ingest feeds Fate signals (32/line) while asking.
    ask(2, "What is Project Aurora's internal codename?", EXTRA)
    ask(3, "What is Project Borealis's internal codename?")
    print("warm+append done", flush=True)
    for attempt in range(4):
        body = call({"questions": [], "proactive_idle": True, "config": config, "context": []})
        print(f"--- idle {attempt} ---", flush=True)
        print("proactive:", json.dumps(body.get("proactive"), ensure_ascii=False)[:1200], flush=True)
        print("rows:", json.dumps(body.get("rows"), ensure_ascii=False)[:1000], flush=True)


if __name__ == "__main__":
    main()
