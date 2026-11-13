"""Live Fate-wiring probe against the internal ValhallaBase runtime.

Sends a small fictional corpus + grounded/trap/repeat questions with the
production chat_config and prints every Fate-related field the engine
returns: fog_decision, proactive telemetry, decode_path, session reuse.
"""
import json
import urllib.request

URL = "http://127.0.0.1:8787/v1/infer"
CORPUS = [
    "Project Aurora's internal codename is AUR-7.",
    "Project Aurora's drone prototype completed its first flight in March.",
    "Project Borealis uses the codename BOR-3 for its satellite program.",
    "The Borealis satellite program is led by Dr. Ines Halvorsen.",
    "Project Aurora's budget for 2026 is 4.2 million euros.",
    "The Aurora drone prototype has a carbon-fiber frame.",
    "Borealis ground station is located near Tromsoe, Norway.",
    "Project Aurora's flight test site is in Nevada.",
]
CONFIG = {
    "decode": "hybrid",
    "fate_ingress_routing": "quad_cycle",
    "fate_qa_feedback": True,
    "core_fusion": True,
    "open_fate": True,
    "answer_mix": "mixed",
    "mcq_decode": "max_pick",
    "incubation_cycles": 2,
    "fate_orchestrate": True,
    "fate_warmup_enabled": True,
    "use_cpu": False,
    "proactive_enabled": True,
    "max_new": 128,
}


def infer(payload):
    req = urllib.request.Request(URL, data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"},
                                 method="POST")
    with urllib.request.urlopen(req, timeout=1200) as resp:
        return json.loads(resp.read().decode())


def show(tag, body):
    rows = body.get("rows") or []
    print(f"== {tag}: top_keys={sorted(body.keys())}")
    for r in rows:
        if not isinstance(r, dict):
            continue
        print(f"   row_keys={sorted(r.keys())}")
        for k in ("answer", "decode_path", "fog_decision", "proactive",
                  "fate", "retrieval_corpus_idx", "score_type"):
            if k in r:
                print(f"   {k} = {json.dumps(r[k], ensure_ascii=False)[:400]}")
    sess = body.get("session")
    print(f"   session: {'present' if isinstance(sess, dict) else sess} "
          f"{sorted(sess.keys()) if isinstance(sess, dict) else ''}")
    return sess


def main():
    b1 = infer({"questions": [{"id": "g1", "text": "What is Project Aurora's internal codename?", "score_type": "open"}],
                "config": CONFIG, "context": CORPUS})
    sess = show("grounded (first, with ingest)", b1)

    payload = {"questions": [{"id": "t1", "text": "What is the wingspan of Project Aurora's drone?", "score_type": "open"}],
               "config": CONFIG, "context": []}
    if sess:
        payload["session"] = sess
    b2 = infer(payload)
    sess = show("trap (session reuse)", b2) or sess

    payload = {"questions": [{"id": "g1b", "text": "What is Project Aurora's internal codename?", "score_type": "open"}],
               "config": CONFIG, "context": []}
    if sess:
        payload["session"] = sess
    b3 = infer(payload)
    show("grounded repeat (fate_qa_feedback check)", b3)


if __name__ == "__main__":
    main()
