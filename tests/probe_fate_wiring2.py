"""Fate probe round 2: dump fate_telemetry / fate_warmup verbatim and
check for an abstain/defer case + protocol declaration."""
import json
import urllib.request

BASE = "http://127.0.0.1:8787"
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
    "decode": "hybrid", "fate_ingress_routing": "quad_cycle",
    "fate_qa_feedback": True, "core_fusion": True, "open_fate": True,
    "answer_mix": "mixed", "mcq_decode": "max_pick", "incubation_cycles": 2,
    "fate_orchestrate": True, "fate_warmup_enabled": True, "use_cpu": False,
    "proactive_enabled": True, "max_new": 128,
}


def post(path, payload, timeout=1200):
    req = urllib.request.Request(BASE + path, data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode())


def get(path):
    with urllib.request.urlopen(BASE + path, timeout=30) as resp:
        return json.loads(resp.read().decode())


def main():
    try:
        proto = get("/v1/protocol")
        print("PROTOCOL:", json.dumps(proto, ensure_ascii=False)[:1200])
    except Exception as exc:  # noqa: BLE001
        print("PROTOCOL ERR:", exc)

    b1 = post("/v1/infer", {"questions": [{"id": "g1", "text": "What is Project Aurora's internal codename?", "score_type": "open"}],
                            "config": CONFIG, "context": CORPUS})
    print("\nWARMUP:", json.dumps(b1.get("fate_warmup"), ensure_ascii=False)[:800])
    print("TELEMETRY#1:", json.dumps(b1.get("fate_telemetry"), ensure_ascii=False)[:1500])
    print("DECODE:", json.dumps(b1.get("decode"), ensure_ascii=False)[:300])
    sess = b1.get("session")

    for qid, q in (("x1", "What is the chemical formula of the Aurora drone's fuel?"),
                   ("x2", "Who leads the Borealis satellite program?")):
        b = post("/v1/infer", {"questions": [{"id": qid, "text": q, "score_type": "open"}],
                               "config": CONFIG, "context": [], "session": sess})
        sess = b.get("session") or sess
        row = (b.get("rows") or [{}])[0]
        print(f"\n{qid} fog={json.dumps(row.get('fog_decision'), ensure_ascii=False)}")
        print(f"{qid} TELEMETRY:", json.dumps(b.get("fate_telemetry"), ensure_ascii=False)[:1500])
        print(f"{qid} fog_evidence:", json.dumps(row.get("fog_evidence"), ensure_ascii=False)[:500])


if __name__ == "__main__":
    main()
