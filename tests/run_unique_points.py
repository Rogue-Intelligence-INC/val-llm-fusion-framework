"""Valhalla unique-points benchmark vs a raw 24B LM.

Same fictional corpus + same questions, three arms:
  M-full   Mistral-24B (llama-server) with the whole corpus dumped in chat
           history, questions asked turn by turn (standard full-context use)
  M-closed Mistral-24B closed book (trap questions only, no corpus)
  V        ValhallaBase via valhalla-server internal runtime: corpus ingested
           once into a persistent Triad session, then questions turn by turn

Subtests: grounded QA under pollution, hallucination traps (unanswerable),
superseded-fact conflicts, interleaved theme drift with return probes.

Scoring is deterministic (expected substring / abstention markers) and every
raw answer is saved for manual audit. Fictional facts only -> no training
data leakage can explain a correct answer.
"""

from __future__ import annotations

import json
import re
import sys
import time
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
SERVER_ROOT = HERE.parent
sys.path.insert(0, str(SERVER_ROOT))

MISTRAL_URL = "http://127.0.0.1:18080/v1/chat/completions"
OUT = HERE / "results_unique_points"
OUT.mkdir(exist_ok=True)

# ---------------------------------------------------------------- corpus
FACTS_A = [  # Project Aurora -- solar drone
    "Project Aurora is a solar drone program led by Dr. Mara Voss.",
    "Project Aurora has a budget of 4.2 million euros.",
    "Project Aurora will hold its first flight test in Reykjavik.",
    "Project Aurora uses battery cells supplied by Helix Cells.",
    "Project Aurora's first flight test is scheduled for March 14.",
    "Project Aurora's internal codename is AUR-7.",
    "Project Aurora's drone has a maximum altitude of 8200 meters.",
    "Project Aurora's team has 17 members.",
]
FACTS_B = [  # Project Borealis -- tidal sensor
    "Project Borealis is a tidal sensor program led by Dr. Ilya Petrov.",
    "Project Borealis has a budget of 2.9 million euros.",
    "Project Borealis will be deployed near the Faroe Islands.",
    "Project Borealis uses pressure sensors supplied by Nordwave.",
    "Project Borealis's first sea trial is scheduled for June 2.",
    "Project Borealis's internal codename is BOR-3.",
    "Project Borealis's sensor has a maximum depth of 340 meters.",
    "Project Borealis's team has 11 members.",
]
UPDATES = [  # superseded facts: later line wins
    "Note: Project Aurora's budget was initially 3.8 million euros and was later revised to 4.2 million euros.",
    "Note: Project Borealis's sea trial was originally planned for May 20 and was later moved to June 2.",
]


def filler_lines() -> list[str]:
    cities = ["Oslo", "Turku", "Gdansk", "Riga", "Tallinn", "Bergen", "Aarhus",
              "Malmo", "Tampere", "Oulu", "Esbjerg", "Klaipeda", "Visby",
              "Hanko", "Pori", "Umea", "Lulea", "Narvik", "Bodo", "Kristiansand"]
    topics = ["harbor crane maintenance", "ferry timetable audit",
              "lighthouse lens cleaning", "warehouse roof inspection",
              "bridge paint survey", "port customs workshop",
              "fish market refrigeration check", "rail siding repair",
              "weather station calibration", "buoy chain replacement"]
    out = []
    k = 0
    for i, city in enumerate(cities):
        for j, topic in enumerate(topics[:6]):
            k += 1
            out.append(f"Log entry {k}: the {city} office completed a routine {topic} in week {i + j + 1}.")
    return out


def build_corpus() -> list[str]:
    fill = filler_lines()
    corpus = []
    corpus += FACTS_A[:4]
    corpus += fill[:40]
    corpus += FACTS_B[:4]
    corpus += fill[40:80]
    corpus += FACTS_A[4:]
    corpus += fill[80:]
    corpus += FACTS_B[4:]
    corpus += UPDATES
    return corpus


# ---------------------------------------------------------------- questions
# kind: grounded | trap | conflict | drift_ask | drift_return
QUESTIONS = [
    # turn order matters: interleaved A/B, traps and conflicts mixed in
    {"id": "g1", "kind": "grounded", "text": "Who leads Project Aurora?", "expect": ["mara voss"]},
    {"id": "g2", "kind": "grounded", "text": "Who leads Project Borealis?", "expect": ["ilya petrov"]},
    {"id": "t1", "kind": "trap", "text": "What is the wingspan of Project Aurora's drone?", "expect": []},
    {"id": "g3", "kind": "grounded", "text": "Where will Project Aurora hold its first flight test?", "expect": ["reykjavik"]},
    {"id": "g4", "kind": "grounded", "text": "Near which islands will Project Borealis be deployed?", "expect": ["faroe"]},
    {"id": "c1", "kind": "conflict", "text": "What is Project Aurora's current budget?", "expect": ["4.2"], "stale": ["3.8"]},
    {"id": "t2", "kind": "trap", "text": "What color is Project Borealis's sensor housing?", "expect": []},
    {"id": "g5", "kind": "grounded", "text": "What is Project Borealis's internal codename?", "expect": ["bor-3"]},
    {"id": "g6", "kind": "grounded", "text": "What is Project Aurora's internal codename?", "expect": ["aur-7"]},
    {"id": "c2", "kind": "conflict", "text": "When is Project Borealis's first sea trial scheduled now?", "expect": ["june 2"], "stale": ["may 20"]},
    {"id": "t3", "kind": "trap", "text": "Who is the CEO of Helix Cells?", "expect": []},
    {"id": "g7", "kind": "grounded", "text": "What is the maximum depth of Project Borealis's sensor?", "expect": ["340"]},
    {"id": "g8", "kind": "grounded", "text": "What is the maximum altitude of Project Aurora's drone?", "expect": ["8200", "8,200"]},
    {"id": "t4", "kind": "trap", "text": "How many hours can Project Borealis's sensor run on one charge?", "expect": []},
    {"id": "t5", "kind": "trap", "text": "On which street in Reykjavik is Project Aurora's office located?", "expect": []},
    {"id": "r1", "kind": "drift_return", "text": "Going back to the first project we discussed: who leads it and what is its codename?", "expect": ["mara voss", "aur-7"]},
    {"id": "r2", "kind": "drift_return", "text": "And for the second project we discussed: who leads it and when is its first sea trial?", "expect": ["ilya petrov", "june 2"]},
    {"id": "t6", "kind": "trap", "text": "What year was Project Aurora founded?", "expect": []},
]

ABSTAIN = ["don't know", "do not know", "not mentioned", "not stated", "no information",
           "isn't mentioned", "is not mentioned", "not provided", "not specified",
           "unknown", "can't tell", "cannot tell", "没有提及", "未提及", "不知道",
           "无法得知", "no mention", "not given", "doesn't say", "does not say",
           "not in the notes", "not in the provided"]


def score(q: dict, answer: str) -> dict:
    a = (answer or "").lower()
    abstained = any(m in a for m in ABSTAIN) or len(a.strip()) < 3
    if q["kind"] == "trap":
        return {"correct": abstained, "abstained": abstained,
                "hallucinated": not abstained}
    hit = any(e in a for e in q["expect"])
    stale_hit = any(s in a for s in q.get("stale", []))
    if q["kind"] == "conflict":
        return {"correct": hit and not (stale_hit and not hit), "abstained": abstained,
                "hallucinated": (stale_hit and not hit) or (not hit and not abstained),
                "stale": stale_hit and not hit}
    return {"correct": hit, "abstained": abstained,
            "hallucinated": not hit and not abstained}


# ---------------------------------------------------------------- mistral
def mistral_chat(messages: list[dict], max_tokens: int = 220) -> str:
    payload = {"model": "mistral-24b", "messages": messages,
               "temperature": 0.2, "max_tokens": max_tokens}
    req = urllib.request.Request(MISTRAL_URL, data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(req, timeout=600) as resp:
        body = json.loads(resp.read().decode())
    return body["choices"][0]["message"]["content"].strip()


def run_mistral(corpus: list[str]) -> list[dict]:
    rows = []
    system = ("Answer questions strictly from the user's project notes. "
              "If the notes do not contain the answer, say you don't know. "
              "Reply in one or two short sentences.")
    messages = [{"role": "system", "content": system},
                {"role": "user", "content": "Here are my project notes:\n" + "\n".join(corpus)}]
    messages.append({"role": "assistant", "content": "Notes received. Ask me anything about them."})
    for q in QUESTIONS:
        messages.append({"role": "user", "content": q["text"]})
        t0 = time.time()
        ans = mistral_chat(messages)
        messages.append({"role": "assistant", "content": ans})
        s = score(q, ans)
        rows.append({"arm": "M-full", **q, "answer": ans, "latency_s": round(time.time() - t0, 2), **s})
        print(f"[M-full] {q['id']} correct={s['correct']} :: {ans[:110]}", flush=True)
    # closed-book traps
    for q in [x for x in QUESTIONS if x["kind"] == "trap"]:
        msgs = [{"role": "system", "content": system},
                {"role": "user", "content": q["text"]}]
        ans = mistral_chat(msgs)
        s = score(q, ans)
        rows.append({"arm": "M-closed", **q, "answer": ans, **s})
        print(f"[M-closed] {q['id']} correct={s['correct']} :: {ans[:110]}", flush=True)
    return rows


# ---------------------------------------------------------------- valhalla
def infer_long(payload: dict, timeout: int = 1800) -> dict:
    from valhalla_server.config import INTERNAL_PORT

    data = json.dumps(payload).encode()
    req = urllib.request.Request(
        f"http://127.0.0.1:{INTERNAL_PORT}/v1/infer", data=data,
        headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode())


def run_valhalla(corpus: list[str]) -> list[dict]:
    from valhalla_server import internal
    from valhalla_server.inference import _materialize

    folder = str(_materialize("unified-core-v1"))
    if not internal.start(folder):
        raise RuntimeError("ValhallaBase internal runtime failed to start")
    config = internal.chat_config(folder)
    rows = []
    session = None
    for i, q in enumerate(QUESTIONS):
        payload = {"questions": [{"id": q["id"], "text": q["text"], "score_type": "open"}],
                   "config": config,
                   "context": corpus if i == 0 else []}
        if session:
            payload["session"] = session
        t0 = time.time()
        body = infer_long(payload)
        session = body.get("session") or session
        r = (body.get("rows") or [{}])[0]
        ans = str(r.get("answer", "")).strip()
        fog = r.get("fog_decision") if isinstance(r.get("fog_decision"), dict) else {}
        fog_kind = fog.get("decision", "")
        effective = "I don't know." if fog_kind in ("abstain", "defer") else ans
        s = score(q, effective)
        rows.append({"arm": "V", **q, "answer": ans, "effective_answer": effective,
                     "fog_decision": fog_kind,
                     "decode_path": r.get("decode_path", ""),
                     "latency_s": round(time.time() - t0, 2), **s})
        print(f"[V] {q['id']} fog={fog_kind} correct={s['correct']} :: {effective[:110]}", flush=True)
    return rows


# ---------------------------------------------------------------- hybrid
def run_hybrid(corpus: list[str]) -> list[dict]:
    """valhalla-transformer policy, completed: Fate verdict + multi-track
    evidence (structure + Tile lexical) + Mistral-24B verified decode."""
    from valhalla_server import hybrid, internal
    from valhalla_server.inference import _materialize

    folder = str(_materialize("unified-core-v1"))
    if not internal.start(folder):
        raise RuntimeError("ValhallaBase internal runtime failed to start")
    config = internal.chat_config(folder)
    rows = []
    session = None
    for i, q in enumerate(QUESTIONS):
        payload = {"questions": [{"id": q["id"], "text": q["text"], "score_type": "open"}],
                   "config": config,
                   "context": corpus if i == 0 else []}
        if session:
            payload["session"] = session
        t0 = time.time()
        body = infer_long(payload)
        session = body.get("session") or session
        r = (body.get("rows") or [{}])[0]
        native = str(r.get("answer", "")).strip()
        fog = r.get("fog_decision") if isinstance(r.get("fog_decision"), dict) else {}
        fog_kind = fog.get("decision", "")
        effective, mode = hybrid.resolve(fog_kind, q["text"], corpus,
                                         r.get("retrieval_corpus_idx"), native)
        s = score(q, effective)
        rows.append({"arm": "V+M24-full", **q, "answer": native, "effective_answer": effective,
                     "fog_decision": fog_kind, "decode_head": mode,
                     "retrieval_corpus_idx": r.get("retrieval_corpus_idx"),
                     "latency_s": round(time.time() - t0, 2), **s})
        print(f"[V+M24-full] {q['id']} fog={fog_kind} mode={mode} correct={s['correct']} :: {effective[:110]}", flush=True)
    return rows


# ---------------------------------------------------------------- summary
def summarize(rows: list[dict]) -> dict:
    out = {}
    for arm in sorted({r["arm"] for r in rows}):
        sub = [r for r in rows if r["arm"] == arm]
        by = {}
        for kind in sorted({r["kind"] for r in sub}):
            ks = [r for r in sub if r["kind"] == kind]
            by[kind] = {"n": len(ks), "correct": sum(r["correct"] for r in ks),
                        "hallucinated": sum(r["hallucinated"] for r in ks)}
        out[arm] = {"total": {"n": len(sub), "correct": sum(r["correct"] for r in sub),
                              "hallucinated": sum(r["hallucinated"] for r in sub)},
                    "by_kind": by}
    return out


def main() -> None:
    corpus = build_corpus()
    print(f"corpus lines={len(corpus)} chars={sum(len(c) for c in corpus)}", flush=True)
    rows = []
    hybrid_only = "--hybrid-only" in sys.argv
    if hybrid_only:
        rows += run_hybrid(corpus)
    else:
        if "--mistral-only" not in sys.argv:
            try:
                rows += run_valhalla(corpus)
            except Exception as exc:  # noqa: BLE001
                print(f"[V] FAILED: {exc}", flush=True)
        if "--valhalla-only" not in sys.argv:
            rows += run_mistral(corpus)
    summary = summarize(rows)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    (OUT / f"run_{stamp}.json").write_text(
        json.dumps({"corpus": corpus, "questions": QUESTIONS, "rows": rows,
                    "summary": summary}, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
