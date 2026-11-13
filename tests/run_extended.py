"""Extended capability arms: numeric / multihop / follow-up / novel ingest /
MCQ routing / proactive idle. Arms: M-full (Mistral solo) vs V+M24-full.

Novel-ingest protocol per arm:
  - ask wingspan BEFORE ingest (expect abstain)
  - append the new fact (Mistral: new user message; Valhalla: context append)
  - ask again (expect 6.5)
Proactive (Valhalla only): after warmup turns, send proactive_idle with no
questions and record whether Fate emits or blocks.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))

from run_unique_points import ABSTAIN, build_corpus, mistral_chat  # noqa: E402
from run_stress_points import infer_long  # noqa: E402

OUT = HERE / "results_unique_points"
OUT.mkdir(exist_ok=True)

NOVEL = "Project Aurora's drone has a wingspan of 6.5 meters."

SEQ = [
    {"id": "n1", "kind": "numeric", "text": "What is the combined budget of Project Aurora and Project Borealis, in million euros?", "expect": ["7.1"]},
    {"id": "n2", "kind": "numeric", "text": "How many more team members does Project Aurora have than Project Borealis?", "expect": ["6"]},
    {"id": "h1", "kind": "multihop", "text": "Which project has the earlier first test date, and in which city or area will that test be held?", "expect": ["aurora", "reykjavik"]},
    {"id": "f1", "kind": "followup", "text": "What is Project Aurora's internal codename?", "expect": ["aur-7"]},
    {"id": "f2", "kind": "followup", "text": "And what is its maximum altitude?", "expect": ["8200", "8,200"]},
    {"id": "x1", "kind": "novel_pre", "text": "What is the wingspan of Project Aurora's drone?", "expect": []},
    {"id": "INGEST", "kind": "ingest", "text": NOVEL, "expect": []},
    {"id": "x2", "kind": "novel_post", "text": "What is the wingspan of Project Aurora's drone?", "expect": ["6.5"]},
    {"id": "m1", "kind": "mcq", "text": "What is Project Borealis's internal codename? A) AUR-7  B) BOR-9  C) BOR-3  D) AUR-3", "expect": ["bor-3"]},
]


def judge(q: dict, ans: str) -> dict:
    a = (ans or "").lower()
    abstained = any(m in a for m in ABSTAIN) or len(a.strip()) < 3
    if q["kind"] == "novel_pre":
        return {"correct": abstained, "abstained": abstained}
    hit = all(e in a for e in q["expect"]) if q["kind"] == "multihop" else any(e in a for e in q["expect"])
    return {"correct": hit, "abstained": abstained}


def run_mistral(corpus: list[str]) -> list[dict]:
    system = ("Answer questions strictly from the user's project notes. "
              "If the notes do not contain the answer, say you don't know. "
              "Reply in one or two short sentences.")
    messages = [{"role": "system", "content": system},
                {"role": "user", "content": "Here are my project notes:\n" + "\n".join(corpus)},
                {"role": "assistant", "content": "Notes received."}]
    rows = []
    for q in SEQ:
        if q["kind"] == "ingest":
            messages.append({"role": "user", "content": f"New note: {NOVEL}"})
            messages.append({"role": "assistant", "content": "Noted."})
            continue
        messages.append({"role": "user", "content": q["text"]})
        ans = mistral_chat(messages)
        messages.append({"role": "assistant", "content": ans})
        s = judge(q, ans)
        rows.append({"arm": "M-full", **q, "answer": ans, **s})
        print(f"[M-ext] {q['id']} correct={s['correct']} :: {ans[:110]}", flush=True)
    return rows


def run_hybrid(corpus: list[str]) -> list[dict]:
    from valhalla_server import hybrid, internal
    from valhalla_server.inference import _materialize

    folder = str(_materialize("unified-core-v1"))
    if not internal.start(folder):
        raise RuntimeError("runtime failed")
    config = internal.chat_config(folder)
    rows, session, live_corpus = [], None, list(corpus)

    def ask(qid, text, score_type="open", context=None):
        nonlocal session
        payload = {"questions": [{"id": qid, "text": text, "score_type": score_type}],
                   "config": config, "context": context or []}
        if session:
            payload["session"] = session
        body = infer_long(payload)
        session = body.get("session") or session
        return (body.get("rows") or [{}])[0]

    for i, q in enumerate(SEQ):
        if q["kind"] == "ingest":
            live_corpus.append(NOVEL)
            ask("ingest", "What is Project Aurora's internal codename?", context=[NOVEL])
            continue
        r = ask(q["id"], q["text"],
                score_type="mcq" if q["kind"] == "mcq" else "open",
                context=corpus if i == 0 else None)
        native = str(r.get("answer", "")).strip()
        fog = r.get("fog_decision") if isinstance(r.get("fog_decision"), dict) else {}
        effective, mode = hybrid.resolve(fog.get("decision", ""), q["text"], live_corpus,
                                         r.get("retrieval_corpus_idx"), native)
        s = judge(q, effective)
        rows.append({"arm": "V+M24-full", **q, "answer": native, "effective_answer": effective,
                     "mode": mode, "fog_decision": fog.get("decision", ""), **s})
        print(f"[V-ext] {q['id']} mode={mode} correct={s['correct']} :: {effective[:110]}", flush=True)

    # ---- proactive idle: no question asked; does Fate speak on its own?
    for attempt in range(3):
        payload = {"questions": [], "proactive_idle": True, "config": config, "context": []}
        if session:
            payload["session"] = session
        try:
            body = infer_long(payload, timeout=600)
            session = body.get("session") or session
            prows = body.get("rows") or []
            detail = json.dumps(prows, ensure_ascii=False)[:500] if prows else "(no rows)"
            emitted = bool(prows) and any(str(r.get("answer", "")).strip() for r in prows)
            rows.append({"arm": "V-proactive", "id": f"p{attempt + 1}", "kind": "proactive",
                         "answer": detail, "correct": emitted, "abstained": not emitted})
            print(f"[V-proactive] attempt{attempt + 1} emitted={emitted} :: {detail[:160]}", flush=True)
            if emitted:
                break
        except Exception as exc:  # noqa: BLE001
            rows.append({"arm": "V-proactive", "id": f"p{attempt + 1}", "kind": "proactive",
                         "answer": f"ERROR {exc}"[:300], "correct": False, "abstained": True})
            print(f"[V-proactive] attempt{attempt + 1} ERROR {exc}", flush=True)
    return rows


def main():
    corpus = build_corpus()
    rows = []
    if "--mistral-only" not in sys.argv:
        rows += run_hybrid(corpus)
    if "--valhalla-only" not in sys.argv:
        rows += run_mistral(corpus)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    (OUT / f"extended_{stamp}.json").write_text(
        json.dumps({"rows": rows}, ensure_ascii=False, indent=2), encoding="utf-8")
    for arm in sorted({r["arm"] for r in rows}):
        sub = [r for r in rows if r["arm"] == arm and r["kind"] != "proactive"]
        print(arm, f"{sum(r['correct'] for r in sub)}/{len(sub)}", flush=True)


if __name__ == "__main__":
    main()
