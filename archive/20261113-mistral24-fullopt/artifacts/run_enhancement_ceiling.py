"""LLM-enhancement ceiling test (NOT the gateway product form).

Question: if Valhalla is used purely as an enhancement layer in front
of a fixed LLM (Mistral-24B decode head), how far does the capability
actually extend? Direct engine calls (8787) + in-process hybrid
resolution; the HTTP gateway is not involved.

Arms (same questions, same decode head):
  hybrid - targeted ingest into a persistent Valhalla session, Fate
           verdict + hybrid.resolve verbalization (the enhancement)
  rag    - plain RAG control: the SAME lexical top-8 lines fed
           directly to the same LLM, no Fate, no session

Beyond the standard 13Q battery (needles/paraphrase/swap/absent/
premise/conflict/injection), a reasoning probe set tests the ceiling:
two-hop chains, budget comparison, counting, listing, temporal
version, negation, multi-fact synthesis. Corpus is the deterministic
extreme corpus with extra unique-phrase facts scattered by depth.
"""
import json
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, r"C:\Ai\rogue-products\valhalla-server\tests")
sys.path.insert(0, r"C:\Ai\rogue-products\valhalla-server")
from run_extreme_stress import CONFIG, QUESTIONS, fabrication, gen_corpus, is_abstain, post, score  # noqa: E402
from run_extreme_targeted import ENGINE, slim, targeted  # noqa: E402
from valhalla_server import hybrid, mistral_head  # noqa: E402

OUT = Path(r"C:\Ai\rogue-products\valhalla-server\tests\results_extreme")
OUT.mkdir(exist_ok=True)
K = 8

EXTRA_FACTS = [
    (0.12, "Project Aurora maintains a lunar calibration beacon."),
    (0.18, "Project Aurora's battery supplier is Helix Cells."),
    (0.22, "Project Aurora has a budget of 12.5 million euros for 2026."),
    (0.30, "Project Aurora uses a titanium alloy frame."),
    (0.36, "Project Aurora's codename in 2025 was AUR-6."),
    (0.42, "Helix Cells is headquartered in Munich, Germany."),
    (0.48, "Project Borealis maintains a lunar calibration beacon."),
    (0.52, "Project Borealis uses a titanium alloy frame."),
    (0.58, "Project Borealis has a budget of 9.5 million euros for 2026."),
    (0.64, "Project Halcyon uses a titanium alloy frame."),
    (0.66, "Dr. Ines Halvorsen is employed by the Nordic Orbital Institute."),
    (0.74, "Project Borealis does not use lithium batteries; it uses nickel-hydrogen batteries."),
    (0.82, "Project Halcyon maintains a lunar calibration beacon."),
    (0.88, "Project Cygnus uses a titanium alloy frame."),
]


def _has(ans, *subs):
    t = (ans or "").lower()
    return all(s.lower() in t for s in subs)


NEW_QUESTIONS = [
    {"id": "hop_supplier", "kind": "multihop",
     "text": "In which city is the headquarters of Project Aurora's battery supplier?",
     "check": lambda a: _has(a, "Munich")},
    {"id": "hop_employer", "kind": "multihop",
     "text": "Which organization employs the lead of the Borealis satellite program?",
     "check": lambda a: _has(a, "Nordic Orbital")},
    {"id": "compare_budget", "kind": "comparison",
     "text": "Which project has the larger 2026 budget, Project Aurora or Project Borealis?",
     "check": lambda a: _has(a, "Aurora") and any(w in a.lower() for w in ("larger", "greater", "higher", "12.5"))},
    {"id": "count_beacons", "kind": "aggregation",
     "text": "How many projects maintain a lunar calibration beacon?",
     "check": lambda a: bool(re.search(r"\b3\b", a or "")) or "three" in (a or "").lower()},
    {"id": "list_frames", "kind": "aggregation",
     "text": "List all projects that use a titanium alloy frame.",
     "check": lambda a: _has(a, "Aurora", "Borealis", "Halcyon", "Cygnus")},
    {"id": "synth_two", "kind": "synthesis",
     "text": "Who leads the Borealis satellite program and where is Project Aurora's flight test site?",
     "check": lambda a: _has(a, "Halvorsen", "Reno")},
    {"id": "temporal_2025", "kind": "temporal",
     "text": "What was Project Aurora's codename in 2025?",
     "check": lambda a: "AUR-6" in (a or "") and "AUR-7" not in (a or "")},
    {"id": "negation_batt", "kind": "negation",
     "text": "What battery chemistry does Project Borealis use?",
     "check": lambda a: "nickel-hydrogen" in (a or "").lower() or "nickel hydrogen" in (a or "").lower()},
    {"id": "absent_supplier", "kind": "absent",
     "text": "Which company supplies solar sails for Project Zephyr?",
     "check": lambda a: is_abstain(a)},
]


def build_corpus(n):
    corpus, _ = gen_corpus(n)
    for frac, line in sorted(EXTRA_FACTS, key=lambda x: -x[0]):
        corpus.insert(int(len(corpus) * frac), line)
    return corpus


def items():
    out = [{"id": q[0], "text": q[1], "kind": q[4], "orig": q} for q in QUESTIONS]
    out += [{"id": q["id"], "text": q["text"], "kind": q["kind"], "new": q}
            for q in NEW_QUESTIONS]
    return out


def judge(it, answer):
    if "orig" in it:
        return score(it["orig"], answer)
    ok = bool(answer) and bool(it["new"]["check"](answer))
    return {"pass": ok, "note": "reasoning_probe"}


def run_hybrid(corpus, its):
    sess = None
    rows = []
    for it in its:
        t1 = time.time()
        lines = targeted(it["text"], corpus)
        if sess is None:
            payload = {"questions": [{"id": it["id"], "text": it["text"], "score_type": "open"}],
                       "config": CONFIG, "context": lines}
        else:
            payload = {"questions": [{"id": it["id"], "text": it["text"], "score_type": "open"}],
                       "config": CONFIG, "context": [], "session": slim(sess),
                       "append_lines": lines}
        body = post(ENGINE, payload)
        sess = body.get("session")
        row = (body.get("rows") or [{}])[0]
        fog = (row.get("fog_decision") or {}).get("decision", "")
        sess_corpus = list((sess or {}).get("corpus") or [])
        answer, mode = hybrid.resolve(fog, it["text"], sess_corpus,
                                      row.get("retrieval_corpus_idx"),
                                      str(row.get("answer", "")))
        sc = judge(it, answer)
        rows.append({"id": it["id"], "kind": it["kind"], "answer": f"[{mode}] {answer}"[:300],
                     "fog": fog, "sec": round(time.time() - t1, 1),
                     "session_lines": len(sess_corpus), **sc})
        print(f"  [hybrid] {it['id']:16s} pass={sc['pass']} fog={str(fog):8s} {rows[-1]['sec']}s", flush=True)
    return rows, sess


def run_rag(corpus, its):
    rows = []
    for it in its:
        t1 = time.time()
        lines = [corpus[i] for i, _s in hybrid.tile_track(it["text"], corpus, k=K)]
        answer = mistral_head.decode(it["text"], lines) or ""
        sc = judge(it, answer)
        rows.append({"id": it["id"], "kind": it["kind"], "answer": answer[:300],
                     "sec": round(time.time() - t1, 1), **sc})
        print(f"  [rag] {it['id']:16s} pass={sc['pass']} {rows[-1]['sec']}s", flush=True)
    return rows


def main():
    sizes = [int(x) for x in sys.argv[1].split(",")] if len(sys.argv) > 1 else [30006]
    arms = tuple(sys.argv[2].split(",")) if len(sys.argv) > 2 else ("hybrid", "rag")
    its = items()
    for n in sizes:
        corpus = build_corpus(n)
        corpus_text = "\n".join(corpus)
        print(f"\n### enhancement-ceiling lines{n}: {len(corpus)} lines, ~{len(corpus_text)//4} est tokens", flush=True)
        res = {"size": n, "lines": len(corpus), "est_tokens": len(corpus_text) // 4,
               "form": "llm_enhancement_direct (no gateway)", "arms": {}}
        final_sess = None
        for arm in arms:
            if arm == "hybrid":
                rows, final_sess = run_hybrid(corpus, its)
            else:
                rows = run_rag(corpus, its)
            for r in rows:
                r["fabricated_tokens"] = fabrication(r["answer"], corpus_text)
            res["arms"][arm] = rows
        if final_sess:
            res["session_wire_mb"] = round(len(json.dumps(final_sess)) / 1e6, 2)
            res["session_lines"] = len(final_sess.get("corpus") or [])
        dest = OUT / f"enhancement_ceiling_lines{n}.json"
        dest.write_text(json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")
        for arm, rows in res["arms"].items():
            scored = [r for r in rows if r["pass"] is not None]
            by = {}
            for r in scored:
                by.setdefault(r["kind"], [0, 0])
                by[r["kind"]][1] += 1
                by[r["kind"]][0] += 1 if r["pass"] else 0
            print(f"  == {arm}: {sum(1 for r in scored if r['pass'])}/{len(scored)} pass, "
                  f"fabricated={sum(len(r['fabricated_tokens']) for r in rows)}", flush=True)
            for kind, (p, t) in sorted(by.items()):
                print(f"     {kind:14s} {p}/{t}", flush=True)
        print(f"  session wire: {res.get('session_wire_mb')} MB, lines={res.get('session_lines')}", flush=True)


if __name__ == "__main__":
    main()
