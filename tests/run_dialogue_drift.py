"""Dialogue-core drift + zero-hallucination test on a big corpus.

Big Valhalla pattern (as deployed): 10,006-line corpus stays client-side;
one persistent engine session accumulates targeted lines turn by turn.
Core needle questions are re-asked at dialogue start / middle / end and
twice more back-to-back at the very end. Distractor questions (real
filler facts) and trap questions (swap / absent / false premise /
conflict / injection) are interleaved. Product policy is applied
(resolve_fate_answer: Fate decides, transformer never does).

Drift  = a core question's pass/answer changes across repetitions.
Hallucination = fabricated codename/number tokens (not in corpus),
forbidden tokens, or a trap answered instead of abstained.
"""
import json
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, r"C:\Ai\rogue-products\valhalla-server\tests")
sys.path.insert(0, r"C:\Ai\rogue-products\valhalla-server")
from run_extreme_stress import (CONFIG, QUESTIONS, fabrication, gen_corpus,  # noqa: E402
                                is_abstain, post, score)
from run_extreme_targeted import ENGINE, targeted  # noqa: E402
from valhalla_server import hybrid, internal  # noqa: E402

OUT = Path(r"C:\Ai\rogue-products\valhalla-server\tests\results_extreme")
N_LINES = int(sys.argv[1]) if len(sys.argv) > 1 else 10006

Q = {q[0]: q for q in QUESTIONS}
CORE = [Q["n_start"], Q["n_mid"], Q["n_end"]]
TRAPS = [Q["swap1"], Q["swap2"], Q["absent_ent"], Q["absent_attr"],
         Q["absent_attr2"], Q["premise"], Q["inject_q"]]
CONFLICT = Q["conflict"]

_CODE_LINE = re.compile(r"(Project \w+) uses the internal codename ([A-Z]{3}-\d+)")


def distractors(corpus, n=40):
    out = []
    seen = set()
    for line in corpus:
        m = _CODE_LINE.search(line)
        if m and m.group(1) not in seen:
            seen.add(m.group(1))
            out.append((f"distr_{len(out)}", f"What is {m.group(1)}'s internal codename?",
                        [m.group(2)], ["AUR-999"], "needle"))
        if len(out) >= n:
            break
    return out


def build_schedule(distr):
    sched = []
    di = ti = 0
    # 6 rounds; each round: core x3, distractors x4, traps x2, conflict x1
    for rnd in range(6):
        for c in CORE:
            sched.append(c)
            sched.append(distr[di % len(distr)]); di += 1
        sched.append(TRAPS[ti % len(TRAPS)]); ti += 1
        sched.append(distr[di % len(distr)]); di += 1
        sched.append(TRAPS[ti % len(TRAPS)]); ti += 1
        sched.append(CONFLICT)
    # final stability: each core twice back-to-back
    for c in CORE:
        sched.append(c); sched.append(c)
    return sched


def main():
    corpus, _ = gen_corpus(N_LINES)
    corpus_text = "\n".join(corpus)
    distr = distractors(corpus)
    sched = build_schedule(distr)
    print(f"corpus={len(corpus)} lines, schedule={len(sched)} turns", flush=True)
    sess = None
    rows = []
    rotations = 0
    t0 = time.time()
    for turn, q in enumerate(sched):
        qid, qtext = q[0], q[1]
        lines = targeted(qtext, corpus)
        if sess is not None:
            # Session wire ceiling (measured): fragments carry cumulative
            # tile state, wire doubles every ~2 turns and 413s at ~43MB.
            # Rotate before the wall: link first, else re-incubate a fresh
            # session from the accumulated corpus lines.
            frags = ((sess or {}).get("snapshot") or {}).get("fragments") or []
            if len(frags) >= internal.LINK_MIN_FRAGMENTS:
                linked = internal.link_session(sess, CONFIG)
                if linked:
                    sess = linked
            wire = len(json.dumps(internal.slim_session(sess)))
            if wire > 12_000_000:
                keep = list((sess or {}).get("corpus") or [])
                body = post(ENGINE, {"questions": [{"id": qid, "text": qtext,
                                                   "score_type": "open"}],
                                     "config": CONFIG, "context": keep + lines})
                sess = body.get("session")
                rotations += 1
                row = (body.get("rows") or [{}])[0]
                fog = (row.get("fog_decision") or {}).get("decision", "")
                sess_corpus = list((sess or {}).get("corpus") or [])
                answer, mode = hybrid.resolve_fate_answer(
                    fog, str(row.get("answer", "")), sess_corpus,
                    row.get("retrieval_corpus_idx"))
                sc = score(q, answer)
                fab = fabrication(answer, corpus_text)
                rows.append({"turn": turn, "id": qid, "kind": q[4], "fog": fog,
                             "mode": mode + "/rotated", "answer": answer[:200],
                             "pass": sc["pass"], "fabricated": fab,
                             "abstain": is_abstain(answer), "wire_mb": round(wire / 1e6, 2),
                             "session_lines": len(sess_corpus)})
                print(f"  t{turn:<3} {qid:<12} ROTATED pass={sc['pass']} fab={fab}", flush=True)
                continue
        if sess is None:
            payload = {"questions": [{"id": qid, "text": qtext, "score_type": "open"}],
                       "config": CONFIG, "context": lines}
        else:
            payload = {"questions": [{"id": qid, "text": qtext, "score_type": "open"}],
                       "config": CONFIG, "context": [],
                       "session": internal.slim_session(sess), "append_lines": lines}
        wire_mb = round(len(json.dumps(payload)) / 1e6, 2)
        body = post(ENGINE, payload)
        sess = body.get("session")
        # Gateway discipline: bundle seg fragments into link chains.
        frags = ((sess or {}).get("snapshot") or {}).get("fragments") or []
        if len(frags) >= internal.LINK_MIN_FRAGMENTS:
            linked = internal.link_session(sess, CONFIG)
            if linked:
                sess = linked
        row = (body.get("rows") or [{}])[0]
        fog = (row.get("fog_decision") or {}).get("decision", "")
        sess_corpus = list((sess or {}).get("corpus") or [])
        answer, mode = hybrid.resolve_fate_answer(
            fog, str(row.get("answer", "")), sess_corpus,
            row.get("retrieval_corpus_idx"))
        sc = score(q, answer)
        fab = fabrication(answer, corpus_text)
        rows.append({"turn": turn, "id": qid, "kind": q[4], "fog": fog,
                     "mode": mode, "answer": answer[:200],
                     "pass": sc["pass"], "fabricated": fab,
                     "abstain": is_abstain(answer), "wire_mb": wire_mb,
                     "session_lines": len(sess_corpus)})
        print(f"  t{turn:<3} {qid:<12} kind={q[4]:<13} pass={sc['pass']} "
              f"fog={fog:<8} fab={fab} sess={len(sess_corpus)}", flush=True)

    core = [r for r in rows if r["id"] in ("n_start", "n_mid", "n_end")]
    traps = [r for r in rows if r["kind"] in ("swap_trap", "absent", "false_premise", "injection")]
    distr_r = [r for r in rows if r["id"].startswith("distr_")]
    fab_total = sum(len(r["fabricated"]) for r in rows)
    summary = {
        "turns": len(rows),
        "core_pass": f"{sum(1 for r in core if r['pass'])}/{len(core)}",
        "core_answers_by_id": {cid: sorted({r["answer"] for r in core if r["id"] == cid})
                               for cid in ("n_start", "n_mid", "n_end")},
        "distractor_pass": f"{sum(1 for r in distr_r if r['pass'])}/{len(distr_r)}",
        "trap_pass": f"{sum(1 for r in traps if r['pass'])}/{len(traps)}",
        "fabricated_tokens_total": fab_total,
        "fabricated_rows": [{"turn": r["turn"], "id": r["id"], "fab": r["fabricated"]}
                            for r in rows if r["fabricated"]],
        "final_session_lines": rows[-1]["session_lines"] if rows else 0,
        "session_rotations": rotations,
        "max_wire_mb": max((r.get("wire_mb") or 0) for r in rows) if rows else 0,
        "minutes": round((time.time() - t0) / 60, 2),
    }
    dest = OUT / f"dialogue_drift_lines{N_LINES}.json"
    dest.write_text(json.dumps({"summary": summary, "rows": rows},
                               ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    print(f"wrote {dest}", flush=True)


if __name__ == "__main__":
    main()
