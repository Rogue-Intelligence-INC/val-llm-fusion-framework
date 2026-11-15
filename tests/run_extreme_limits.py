"""Context + zero-hallucination LIMIT test on the landed Fate gate.

One persistent engine session, 300 turns over a 10,006-line corpus
(client-side, targeted appends — the production pattern). Every 5th
round adds bulk fresh-line appends to push session volume. Questions:
is the limit behaviour produced by the new mechanism (block-form
state + block gate + silence-legal egress) or by leftover tuning?
Metrics are reported in turn-thirds so degradation is visible.
"""
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, r"C:\Ai\rogue-products\valhalla-server\tests")
sys.path.insert(0, r"C:\Ai\rogue-products\valhalla-server")
from run_extreme_stress import CONFIG, fabrication, gen_corpus, is_abstain, post, score  # noqa: E402
from run_extreme_targeted import ENGINE, targeted  # noqa: E402
from run_dialogue_drift import CONFLICT, CORE, TRAPS, distractors  # noqa: E402
from valhalla_server import hybrid, internal  # noqa: E402

OUT = Path(r"C:\Ai\rogue-products\valhalla-server\tests\results_extreme")
N_LINES = 10006
ROUNDS = int(sys.argv[1]) if len(sys.argv) > 1 else 30


def main():
    corpus, _ = gen_corpus(N_LINES)
    corpus_text = "\n".join(corpus)
    distr = distractors(corpus, n=200)
    print(f"corpus={len(corpus)} distractors={len(distr)} rounds={ROUNDS}", flush=True)

    sched = []
    di = ti = 0
    bulk_at = {}
    for rnd in range(ROUNDS):
        for c in CORE:
            sched.append(c)
            sched.append(distr[di % len(distr)]); di += 1
        sched.append(TRAPS[ti % len(TRAPS)]); ti += 1
        sched.append(distr[di % len(distr)]); di += 1
        sched.append(TRAPS[ti % len(TRAPS)]); ti += 1
        sched.append(CONFLICT)
        if (rnd + 1) % 5 == 0:
            # Bulk context push: two turns carrying 30 fresh lines each.
            # (150-line bulks measured >10 min at ~300-line sessions —
            # intake cost grows superlinearly; that wall is reported
            # separately from the turn-count behaviour limit.)
            base = (rnd // 5) * 60
            bulk_at[len(sched)] = corpus[base:base + 30]
            sched.append(CORE[1])
            bulk_at[len(sched)] = corpus[base + 30:base + 60]
            sched.append(CORE[1])
    print(f"schedule={len(sched)} turns", flush=True)

    sess = None
    rows = []
    rotations = 0
    t0 = time.time()
    for turn, q in enumerate(sched):
        qid, qtext = q[0], q[1]
        lines = targeted(qtext, corpus)
        if turn in bulk_at:
            lines = lines + bulk_at[turn]
        if sess is not None:
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
                rows.append({"turn": turn, "id": qid, "kind": q[4], "fog": fog,
                             "reason": (row.get("fog_decision") or {}).get("reason", ""),
                             "mode": mode + "/rotated", "answer": answer[:200],
                             "pass": sc["pass"], "fabricated": fabrication(answer, corpus_text),
                             "abstain": is_abstain(answer), "wire_mb": round(wire / 1e6, 2),
                             "session_lines": len(sess_corpus)})
                print(f"  t{turn:<3} {qid:<12} ROTATED wire={wire/1e6:.1f}MB", flush=True)
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
        frags = ((sess or {}).get("snapshot") or {}).get("fragments") or []
        if len(frags) >= internal.LINK_MIN_FRAGMENTS:
            linked = internal.link_session(sess, CONFIG)
            if linked:
                sess = linked
        row = (body.get("rows") or [{}])[0]
        fog = (row.get("fog_decision") or {}).get("decision", "")
        reason = (row.get("fog_decision") or {}).get("reason", "")
        sess_corpus = list((sess or {}).get("corpus") or [])
        answer, mode = hybrid.resolve_fate_answer(
            fog, str(row.get("answer", "")), sess_corpus,
            row.get("retrieval_corpus_idx"))
        sc = score(q, answer)
        rows.append({"turn": turn, "id": qid, "kind": q[4], "fog": fog,
                     "reason": reason, "mode": mode, "answer": answer[:200],
                     "pass": sc["pass"], "fabricated": fabrication(answer, corpus_text),
                     "abstain": is_abstain(answer), "wire_mb": wire_mb,
                     "session_lines": len(sess_corpus)})
        if turn % 25 == 0 or turn == len(sched) - 1:
            print(f"  t{turn:<3} sess_lines={len(sess_corpus)} wire={wire_mb}MB "
                  f"elapsed={round((time.time()-t0)/60, 1)}min", flush=True)
            # Stream partial results so a stopped run still yields data.
            (OUT / "extreme_limits_300turns.partial.json").write_text(
                json.dumps({"rows": rows}, ensure_ascii=False, indent=2),
                encoding="utf-8")

    def seg_stats(rs):
        core = [r for r in rs if r["id"] in ("n_start", "n_mid", "n_end")]
        traps = [r for r in rs if r["kind"] in ("swap_trap", "absent", "false_premise", "injection")]
        dis = [r for r in rs if r["id"].startswith("distr_")]
        return {
            "turns": len(rs),
            "core": f"{sum(1 for r in core if r['pass'])}/{len(core)}",
            "traps": f"{sum(1 for r in traps if r['pass'])}/{len(traps)}",
            "distractors": f"{sum(1 for r in dis if r['pass'])}/{len(dis)}",
            "fabricated": sum(len(r["fabricated"]) for r in rs),
            "gate_emits": sum(1 for r in rs if "fate_block_ready" in (r.get("reason") or "")),
        }

    n = len(rows)
    third = n // 3
    summary = {
        "turns": n,
        "all": seg_stats(rows),
        "third_1": seg_stats(rows[:third]),
        "third_2": seg_stats(rows[third:2 * third]),
        "third_3": seg_stats(rows[2 * third:]),
        "core_answers_by_id": {cid: sorted({r["answer"] for r in rows
                                             if r["id"] == cid and r["pass"]})
                               for cid in ("n_start", "n_mid", "n_end")},
        "core_fail_turns": [{"turn": r["turn"], "id": r["id"], "fog": r["fog"],
                             "answer": r["answer"][:60]}
                            for r in rows if r["id"] in ("n_start", "n_mid", "n_end")
                            and not r["pass"]],
        "trap_fail_turns": [{"turn": r["turn"], "id": r["id"], "fog": r["fog"],
                             "answer": r["answer"][:60]}
                            for r in rows if r["kind"] in ("swap_trap", "absent", "false_premise", "injection")
                            and not r["pass"]],
        "fabricated_rows": [{"turn": r["turn"], "id": r["id"], "fab": r["fabricated"]}
                            for r in rows if r["fabricated"]],
        "final_session_lines": rows[-1]["session_lines"] if rows else 0,
        "max_session_lines": max((r["session_lines"] for r in rows), default=0),
        "session_rotations": rotations,
        "max_wire_mb": max((r.get("wire_mb") or 0) for r in rows) if rows else 0,
        "minutes": round((time.time() - t0) / 60, 2),
    }
    dest = OUT / "extreme_limits_300turns.json"
    dest.write_text(json.dumps({"summary": summary, "rows": rows},
                               ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    print(f"wrote {dest}", flush=True)


if __name__ == "__main__":
    main()
