"""Two probes: (1) session wire-size growth vs ingested lines (HTTP
ceiling), (2) 60-turn long-session drift on the targeted pattern."""
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, r"C:\Ai\rogue-products\valhalla-server\tests")
sys.path.insert(0, r"C:\Ai\rogue-products\valhalla-server")
from run_extreme_stress import CONFIG, QUESTIONS, gen_corpus, post  # noqa: E402
from run_extreme_targeted import ENGINE, slim, targeted  # noqa: E402
from valhalla_server import hybrid  # noqa: E402

OUT = Path(r"C:\Ai\rogue-products\valhalla-server\tests\results_extreme")


def wire_mb(sess):
    return round(len(json.dumps(sess)) / 1e6, 2)


def ceiling():
    corpus, _ = gen_corpus(400)
    print("== ceiling probe (batch 25, full-context bootstrap path)")
    body = post(ENGINE, {"questions": [{"id": "w", "text": QUESTIONS[0][1], "score_type": "open"}],
                         "config": CONFIG, "context": corpus[:25]})
    sess = body.get("session")
    print(f"  lines={len(sess.get('corpus') or [])} wire={wire_mb(sess)}MB")
    for i in range(25, 400, 25):
        try:
            body = post(ENGINE, {"questions": [], "session_extend_only": True,
                                 "session": slim(sess), "append_lines": corpus[i:i + 25],
                                 "config": CONFIG}, timeout=600)
        except Exception as exc:  # noqa: BLE001
            print(f"  extend at {i} FAILED: {str(exc)[:120]}")
            return {"failed_at_lines": i, "wire_mb_last_ok": wire_mb(sess)}
        sess = body.get("session")
        print(f"  lines={len(sess.get('corpus') or [])} wire={wire_mb(sess)}MB", flush=True)
    return {"failed_at_lines": None, "wire_mb_last_ok": wire_mb(sess)}


def drift(turns=60):
    corpus, _ = gen_corpus(3000)
    print(f"== drift probe ({turns} turns, one session, hybrid)")
    sess = None
    checks = []
    fillers = [q for q in QUESTIONS if q[0] not in ("n_start", "para1")]
    for t in range(turns):
        probe_turn = (t % 10 == 9) or t == 0
        q = QUESTIONS[0] if probe_turn else fillers[t % len(fillers)]
        qid, qtext = ("n_start", QUESTIONS[0][1]) if probe_turn else (q[0], q[1])
        lines = targeted(qtext, corpus)
        if sess is None:
            payload = {"questions": [{"id": qid, "text": qtext, "score_type": "open"}],
                       "config": CONFIG, "context": lines}
        else:
            payload = {"questions": [{"id": qid, "text": qtext, "score_type": "open"}],
                       "config": CONFIG, "context": [], "session": slim(sess),
                       "append_lines": lines}
        t1 = time.time()
        body = post(ENGINE, payload)
        sess = body.get("session")
        row = (body.get("rows") or [{}])[0]
        fog = (row.get("fog_decision") or {}).get("decision", "")
        sess_corpus = list((sess or {}).get("corpus") or [])
        def _regrasp(variant, _c=list(sess_corpus)):
            b = post(ENGINE, {"questions": [{"id": "rg", "text": variant, "score_type": "open"}],
                              "config": CONFIG, "context": _c})
            return (b.get("rows") or [{}])[0]

        ans, mode = hybrid.resolve(fog, qtext, sess_corpus,
                                       row.get("retrieval_corpus_idx"),
                                       str(row.get("answer", "")),
                                       fog_decision=row.get("fog_decision"),
                                       fog_evidence=row.get("fog_evidence"),
                                       regrasp=_regrasp)
        if probe_turn:
            ok = "AUR-7" in ans and "AUR-999" not in ans
            checks.append({"turn": t + 1, "ok": ok, "answer": ans[:80], "mode": mode,
                           "sec": round(time.time() - t1, 2),
                           "wire_mb": wire_mb(sess), "sess_lines": len(sess_corpus)})
            print(f"  turn {t + 1:2d} needle ok={ok} mode={mode} wire={checks[-1]['wire_mb']}MB {checks[-1]['sec']}s", flush=True)
    return {"checks": checks,
            "stable": all(c["ok"] for c in checks)}


if __name__ == "__main__":
    out = {}
    out["ceiling"] = ceiling()
    out["drift"] = drift()
    dest = OUT / "extreme_ceiling_drift.json"
    dest.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"-> {dest}")
