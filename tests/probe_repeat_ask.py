"""Repeat-ask feedback-inflation probe.

One persistent engine session; the same core question asked 10 times
(targeted lines re-appended each turn, the dialogue pattern). Record
fog margin + block reading per repetition. Feedback inflation shows
as a rising margin trajectory for an unchanged question/corpus.
"""
import json, os, sys
from pathlib import Path

sys.path.insert(0, r"C:\Ai\rogue-products\valhalla-server\tests")
sys.path.insert(0, r"C:\Ai\rogue-products\valhalla-server")
from run_extreme_stress import CONFIG, QUESTIONS, gen_corpus, post  # noqa: E402
from run_extreme_targeted import ENGINE, targeted  # noqa: E402
from valhalla_server import internal  # noqa: E402

TAG = os.environ.get("REPEAT_TAG", "run")
Q = {q[0]: q for q in QUESTIONS}
corpus, _ = gen_corpus(806)
qid, qtext = "n_mid", Q["n_mid"][1]
lines = targeted(qtext, corpus)

sess = None
out = []
for t in range(10):
    if sess is None:
        payload = {"questions": [{"id": qid, "text": qtext, "score_type": "open"}],
                   "config": CONFIG, "context": lines}
    else:
        payload = {"questions": [{"id": qid, "text": qtext, "score_type": "open"}],
                   "config": CONFIG, "context": [],
                   "session": internal.slim_session(sess), "append_lines": lines}
    body = post(ENGINE, payload)
    sess = body.get("session")
    frags = ((sess or {}).get("snapshot") or {}).get("fragments") or []
    if len(frags) >= internal.LINK_MIN_FRAGMENTS:
        linked = internal.link_session(sess, CONFIG)
        if linked:
            sess = linked
    r = (body.get("rows") or [{}])[0]
    e = r.get("fog_evidence") or {}
    g = r.get("fate_grasp") or {}
    rec = {"turn": t, "fog": (r.get("fog_decision") or {}).get("decision"),
           "margin": e.get("margin"), "top": e.get("top_score"),
           "rel": g.get("fate_block_rel"),
           "answer": str(r.get("answer", ""))[:60]}
    out.append(rec)
    print(TAG, rec, flush=True)

Path(rf"C:\Ai\rogue-products\valhalla-server\tests\results_extreme\repeat_ask_{TAG}.json").write_text(
    json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
print("wrote repeat_ask_" + TAG)
