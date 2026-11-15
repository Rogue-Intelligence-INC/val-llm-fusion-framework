"""Which session key doubles every two turns?"""
import json
import sys

sys.path.insert(0, r"C:\Ai\rogue-products\valhalla-server\tests")
sys.path.insert(0, r"C:\Ai\rogue-products\valhalla-server")
from run_extreme_stress import CONFIG, gen_corpus, post  # noqa: E402
from run_extreme_targeted import ENGINE, targeted  # noqa: E402
from run_dialogue_drift import build_schedule, distractors  # noqa: E402
from valhalla_server import internal  # noqa: E402

corpus, _ = gen_corpus(10006)
sched = build_schedule(distractors(corpus))[:16]
sess = None
for turn, q in enumerate(sched):
    lines = targeted(q[1], corpus)
    if sess is None:
        payload = {"questions": [{"id": q[0], "text": q[1], "score_type": "open"}],
                   "config": CONFIG, "context": lines}
    else:
        payload = {"questions": [{"id": q[0], "text": q[1], "score_type": "open"}],
                   "config": CONFIG, "context": [],
                   "session": internal.slim_session(sess), "append_lines": lines}
    body = post(ENGINE, payload)
    sess = body.get("session")
    if turn in (8, 11, 13, 15):
        print(f"--- after t{turn} ---")
        for k, v in sess.items():
            print(f"  session.{k}: {len(json.dumps(v))/1e6:.2f}MB")
        snap = sess.get("snapshot") or {}
        for k, v in snap.items():
            s = len(json.dumps(v)) / 1e6
            if s > 0.05:
                print(f"    snapshot.{k}: {s:.2f}MB (n={len(v) if isinstance(v, list) else '-'})")
