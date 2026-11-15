"""Does fate_link_only actually shrink the session wire mid-dialogue?
Run turns, link whenever fragments >= 16, print wire before/after."""
import json
import sys
import urllib.error

sys.path.insert(0, r"C:\Ai\rogue-products\valhalla-server\tests")
sys.path.insert(0, r"C:\Ai\rogue-products\valhalla-server")
from run_extreme_stress import CONFIG, gen_corpus, post  # noqa: E402
from run_extreme_targeted import ENGINE, targeted  # noqa: E402
from run_dialogue_drift import build_schedule, distractors  # noqa: E402
from valhalla_server import internal  # noqa: E402

corpus, _ = gen_corpus(10006)
sched = build_schedule(distractors(corpus))[:40]
sess = None
for turn, q in enumerate(sched):
    lines = targeted(q[1], corpus)
    if sess is None:
        payload = {"questions": [{"id": q[0], "text": q[1], "score_type": "open"}],
                   "config": CONFIG, "context": lines}
    else:
        frags = len(((sess.get("snapshot") or {}).get("fragments") or []))
        if frags >= internal.LINK_MIN_FRAGMENTS:
            before = len(json.dumps(internal.slim_session(sess))) / 1e6
            linked = internal.link_session(sess, CONFIG)
            if linked:
                sess = linked
                after_frags = len(((sess.get("snapshot") or {}).get("fragments") or []))
                after = len(json.dumps(internal.slim_session(sess))) / 1e6
                print(f"  [link @t{turn}] frags {frags}->{after_frags} "
                      f"wire {before:.2f}->{after:.2f}MB", flush=True)
            else:
                print(f"  [link @t{turn}] returned None (frags={frags})", flush=True)
        payload = {"questions": [{"id": q[0], "text": q[1], "score_type": "open"}],
                   "config": CONFIG, "context": [],
                   "session": internal.slim_session(sess), "append_lines": lines}
    wire = len(json.dumps(payload)) / 1e6
    try:
        body = post(ENGINE, payload)
    except urllib.error.HTTPError as e:
        print(f"t{turn} {q[0]}: HTTP {e.code} at wire={wire:.2f}MB", flush=True)
        break
    sess = body.get("session")
    print(f"t{turn:<3} {q[0]:<12} wire={wire:.2f}MB", flush=True)
else:
    print("completed all turns without 413", flush=True)
