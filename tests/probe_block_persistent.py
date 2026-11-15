"""Block activation in a persistent session (blocks accumulate).

Per question: engine's selected fragment ids (decode_path), their line
ranges in the session corpus, whether the needle line lies inside an
activated block, retrieval idx, and FOG. Over-silence = needle inside
an activated block (or idx correct) while FOG abstains.
"""
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, r"C:\Ai\rogue-products\valhalla-server\tests")
sys.path.insert(0, r"C:\Ai\rogue-products\valhalla-server")
from run_extreme_stress import CONFIG, QUESTIONS, gen_corpus, post  # noqa: E402
from run_extreme_targeted import ENGINE, targeted  # noqa: E402
from valhalla_server import internal  # noqa: E402

OUT = Path(r"C:\Ai\rogue-products\valhalla-server\tests\results_extreme\block_activation_persistent.json")
FRAG_RE = re.compile(r"query_frag/k(\d+)/ids=([a-z0-9_+]+)")
NEEDLE = {
    "n_start": "Project Aurora's internal codename is AUR-7.",
    "para1": "Project Aurora's internal codename is AUR-7.",
    "para2": "Project Aurora's internal codename is AUR-7.",
    "inject_q": "Project Aurora's internal codename is AUR-7.",
    "n_mid": "Project Borealis's satellite lead is Dr. Ines Halvorsen.",
    "n_end": "Project Aurora's flight test site is located near Reno, Nevada.",
}

corpus, _ = gen_corpus(806)
sess = None
out = []
for qid, qtext, *_ in QUESTIONS:
    lines = targeted(qtext, corpus)
    if sess is None:
        payload = {"questions": [{"id": qid, "text": qtext, "score_type": "open"}],
                   "config": CONFIG, "context": lines}
    else:
        payload = {"questions": [{"id": qid, "text": qtext, "score_type": "open"}],
                   "config": CONFIG, "context": [],
                   "session": internal.slim_session(sess), "append_lines": lines}
    body = post(ENGINE, payload)
    row = (body.get("rows") or [{}])[0]
    # Selected ids refer to fragments of the session AS SENT (pre-question);
    # match ranges there, not in the post-QA export.
    pre_frags = []
    if sess is not None:
        pre_frags = ((sess.get("snapshot") or {}).get("fragments") or [])
    pre_corpus = list((sess or {}).get("corpus") or []) if sess is not None else lines
    sess = body.get("session")
    snap = (sess or {}).get("snapshot") or {}
    frags = snap.get("fragments") or []
    sess_corpus = list((sess or {}).get("corpus") or [])
    m = FRAG_RE.search(str(row.get("decode_path", "")))
    ids = m.group(2).split("+") if m else []
    ranges = [(f.get("line_start"), f.get("line_end")) for f in pre_frags if f.get("id") in ids]
    needle = NEEDLE.get(qid)
    nidx = pre_corpus.index(needle) if needle in pre_corpus else (
        sess_corpus.index(needle) if needle in sess_corpus else None)
    idx = row.get("retrieval_corpus_idx")
    fog = (row.get("fog_decision") or {}).get("decision")
    ev = row.get("fog_evidence") or {}
    in_block = nidx is not None and any(a is not None and a <= nidx <= b for a, b in ranges)
    rec = {"id": qid, "fog": fog, "idx": idx, "needle_idx": nidx,
           "needle_in_block": in_block, "selected_ids": ids,
           "ranges": ranges, "n_fragments": len(frags),
           "margin": ev.get("margin")}
    out.append(rec)
    print(f"{qid:<12} fog={fog:<8} needle@{nidx} in_block={in_block} idx={idx} "
          f"margin={rec['margin']} frags={len(frags)} sel={ids}", flush=True)

OUT.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
print(f"\nwrote {OUT}")
