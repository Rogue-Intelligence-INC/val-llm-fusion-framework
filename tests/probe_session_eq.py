"""Does Fate's live equation move after a big-corpus incubated ingest?

Warmup-ingest the corpus into an engine session (no QA), then ask on
the returned session. Read eq coefs/threshold/adaptations per row.
"""
import json, sys, time
from pathlib import Path

sys.path.insert(0, r"C:\Ai\rogue-products\valhalla-server\tests")
sys.path.insert(0, r"C:\Ai\rogue-products\valhalla-server")
from run_extreme_stress import CONFIG, QUESTIONS, gen_corpus, post  # noqa: E402
from run_extreme_targeted import ENGINE  # noqa: E402

N = int(sys.argv[1]) if len(sys.argv) > 1 else 2000
corpus, _ = gen_corpus(N)
print(f"corpus={len(corpus)}", flush=True)

ASK = ["n_start", "n_mid", "n_end", "conflict", "absent_ent", "inject_q"]
qs = {qid: text for qid, text, *_ in QUESTIONS}

t0 = time.time()
b = post(ENGINE, {"questions": [], "config": CONFIG, "context": corpus,
                  "fate_warmup_only": True})
sess = b.get("session")
print(f"warmup ingest done {time.time()-t0:.1f}s session={'yes' if sess else 'NO'}", flush=True)

out = []
for qid in ASK:
    t0 = time.time()
    b = post(ENGINE, {"questions": [{"id": qid, "text": qs[qid], "score_type": "open"}],
                      "config": CONFIG, "session": sess})
    sess = b.get("session") or sess
    r = (b.get("rows") or [{}])[0]
    g = r.get("fate_grasp") or {}
    row = {"id": qid, "fog": (r.get("fog_decision") or {}).get("decision"),
           "idx": r.get("retrieval_corpus_idx"), "rel": g.get("fate_block_rel"),
           "block": g.get("grasp_block"), "eq": g.get("eq_coefs"),
           "eq_thr": g.get("eq_threshold"), "eq_adapt": g.get("eq_adaptations"),
           "alg_rel": g.get("alg_delta_rel"), "secs": round(time.time() - t0, 1)}
    out.append(row)
    print(row, flush=True)

Path(r"C:\Ai\rogue-products\valhalla-server\tests\results_extreme\session_eq_probe.json").write_text(
    json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
print("wrote session_eq_probe.json")
