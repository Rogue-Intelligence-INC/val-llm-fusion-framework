"""Chunking follow-up: correct span fields (line_start/line_end),
needle-block granularity and engine-exact retrieval before/after
fate_link bundling, same protocol as run_resource_chunk_test part B."""
import json
import sys
import time
import urllib.request
from pathlib import Path

sys.path.insert(0, r"C:\Ai\rogue-products\valhalla-server\tests")
sys.path.insert(0, r"C:\Ai\rogue-products\valhalla-server")
from run_extreme_stress import CONFIG, gen_corpus  # noqa: E402
from run_dynamic_unload import spawn  # noqa: E402
from valhalla_server import internal  # noqa: E402

OUT = Path(r"C:\Ai\rogue-products\valhalla-server\tests\results_extreme\chunk_spans.json")
URL = "http://127.0.0.1:8794/v1/infer"
QUESTIONS = {"n_start": ("What is Project Aurora's internal codename?", "AUR-7", 28),
             "n_mid": ("Who leads the Borealis satellite program?", "Halvorsen", 282),
             "n_end": ("Where is Project Aurora's flight test site?", "Reno", 537)}


def post(payload, timeout=600):
    req = urllib.request.Request(URL, data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode())


def spans_of(sess):
    fr = (sess.get("snapshot") or {}).get("fragments") or []
    spans = [(f.get("line_start"), f.get("line_end")) for f in fr]
    spans = [(a, b) for a, b in spans if isinstance(a, int) and isinstance(b, int)]
    return fr, spans


def stats(spans):
    sizes = [b - a + 1 for a, b in spans]
    if not sizes:
        return None
    return {"n": len(sizes), "min": min(sizes), "max": max(sizes),
            "mean": round(sum(sizes) / len(sizes), 1),
            "total_covered": sum(sizes)}


def needle_block(spans, gold):
    hits = [b - a + 1 for a, b in spans if a <= gold <= b]
    return hits


def qa(sess, corpus):
    out = {}
    for nid, (qtext, kw, gold) in QUESTIONS.items():
        body = post({"questions": [{"id": nid, "text": qtext, "score_type": "open"}],
                     "config": CONFIG, "context": [],
                     "session": internal.slim_session(sess)})
        sess = body.get("session")
        row = (body.get("rows") or [{}])[0]
        idx = row.get("retrieval_corpus_idx")
        line = corpus[idx] if isinstance(idx, int) and 0 <= idx < len(corpus) else ""
        out[nid] = {"idx": idx, "gold": gold, "exact": idx == gold,
                    "line_has_kw": kw in line,
                    "fog": (row.get("fog_decision") or {}).get("decision", "")}
        print(f"[spans] QA {nid}: idx={idx} gold={gold} exact={idx == gold} "
              f"kw_line={kw in line} fog={out[nid]['fog']}", flush=True)
    return sess, out


def main():
    proc = spawn(8794, {"VALHALLA_QA_PASSIVE_FATE": "1",
                        "VALHALLA_FATE_PULSE_SCOPE": "segment",
                        "VALHALLA_DYNAMIC_CORPUS_FRAGMENTS": "1"})
    try:
        corpus, _ = gen_corpus(560)
        t0 = time.time()
        body = post({"questions": [{"id": "w", "text": QUESTIONS["n_start"][0],
                                    "score_type": "open"}],
                     "config": CONFIG, "context": corpus[:30]})
        sess = body.get("session")
        for i in range(30, len(corpus), 30):
            body = post({"questions": [], "session_extend_only": True,
                         "session": internal.slim_session(sess),
                         "append_lines": corpus[i:i + 30], "config": CONFIG})
            sess = body.get("session")
        print(f"[spans] ingest {round(time.time()-t0,1)}s", flush=True)
        fr, spans = spans_of(sess)
        st = stats(spans)
        print(f"[spans] fragments={len(fr)} span_stats={st}", flush=True)
        blocks = {nid: needle_block(spans, g) for nid, (_q, _k, g) in QUESTIONS.items()}
        print(f"[spans] needle block sizes pre-link: {blocks}", flush=True)
        sess, qa_pre = qa(sess, corpus)

        body = post({"questions": [], "fate_link_only": True,
                     "session": internal.slim_session(sess), "config": CONFIG})
        linked = body.get("session")
        fr2, spans2 = spans_of(linked)
        st2 = stats(spans2)
        print(f"[spans] after link: fragments={len(fr2)} span_stats={st2} "
              f"features={body.get('features')}", flush=True)
        blocks2 = {nid: needle_block(spans2, g) for nid, (_q, _k, g) in QUESTIONS.items()}
        print(f"[spans] needle block sizes post-link: {blocks2}", flush=True)
        _s, qa_post = qa(linked, corpus)

        doc = {"pre": {"fragments": len(fr), "span_stats": st, "needle_blocks": blocks,
                       "qa": qa_pre},
               "post": {"fragments": len(fr2), "span_stats": st2, "needle_blocks": blocks2,
                        "qa": qa_post},
               "fixed_baseline": {"fixed_20": {"chunks": 28, "needle_block": 20},
                                  "fixed_50": {"chunks": 12, "needle_block": 50}}}
        OUT.write_text(json.dumps(doc, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"wrote {OUT}", flush=True)
    finally:
        proc.terminate()


if __name__ == "__main__":
    main()
