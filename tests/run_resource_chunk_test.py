"""Resource-relief + chunking test (2026-11-15, consolidation build).

Part A: dynamic-unload A/B at 48 turns over the 10,006-line corpus —
per-turn wire MB, engine RSS (psutil), seconds, fog/pass/fabrication.
Part B: chunking quality on a private engine in product config
(passive Fate, segment pulse, link group 8): dynamic fragments vs
fate_link bundling vs fixed 20/50-line hard chunks — block counts,
needle-block granularity, query-time fragment selection ratio.
"""
import json
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

import psutil

sys.path.insert(0, r"C:\Ai\rogue-products\valhalla-server\tests")
sys.path.insert(0, r"C:\Ai\rogue-products\valhalla-server")
from run_extreme_stress import CONFIG, fabrication, gen_corpus, score  # noqa: E402
from run_extreme_targeted import targeted  # noqa: E402
from run_dialogue_drift import build_schedule, distractors  # noqa: E402
from run_dynamic_unload import spawn  # noqa: E402
from valhalla_server import hybrid, internal  # noqa: E402

OUT = Path(r"C:\Ai\rogue-products\valhalla-server\tests\results_extreme\resource_chunk_test.json")
TURNS_A = 48


def post(url, payload, timeout=600):
    req = urllib.request.Request(url, data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode())


def run_arm(name, port, extra_env, sched, corpus, corpus_text):
    proc = spawn(port, extra_env)
    p = psutil.Process(proc.pid)
    url = f"http://127.0.0.1:{port}/v1/infer"
    rows, sess = [], None
    rss_peak = 0
    try:
        for turn, q in enumerate(sched[:TURNS_A]):
            lines = targeted(q[1], corpus)
            if sess is None:
                payload = {"questions": [{"id": q[0], "text": q[1], "score_type": "open"}],
                           "config": CONFIG, "context": lines}
            else:
                payload = {"questions": [{"id": q[0], "text": q[1], "score_type": "open"}],
                           "config": CONFIG, "context": [],
                           "session": internal.slim_session(sess), "append_lines": lines}
            wire = len(json.dumps(payload)) / 1e6
            t0 = time.time()
            body = post(url, payload)
            sec = time.time() - t0
            sess = body.get("session")
            row = (body.get("rows") or [{}])[0]
            fog = (row.get("fog_decision") or {}).get("decision", "")
            sess_corpus = list((sess or {}).get("corpus") or [])
            answer, _mode = hybrid.resolve_fate_answer(
                fog, str(row.get("answer", "")), sess_corpus, row.get("retrieval_corpus_idx"))
            sc = score(q, answer)
            try:
                rss = p.memory_info().rss / 1e6
            except Exception:
                rss = 0
            rss_peak = max(rss_peak, rss)
            rows.append({"turn": turn, "id": q[0], "kind": q[4], "fog": fog,
                         "pass": sc["pass"], "wire_mb": round(wire, 3),
                         "sec": round(sec, 2), "rss_mb": round(rss, 1),
                         "fabricated": fabrication(answer, corpus_text),
                         "frags": len(((sess or {}).get("snapshot") or {}).get("fragments") or [])})
            print(f"[{name}] t{turn:<3} {q[0]:<12} wire={wire:.2f}MB rss={rss:.0f}MB "
                  f"pass={sc['pass']} fog={fog} {sec:.1f}s frags={rows[-1]['frags']}", flush=True)
    finally:
        proc.terminate()
    core = [r for r in rows if r["id"] in ("n_start", "n_mid", "n_end")]
    return {"arm": name, "turns": len(rows),
            "core_pass": f"{sum(1 for r in core if r['pass'])}/{len(core)}",
            "max_wire_mb": max((r["wire_mb"] for r in rows), default=0),
            "rss_peak_mb": round(rss_peak, 1),
            "mean_sec": round(sum(r["sec"] for r in rows) / max(1, len(rows)), 2),
            "fabricated_total": sum(len(r["fabricated"]) for r in rows),
            "last_frags": rows[-1]["frags"] if rows else 0, "rows": rows}


def frag_spans(sess):
    """Best-effort line-span per fragment from whatever range fields exist."""
    fr = (sess.get("snapshot") or {}).get("fragments") or []
    spans = []
    for f in fr:
        span = None
        for k in ("line_range", "corpus_span", "span"):
            v = f.get(k)
            if isinstance(v, (list, tuple)) and len(v) == 2:
                span = (int(v[0]), int(v[1])); break
        if span is None and isinstance(f.get("start"), int) and isinstance(f.get("end"), int):
            span = (f["start"], f["end"])
        if span is None and isinstance(f.get("lines"), list) and f["lines"]:
            span = (min(f["lines"]), max(f["lines"]))
        spans.append(span)
    return fr, spans


def part_b():
    port = 8793
    proc = spawn(port, {"VALHALLA_QA_PASSIVE_FATE": "1",
                        "VALHALLA_FATE_PULSE_SCOPE": "segment",
                        "VALHALLA_DYNAMIC_CORPUS_FRAGMENTS": "1"})
    url = f"http://127.0.0.1:{port}/v1/infer"
    try:
        corpus, _ = gen_corpus(560)
        corpus_text = "\n".join(corpus)
        needles = {"n_start": ("What is Project Aurora's internal codename?", "AUR-7"),
                   "n_mid": ("Who leads the Borealis satellite program?", "Halvorsen"),
                   "n_end": ("Where is Project Aurora's flight test site?", "Reno")}
        gold_idx = {}
        for nid, (_q, kw) in needles.items():
            gold_idx[nid] = next(i for i, l in enumerate(corpus) if kw in l)
        print(f"[chunk] gold lines: {gold_idx}", flush=True)

        t0 = time.time()
        body = post(url, {"questions": [{"id": "w", "text": needles["n_start"][0],
                                         "score_type": "open"}],
                          "config": CONFIG, "context": corpus[:30]})
        sess = body.get("session")
        for i in range(30, len(corpus), 30):
            body = post(url, {"questions": [], "session_extend_only": True,
                              "session": internal.slim_session(sess),
                              "append_lines": corpus[i:i + 30], "config": CONFIG})
            sess = body.get("session")
        ingest_sec = round(time.time() - t0, 1)
        fr, spans = frag_spans(sess)
        kinds = {}
        for f in fr:
            kinds[f["id"].split("_")[0]] = kinds.get(f["id"].split("_")[0], 0) + 1
        full_mb = round(len(json.dumps(sess)) / 1e6, 2)
        slim_mb = round(len(json.dumps(internal.slim_session(sess))) / 1e6, 2)
        print(f"[chunk] ingest 560 lines in {ingest_sec}s; fragments={len(fr)} {kinds} "
              f"full={full_mb}MB slim={slim_mb}MB", flush=True)
        if fr:
            print(f"[chunk] fragment[0] keys: {sorted(fr[0].keys())}", flush=True)
            print(f"[chunk] fragment[0]: {json.dumps(fr[0])[:500]}", flush=True)
        sized = [s for s in spans if s]
        span_sizes = [b - a + 1 for a, b in sized]
        print(f"[chunk] spans known for {len(sized)}/{len(fr)} fragments; "
              f"span lines min/mean/max = "
              f"{(min(span_sizes), round(sum(span_sizes)/max(1,len(span_sizes)),1), max(span_sizes)) if span_sizes else None}",
              flush=True)

        def needle_block(nid):
            gi = gold_idx[nid]
            for (a, b) in sized:
                if a <= gi <= b:
                    return b - a + 1
            return None

        qa = {}
        for nid, (qtext, kw) in needles.items():
            body = post(url, {"questions": [{"id": nid, "text": qtext, "score_type": "open"}],
                              "config": CONFIG, "context": [],
                              "session": internal.slim_session(sess)})
            sess = body.get("session")
            row = (body.get("rows") or [{}])[0]
            fog = (row.get("fog_decision") or {}).get("decision", "")
            sess_corpus = list((sess or {}).get("corpus") or [])
            answer, _m = hybrid.resolve_fate_answer(
                fog, str(row.get("answer", "")), sess_corpus, row.get("retrieval_corpus_idx"))
            grasp = row.get("fate_grasp") or {}
            sel = grasp.get("selected_fragments")
            qa[nid] = {"kw_in_answer": kw in answer, "fog": fog,
                       "retrieval_idx": row.get("retrieval_corpus_idx"),
                       "gold_idx": gold_idx[nid],
                       "selected_fragments": sel,
                       "total_fragments": len(fr),
                       "needle_block_lines": needle_block(nid)}
            print(f"[chunk] QA {nid}: kw={kw in answer} fog={fog} "
                  f"idx={row.get('retrieval_corpus_idx')} gold={gold_idx[nid]} "
                  f"selected={sel}/{len(fr)} block={needle_block(nid)}", flush=True)

        linked_body = post(url, {"questions": [], "fate_link_only": True,
                                 "session": internal.slim_session(sess), "config": CONFIG})
        linked = linked_body.get("session")
        fr2, _sp2 = frag_spans(linked)
        print(f"[chunk] fate_link: fragments {len(fr)} -> {len(fr2)} "
              f"features={linked_body.get('features')}", flush=True)

        fixed = {}
        for size in (20, 50):
            nb = {}
            for nid in needles:
                gi = gold_idx[nid]
                nb[nid] = size  # every needle sits in a full fixed block
            fixed[f"fixed_{size}"] = {"chunks": -(-len(corpus) // size),
                                      "needle_block_lines": nb}
        return {"ingest_sec": ingest_sec, "fragments": len(fr), "kinds": kinds,
                "full_mb": full_mb, "slim_mb": slim_mb,
                "span_sizes": span_sizes, "qa": qa,
                "after_link_fragments": len(fr2),
                "link_features": linked_body.get("features"),
                "fixed_baseline": fixed,
                "fabricated_check": fabrication(corpus_text[:0], corpus_text)}
    finally:
        proc.terminate()


def main():
    corpus, _ = gen_corpus(10006)
    corpus_text = "\n".join(corpus)
    sched = build_schedule(distractors(corpus))
    arms = [run_arm("relief_on", 8791, {}, sched, corpus, corpus_text),
            run_arm("relief_off", 8792,
                    {"VALHALLA_DYNAMIC_CORPUS_FRAGMENTS": "0",
                     "VALHALLA_FRAGMENT_ONLY": "0",
                     "VALHALLA_QUERY_FRAGMENT_MERGE": "0"},
                    sched, corpus, corpus_text)]
    for a in arms:
        print(json.dumps({k: a[k] for k in ("arm", "turns", "core_pass", "max_wire_mb",
                                            "rss_peak_mb", "mean_sec", "fabricated_total",
                                            "last_frags")}, ensure_ascii=False), flush=True)
    chunk = part_b()
    doc = {"arms": [{k: v for k, v in a.items() if k != "rows"} | {"rows": a["rows"]} for a in arms],
           "chunking": chunk}
    OUT.write_text(json.dumps(doc, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"wrote {OUT}", flush=True)


if __name__ == "__main__":
    main()
