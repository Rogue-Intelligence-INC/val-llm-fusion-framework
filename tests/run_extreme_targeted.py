"""Extreme stress v2 -- production pattern (targeted ingest).

The corpus NEVER enters the engine wholesale. It stays client-side;
per question the harness selects top-k lines (IDF lexical track, the
same principle as hub-f64 context_routing::targeted_context, whose
Python twin is absent from this checkout) and appends only those to a
persistent session (append_lines, deduped engine-side). Engine work is
therefore independent of client corpus size; we also record the session
wire size, the real HTTP ceiling metric.

Arms: native / hybrid share the protocol (fresh session per arm),
mistral gets the full corpus and is expected to die past its window.
Corpus generation, questions and scoring are imported from
run_extreme_stress so both suites are directly comparable.
"""
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, r"C:\Ai\rogue-products\valhalla-server\tests")
sys.path.insert(0, r"C:\Ai\rogue-products\valhalla-server")
from run_extreme_stress import (CONFIG, QUESTIONS, fabrication,  # noqa: E402
                                gen_corpus, is_abstain, post, score)
from valhalla_server import hybrid, internal  # noqa: E402

ENGINE = "http://127.0.0.1:8787/v1/infer"
HEAD = "http://127.0.0.1:18080/v1/chat/completions"
OUT = Path(r"C:\Ai\rogue-products\valhalla-server\tests\results_extreme")
OUT.mkdir(exist_ok=True)
K = 8


def slim(sess):
    if not isinstance(sess, dict):
        return sess
    out = dict(sess)
    out.pop("fate_rollback_stack", None)
    qh = out.get("question_history")
    if isinstance(qh, list) and len(qh) > 32:
        out["question_history"] = qh[-32:]
    return out


def targeted(question, corpus):
    lines = [corpus[i] for i, _s in hybrid.tile_track(question, corpus, k=K)]
    # Ingest bundling: near-duplicate template lines form one bundle;
    # only the strongest representative is appended to the session.
    return internal.bundle_lines(lines)


def run_arm(arm, corpus):
    sess = None
    rows = []
    for q in QUESTIONS:
        qid, qtext = q[0], q[1]
        t1 = time.time()
        lines = targeted(qtext, corpus)
        if sess is None:
            payload = {"questions": [{"id": qid, "text": qtext, "score_type": "open"}],
                       "config": CONFIG, "context": lines}
        else:
            payload = {"questions": [{"id": qid, "text": qtext, "score_type": "open"}],
                       "config": CONFIG, "context": [],
                       "session": slim(sess), "append_lines": lines}
        body = post(ENGINE, payload)
        sess = body.get("session")
        row = (body.get("rows") or [{}])[0]
        fog = (row.get("fog_decision") or {}).get("decision", "")
        native = str(row.get("answer", ""))
        if arm == "hybrid":
            # Production-faithful evidence: only lines actually ingested
            # into the session, never the whole client-side corpus.
            sess_corpus = list((sess or {}).get("corpus") or [])
            def _regrasp(variant, _c=list(sess_corpus)):
                b = post(ENGINE, {"questions": [{"id": "rg", "text": variant, "score_type": "open"}],
                                  "config": CONFIG, "context": _c})
                return (b.get("rows") or [{}])[0]

            answer, mode = hybrid.resolve(fog, qtext, sess_corpus,
                                          row.get("retrieval_corpus_idx"), native,
                                          fog_decision=row.get("fog_decision"),
                                          fog_evidence=row.get("fog_evidence"),
                                          regrasp=_regrasp)
            answer = f"[{mode}] {answer}"
        else:
            answer = native
        sc = score(q, answer)
        rows.append({"id": qid, "kind": q[4], "answer": answer[:300], "fog": fog,
                     "sec": round(time.time() - t1, 1),
                     "session_lines": len((sess or {}).get("corpus") or []),
                     "fabricated_tokens": fabrication(answer, "\n".join(corpus)), **sc})
        print(f"  [{arm}] {qid:12s} pass={sc['pass']} fog={str(fog):8s} sess_lines={rows[-1]['session_lines']} {rows[-1]['sec']}s", flush=True)
    return rows, sess


def mistral_full(corpus, qtext):
    prompt = "Answer using ONLY the documents.\n\nDOCUMENTS:\n" + "\n".join(corpus) + f"\n\nQuestion: {qtext}"
    try:
        body = post(HEAD, {"model": "mistral-24b", "messages": [
            {"role": "system", "content": "Answer from the documents only. If not in the documents, say you don't know."},
            {"role": "user", "content": prompt}], "temperature": 0.2, "max_tokens": 200}, timeout=600)
        return body["choices"][0]["message"]["content"].strip(), None
    except Exception as exc:  # noqa: BLE001
        return None, str(exc)[:160]


def main():
    sizes = [int(x) for x in sys.argv[1].split(",")] if len(sys.argv) > 1 else [800]
    arms = tuple(sys.argv[2].split(",")) if len(sys.argv) > 2 else ("native", "hybrid", "mistral")
    for n in sizes:
        corpus, _ = gen_corpus(n)
        corpus_text = "\n".join(corpus)
        print(f"\n### targeted lines{n}: {len(corpus)} client lines, ~{len(corpus_text)//4} est tokens", flush=True)
        res = {"size": n, "lines": len(corpus), "est_tokens": len(corpus_text) // 4,
               "mode": "targeted_append_only", "k": K, "arms": {}}
        final_sess = None
        for arm in arms:
            if arm == "mistral":
                rows = []
                for q in QUESTIONS:
                    t1 = time.time()
                    ans, err = mistral_full(corpus, q[1])
                    sc = score(q, ans if err is None else None)
                    rows.append({"id": q[0], "kind": q[4], "answer": (ans or "")[:300],
                                 "err": err, "sec": round(time.time() - t1, 1),
                                 "fabricated_tokens": fabrication(ans or "", corpus_text), **sc})
                    print(f"  [mistral] {q[0]:12s} pass={sc['pass']} err={err} {rows[-1]['sec']}s", flush=True)
                res["arms"][arm] = rows
            else:
                rows, final_sess = run_arm(arm, corpus)
                res["arms"][arm] = rows
        if final_sess:
            res["session_wire_mb"] = round(len(json.dumps(final_sess)) / 1e6, 2)
            res["session_lines"] = len(final_sess.get("corpus") or [])
        dest = OUT / f"extreme_targeted_lines{n}.json"
        dest.write_text(json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")
        for arm, rows in res["arms"].items():
            scored = [r for r in rows if r["pass"] is not None]
            print(f"  == {arm}: {sum(1 for r in scored if r['pass'])}/{len(scored)} pass, "
                  f"fabricated={sum(len(r['fabricated_tokens']) for r in rows)}", flush=True)
        print(f"  session wire: {res.get('session_wire_mb')} MB, lines={res.get('session_lines')}", flush=True)


if __name__ == "__main__":
    main()
