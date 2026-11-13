"""Step-1 baseline: fate_link_only bundling on a real session.
Records fragments in/out, links_merged, wire MB and needle QA before
vs after linking. No production code touched."""
import json
import sys
from pathlib import Path

sys.path.insert(0, r"C:\Ai\rogue-products\valhalla-server\tests")
sys.path.insert(0, r"C:\Ai\rogue-products\valhalla-server")
from run_extreme_stress import CONFIG, gen_corpus, post  # noqa: E402
from valhalla_server.internal import slim_session  # noqa: E402

ENGINE = "http://127.0.0.1:8787/v1/infer"
OUT = Path(r"C:\Ai\rogue-products\valhalla-server\tests\results_extreme")


def mb(x):
    return round(len(json.dumps(x)) / 1e6, 2)


def frag_info(sess):
    fr = (sess.get("snapshot") or {}).get("fragments") or []
    kinds = {}
    for f in fr:
        kinds[f["id"].split("_")[0]] = kinds.get(f["id"].split("_")[0], 0) + 1
    return len(fr), kinds


def ask(sess, qtext, qid="q"):
    body = post(ENGINE, {"questions": [{"id": qid, "text": qtext, "score_type": "open"}],
                         "config": CONFIG, "context": [], "session": slim_session(sess)})
    row = (body.get("rows") or [{}])[0]
    return body.get("session"), str(row.get("answer", "")), row.get("retrieval_corpus_idx")


def main():
    corpus, _ = gen_corpus(120)
    body = post(ENGINE, {"questions": [{"id": "w", "text": "What is Project Aurora's internal codename?", "score_type": "open"}],
                         "config": CONFIG, "context": corpus[:30]})
    sess = body.get("session")
    for i in range(30, 90, 30):
        body = post(ENGINE, {"questions": [], "session_extend_only": True,
                             "session": slim_session(sess),
                             "append_lines": corpus[i:i + 30], "config": CONFIG})
        sess = body.get("session")
    n0, kinds0 = frag_info(sess)
    print(f"BEFORE link: lines={len(sess.get('corpus') or [])} fragments={n0} {kinds0} "
          f"full={mb(sess)}MB slim={mb(slim_session(sess))}MB", flush=True)
    sess_q, a1, _ = ask(sess, "What is Project Aurora's internal codename?", "n_start")
    sess_q, a2, idx2 = ask(sess_q, "Who leads the Borealis satellite program?", "n_mid")
    line2 = (sess_q.get("corpus") or [])[idx2] if isinstance(idx2, int) else ""
    print(f"BEFORE QA: n_start AUR-7={'AUR-7' in a1} n_mid Halvorsen={'Halvorsen' in a2 or 'Halvorsen' in line2}", flush=True)

    body = post(ENGINE, {"questions": [], "fate_link_only": True,
                         "session": slim_session(sess), "config": CONFIG})
    linked = body.get("session")
    print(f"link features: {body.get('features')}")
    n1, kinds1 = frag_info(linked)
    print(f"AFTER link: fragments={n1} {kinds1} full={mb(linked)}MB slim={mb(slim_session(linked))}MB", flush=True)
    sess_q, b1, _ = ask(linked, "What is Project Aurora's internal codename?", "n_start")
    sess_q, b2, idxb = ask(sess_q, "Who leads the Borealis satellite program?", "n_mid")
    lineb = (sess_q.get("corpus") or [])[idxb] if isinstance(idxb, int) else ""
    print(f"AFTER QA: n_start AUR-7={'AUR-7' in b1} n_mid Halvorsen={'Halvorsen' in b2 or 'Halvorsen' in lineb}", flush=True)

    doc = {"before": {"fragments": n0, "kinds": kinds0, "full_mb": mb(sess), "slim_mb": mb(slim_session(sess))},
           "after": {"fragments": n1, "kinds": kinds1, "full_mb": mb(linked), "slim_mb": mb(slim_session(linked))},
           "features": body.get("features"),
           "qa_before": {"n_start": "AUR-7" in a1, "n_mid": "Halvorsen" in a2 or "Halvorsen" in line2},
           "qa_after": {"n_start": "AUR-7" in b1, "n_mid": "Halvorsen" in b2 or "Halvorsen" in lineb}}
    dest = OUT / "fate_link_baseline.json"
    dest.write_text(json.dumps(doc, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"-> {dest}")


if __name__ == "__main__":
    main()
