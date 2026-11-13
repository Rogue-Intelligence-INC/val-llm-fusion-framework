"""Stress round with the completed hybrid: Valhalla session (super-window
corpus) + Fate verdict + multi-track evidence + Mistral-24B decode."""
import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))

from run_stress_points import STRESS_IDS, build_stress_corpus, infer_long  # noqa: E402
from run_unique_points import QUESTIONS, score  # noqa: E402

OUT = HERE / "results_unique_points"


def main():
    from valhalla_server import hybrid, internal
    from valhalla_server.inference import _materialize

    corpus = build_stress_corpus()
    print(f"stress corpus lines={len(corpus)} chars={sum(len(c) for c in corpus)}", flush=True)
    folder = str(_materialize("unified-core-v1"))
    if not internal.start(folder):
        raise RuntimeError("runtime failed")
    config = internal.chat_config(folder)
    questions = [q for q in QUESTIONS if q["id"] in STRESS_IDS]
    rows, session = [], None
    for i, q in enumerate(questions):
        payload = {"questions": [{"id": q["id"], "text": q["text"], "score_type": "open"}],
                   "config": config, "context": corpus if i == 0 else []}
        if session:
            payload["session"] = session
        t0 = time.time()
        body = infer_long(payload)
        session = body.get("session") or session
        r = (body.get("rows") or [{}])[0]
        native = str(r.get("answer", "")).strip()
        fog = r.get("fog_decision") if isinstance(r.get("fog_decision"), dict) else {}
        effective, mode = hybrid.resolve(fog.get("decision", ""), q["text"], corpus,
                                         r.get("retrieval_corpus_idx"), native)
        s = score(q, effective)
        rows.append({"arm": "V+M24-stress", **q, "answer": native,
                     "effective_answer": effective, "mode": mode,
                     "fog_decision": fog.get("decision", ""),
                     "latency_s": round(time.time() - t0, 2), **s})
        print(f"[V+M24-stress] {q['id']} mode={mode} correct={s['correct']} :: {effective[:110]}", flush=True)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    (OUT / f"stress_hybrid_{stamp}.json").write_text(
        json.dumps({"rows": rows}, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"TOTAL {sum(r['correct'] for r in rows)}/{len(rows)}", flush=True)


if __name__ == "__main__":
    main()
