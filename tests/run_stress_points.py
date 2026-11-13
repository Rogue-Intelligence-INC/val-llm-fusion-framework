"""Round 2 stress: corpus far beyond the local LM context window.

Key facts sit at the very beginning (~45k tokens of long filler lines follow).
Arm M-full must swallow the whole corpus as prompt text (16k window here);
arm V ingests the same lines once into a persistent Valhalla session.
Also classifies Valhalla trap behaviour: invented content vs grounded
(but possibly irrelevant) corpus retrieval.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))

from run_unique_points import FACTS_A, FACTS_B, UPDATES, QUESTIONS, score  # noqa: E402

OUT = HERE / "results_unique_points"
OUT.mkdir(exist_ok=True)

WORDS = ("harbor office routine inspection report filed forms signed copies archived "
         "shelf box folder cabinet drawer stamp seal envelope ledger binder folder "
         "meeting minutes agenda attendance signature approval copy district region")


def build_stress_corpus(n_filler: int = 180) -> list[str]:
    corpus = FACTS_A + FACTS_B + UPDATES  # needles first: lines 0..17
    for i in range(n_filler):
        w = " ".join([WORDS.split()[((i * 7 + j) % len(WORDS.split()))]
                      for j in range(90)])
        corpus.append(f"Log entry {1000 + i}: weekly administrative record {w} entry closed.")
    return corpus


STRESS_IDS = ["g1", "g2", "c1", "g6", "t1", "g7", "c2", "t3"]


def corpus_vocab(corpus: list[str]) -> set[str]:
    vocab: set[str] = set()
    for line in corpus:
        vocab.update(line.lower().replace("'", " ").split())
    return vocab


def invented_numbers(answer: str, corpus_text: str) -> list[str]:
    import re
    return [n for n in re.findall(r"\d[\d,\.]*", answer or "") if n not in corpus_text]


def infer_long(payload: dict, timeout: int = 1800) -> dict:
    import urllib.request as u
    from valhalla_server.config import INTERNAL_PORT

    data = json.dumps(payload).encode()
    req = u.Request(f"http://127.0.0.1:{INTERNAL_PORT}/v1/infer", data=data,
                    headers={"Content-Type": "application/json"}, method="POST")
    with u.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode())


def main() -> None:
    corpus = build_stress_corpus()
    chars = sum(len(c) for c in corpus)
    print(f"stress corpus lines={len(corpus)} chars={chars} approx_tokens={chars // 4}", flush=True)
    questions = [q for q in QUESTIONS if q["id"] in STRESS_IDS]
    rows = []

    # --- Valhalla arm first (Mistral server may be down while it runs) ---
    if "--mistral-only" not in sys.argv:
        from run_unique_points import run_valhalla  # noqa: F401  (keeps import graph simple)
        from valhalla_server import internal
        from valhalla_server.inference import _materialize

        folder = str(_materialize("unified-core-v1"))
        if not internal.start(folder):
            raise RuntimeError("ValhallaBase runtime failed")
        config = internal.chat_config(folder)
        session = None
        corpus_text = "\n".join(corpus).lower()
        for i, q in enumerate(questions):
            payload = {"questions": [{"id": q["id"], "text": q["text"], "score_type": "open"}],
                       "config": config, "context": corpus if i == 0 else []}
            if session:
                payload["session"] = session
            t0 = time.time()
            body = infer_long(payload)
            session = body.get("session") or session
            r = (body.get("rows") or [{}])[0]
            ans = str(r.get("answer", "")).strip()
            s = score(q, ans)
            inv = invented_numbers(ans, corpus_text)
            rows.append({"arm": "V-stress", **q, "answer": ans,
                         "decode_path": r.get("decode_path", ""),
                         "latency_s": round(time.time() - t0, 2),
                         "invented_numbers": inv, **s})
            print(f"[V-stress] {q['id']} correct={s['correct']} invented={inv} :: {ans[:110]}", flush=True)

    # --- Mistral arm ---
    if "--valhalla-only" not in sys.argv:
        from run_unique_points import mistral_chat

        system = ("Answer questions strictly from the user's project notes. "
                  "If the notes do not contain the answer, say you don't know. "
                  "Reply in one or two short sentences.")
        messages = [{"role": "system", "content": system},
                    {"role": "user", "content": "Here are my project notes:\n" + "\n".join(corpus)}]
        corpus_text = "\n".join(corpus).lower()
        for q in questions:
            msgs = messages + [{"role": "user", "content": q["text"]}]
            t0 = time.time()
            try:
                ans = mistral_chat(msgs, max_tokens=160)
                err = ""
            except Exception as exc:  # noqa: BLE001
                ans, err = "", str(exc)[:200]
            s = score(q, ans) if ans else {"correct": False, "abstained": False, "hallucinated": False}
            rows.append({"arm": "M-full-stress", **q, "answer": ans, "error": err,
                         "latency_s": round(time.time() - t0, 2),
                         "invented_numbers": invented_numbers(ans, corpus_text), **s})
            print(f"[M-stress] {q['id']} correct={s['correct']} err={err[:60]} :: {ans[:110]}", flush=True)

    stamp = time.strftime("%Y%m%d_%H%M%S")
    (OUT / f"stress_{stamp}.json").write_text(
        json.dumps({"corpus_lines": len(corpus), "corpus_chars": chars,
                    "rows": rows}, ensure_ascii=False, indent=2), encoding="utf-8")
    for r in rows:
        print(r["arm"], r["id"], r["kind"], "correct=", r["correct"], flush=True)


if __name__ == "__main__":
    main()
