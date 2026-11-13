"""Multimodal fix acceptance battery.

Valhalla arm goes through the NEW gateway media endpoints: items are
co-ingested (text + real TTS wav + real SD png, encoded server-side by
the repo DL-front encoder). Competitor arms: caption-RAG (Mistral with
all captions as documents -- the Mem0/RAG class) and closed-book.
Feature-anchored queries (q6/q7) are scored structurally via the
retrieved corpus line; caption-RAG cannot accept features at all.
"""
import base64
import json
import sys
import time
import urllib.request
from pathlib import Path

sys.path.insert(0, r"C:\Ai\rogue-products\valhalla-server\tests")
from run_extreme_stress import is_abstain  # noqa: E402

GW = "http://127.0.0.1:8780"
HEAD = "http://127.0.0.1:18080/v1/chat/completions"
MEDIA = Path(r"C:\Ai\rogue-products\valhalla-server\tests\media")
OUT = Path(r"C:\Ai\rogue-products\valhalla-server\tests\results_extreme")
MODEL = "unified-core-v1"

ITEMS = [
    {"id": "s1", "text": "Project Aurora is a solar drone program led by Dr. Mara Voss.", "wav": "s1.wav", "png": "s1.png"},
    {"id": "s2", "text": "Project Borealis is a tidal sensor program led by Dr. Ilya Petrov.", "wav": "s2.wav", "png": "s2.png"},
    {"id": "s3", "text": "Project Aurora uses battery cells supplied by Helix Cells.", "wav": "s3.wav", "png": "s3.png"},
    {"id": "s4", "text": "Project Borealis uses pressure sensors supplied by Nordwave.", "wav": "s4.wav", "png": "s4.png"},
]
DISTRACTORS = [
    "Project Cygnus is led by Dr. Kenji Sato.",
    "Project Draco has a budget of 9.4 million euros.",
    "Project Lyra completed a review milestone in March.",
]


def post(url, payload, timeout=1200):
    req = urllib.request.Request(url, data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode())


def b64(path):
    return base64.b64encode(Path(path).read_bytes()).decode()


def mistral(messages, timeout=600):
    body = post(HEAD, {"model": "mistral-24b", "messages": messages,
                       "temperature": 0.2, "max_tokens": 220}, timeout=timeout)
    return body["choices"][0]["message"]["content"].strip()


QUESTIONS = [
    ("q1_text", "Who leads Project Aurora?", ["Mara Voss"], None),
    ("q2_text", "Who leads Project Borealis?", ["Ilya Petrov"], None),
    ("q3_text", "Which company supplies battery cells for Project Aurora?", ["Helix"], None),
    ("q4_text", "Which company supplies pressure sensors for Project Borealis?", ["Nordwave"], None),
    ("q5_mixed", "Who leads Project Aurora and which company supplies its battery cells?", ["Mara Voss", "Helix"], None),
    ("q6_imgfeat", "Which project is this image about?", [], "s2"),
    ("q7_audfeat", "Which project is this audio about?", [], "s3"),
    ("q8_trap", "Who leads Project Zephyr?", [], "ABSTAIN"),
    ("q9_distractor", "Who leads Project Cygnus?", ["Kenji Sato"], None),
]


def main():
    texts_order = [it["text"] for it in ITEMS] + DISTRACTORS
    res = {"arms": {}}

    # ---- Valhalla arm: ingest via gateway media endpoint
    items_payload = []
    for it in ITEMS:
        items_payload.append({"text": it["text"],
                              "audio_b64": b64(MEDIA / it["wav"]),
                              "image_b64": b64(MEDIA / it["png"])})
    for d in DISTRACTORS:
        items_payload.append({"text": d})
    t0 = time.time()
    ing = post(f"{GW}/v1/media/ingest", {"model": MODEL, "items": items_payload})
    print(f"ingest: {ing} ({round(time.time() - t0, 1)}s)", flush=True)
    res["ingest"] = ing

    rows = []
    for qid, qtext, must, special in QUESTIONS:
        payload = {"model": MODEL, "question": qtext}
        if special == "s2":
            payload["image_b64"] = b64(MEDIA / "s2.png")
        if special == "s3":
            payload["audio_b64"] = b64(MEDIA / "s3.wav")
        t1 = time.time()
        r = post(f"{GW}/v1/media/ask", payload)
        ans = r.get("answer", "")
        idx = r.get("retrieval_corpus_idx")
        line = texts_order[idx] if isinstance(idx, int) and 0 <= idx < len(texts_order) else ""
        if special == "ABSTAIN":
            ok = is_abstain(ans)
        elif special in ("s2", "s3"):
            want = "Borealis" if special == "s2" else "Aurora"
            ok = want in ans or want in line
        else:
            ok = all(m.lower() in ans.lower() for m in must)
        rows.append({"id": qid, "pass": ok, "answer": ans[:160], "mode": r.get("mode"),
                     "fog": (r.get("fog_decision") or {}).get("decision"),
                     "retrieval_idx": idx, "retrieved_line": line[:80],
                     "sec": round(time.time() - t1, 1)})
        print(f"  [valhalla] {qid:14s} pass={ok} fog={rows[-1]['fog']} idx={idx} {ans[:60]!r}", flush=True)
    res["arms"]["valhalla"] = rows

    # ---- caption-RAG arm (Mistral + captions as documents)
    docs = "\n".join(texts_order)
    rows = []
    for qid, qtext, must, special in QUESTIONS:
        if special in ("s2", "s3"):
            rows.append({"id": qid, "pass": False, "note": "cannot_accept_features", "answer": ""})
            print(f"  [rag] {qid:14s} pass=False (features not accepted)", flush=True)
            continue
        t1 = time.time()
        ans = mistral([{"role": "system", "content": "Answer using ONLY the documents. If not in the documents, say you don't know."},
                       {"role": "user", "content": f"DOCUMENTS:\n{docs}\n\nQuestion: {qtext}"}])
        ok = is_abstain(ans) if special == "ABSTAIN" else all(m.lower() in ans.lower() for m in must)
        rows.append({"id": qid, "pass": ok, "answer": ans[:160], "sec": round(time.time() - t1, 1)})
        print(f"  [rag] {qid:14s} pass={ok} {ans[:60]!r}", flush=True)
    res["arms"]["caption_rag"] = rows

    # ---- closed-book arm
    rows = []
    for qid, qtext, must, special in QUESTIONS:
        if special in ("s2", "s3"):
            rows.append({"id": qid, "pass": False, "note": "cannot_accept_features", "answer": ""})
            continue
        ans = mistral([{"role": "user", "content": qtext}])
        ok = is_abstain(ans) if special == "ABSTAIN" else all(m.lower() in ans.lower() for m in must)
        rows.append({"id": qid, "pass": ok, "answer": ans[:160]})
        print(f"  [closed] {qid:14s} pass={ok} {ans[:60]!r}", flush=True)
    res["arms"]["closed_book"] = rows

    dest = OUT / "multimodal_fixed.json"
    dest.write_text(json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")
    for arm, rr in res["arms"].items():
        print(f"== {arm}: {sum(1 for r in rr if r['pass'])}/{len(rr)}")
    print(f"-> {dest}")


if __name__ == "__main__":
    main()
