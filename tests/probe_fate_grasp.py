"""Probe: what does Fate actually grasp for the paraphrase questions
that fail under Fate-only evidence? Fresh targeted context per question,
print row idx/fog/native/decode_path and the grasped line; then the
transformer variants and their re-grasp rows."""
import sys

sys.path.insert(0, r"C:\Ai\rogue-products\valhalla-server\tests")
sys.path.insert(0, r"C:\Ai\rogue-products\valhalla-server")
from run_extreme_stress import CONFIG, gen_corpus, post  # noqa: E402
from run_extreme_targeted import ENGINE, targeted  # noqa: E402
from valhalla_server import mistral_head  # noqa: E402

corpus, _ = gen_corpus(806)
for q in ("Tell me the codename used by Project Aurora.",
          "What is Project Aurora's codename? Answer with the codename only."):
    lines = targeted(q, corpus)
    body = post(ENGINE, {"questions": [{"id": "p", "text": q, "score_type": "open"}],
                         "config": CONFIG, "context": lines})
    row = (body.get("rows") or [{}])[0]
    idx = row.get("retrieval_corpus_idx")
    print(f"\nQ: {q}")
    print(f"  fog={row.get('fog_decision')} ev={row.get('fog_evidence')}")
    print(f"  idx={idx} grasped={lines[idx] if isinstance(idx, int) and 0 <= idx < len(lines) else None!r}")
    print(f"  native={row.get('answer')!r} path={row.get('decode_path')}")
    vs = mistral_head.variants(q)
    print(f"  variants={vs}")
    for v in vs:
        b2 = post(ENGINE, {"questions": [{"id": "p2", "text": v, "score_type": "open"}],
                           "config": CONFIG, "context": lines})
        r2 = (b2.get("rows") or [{}])[0]
        i2 = r2.get("retrieval_corpus_idx")
        print(f"    v={v!r} fog={(r2.get('fog_decision') or {}).get('decision')} "
              f"grasped={lines[i2] if isinstance(i2, int) and 0 <= i2 < len(lines) else None!r} "
              f"native={r2.get('answer')!r}")
