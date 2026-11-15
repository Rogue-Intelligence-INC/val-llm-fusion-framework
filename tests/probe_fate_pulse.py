"""Fate-only dual-pulse probe (diagnostic, no production changes).

Pulse 1: fresh targeted session, record fog + fog_evidence + grasped line.
Pulse 2: same question, grasped line REMOVED from context. If a strong
competitor emerges, the field was genuinely ambiguous; if the field
collapses, pulse-1 grasp was uniquely supported. Only Fate scores are
used -- no lexical rescoring, no transformer.

Grasp labels are exact needle lines from the synthetic corpus, so grasp
correctness is measured separately from native-answer correctness.
"""
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, r"C:\Ai\rogue-products\valhalla-server\tests")
sys.path.insert(0, r"C:\Ai\rogue-products\valhalla-server")
from run_extreme_stress import CONFIG, QUESTIONS, gen_corpus, post  # noqa: E402
from run_extreme_targeted import ENGINE, targeted  # noqa: E402

OUT = Path(r"C:\Ai\rogue-products\valhalla-server\tests\results_extreme\fate_pulse_probe.json")

NEEDLE = {
    "n_start": "Project Aurora's internal codename is AUR-7.",
    "para1": "Project Aurora's internal codename is AUR-7.",
    "para2": "Project Aurora's internal codename is AUR-7.",
    "inject_q": "Project Aurora's internal codename is AUR-7.",
    "n_mid": "Project Borealis's satellite lead is Dr. Ines Halvorsen.",
    "n_end": "Project Aurora's flight test site is located near Reno, Nevada.",
    "conflict": None,  # two valid competing lines
}


def ask(qtext, lines):
    body = post(ENGINE, {"questions": [{"id": "p", "text": qtext, "score_type": "open"}],
                         "config": CONFIG, "context": lines})
    return (body.get("rows") or [{}])[0]


def ev(row):
    e = row.get("fog_evidence") or {}
    return {k: e.get(k) for k in ("top_score", "second_score", "margin", "keyword_count")}


def main():
    corpus, _ = gen_corpus(806)
    out = []
    for qid, qtext, *_rest in QUESTIONS:
        lines = targeted(qtext, corpus)
        t0 = time.time()
        r1 = ask(qtext, lines)
        idx = r1.get("retrieval_corpus_idx")
        grasped = lines[idx] if isinstance(idx, int) and 0 <= idx < len(lines) else None
        needle = NEEDLE.get(qid)
        grasp_ok = (grasped == needle) if needle else None
        # pulse 2: remove grasped line
        rest = [l for i, l in enumerate(lines) if i != idx]
        r2 = ask(qtext, rest) if rest else {}
        idx2 = r2.get("retrieval_corpus_idx")
        grasped2 = rest[idx2] if isinstance(idx2, int) and 0 <= idx2 < len(rest) else None
        e1, e2 = ev(r1), ev(r2)
        ratio = None
        if e1.get("top_score") and e2.get("top_score"):
            ratio = round(e2["top_score"] / e1["top_score"], 3)
        rec = {
            "id": qid, "kind": _rest[2] if len(_rest) > 2 else None,
            "fog": (r1.get("fog_decision") or {}).get("decision"),
            "ev1": e1, "grasped": grasped, "grasp_ok": grasp_ok,
            "native": str(r1.get("answer", ""))[:120],
            "pulse2": {"fog": (r2.get("fog_decision") or {}).get("decision"),
                       "ev": e2, "grasped": grasped2, "top_ratio_p2_p1": ratio},
            "sec": round(time.time() - t0, 1),
        }
        out.append(rec)
        print(f"{qid:<12} fog={rec['fog']:<8} grasp_ok={grasp_ok} "
              f"m1={e1.get('margin')} top1={e1.get('top_score')} "
              f"p2top_ratio={ratio} p2fog={rec['pulse2']['fog']}", flush=True)
    OUT.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
