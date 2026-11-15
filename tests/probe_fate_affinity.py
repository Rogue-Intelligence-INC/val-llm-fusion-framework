"""Check whether the grasped line's corpus affinity (a Fate field signal
the FOG gate never reads) separates grasp-correct abstentions from traps."""
import statistics
import sys

sys.path.insert(0, r"C:\Ai\rogue-products\valhalla-server\tests")
sys.path.insert(0, r"C:\Ai\rogue-products\valhalla-server")
from run_extreme_stress import CONFIG, QUESTIONS, gen_corpus, post  # noqa: E402
from run_extreme_targeted import ENGINE, targeted  # noqa: E402

NEEDLE = {
    "n_start": "Project Aurora's internal codename is AUR-7.",
    "para1": "Project Aurora's internal codename is AUR-7.",
    "para2": "Project Aurora's internal codename is AUR-7.",
    "inject_q": "Project Aurora's internal codename is AUR-7.",
    "n_mid": "Project Borealis's satellite lead is Dr. Ines Halvorsen.",
    "n_end": "Project Aurora's flight test site is located near Reno, Nevada.",
}

corpus, _ = gen_corpus(806)
for qid, qtext, *_ in QUESTIONS:
    lines = targeted(qtext, corpus)
    b = post(ENGINE, {"questions": [{"id": qid, "text": qtext, "score_type": "open"}],
                      "config": CONFIG, "context": lines})
    r = (b.get("rows") or [{}])[0]
    idx = r.get("retrieval_corpus_idx")
    aff = ((b.get("session") or {}).get("snapshot") or {}).get("corpus_line_affinity") or []
    grasped = lines[idx] if isinstance(idx, int) and 0 <= idx < len(lines) else None
    ok = (grasped == NEEDLE[qid]) if qid in NEEDLE else None
    e = r.get("fog_evidence") or {}
    at = aff[idx] if isinstance(idx, int) and idx < len(aff) else None
    rank = 1 + sum(1 for a in aff if at is not None and a > at) if at is not None else None
    print(f"{qid:<12} fog={(r.get('fog_decision') or {}).get('decision'):<8} "
          f"grasp_ok={ok} margin={round(e.get('margin') or 0, 4)} "
          f"aff_top={round(at, 3) if at is not None else None} "
          f"aff_rank={rank}/{len(aff)} aff_mean={round(statistics.mean(aff), 3) if aff else None}")
