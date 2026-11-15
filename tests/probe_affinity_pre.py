"""Pre-question affinity: ingest first (no QA), snapshot affinity,
then ask. Is the needle already the field peak BEFORE any feedback?"""
import statistics
import sys

sys.path.insert(0, r"C:\Ai\rogue-products\valhalla-server\tests")
sys.path.insert(0, r"C:\Ai\rogue-products\valhalla-server")
from run_extreme_stress import CONFIG, QUESTIONS, gen_corpus, post  # noqa: E402
from run_extreme_targeted import ENGINE, targeted  # noqa: E402
from valhalla_server import internal  # noqa: E402

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
    # ingest: context call (incubation forms affinity); warm question is
    # a throwaway so the snapshot is pre-question for the real one
    b0 = post(ENGINE, {"questions": [{"id": "warm", "text": "Warmup.", "score_type": "open"}],
                       "config": CONFIG, "context": lines})
    sess = b0.get("session")
    aff_pre = list(((sess or {}).get("snapshot") or {}).get("corpus_line_affinity") or [])
    sess_corpus = list((sess or {}).get("corpus") or lines)
    b = post(ENGINE, {"questions": [{"id": qid, "text": qtext, "score_type": "open"}],
                      "config": CONFIG, "context": [],
                      "session": internal.slim_session(sess), "append_lines": []})
    r = (b.get("rows") or [{}])[0]
    idx = r.get("retrieval_corpus_idx")
    needle = NEEDLE.get(qid)
    nidx = sess_corpus.index(needle) if needle in sess_corpus else None
    def rank_of(i):
        if i is None or i >= len(aff_pre):
            return None
        return 1 + sum(1 for a in aff_pre if a > aff_pre[i])
    fog = (r.get("fog_decision") or {}).get("decision")
    print(f"{qid:<12} fog={fog:<8} grasp_idx={idx} needle_idx={nidx} "
          f"aff_pre[grasp]={aff_pre[idx] if isinstance(idx, int) and idx < len(aff_pre) else None} "
          f"rank_grasp={rank_of(idx)} rank_needle={rank_of(nidx)} "
          f"aff_mean={round(statistics.mean(aff_pre), 3) if aff_pre else None}", flush=True)
