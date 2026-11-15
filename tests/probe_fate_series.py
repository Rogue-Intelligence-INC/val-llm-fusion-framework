"""Collect fate_grasp candidate series with grasp labels for gate analysis."""
import json
import sys
from pathlib import Path

sys.path.insert(0, r"C:\Ai\rogue-products\valhalla-server\tests")
sys.path.insert(0, r"C:\Ai\rogue-products\valhalla-server")
from run_extreme_stress import CONFIG, QUESTIONS, gen_corpus, post  # noqa: E402
from run_extreme_targeted import ENGINE, targeted  # noqa: E402

import os

OUT = Path(os.environ.get(
    "FATE_SERIES_OUT",
    r"C:\Ai\rogue-products\valhalla-server\tests\results_extreme\fate_series_labeled.json"))
NEEDLE = {
    "n_start": "Project Aurora's internal codename is AUR-7.",
    "para1": "Project Aurora's internal codename is AUR-7.",
    "para2": "Project Aurora's internal codename is AUR-7.",
    "inject_q": "Project Aurora's internal codename is AUR-7.",
    "n_mid": "Project Borealis's satellite lead is Dr. Ines Halvorsen.",
    "n_end": "Project Aurora's flight test site is located near Reno, Nevada.",
}
MM_LINES = [
    "Project Aurora is a solar drone program led by Dr. Mara Voss.",
    "Project Borealis is a tidal sensor program led by Dr. Ilya Petrov.",
    "Project Aurora uses battery cells supplied by Helix Cells.",
    "Project Borealis uses pressure sensors supplied by Nordwave.",
    "Project Cygnus is led by Dr. Kenji Sato.",
    "Project Draco has a budget of 9.4 million euros.",
    "Project Lyra completed a review milestone in March.",
]
MM = [("q1", "Who leads Project Aurora?", 0), ("q2", "Who leads Project Borealis?", 1),
      ("q3", "Which company supplies battery cells for Project Aurora?", 2),
      ("q4", "Which company supplies pressure sensors for Project Borealis?", 3),
      ("q5", "Who leads Project Aurora and which company supplies its battery cells?", None),
      ("q8", "Who leads Project Zephyr?", None), ("q9", "Who leads Project Cygnus?", 4)]


def block_rel_margin(g, idx):
    series = g.get("candidate_scores") or []
    cand = g.get("candidate_indices") or []
    active = set(g.get("activated_indices") or [])
    pairs = [(i, s) for i, s in zip(cand, series) if i is not None and i in active]
    sg = next((s for i, s in pairs if i == idx), None)
    if sg is None or sg <= 0:
        return None
    others = [s for i, s in pairs if i != idx]
    return round(((sg - max(others)) if others else sg) / sg, 4)


def ask(qid, qtext, lines):
    b = post(ENGINE, {"questions": [{"id": qid, "text": qtext, "score_type": "open"}],
                      "config": CONFIG, "context": lines})
    return (b.get("rows") or [{}])[0]


out = []
for qid, q, exp in MM:
    r = ask(qid, q, MM_LINES)
    g = r.get("fate_grasp") or {}
    idx = r.get("retrieval_corpus_idx")
    out.append({"set": "mm", "id": qid, "fog": (r.get("fog_decision") or {}).get("decision"),
                "reason": (r.get("fog_decision") or {}).get("reason"),
                "series": g.get("candidate_scores"), "frag_scores": g.get("fragment_scores"),
                "cand_idx": g.get("candidate_indices"), "activated": g.get("activated_indices"),
                "block_rel": block_rel_margin(g, idx),
                "grasp_block": g.get("grasp_block"), "fate_block_rel": g.get("fate_block_rel"),
                "n_block_cand": g.get("block_candidate_count"),
                "alg_std": g.get("alg_affinity_std"), "alg_delta_rel": g.get("alg_delta_rel"),
                "blk_thr": g.get("block_cluster_threshold"),
                "eq": g.get("eq_coefs"), "eq_thr": g.get("eq_threshold"), "eq_adapt": g.get("eq_adaptations"),
                "idx": idx, "needle_idx": exp, "grasp_ok": (idx == exp) if exp is not None else None})
    print("mm", qid, out[-1]["fog"], out[-1]["grasp_ok"],
          "block", out[-1]["grasp_block"], "rel", out[-1]["fate_block_rel"],
          "alg_rel", out[-1]["alg_delta_rel"], "blk_thr", out[-1]["blk_thr"], "eq", out[-1]["eq"], out[-1]["eq_thr"], out[-1]["eq_adapt"], "eq", out[-1]["eq"], out[-1]["eq_thr"], out[-1]["eq_adapt"], flush=True)
corpus, _ = gen_corpus(806)
for qid, qtext, *_ in QUESTIONS:
    lines = targeted(qtext, corpus)
    r = ask(qid, qtext, lines)
    g = r.get("fate_grasp") or {}
    idx = r.get("retrieval_corpus_idx")
    needle = NEEDLE.get(qid)
    nidx = lines.index(needle) if needle in lines else None
    grasped = lines[idx] if isinstance(idx, int) and 0 <= idx < len(lines) else None
    out.append({"set": "ex", "id": qid, "fog": (r.get("fog_decision") or {}).get("decision"),
                "reason": (r.get("fog_decision") or {}).get("reason"),
                "series": g.get("candidate_scores"), "frag_scores": g.get("fragment_scores"),
                "cand_idx": g.get("candidate_indices"), "activated": g.get("activated_indices"),
                "block_rel": block_rel_margin(g, idx),
                "grasp_block": g.get("grasp_block"), "fate_block_rel": g.get("fate_block_rel"),
                "n_block_cand": g.get("block_candidate_count"),
                "alg_std": g.get("alg_affinity_std"), "alg_delta_rel": g.get("alg_delta_rel"),
                "blk_thr": g.get("block_cluster_threshold"),
                "eq": g.get("eq_coefs"), "eq_thr": g.get("eq_threshold"), "eq_adapt": g.get("eq_adaptations"),
                "idx": idx, "needle_idx": nidx,
                "grasp_ok": (grasped == needle) if needle else None})
    print(qid, out[-1]["fog"], out[-1]["grasp_ok"],
          "block", out[-1]["grasp_block"], "rel", out[-1]["fate_block_rel"],
          "alg_rel", out[-1]["alg_delta_rel"], "blk_thr", out[-1]["blk_thr"], "eq", out[-1]["eq"], out[-1]["eq_thr"], out[-1]["eq_adapt"],
          out[-1]["series"][:3] if out[-1]["series"] else [], flush=True)
OUT.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
print(f"wrote {OUT}")
