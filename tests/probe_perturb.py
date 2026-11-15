"""Perturbation audit: is the gate mechanism-driven or fit to the set?

Same questions, perturbed corpora:
  A) multimodal lines REVERSED (needles move to other blocks),
  B) multimodal lines shuffled (seed 99),
  C) extreme corpus regenerated with seed 777 (different fillers).
A mechanism (block membership + Fate bundle judgement) should keep
deciding from the readings; a fitted constant/case logic would crack.
"""
import json, random, sys
from pathlib import Path

sys.path.insert(0, r"C:\Ai\rogue-products\valhalla-server\tests")
sys.path.insert(0, r"C:\Ai\rogue-products\valhalla-server")
from run_extreme_stress import CONFIG, QUESTIONS, gen_corpus, post  # noqa: E402
from run_extreme_targeted import ENGINE, targeted  # noqa: E402

MM_LINES = [
    "Project Aurora is a solar drone program led by Dr. Mara Voss.",
    "Project Borealis is a tidal sensor program led by Dr. Ilya Petrov.",
    "Project Aurora uses battery cells supplied by Helix Cells.",
    "Project Borealis uses pressure sensors supplied by Nordwave.",
    "Project Cygnus is led by Dr. Kenji Sato.",
    "Project Draco has a budget of 9.4 million euros.",
    "Project Lyra completed a review milestone in March.",
]
# (qid, question, unique substring of the correct line or None)
MM = [("q1", "Who leads Project Aurora?", "led by Dr. Mara Voss"),
      ("q2", "Who leads Project Borealis?", "led by Dr. Ilya Petrov"),
      ("q3", "Which company supplies battery cells for Project Aurora?", "Helix Cells"),
      ("q4", "Which company supplies pressure sensors for Project Borealis?", "Nordwave"),
      ("q8", "Who leads Project Zephyr?", None),
      ("q9", "Who leads Project Cygnus?", "Dr. Kenji Sato")]
NEEDLE = {
    "n_start": "Project Aurora's internal codename is AUR-7.",
    "para1": "Project Aurora's internal codename is AUR-7.",
    "para2": "Project Aurora's internal codename is AUR-7.",
    "inject_q": "Project Aurora's internal codename is AUR-7.",
    "n_mid": "Project Borealis's satellite lead is Dr. Ines Halvorsen.",
    "n_end": "Project Aurora's flight test site is located near Reno, Nevada.",
}


def ask(qid, qtext, lines):
    b = post(ENGINE, {"questions": [{"id": qid, "text": qtext, "score_type": "open"}],
                      "config": CONFIG, "context": lines})
    return (b.get("rows") or [{}])[0]


def summarize(tag, qid, r, lines, needle_text):
    g = r.get("fate_grasp") or {}
    idx = r.get("retrieval_corpus_idx")
    grasped = lines[idx] if isinstance(idx, int) and 0 <= idx < len(lines) else None
    ok = (grasped == needle_text) if needle_text else None
    print(f"{tag} {qid:13} {(r.get('fog_decision') or {}).get('decision'):8} "
          f"grasp_ok={ok} block={g.get('grasp_block')} rel={g.get('fate_block_rel')} "
          f"reason={((r.get('fog_decision') or {}).get('reason') or '')[:60]}", flush=True)
    return {"variant": tag, "id": qid, "fog": (r.get("fog_decision") or {}).get("decision"),
            "grasp_ok": ok, "block": g.get("grasp_block"), "rel": g.get("fate_block_rel")}


out = []
rev = list(reversed(MM_LINES))
for qid, q, sub in MM:
    exp = next((ln for ln in rev if sub and sub in ln), None)
    out.append(summarize("A-rev", qid, ask(qid, q, rev), rev, exp))

sh = MM_LINES[:]
random.Random(99).shuffle(sh)
for qid, q, sub in MM:
    exp = next((ln for ln in sh if sub and sub in ln), None)
    out.append(summarize("B-shuf", qid, ask(qid, q, sh), sh, exp))

corpus, _ = gen_corpus(806, seed=777)
for qid, qtext, *_ in QUESTIONS:
    lines = targeted(qtext, corpus)
    needle = NEEDLE.get(qid)
    out.append(summarize("C-seed777", qid, ask(qid, qtext, lines), lines, needle))

Path(r"C:\Ai\rogue-products\valhalla-server\tests\results_extreme\perturb_audit.json").write_text(
    json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
print("wrote perturb_audit.json")
