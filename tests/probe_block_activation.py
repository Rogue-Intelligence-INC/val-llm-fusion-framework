"""Block-activation grasp probe (owner's design, read-only readout).

The body is organised in blocks (session fragments, each carrying its
own tiles/stem cells and a corpus line range). A question is a stimulus:
the engine passively selects which blocks light up (decode_path
query_frag ids). We measure, per question:
  - needle line inside an activated block?   (block grasp)
  - engine retrieval idx inside activated?   (consistency)
  - FOG decision                             (the silence gate)
Over-silence = block grasp correct but FOG abstains.
"""
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, r"C:\Ai\rogue-products\valhalla-server\tests")
sys.path.insert(0, r"C:\Ai\rogue-products\valhalla-server")
from run_extreme_stress import CONFIG, QUESTIONS, gen_corpus, post  # noqa: E402
from run_extreme_targeted import ENGINE, targeted  # noqa: E402

OUT = Path(r"C:\Ai\rogue-products\valhalla-server\tests\results_extreme\block_activation_probe.json")
FRAG_RE = re.compile(r"query_frag/k(\d+)/ids=([a-z0-9_+]+)")

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
MM = [("q1", "Who leads Project Aurora?", 0),
      ("q2", "Who leads Project Borealis?", 1),
      ("q3", "Which company supplies battery cells for Project Aurora?", 2),
      ("q4", "Which company supplies pressure sensors for Project Borealis?", 3),
      ("q5", "Who leads Project Aurora and which company supplies its battery cells?", None),
      ("q8", "Who leads Project Zephyr?", None),
      ("q9", "Who leads Project Cygnus?", 4)]


def ask(qid, qtext, lines):
    body = post(ENGINE, {"questions": [{"id": qid, "text": qtext, "score_type": "open"}],
                         "config": CONFIG, "context": lines})
    row = (body.get("rows") or [{}])[0]
    snap = (body.get("session") or {}).get("snapshot") or {}
    frags = snap.get("fragments") or []
    m = FRAG_RE.search(str(row.get("decode_path", "")))
    ids = m.group(2).split("+") if m else []
    # activated line ranges: fragments whose id was selected (ids may
    # repeat across appends; match by id in order of appearance)
    ranges = [(f.get("line_start"), f.get("line_end")) for f in frags
              if f.get("id") in ids]
    return row, ranges, len(frags)


def run_set(name, items):
    out = []
    for qid, qtext, lines, needle_idx in items:
        row, ranges, nfrag = ask(qid, qtext, lines)
        idx = row.get("retrieval_corpus_idx")
        fog = (row.get("fog_decision") or {}).get("decision")
        ev = row.get("fog_evidence") or {}
        in_block = (needle_idx is not None and any(
            a is not None and b is not None and a <= needle_idx <= b
            for a, b in ranges))
        idx_in = (isinstance(idx, int) and any(
            a is not None and b is not None and a <= idx <= b for a, b in ranges))
        rec = {"set": name, "id": qid, "fog": fog, "idx": idx,
               "needle_idx": needle_idx, "needle_in_block": in_block,
               "idx_in_block": idx_in, "blocks": ranges, "n_fragments": nfrag,
               "margin": ev.get("margin"), "top": ev.get("top_score")}
        out.append(rec)
        print(f"[{name}] {qid:<10} fog={fog:<8} needle_in_block={in_block} "
              f"idx={idx} idx_in_block={idx_in} margin={rec['margin']} blocks={ranges}", flush=True)
    return out


def main():
    results = []
    mm_items = [(qid, q, MM_LINES, exp) for qid, q, exp in MM]
    results += run_set("multimodal", mm_items)
    corpus, _ = gen_corpus(806)
    ex_items = []
    for qid, qtext, *_ in QUESTIONS:
        lines = targeted(qtext, corpus)
        needle = NEEDLE.get(qid)
        nidx = lines.index(needle) if needle in lines else None
        ex_items.append((qid, qtext, lines, nidx))
    results += run_set("extreme806", ex_items)
    OUT.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
