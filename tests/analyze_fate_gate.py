"""Offline: simulate Fate-controlled gate rules on recorded block series.

Rule A (session algebra): fire iff block margin >= delta_required(s1),
  delta_required(s1) = s1 * (affinity_std + 0.025) * relax  -> rel >= scale.
  We don't have affinity_std recorded; report the rel each rule needs.
Rule B (equation): within-block score diffs d1>=d2..., x = d1/d2,
  fire iff NOT bundle_fits i.e. (x-1)^2 >= threshold (grasp separated
  from runner-up = not one bundle). Report (x-1)^2 per row.
"""
import json, os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
rows = json.load(open(os.path.join(
    ROOT, "tests", "results_extreme", "fate_series_block.json"), encoding="utf-8"))

BASE = {("mm", "q1"): "abstain", ("mm", "q2"): "abstain", ("mm", "q3"): "emit",
        ("mm", "q4"): "emit", ("mm", "q5"): "emit", ("mm", "q8"): "abstain",
        ("mm", "q9"): "emit", ("ex", "n_start"): "abstain", ("ex", "n_mid"): "emit",
        ("ex", "n_end"): "emit", ("ex", "para1"): "abstain", ("ex", "para2"): "abstain",
        ("ex", "swap1"): "abstain", ("ex", "swap2"): "emit",
        ("ex", "absent_ent"): "abstain", ("ex", "absent_attr"): "abstain",
        ("ex", "absent_attr2"): "emit", ("ex", "premise"): "emit",
        ("ex", "conflict"): "abstain", ("ex", "inject_q"): "abstain"}

print(f"{'id':13} {'base':8} {'now':8} {'grasp_ok':8} {'rel':>7} {'blk_series':38} {'(x-1)^2':>8}")
for r in rows:
    blk = r.get("grasp_block")
    ser = []
    if blk:
        pairs = [(i, s) for i, s in zip(r.get("cand_idx") or [], r.get("series") or [])
                 if i is not None and blk[0] <= i <= blk[1]]
        pairs.sort(key=lambda p: -p[1])
        ser = [round(s, 4) for _, s in pairs]
    diffs = [ser[i] - ser[i + 1] for i in range(len(ser) - 1)]
    x2 = None
    if len(diffs) >= 2 and diffs[1] > 1e-12:
        x = diffs[0] / diffs[1]
        x2 = round((x - 1) ** 2, 4)
    print(f"{r['id']:13} {BASE[(r['set'], r['id'])]:8} {str(r.get('fog')):8} "
          f"{str(r.get('grasp_ok')):8} {r.get('fate_block_rel') or 0:>7.4f} "
          f"{str(ser):38} {x2 if x2 is not None else '-':>8}")
