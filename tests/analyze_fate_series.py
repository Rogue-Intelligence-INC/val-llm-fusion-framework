"""Offline gate readings over recorded fate_grasp candidate series."""
import json

rows = json.load(open(r"C:\Ai\rogue-products\valhalla-server\tests\results_extreme\fate_series_labeled.json", encoding="utf-8"))


def feats(s):
    d = [s[i] - s[i + 1] for i in range(len(s) - 1)]
    return {
        "d1": round(d[0], 4) if d else None,
        "rel": round(d[0] / s[0], 3) if d else None,
        "d1_d2": round(d[0] / d[1], 2) if len(d) > 1 and d[1] > 1e-9 else None,
        "d1_tail": round(d[0] / (sum(d[1:]) / max(1, len(d) - 1)), 2) if len(d) > 1 else None,
    }


for r in rows:
    s = r["series"] or []
    print(f"{r['set']} {r['id']:<12} fog={r['fog']:<8} grasp_ok={r['grasp_ok']} "
          f"{feats(s) if len(s) >= 2 else s}")

print("\nSeparation check (extreme set): should-emit = grasp_ok True")
pos = [r for r in rows if r["set"] == "ex" and r["grasp_ok"]]
neg = [r for r in rows if r["set"] == "ex" and not r["grasp_ok"]]
for key in ("d1", "rel", "d1_d2", "d1_tail"):
    pv = [feats(r["series"])[key] for r in pos if feats(r["series"])[key] is not None]
    nv = [feats(r["series"])[key] for r in neg if feats(r["series"])[key] is not None]
    print(f"  {key}: emit-min={min(pv) if pv else None} silence-max={max(nv) if nv else None}")
