"""Analyze within-block grasp readings vs global margin separation."""
import json, os, sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
paths = sys.argv[1:] or [os.path.join(ROOT, "tests", "results_extreme", "fate_series_labeled.json")]
rows = []
for p in paths:
    rows += json.load(open(p, encoding="utf-8"))

print(f"{'set':3} {'id':13} {'fog':8} {'grasp_ok':8} {'idx':>5} {'in_act':6} "
      f"{'block_rel':>9} {'glob_rel':>8} {'n_act':>5}")
for r in rows:
    s = r.get("series") or []
    glob_rel = ((s[0] - s[1]) / s[0]) if len(s) >= 2 and s[0] > 0 else 0.0
    act = r.get("activated") or []
    idx = r.get("idx")
    print(f"{r['set']:3} {r['id']:13} {str(r.get('fog')):8} {str(r.get('grasp_ok')):8} "
          f"{idx if idx is not None else -1:>5} {str(idx in act) if idx is not None else '?':6} "
          f"{r.get('block_rel') if r.get('block_rel') is not None else -1:>9.4f} {glob_rel:>8.4f} {len(act):>5}")

print("\n== separation check (extreme needles should-emit vs should-silence) ==")
needles = [r for r in rows if r["set"] == "ex" and r.get("grasp_ok") is True]
silence = [r for r in rows if r["set"] == "ex" and r.get("grasp_ok") is not True]
for name, key in [("block_rel", lambda r: r.get("block_rel") or 0.0),
                  ("glob_rel", lambda r: ((r["series"][0]-r["series"][1])/r["series"][0]) if r.get("series") and len(r["series"])>1 else 0.0)]:
    nmin = min((key(r) for r in needles), default=0)
    smax = max((key(r) for r in silence), default=0)
    print(f"{name}: needle_min={nmin:.4f} silence_max={smax:.4f} separated={nmin > smax}")
