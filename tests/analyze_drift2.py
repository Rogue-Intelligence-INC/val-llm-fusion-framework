import json

d = json.load(open(r"C:\Ai\rogue-products\valhalla-server\tests\results_extreme\dialogue_drift_lines10006.json", encoding="utf-8"))
rows = d["rows"]
print(json.dumps(d["summary"], ensure_ascii=False, indent=1)[:600])
print("trap fails:", [(r["turn"], r["id"], r["answer"][:70]) for r in rows
                      if r["kind"] in ("swap_trap", "absent", "false_premise", "injection") and not r["pass"]])
print("core timeline:")
for cid in ("n_start", "n_mid", "n_end"):
    print(" ", cid, [(r["turn"], r["pass"]) for r in rows if r["id"] == cid])
print("max wire:", max(r.get("wire_mb") or 0 for r in rows))
