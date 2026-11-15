import json

arms = json.load(open(r"C:\Ai\rogue-products\valhalla-server\tests\results_extreme\dynamic_unload_ab.json", encoding="utf-8"))
on, off = arms
ids = [r["id"] for r in on["rows"]]
for i, (a, b) in enumerate(zip(on["rows"], off["rows"])):
    flag = "" if a["pass"] == b["pass"] else "  <-- DIFF"
    print(f"t{i:<3} {a['id']:<12} on pass={a['pass']} fog={a['fog']:<8} wire={a['wire_mb']} | "
          f"off pass={b['pass']} fog={b['fog']:<8} wire={b['wire_mb']}{flag}")
