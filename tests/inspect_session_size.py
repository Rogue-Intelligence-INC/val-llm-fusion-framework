import json

d = json.load(open(r"C:\Ai\rogue-products\valhalla-server\tests\results_extreme\session_lines120.json", encoding="utf-8"))
s = d["session"]


def mb(x):
    return round(len(json.dumps(x)) / 1e6, 2)


print("session total MB:", mb(s))
for k, v in sorted(s.items(), key=lambda kv: -len(json.dumps(kv[1]))):
    print(f"  {k}: {mb(v)} MB")
snap = s.get("snapshot") or {}
print("snapshot keys by size:")
for k, v in sorted(snap.items(), key=lambda kv: -len(json.dumps(kv[1])))[:10]:
    n = len(v) if isinstance(v, list) else "-"
    print(f"  {k}: {mb(v)} MB len={n}")
fr = snap.get("fragments") or []
if fr:
    print("fragment count:", len(fr), "first fragment MB:", mb(fr[0]),
          "keys:", sorted(fr[0].keys()))
