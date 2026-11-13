"""Respawn the internal engine via the gateway after an engine kill."""
import json
import urllib.request

payload = {"questions": [{"id": "boot", "text": "What is Project Aurora's internal codename?", "score_type": "open"}],
           "context": ["Project Aurora's internal codename is AUR-7."],
           "config": {"decode": "hybrid", "max_new": 16}}
req = urllib.request.Request("http://127.0.0.1:8780/v1/infer",
                             data=json.dumps(payload).encode(),
                             headers={"Content-Type": "application/json"}, method="POST")
with urllib.request.urlopen(req, timeout=600) as resp:
    body = json.loads(resp.read().decode())
print("rows:", json.dumps((body.get("rows") or [{}])[:1], ensure_ascii=False)[:200])
print("keys:", sorted(body.keys()))
