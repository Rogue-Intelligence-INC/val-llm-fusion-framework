"""Corpus-switch regression: after a different conversation used the
same model, a new corpus must be answered from ITS facts only."""
import json
import urllib.request

URL = "http://127.0.0.1:8780/v1/models/unified-core-v1/chat/completions"
CORPUS_B = ("Project Zephyr's internal codename is ZEP-9. "
            "Project Zephyr's wind tunnel is located in Bristol, England. "
            "Dr. Mara Kessler leads Project Zephyr.")


def chat(messages):
    req = urllib.request.Request(URL, data=json.dumps({"messages": messages}).encode(),
                                 headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(req, timeout=1200) as resp:
        return json.loads(resp.read().decode())


def ask(tag, q):
    body = chat([{"role": "user", "content": CORPUS_B}, {"role": "user", "content": q}])
    v = body.get("valhalla") or {}
    ans = ((body.get("choices") or [{}])[0].get("message") or {}).get("content", "")
    tel = v.get("fate_telemetry") or {}
    print(f"[{tag}] answer={ans[:120]!r}")
    print(f"   fog={(v.get('fog_decision') or {}).get('decision')} ingress={tel.get('fate_ingress_routing')} "
          f"spread={tel.get('hub_prefs_spread')}")


if __name__ == "__main__":
    ask("new-corpus grounded", "What is Project Zephyr's internal codename?")
    ask("old-corpus leak check", "What is Project Aurora's internal codename?")
