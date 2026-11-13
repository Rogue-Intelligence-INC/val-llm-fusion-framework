"""Gateway end-to-end Fate check via chat completions (port 8780)."""
import json
import urllib.request

URL = "http://127.0.0.1:8780/v1/models/unified-core-v1/chat/completions"
CORPUS = ("Project Aurora's internal codename is AUR-7. "
          "Project Borealis uses the codename BOR-3 for its satellite program. "
          "The Borealis satellite program is led by Dr. Ines Halvorsen. "
          "Project Aurora's budget for 2026 is 4.2 million euros.")


def chat(messages):
    req = urllib.request.Request(URL, data=json.dumps({"messages": messages}).encode(),
                                 headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(req, timeout=1200) as resp:
        return json.loads(resp.read().decode())


def main():
    for tag, q in (("grounded", "What is Project Aurora's internal codename?"),
                   ("trap", "What is the wingspan of Project Aurora's drone?"),
                   ("close-margin", "Who leads the Borealis satellite program?")):
        body = chat([{"role": "user", "content": CORPUS},
                     {"role": "user", "content": q}])
        v = body.get("valhalla") or {}
        ans = ((body.get("choices") or [{}])[0].get("message") or {}).get("content", "")
        print(f"[{tag}] answer={ans[:120]!r}")
        print(f"   decode_path={v.get('decode_path')} fog={json.dumps(v.get('fog_decision'), ensure_ascii=False)}")
        print(f"   decode_head={v.get('decode_head')} model_fate={json.dumps(v.get('model_fate'), ensure_ascii=False)[:200]}")


if __name__ == "__main__":
    main()
