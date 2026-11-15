"""Read live-equation telemetry through the gateway (8780) path."""
import sys

sys.path.insert(0, r"C:\Ai\rogue-products\valhalla-server\tests")
sys.path.insert(0, r"C:\Ai\rogue-products\valhalla-server")
from run_extreme_stress import CONFIG, post  # noqa: E402
from probe_fate_series import MM_LINES, MM  # noqa: E402

GW = "http://127.0.0.1:8780/v1/infer"
for qid, q, _exp in MM[:3]:
    b = post(GW, {"questions": [{"id": qid, "text": q, "score_type": "open"}],
                  "config": CONFIG, "context": MM_LINES})
    r = (b.get("rows") or [{}])[0]
    g = r.get("fate_grasp") or {}
    print(qid, (r.get("fog_decision") or {}).get("decision"),
          "eq", g.get("eq_coefs"), "thr", g.get("eq_threshold"),
          "adapt", g.get("eq_adaptations"), "rel", g.get("fate_block_rel"),
          "keys", sorted(b.keys())[:8], flush=True)
