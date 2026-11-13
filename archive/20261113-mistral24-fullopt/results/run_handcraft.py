"""Drive ONLY battery_handcraft() from the official test module.

The full main() also runs battery_fair, which needs data absent from
this checkout (incubation_benchmark_suite.json). Handcraft uses the
module's own SWAP_SCENARIOS and trial_proactive unchanged.
"""
import importlib.util
import json
import sys
from pathlib import Path

import valhalla_base

valhalla_base.BIN = Path(r"C:\Ai\Valhalla-main\target\release\valhalla_base.exe")
for _p in Path(r"C:\Ai\Valhalla-main\tools").iterdir():
    if _p.is_dir():
        sys.path.insert(0, str(_p))
sys.path.insert(0, r"C:\Ai\rogue-products\valhalla-server\tests\stubs")
sys.path.insert(0, r"C:\Ai\Valhalla-main\tools\valhalla_inference")

spec = importlib.util.spec_from_file_location(
    "test_proactive_fate_v2",
    r"C:\Ai\Valhalla-main\tools\valhalla_inference\test_proactive_fate_v2.py",
)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)

rows = mod.battery_handcraft()
metrics = mod.compute_metrics(rows)
out = {"metrics": metrics, "rows": rows}
dest = Path(r"C:\Ai\rogue-products\valhalla-server\tests\results_handcraft.json")
dest.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
print(json.dumps(metrics, indent=2, ensure_ascii=False))
print(f"passed {sum(1 for r in rows if r.get('pass'))}/{len(rows)} -> {dest}")
