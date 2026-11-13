"""Probe use_cpu=false (CUDA LM decode in valhalla_base) on a tiny corpus."""
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))

from run_unique_points import FACTS_A, FACTS_B  # noqa: E402
from run_stress_points import infer_long  # noqa: E402


def main():
    from valhalla_server import internal
    from valhalla_server.inference import _materialize

    folder = str(_materialize("unified-core-v1"))
    if not internal.start(folder):
        raise RuntimeError("runtime failed")
    config = internal.chat_config(folder)
    config["use_cpu"] = False
    corpus = FACTS_A + FACTS_B
    payload = {"questions": [{"id": "g6", "text": "What is Project Aurora's internal codename?", "score_type": "open"}],
               "config": config, "context": corpus}
    t0 = time.time()
    try:
        body = infer_long(payload, timeout=600)
        r = (body.get("rows") or [{}])[0]
        print(f"GPU_OK {time.time()-t0:.1f}s :: {str(r.get('answer'))[:150]}", flush=True)
    except Exception as exc:  # noqa: BLE001
        print(f"GPU_FAIL {time.time()-t0:.1f}s :: {exc}", flush=True)


if __name__ == "__main__":
    main()
