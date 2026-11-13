"""python -m valhalla_server            serve the API
python -m valhalla_server provision    encrypt the Valhalla engines
python -m valhalla_server build [id]   run decompose->fusion->bake->publish once
"""

from __future__ import annotations

import sys
import time

from valhalla_server import runtime
from valhalla_server.app import serve
from valhalla_server.pipeline import DEFAULT_SAMPLES, job_status, start_pipeline


def build(model_id: str) -> int:
    job = start_pipeline(model_id, "", 0.08, DEFAULT_SAMPLES)
    job_id = job["job_id"]
    while True:
        row = job_status(job_id)
        print(f"{row.get('status')} stage={row.get('stage')}", flush=True)
        if row.get("status") == "done":
            print(f"endpoint {row.get('endpoint')}")
            return 0
        if row.get("status") == "failed":
            print(f"error {row.get('error')}")
            return 1
        time.sleep(5)


def modelops(argv: list[str]) -> int:
    import argparse

    from valhalla_server.modelops import start_modelops
    from valhalla_server.pipeline import job_status as _status

    parser = argparse.ArgumentParser(prog="modelops")
    parser.add_argument("head_id")
    parser.add_argument("--base", required=True)
    parser.add_argument("--strength", type=float, default=0.08)
    parser.add_argument("--quant", default="Q4_K_M")
    parser.add_argument("--no-deploy", action="store_true")
    args = parser.parse_args(argv)
    job = start_modelops(args.head_id, args.base, args.strength, args.quant,
                         [], not args.no_deploy)
    job_id = job["job_id"]
    while True:
        row = _status(job_id)
        print(f"{row.get('status')} stage={row.get('stage')}", flush=True)
        if row.get("status") == "done":
            print(f"gguf {row.get('gguf')} verify={row.get('verify')}")
            return 0
        if row.get("status") == "failed":
            print(f"error {row.get('error')}")
            return 1
        time.sleep(10)


def main() -> None:
    if len(sys.argv) > 1 and sys.argv[1] == "provision":
        for row in runtime.provision():
            print(f"{row['engine']} {row['format']} bytes={row['bytes']}")
        return
    if len(sys.argv) > 1 and sys.argv[1] == "build":
        raise SystemExit(build(sys.argv[2] if len(sys.argv) > 2 else "unified-core-v1"))
    if len(sys.argv) > 1 and sys.argv[1] == "modelops":
        raise SystemExit(modelops(sys.argv[2:]))
    serve()


if __name__ == "__main__":
    main()

