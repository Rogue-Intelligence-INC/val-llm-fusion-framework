"""Strength sweep point s=0.02: bake -> convert -> quantize -> swap head ->
solo benchmark. Restores the s0.08 head afterwards if SWEEP_RESTORE=1."""
import os
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
PY = sys.executable
BASE = r"C:\Ai\models\Mistral-Small-3.1-24B-Instruct-2503"
EXPORT = str(HERE / "mistral24" / "fusion_export.json")
CORPUS = str(HERE / "mistral24" / "corpus.txt")
OUT = r"C:\Ai\models\Mistral-Small-3.1-24B-Valhalla-s0.02"
CONV = r"C:\Ai\tools\llama-src\llama.cpp-b12345\convert_hf_to_gguf.py"
QUANT = r"C:\Ai\tools\llama.cpp\llama-quantize.exe"
SERVER = r"C:\Ai\tools\llama.cpp\llama-server.exe"
PATCHED_Q4 = r"C:\Ai\models\Mistral-Small-3.1-24B-Valhalla-s0.08-Q4_K_M.gguf"


def run(cmd, timeout=1800):
    done = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8",
                          errors="replace", timeout=timeout, cwd=str(ROOT))
    if done.returncode != 0:
        raise RuntimeError((done.stderr or done.stdout)[-500:])
    return done.stdout


def serve(model: str):
    subprocess.run(["taskkill", "/F", "/IM", "llama-server.exe"],
                   capture_output=True)
    time.sleep(3)
    subprocess.Popen([SERVER, "-m", model, "-ngl", "99", "-c", "16384",
                      "--cache-type-k", "q8_0", "--cache-type-v", "q8_0",
                      "--host", "127.0.0.1", "--port", "18080",
                      "--alias", "mistral-24b"],
                     stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    for _ in range(60):
        try:
            with urllib.request.urlopen("http://127.0.0.1:18080/health", timeout=5) as r:
                if r.status == 200:
                    return
        except Exception:
            time.sleep(2)
    raise RuntimeError("llama-server did not come up")


def main():
    if not Path(OUT).exists():
        print(run([PY, "-m", "valhalla_server.bake_mistral", "--base", BASE,
                   "--export", EXPORT, "--out", OUT, "--strength", "0.02",
                   "--corpus", CORPUS]), flush=True)
    f16 = OUT + "-F16.gguf"
    q4 = OUT + "-Q4_K_M.gguf"
    if not Path(q4).exists():
        print(run([PY, CONV, OUT, "--outfile", f16, "--outtype", "f16"]), flush=True)
        print(run([QUANT, f16, q4, "Q4_K_M"]), flush=True)
    serve(q4)
    print(run([PY, str(HERE / "run_unique_points.py"), "--mistral-only"], timeout=1800), flush=True)
    if os.environ.get("SWEEP_RESTORE", "1") == "1":
        serve(PATCHED_Q4)
        print("RESTORED s0.08 head", flush=True)
    print("SWEEP_DONE", flush=True)


if __name__ == "__main__":
    main()
