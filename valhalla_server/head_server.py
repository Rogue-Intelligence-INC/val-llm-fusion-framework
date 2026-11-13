"""Decode-head server process management (llama.cpp llama-server).

One head at a time on HEAD_PORT. Starting a head replaces the previous one.
Used by modelops deploy stage and by POST /v1/decode-heads/<id>/deploy.
"""

from __future__ import annotations

import subprocess
import threading
import time
import urllib.request

from valhalla_server.config import HEAD_HOST, HEAD_PORT, LLAMA_BIN

_proc: subprocess.Popen | None = None
_lock = threading.Lock()
_current: str = ""


def _health() -> bool:
    try:
        with urllib.request.urlopen(f"http://{HEAD_HOST}:{HEAD_PORT}/health", timeout=3) as resp:
            return resp.status == 200
    except Exception:
        return False


def start(gguf_path: str, alias: str = "mistral-24b", ctx: int = 16384,
          wait_s: int = 180) -> bool:
    global _proc, _current
    with _lock:
        if _proc is not None and _proc.poll() is None:
            _proc.terminate()
            try:
                _proc.wait(timeout=10)
            except Exception:
                _proc.kill()
            _proc = None
        exe = LLAMA_BIN / "llama-server.exe"
        _proc = subprocess.Popen(
            [str(exe), "-m", gguf_path, "-ngl", "99", "-c", str(ctx),
             "--cache-type-k", "q8_0", "--cache-type-v", "q8_0",
             "--host", HEAD_HOST, "--port", str(HEAD_PORT), "--alias", alias],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
        _current = gguf_path
        for _ in range(max(1, wait_s // 2)):
            if _health():
                return True
            if _proc.poll() is not None:
                return False
            time.sleep(2)
        return False


def status() -> dict:
    return {"running": bool(_proc is not None and _proc.poll() is None),
            "healthy": _health(), "gguf": _current,
            "url": f"http://{HEAD_HOST}:{HEAD_PORT}/v1/chat/completions"}
