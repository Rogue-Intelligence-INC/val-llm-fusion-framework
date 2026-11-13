"""Valhalla engines at rest are VALENC1. They are decrypted only inside this project.

provision() copies each built engine from the Valhalla checkout into
runtime/<name>.enc. engine() decrypts one into runtime/.cache for this
process and removes it at exit. Plaintext engines are never written here.
"""

from __future__ import annotations

import atexit
import os
import tempfile
import threading
from pathlib import Path

from valhalla_server.config import CACHE, ENGINES, RUNTIME_ENC, VALHALLA_HOME
from valhalla_server.valenc import decrypt_file, encrypt_file, load_key

_lock = threading.Lock()
_live: dict[str, Path] = {}


def _source(name: str) -> Path:
    path = VALHALLA_HOME / "target" / "release" / name
    if not path.is_file():
        raise FileNotFoundError(f"找不到引擎 {name}，先在 Valhalla 仓库构建它。")
    return path


def provision() -> list[dict]:
    RUNTIME_ENC.mkdir(parents=True, exist_ok=True)
    key = load_key()
    out = []
    for name in ENGINES:
        src = _source(name)
        dest = RUNTIME_ENC / f"{name}.enc"
        if not dest.is_file() or dest.stat().st_mtime < src.stat().st_mtime:
            encrypt_file(src, dest, key)
        out.append({"engine": name, "format": "valenc1", "bytes": dest.stat().st_size})
    return out


def _cleanup() -> None:
    for path in list(_live.values()):
        try:
            path.unlink(missing_ok=True)
        except OSError:
            pass
    _live.clear()


def engine(name: str) -> Path:
    with _lock:
        if name in _live and _live[name].is_file():
            return _live[name]
        enc = RUNTIME_ENC / f"{name}.enc"
        if not enc.is_file():
            provision()
        if not enc.is_file():
            raise FileNotFoundError(f"引擎没有加密副本：{name}")
        CACHE.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(prefix=f"{Path(name).stem}_{os.getpid()}_", suffix=".exe", dir=CACHE)
        os.close(fd)
        dest = Path(tmp)
        try:
            decrypt_file(enc, dest, load_key())
        except Exception:
            dest.unlink(missing_ok=True)
            raise
        if dest.read_bytes()[:2] != b"MZ":
            dest.unlink(missing_ok=True)
            raise RuntimeError(f"解密后的引擎不是可执行文件：{name}")
        _live[name] = dest
        atexit.register(_cleanup)
        return dest


def status() -> dict:
    rows = []
    for name in ENGINES:
        enc = RUNTIME_ENC / f"{name}.enc"
        rows.append({"engine": name, "encrypted": enc.is_file(), "bytes": enc.stat().st_size if enc.is_file() else 0})
    return {"format": "valenc1", "engines": rows}
