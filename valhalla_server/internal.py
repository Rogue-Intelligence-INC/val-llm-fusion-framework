"""Internal ValhallaBase runtime (newer unified-core deployment pattern).

ValhallaBase is the body; the baked LM is only a decode head. The gateway
decrypts valhalla_base.exe from its VALENC1 copy, runs `serve` on an
internal port, and talks valhalla-base-v1 to it. The subprocess is stopped
when this process exits.
"""

from __future__ import annotations

import atexit
import json
import os
import subprocess
import threading
import time
import urllib.request

from valhalla_server import runtime
from valhalla_server.config import INTERNAL_PORT, ROOT

_proc: subprocess.Popen | None = None
_lock = threading.Lock()
_sessions: dict[str, dict] = {}
_corpora: dict[str, list[str]] = {}


def _health() -> bool:
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{INTERNAL_PORT}/health", timeout=2) as resp:
            return resp.status == 200
    except Exception:
        return False


def start(lm_model_dir: str) -> bool:
    global _proc
    with _lock:
        if _proc is not None and _proc.poll() is None and _health():
            return True
        try:
            exe = runtime.engine("valhalla_base.exe")
        except Exception:
            return False
        env = {
            **os.environ,
            "VALHALLA_LM_MODEL": lm_model_dir,
            "VALHALLA_DYNAMIC_CORPUS_FRAGMENTS": "1",
            # Bundling (fate_link): measured 40 -> 5 fragments, wire -20%,
            # QA unchanged (tests/measure_fate_link.py). Default group 2
            # only halves the count; 8 is the adopted operating point.
            "VALHALLA_FATE_LINK_MAX_GROUP": os.environ.get(
                "VALHALLA_FATE_LINK_MAX_GROUP", "8"),
        }
        _proc = subprocess.Popen(
            [str(exe), "serve", "--port", str(INTERNAL_PORT)],
            cwd=ROOT,
            env=env,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        atexit.register(stop)
        for _ in range(60):
            if _health():
                return True
            if _proc.poll() is not None:
                return False
            time.sleep(0.5)
        return False


def stop() -> None:
    global _proc
    if _proc is not None and _proc.poll() is None:
        _proc.terminate()
        try:
            _proc.wait(timeout=5)
        except Exception:
            _proc.kill()
    _proc = None


def infer(payload: dict) -> dict:
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        f"http://127.0.0.1:{INTERNAL_PORT}/v1/infer",
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=300) as resp:
        body = json.loads(resp.read().decode("utf-8"))
    if not isinstance(body, dict):
        raise RuntimeError("ValhallaBase 返回的不是 JSON。")
    return body


def chat_config(lm_model_dir: str) -> dict:
    """Production knobs from the unified-core stack table, completed with the
    hybrid routing policy from valhalla-transformer (open->LM decode,
    mcq->max-pick of structure/LM tracks, Fate warmup + orchestration on)."""
    return {
        "decode": "hybrid",
        "lm_model": lm_model_dir,
        "follow_up_decode": "native_follow_up_aware",
        "fate_ingress_routing": "quad_cycle",
        "fate_qa_feedback": True,
        "core_fusion": True,
        "open_fate": True,
        "answer_mix": "mixed",
        "mcq_decode": "max_pick",
        "incubation_cycles": 2,
        "fate_orchestrate": True,
        "fate_warmup_enabled": True,
        "use_cpu": False,
        "proactive_enabled": True,
        "max_new": 128,
    }


def slim_session(session: dict) -> dict:
    """Wire dedupe (main-project redundancy-removal discipline): the
    engine's HTTP paths rebuild state from `snapshot` + `corpus` and
    never read the top-level `fragments` copy, which duplicates
    `snapshot.fragments` byte for byte (~half the wire size). Empty it
    (serde-defaulted engine-side), drop any rollback stack, and cap
    question history at 32, mirroring valhalla_base.py's
    slim_session_for_wire."""
    out = dict(session)
    snap = out.get("snapshot")
    if isinstance(snap, dict) and snap.get("fragments"):
        out["fragments"] = []
    out.pop("fate_rollback_stack", None)
    qh = out.get("question_history")
    if isinstance(qh, list) and len(qh) > 32:
        out["question_history"] = qh[-32:]
    return out


LINK_MIN_FRAGMENTS = int(os.environ.get("VALHALLA_LINK_MIN_FRAGMENTS", "16"))


def link_session(session: dict | None, config: dict) -> dict | None:
    """Bundle a session's seg fragments into link chains via the
    engine's fate_link_only request (pipeline -> bundle). Returns the
    linked session, or None on any failure (caller keeps the old one)."""
    if not isinstance(session, dict):
        return None
    try:
        body = infer({"questions": [], "fate_link_only": True,
                      "session": slim_session(session), "config": config})
    except Exception:  # noqa: BLE001 - linking is an optimization
        return None
    linked = body.get("session")
    return linked if isinstance(linked, dict) else None


def bundle_lines(lines: list[str], threshold: float = 0.85) -> list[str]:
    """Ingest bundling (incubation-project bundling discipline): lines
    whose token sets are near-identical (Jaccard >= threshold) form one
    bundle; only the first (strongest, caller-ordered) representative is
    kept. Used by targeted-ingest callers before append so near-duplicate
    template lines never grow the session."""
    kept: list[set] = []
    out: list[str] = []
    for line in lines:
        toks = set(line.lower().split())
        if any(toks and len(toks & k) / max(1, len(toks | k)) >= threshold
               for k in kept):
            continue
        kept.append(toks)
        out.append(line)
    return out


_media_items: dict[str, list[dict]] = {}


def keep_media_items(key: str, items: list[dict]) -> None:
    """Registry of co-ingested media items per model, in engine corpus
    order (only text-bearing items occupy a corpus line)."""
    if items:
        _media_items[key] = items


def media_items(key: str) -> list[dict]:
    return list(_media_items.get(key) or [])


def session_for(key: str) -> dict | None:
    return _sessions.get(key)


def drop_session(key: str) -> None:
    _sessions.pop(key, None)
    _corpora.pop(key, None)


def keep_session(key: str, session: dict | None) -> None:
    if isinstance(session, dict):
        _sessions[key] = slim_session(session)


def keep_corpus(key: str, lines: list[str]) -> None:
    if lines:
        _corpora[key] = list(lines)


def corpus_line(key: str, idx: object) -> str | None:
    lines = _corpora.get(key) or []
    if isinstance(idx, int) and 0 <= idx < len(lines):
        return lines[idx]
    return None


def corpus_lines(key: str) -> list[str]:
    return list(_corpora.get(key) or [])
