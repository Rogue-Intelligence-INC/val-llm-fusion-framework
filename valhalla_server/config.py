"""Paths for the server-side Valhalla pipeline. Nothing here points into client installs."""

from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
VALHALLA_HOME = Path(os.environ.get("VALHALLA_HOME", r"C:\Ai\Valhalla-main"))
VENV_PY = VALHALLA_HOME / ".venv-llm" / "Scripts" / "python.exe"
BASE_MODEL = os.environ.get("VALHALLA_BASE_MODEL", r"C:\Ai\models\Qwen2.5-0.5B-Instruct")

DATA = ROOT / "data"
VAULT = DATA / "vault"
JOBS = DATA / "jobs"
BUILDS = DATA / "builds"
REGISTRY = DATA / "models.json"
RUNTIME_ENC = ROOT / "runtime"
CACHE = RUNTIME_ENC / ".cache"

ENGINES = (
    "valhalla_base.exe",
    "valhalla_model_fate.exe",
    "valhalla_free_multimodal_incubate.exe",
    "valhalla_trimodal_incubate.exe",
    "valhalla_fused_lm.exe",
)

# Newer deployment pattern (valhalla-unified-core, 2026-07): ValhallaBase is
# the body, the LM is only a decode head. Internal Rust HTTP runtime port.
INTERNAL_PORT = int(os.environ.get("VALHALLA_INTERNAL_PORT", "8787"))

HOST = os.environ.get("VALHALLA_SERVER_HOST", "127.0.0.1")
PORT = int(os.environ.get("VALHALLA_SERVER_PORT", "8780"))
API_KEY = os.environ.get("VALHALLA_SERVER_KEY", "").strip()

# Automated model processing (modelops): decode-head toolchain.
LLAMA_BIN = Path(os.environ.get("VALHALLA_LLAMA_BIN", r"C:\Ai\tools\llama.cpp"))
LLAMA_SRC = Path(os.environ.get("VALHALLA_LLAMA_SRC", r"C:\Ai\tools\llama-src\llama.cpp-b12345"))
DECODE_HEADS = DATA / "decode_heads.json"
HEAD_HOST = os.environ.get("VALHALLA_HEAD_HOST", "127.0.0.1")
HEAD_PORT = int(os.environ.get("VALHALLA_HEAD_PORT", "18080"))
