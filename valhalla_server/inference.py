"""Serve one baked checkpoint. Weights are decrypted from the vault on load.

The plaintext copy lives under runtime/.cache/<model_id> only while this
process holds the model. Run with the Valhalla venv python (torch).
"""

from __future__ import annotations

import atexit
import shutil
import threading
from pathlib import Path

from valhalla_server.config import CACHE, VAULT
from valhalla_server.pipeline import model_row
from valhalla_server.valenc import decrypt_file, load_key

_lock = threading.Lock()
_loaded: dict[str, tuple] = {}


def _cleanup_cache() -> None:
    for model_id in list(_loaded):
        weights = CACHE / model_id / "model.safetensors"
        try:
            weights.unlink(missing_ok=True)
        except OSError:
            pass


atexit.register(_cleanup_cache)


def _materialize(model_id: str) -> Path:
    dest = CACHE / model_id
    weights = dest / "model.safetensors"
    if weights.is_file():
        return dest
    vault = VAULT / model_id
    enc = vault / "model.safetensors.enc"
    if not enc.is_file():
        raise FileNotFoundError(f"模型 {model_id} 还没有加密权重。")
    dest.mkdir(parents=True, exist_ok=True)
    for item in vault.iterdir():
        if item.is_file() and item.suffix != ".enc":
            shutil.copy(item, dest / item.name)
    decrypt_file(enc, weights, load_key())
    if weights.read_bytes()[:8] == b"VALENC1\x00":
        weights.unlink(missing_ok=True)
        raise RuntimeError("权重解密失败。")
    return dest


def _load(model_id: str):
    with _lock:
        if model_id in _loaded:
            return _loaded[model_id]
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        folder = _materialize(model_id)
        tokenizer = AutoTokenizer.from_pretrained(str(folder))
        model = AutoModelForCausalLM.from_pretrained(str(folder), torch_dtype=torch.float32)
        model.eval()
        pair = (tokenizer, model, torch)
        _loaded[model_id] = pair
        return pair


def generate(model_id: str, messages: list[dict], max_tokens: int = 256, temperature: float = 0.3) -> str:
    row = model_row(model_id)
    tokenizer, model, torch = _load(model_id)
    try:
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    except Exception:
        prompt = "\n".join(f"{m.get('role', 'user')}: {m.get('content', '')}" for m in messages) + "\nassistant:"
    inputs = tokenizer(prompt, return_tensors="pt")
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=max(1, min(int(max_tokens), 1024)),
            do_sample=temperature > 0,
            temperature=max(temperature, 0.05),
            pad_token_id=tokenizer.eos_token_id,
        )
    text = tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    return text.strip(), row
