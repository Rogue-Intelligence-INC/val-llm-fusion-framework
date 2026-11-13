"""Media encoding for the multimodal fix (official bridge contract).

The engine only accepts 384-d normalized feature vectors and binds
modalities by CO-INGESTION: text + audio + image of one item pulse
into the same fragment. The production encoder is the repo's DL-front
sidecar (tools/valhalla_dl_front/encode.py); we call its own functions
in the repo venv so our features live in exactly the family the engine
expects (mel-band audio, ResNet/grayscale image, pad/truncate 384).
Results are cached by file hash under data/.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import tempfile
from pathlib import Path

from valhalla_server.config import DATA, VALHALLA_HOME, VENV_PY

ENCODE_PY = VALHALLA_HOME / "tools" / "valhalla_dl_front" / "encode.py"
_CACHE_FILE = DATA / "media_feature_cache.json"

_RUNNER = r"""
import importlib.util, json, sys
spec = importlib.util.spec_from_file_location("dlfront", sys.argv[1])
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
kind, path = sys.argv[2], sys.argv[3]
if kind == "audio":
    emb, enc = mod.encode_audio_path(path)
else:
    emb, enc = mod.encode_image_path(path)
print(json.dumps({"embedding": emb, "encoder": enc}))
"""

# v2 encoders (measured reason): encode.py's fallbacks have no grasp
# discrimination -- grayscale image features sit at cos 0.87-0.97 to
# each other and mean-mel audio features at cos 1.000 for distinct
# sentences. Images use a real learned vision encoder, audio uses
# segment-temporal mel statistics (24 bands x 16 segments = 384),
# which preserves phoneme sequence shape instead of averaging it away.
SD_PY = Path(r"C:\Ai\models\.venv-sd\Scripts\python.exe")
CLIP_VISION = Path(r"C:\Ai\models\ip-adapter\models\image_encoder")

# ResNet18 weights are not downloadable on this network (proxy SSL),
# so images use the locally installed CLIP ViT-H/14 vision encoder
# (ip-adapter bundle): 1024-d image_embeds, fixed seeded Gaussian
# projection to the engine's 384 contract, then normalized.
_IMAGE_RUNNER = r"""
import json, sys
import numpy as np
import torch
from PIL import Image
from transformers import CLIPVisionModelWithProjection
img = Image.open(sys.argv[1]).convert("RGB")
img = img.resize((224, 224))
x = torch.from_numpy(np.asarray(img, dtype=np.float32) / 255.0).permute(2, 0, 1)
mean = torch.tensor([0.48145466, 0.4578275, 0.40821073]).view(3, 1, 1)
std = torch.tensor([0.26862954, 0.26130258, 0.27577711]).view(3, 1, 1)
x = ((x - mean) / std).unsqueeze(0)
net = CLIPVisionModelWithProjection.from_pretrained(sys.argv[2])
net.eval()
with torch.no_grad():
    emb = net(pixel_values=x).image_embeds.squeeze().numpy().astype(np.float64)
rng = np.random.default_rng(20261113)
proj = rng.standard_normal((len(emb), 384)) / np.sqrt(384)
out = emb @ proj
n = float(np.linalg.norm(out)) or 1e-9
print(json.dumps({"embedding": (out / n).tolist(), "encoder": "clip-vit-h14-proj384"}))
"""

_AUDIO_RUNNER = r"""
import json, sys
import numpy as np
from scipy.io import wavfile
rate, data = wavfile.read(sys.argv[1])
x = data.astype(np.float64)
if x.ndim > 1:
    x = x.mean(axis=1)
if np.max(np.abs(x)) > 1.5:
    x = x / (np.max(np.abs(x)) + 1e-9)
BANDS, SEGS, NFFT, HOP = 24, 16, 512, 256
if len(x) < NFFT:
    x = np.pad(x, (0, NFFT - len(x)))
frames = []
for s in range(0, len(x) - NFFT + 1, HOP):
    frames.append(np.abs(np.fft.rfft(x[s:s + NFFT] * np.hanning(NFFT))) ** 2)
P = np.stack(frames)                       # T x F power spectra
edges = np.linspace(0, P.shape[1], BANDS + 1, dtype=int)
mel = np.stack([np.log1p(P[:, edges[i]:edges[i + 1]].mean(axis=1) + 1e-9)
                for i in range(BANDS)], axis=1)          # T x BANDS
seg_idx = np.linspace(0, len(mel), SEGS + 1, dtype=int)
emb = []
for i in range(SEGS):
    emb.extend(mel[seg_idx[i]:max(seg_idx[i] + 1, seg_idx[i + 1])].mean(axis=0).tolist())
emb = (emb + [0.0] * 384)[:384]
n = sum(v * v for v in emb) ** 0.5 or 1e-9
print(json.dumps({"embedding": [v / n for v in emb], "encoder": "wav_mel_segments24x16"}))
"""


def _cache() -> dict:
    try:
        return json.loads(_CACHE_FILE.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001
        return {}


def encode_file(path: str | Path, kind: str) -> list[float]:
    """kind: 'audio' | 'image'. Returns 384 floats (engine contract).

    v2 routing: images via torchvision ResNet18 when available (the
    official sidecar's real encoder; its grayscale fallback has no
    grasp discrimination), audio via segment-temporal mel statistics.
    """
    p = Path(path)
    digest = hashlib.sha256(p.read_bytes()).hexdigest()
    cache = _cache()
    key = f"v2:{kind}:{digest}"
    if key in cache:
        return cache[key]
    if kind == "image" and SD_PY.exists() and CLIP_VISION.exists():
        args = [str(SD_PY), "-c", _IMAGE_RUNNER, str(p), str(CLIP_VISION)]
    elif kind == "audio":
        args = [str(VENV_PY), "-c", _AUDIO_RUNNER, str(p)]
    else:
        args = [str(VENV_PY), "-c", _RUNNER, str(ENCODE_PY), kind, str(p)]
    out = subprocess.run(args, capture_output=True, text=True, timeout=600)
    if out.returncode != 0:
        raise RuntimeError(f"媒体编码失败：{out.stderr[-500:]}")
    emb = json.loads(out.stdout.strip().splitlines()[-1])["embedding"]
    cache[key] = emb
    _CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)
    _CACHE_FILE.write_text(json.dumps(cache), encoding="utf-8")
    return emb


def cosine(a: list[float], b: list[float]) -> float:
    n = min(len(a), len(b))
    if not n:
        return 0.0
    dot = sum(a[i] * b[i] for i in range(n))
    na = sum(x * x for x in a[:n]) ** 0.5 or 1e-9
    nb = sum(x * x for x in b[:n]) ** 0.5 or 1e-9
    return dot / (na * nb)


def grasp(items: list[dict], modality: str, query_features: list[float],
          threshold: float = 0.85, margin: float = 0.05) -> tuple[int, float] | None:
    """Feature grasp (product layer): the engine binds modalities by
    co-ingestion but has no feature->line retrieval, so the gateway
    grasps in encoder space -- same-family cosine over ingested item
    features. Only a clear winner (score >= threshold and ahead of the
    runner-up by margin) counts as a grasp; anything else is silence."""
    key = f"{modality}_features"
    scored = []
    for i, it in enumerate(items):
        feats = it.get(key)
        if isinstance(feats, list) and feats:
            scored.append((cosine(query_features, feats), i))
    if not scored:
        return None
    scored.sort(reverse=True)
    best, idx = scored[0]
    second = scored[1][0] if len(scored) > 1 else 0.0
    if best >= threshold and best - second >= margin:
        return idx, best
    return None


def encode_bytes(data: bytes, kind: str, suffix: str) -> list[float]:
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix,
                                     dir=tempfile.gettempdir()) as fh:
        fh.write(data)
        tmp = fh.name
    try:
        return encode_file(tmp, kind)
    finally:
        Path(tmp).unlink(missing_ok=True)
