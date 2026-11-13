"""Generate REAL multimodal samples for the fusion attempt.

TTS: facebook mms-tts-eng (already on this machine, the model family used by
the repo's production bakes). Image: sd-turbo (also already local).
Features are computed from the actual waveforms / pixels (mel statistics and
pixel projections), 384-dim L2-normalized to match the Triad port width.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

HERE = Path(__file__).resolve().parent
MEDIA = HERE / "media"
MEDIA.mkdir(exist_ok=True)

SAMPLES = [
    {"id": "s1", "text": "Project Aurora is a solar drone program led by Dr. Mara Voss.",
     "prompt": "a solar-powered drone flying over a coastal city at dawn, photorealistic"},
    {"id": "s2", "text": "Project Borealis is a tidal sensor program led by Dr. Ilya Petrov.",
     "prompt": "an underwater tidal sensor on the seabed near rocky islands, photorealistic"},
    {"id": "s3", "text": "Project Aurora uses battery cells supplied by Helix Cells.",
     "prompt": "close-up of advanced battery cells in a laboratory, photorealistic"},
    {"id": "s4", "text": "Project Borealis uses pressure sensors supplied by Nordwave.",
     "prompt": "industrial pressure sensors on a research vessel deck, photorealistic"},
]

DIM = 384


def audio_features(wav_path: Path) -> list[float]:
    import numpy as np
    import soundfile as sf

    y, sr = sf.read(str(wav_path))
    if y.ndim > 1:
        y = y[:, 0]
    y = y.astype(np.float64)
    frame, hop, nfft = int(sr * 0.025), int(sr * 0.010), 512
    if len(y) < frame:
        y = np.pad(y, (0, frame - len(y)))
    frames = np.lib.stride_tricks.sliding_window_view(y, frame)[::hop]
    spec = np.abs(np.fft.rfft(frames * np.hanning(frame), n=nfft, axis=1)) + 1e-9

    def hz2mel(f):
        return 2595 * math.log10(1 + f / 700)

    def mel2hz(m):
        return 700 * (10 ** (m / 2595) - 1)

    n_mels, fmax = 64, sr / 2
    pts = [mel2hz(v) for v in
           [hz2mel(0) + i * (hz2mel(fmax) - hz2mel(0)) / (n_mels + 1) for i in range(n_mels + 2)]]
    bins = [min(nfft // 2, int(p / sr * nfft)) for p in pts]
    fb = np.zeros((n_mels, spec.shape[1]))
    for i in range(n_mels):
        a, b, c = bins[i], bins[i + 1], bins[i + 2]
        for k in range(a, b):
            fb[i, k] = (k - a) / max(1, b - a)
        for k in range(b, c):
            fb[i, k] = (c - k) / max(1, c - b)
    mel = np.log(spec @ fb.T + 1e-6)
    feats = np.concatenate([mel.mean(axis=0), mel.std(axis=0)]).tolist()  # 128
    feats += [float(np.mean(y)), float(np.std(y)),
              float(np.mean(np.abs(np.diff(y)))), float(np.max(np.abs(y)))]
    feats = (feats + [0.0] * DIM)[:DIM]
    norm = math.sqrt(sum(v * v for v in feats)) or 1.0
    return [v / norm for v in feats]


def image_features(png_path: Path) -> list[float]:
    import numpy as np
    from PIL import Image

    img = Image.open(png_path).convert("RGB").resize((16, 8))
    feats = (np.asarray(img, dtype=np.float64) / 255.0).reshape(-1).tolist()  # 384
    norm = math.sqrt(sum(v * v for v in feats)) or 1.0
    return [v / norm for v in feats]


def gen_tts() -> None:
    import soundfile as sf
    import torch
    from transformers import AutoTokenizer, VitsModel

    model_id = r"C:\Ai\models\mms-tts-eng"
    tok = AutoTokenizer.from_pretrained(model_id)
    model = VitsModel.from_pretrained(model_id)
    model.eval()
    for s in SAMPLES:
        out = MEDIA / f"{s['id']}.wav"
        if out.exists():
            continue
        inputs = tok(s["text"], return_tensors="pt")
        with torch.no_grad():
            wav = model(**inputs).waveform[0].numpy()
        sf.write(str(out), wav, samplerate=model.config.sampling_rate)
        print(f"tts {out.name} sr={model.config.sampling_rate} n={len(wav)}", flush=True)


def main() -> None:
    gen_tts()
    feats = {}
    for s in SAMPLES:
        wav, png = MEDIA / f"{s['id']}.wav", MEDIA / f"{s['id']}.png"
        feats[s["id"]] = {
            "text": s["text"],
            "audio_features": audio_features(wav),
            "image_features": image_features(png) if png.exists() else None,
        }
        print(f"features {s['id']} audio=ok image={'ok' if png.exists() else 'pending'}", flush=True)
    (MEDIA / "features.json").write_text(json.dumps(feats), encoding="utf-8")
    print("MEDIA_DONE", flush=True)


if __name__ == "__main__":
    main()
