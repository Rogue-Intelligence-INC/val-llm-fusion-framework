import sys
from pathlib import Path

sys.path.insert(0, r"C:\Ai\rogue-products\valhalla-server")
from valhalla_server import media  # noqa: E402

MEDIA = Path(r"C:\Ai\rogue-products\valhalla-server\tests\media")
for kind, ext in (("image", "png"), ("audio", "wav")):
    feats = {f"s{i}": media.encode_file(MEDIA / f"s{i}.{ext}", kind) for i in range(1, 5)}
    print(kind)
    for a in feats:
        row = " ".join(f"{media.cosine(feats[a], feats[b]):.3f}" for b in feats)
        print(f"  {a}: {row}")
