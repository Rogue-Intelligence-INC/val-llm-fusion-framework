"""Generate the 4 sample images with sd-turbo (run with the SD venv)."""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from gen_media import MEDIA, SAMPLES  # noqa: E402


def main():
    import torch
    from diffusers import AutoPipelineForText2Image

    pipe = AutoPipelineForText2Image.from_pretrained(
        r"C:\Ai\models\sd-turbo", torch_dtype=torch.float16, variant="fp16")
    pipe.to("cuda")
    for s in SAMPLES:
        out = MEDIA / f"{s['id']}.png"
        if out.exists():
            continue
        img = pipe(prompt=s["prompt"], num_inference_steps=4,
                   guidance_scale=0.0).images[0]
        img.save(out)
        print(f"img {out.name} {img.size}", flush=True)
    print("IMG_DONE", flush=True)


if __name__ == "__main__":
    main()
