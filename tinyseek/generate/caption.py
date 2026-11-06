"""Caption real camera photos for domain-LoRA training.

    python -m tinyseek.generate.caption --coco data/esp32cam/kept_coco.json \
        --images data/esp32cam/raw --out data/esp32cam/captions.jsonl

Each caption is "<trigger>, <BLIP description>". The description names the content
(rooms, people, objects) so the LoRA attributes those to the text and learns only what
every caption shares, the trigger, as the camera's look: wide-angle distortion, sensor
noise, color and exposure. Generation then adds the trigger to its prompts.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from PIL import Image
from tqdm import tqdm

CAPTIONER = "Salesforce/blip-image-captioning-large"
TRIGGER = "esp32cam photo"


@torch.no_grad()
def caption_images(paths: list[Path], device: torch.device, batch_size: int = 8) -> list[str]:
    from transformers import BlipForConditionalGeneration, BlipProcessor

    processor = BlipProcessor.from_pretrained(CAPTIONER)
    model = BlipForConditionalGeneration.from_pretrained(CAPTIONER).to(device).eval()
    captions = []
    for start in tqdm(range(0, len(paths), batch_size), desc="captions", mininterval=10):
        images = [Image.open(p).convert("RGB") for p in paths[start : start + batch_size]]
        inputs = processor(images=images, return_tensors="pt").to(device)
        out = model.generate(**inputs, max_new_tokens=40, num_beams=3)
        captions += [c.strip() for c in processor.batch_decode(out, skip_special_tokens=True)]
    return captions


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--coco", type=Path, required=True, help="COCO image list of the photos to caption")
    parser.add_argument("--images", type=Path, required=True, help="image root the file names are relative to")
    parser.add_argument("--out", type=Path, required=True, help="JSONL: file, description, caption")
    parser.add_argument("--trigger", default=TRIGGER)
    args = parser.parse_args()

    files = [img["file_name"] for img in json.loads(args.coco.read_text())["images"]]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    descriptions = caption_images([args.images / f for f in files], device)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w") as f:
        for file, desc in zip(files, descriptions):
            f.write(json.dumps({"file": file, "description": desc, "caption": f"{args.trigger}, {desc}"}) + "\n")
    print(f"{len(files)} captions -> {args.out}")


if __name__ == "__main__":
    main()
