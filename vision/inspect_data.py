"""
Visualise a sample of the collected vision dataset.

Picks 16 random frames and renders them in a 4x4 grid with their labels
overlaid (local_x, local_y, visible). Saves to data/vision_sample.png.

This is a sanity check — before training the CNN we should confirm:
  - visible=1 frames really show an obstacle
  - visible=0 frames really don't (or obstacle is behind/out-of-frame)
  - local_x/y values look sensible for what's in the image
  - the distance distribution covers the close-range regime we wanted

Run from src/ or anywhere:
    python3 ../vision/inspect_data.py
"""

import argparse
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))

import numpy as np
from PIL import Image, ImageDraw, ImageFont


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str,
                        default=os.path.join(ROOT, "data", "vision_dataset.npz"))
    parser.add_argument("--out",  type=str,
                        default=os.path.join(ROOT, "data", "vision_sample.png"))
    parser.add_argument("--n",    type=int, default=16, help="frames to show (must be a perfect square)")
    args = parser.parse_args()

    grid_side = int(np.sqrt(args.n))
    if grid_side * grid_side != args.n:
        raise SystemExit("--n must be a perfect square (e.g. 9, 16, 25)")

    print(f"Loading {args.data}")
    d = np.load(args.data, allow_pickle=True)
    images, labels, dists = d["images"], d["labels"], d["dists"]
    print(f"  shape: images={images.shape}, labels={labels.shape}")
    print(f"  visible fraction:        {labels[:, 2].mean():.3f}")
    print(f"  distance mean (visible): "
          f"{dists[labels[:, 2] == 1.0].mean():.2f} m")
    print(f"  distance < 2m fraction:  {(dists < 2.0).mean():.3f}")

    rng = np.random.default_rng(0)
    idxs = rng.choice(len(images), size=args.n, replace=False)

    # Upscale each 84x84 frame to 168x168 so the labels are legible.
    tile = 168
    grid = np.zeros((grid_side * tile, grid_side * tile, 3), dtype=np.uint8)

    for k, i in enumerate(idxs):
        row, col = divmod(k, grid_side)
        img = Image.fromarray(images[i]).resize((tile, tile), Image.NEAREST)
        draw = ImageDraw.Draw(img)
        lx, ly, vis = labels[i]
        text = f"x={lx:+.1f} y={ly:+.1f}\nvis={int(vis)} d={dists[i]:.1f}"
        # White text with a dark shadow for legibility
        for dx, dy in ((1, 1), (-1, 1), (1, -1), (-1, -1)):
            draw.text((2 + dx, 2 + dy), text, fill=(0, 0, 0))
        draw.text((2, 2), text, fill=(255, 255, 255))
        grid[row * tile:(row + 1) * tile, col * tile:(col + 1) * tile] = np.array(img)

    Image.fromarray(grid).save(args.out)
    print(f"\nSaved sample sheet to {args.out}")


if __name__ == "__main__":
    main()