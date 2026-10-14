"""
demo/make_demo.py - Creates a tiny SYNTHETIC dataset + external set so the whole pipeline can be
smoke-tested in a couple of minutes on any CPU (no real model or dataset needed).

    python demo/make_demo.py            # creates demo/data/...
    python run_all.py --config demo/demo_config.py
"""
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFilter

HERE = Path(__file__).resolve().parent
COLORS = np.array([(200, 60, 60), (60, 200, 60), (60, 60, 200), (200, 200, 60), (60, 200, 200)], dtype=float)
SIZE = 96


def make_image(rng, label, mix=0.0, other=None, noise=0.04):
    other = (label + 1) % 5 if other is None else other
    col = (1 - mix) * COLORS[label] + mix * COLORS[other]
    arr = np.ones((SIZE, SIZE, 3)) * col
    img = Image.fromarray(arr.astype(np.uint8))
    d = ImageDraw.Draw(img)
    for _ in range(3):
        x, y, r = rng.integers(10, SIZE - 10, 2).tolist() + [int(rng.integers(5, 14))]
        shade = np.clip(col * rng.uniform(0.85, 1.15), 0, 255).astype(int).tolist()
        d.ellipse([x - r, y - r, x + r, y + r], fill=tuple(shade))
    a = np.asarray(img, dtype=np.float32) / 255.0 + rng.normal(0, noise, (SIZE, SIZE, 3))
    return Image.fromarray((np.clip(a, 0, 1) * 255).astype(np.uint8))


def main(n=700, n_ext=40, seed=0):
    rng = np.random.default_rng(seed)
    img_dir = HERE / "data" / "images"
    ext_dir = HERE / "data" / "external" / "images"
    img_dir.mkdir(parents=True, exist_ok=True)
    ext_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for i in range(n):
        lab = int(rng.integers(0, 5))
        mix = float(rng.beta(1.2, 3.0) * 0.75)               # ambiguity: how much of a neighbour class
        img = make_image(rng, lab, mix)
        true = lab if rng.random() > 0.04 else int(rng.integers(0, 5))   # ~4% label noise
        p = img_dir / f"img_{i:04d}.png"
        img.save(p)
        rows.append(dict(image_path=str(p), label=true))
    pd.DataFrame(rows).to_csv(HERE / "data" / "dataset.csv", index=False)

    ext_rows = []
    for i in range(n_ext):                                    # external = shifted domain (blur + colour cast)
        lab = int(rng.integers(0, 5))
        img = make_image(rng, lab, float(rng.beta(1.2, 3.0) * 0.6), noise=0.07).filter(ImageFilter.GaussianBlur(1.2))
        arr = np.asarray(img, dtype=np.float32)
        arr[..., 0] *= 1.18
        arr[..., 2] *= 0.82
        img = Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))
        name = f"ext_{i:03d}.png"
        img.save(ext_dir / name)
        ext_rows.append(dict(filename=name, label=lab))
    pd.DataFrame(ext_rows).to_csv(HERE / "data" / "external" / "labels.csv", index=False)
    print(f"[demo] wrote {n} dataset images and {n_ext} external images under {HERE / 'data'}")


if __name__ == "__main__":
    main()
