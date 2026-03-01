"""
save_downsampled.py — Save bicubic-downsampled versions of one image at all token budgets.

Usage:
    python MVV/Phase_1_1/shared/save_downsampled.py
"""

from pathlib import Path
from PIL import Image

BUDGETS = {
    729: (378, 378),
    441: (294, 294),
    256: (224, 224),
    121: (154, 154),
}

SRC = Path(__file__).parent.parent / "data_mvv" / "images" / "black__action__main_py.png"
OUT = Path(__file__).parent.parent / "data_mvv" / "images" / "downsampled"

OUT.mkdir(exist_ok=True)

img = Image.open(SRC)
print(f"Source: {SRC.name}  ({img.size[0]}×{img.size[1]}px, mode={img.mode})")

for budget, (h, w) in BUDGETS.items():
    resized = img.resize((w, h), Image.BICUBIC)
    out_path = OUT / f"black__action__main_py_budget{budget}_{w}x{h}.png"
    resized.save(out_path)
    print(f"  budget_{budget:3d} → {w}×{h}px  saved: {out_path.name}")

print(f"\nAll saved to {OUT}/")
