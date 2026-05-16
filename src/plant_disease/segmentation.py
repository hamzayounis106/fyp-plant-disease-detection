from __future__ import annotations

from io import BytesIO
from typing import Optional, Tuple

import cv2
import numpy as np
from PIL import Image

try:
    from rembg import remove
except BaseException:  # pragma: no cover
    remove = None


def _rgba_to_rgb_white_bg(rgba: Image.Image) -> Image.Image:
    if rgba.mode != "RGBA":
        return rgba.convert("RGB")
    background = Image.new("RGBA", rgba.size, (255, 255, 255, 255))
    composed = Image.alpha_composite(background, rgba)
    return composed.convert("RGB")


def segment_with_rembg(image: Image.Image) -> Optional[Image.Image]:
    if remove is None:
        return None
    try:
        buf = BytesIO()
        image.save(buf, format="PNG")
        output_bytes = remove(buf.getvalue())
        out = Image.open(BytesIO(output_bytes)).convert("RGBA")
        return _rgba_to_rgb_white_bg(out)
    except Exception:
        return None


def segment_with_grabcut(image: Image.Image, iter_count: int = 5, max_side: int = 1024) -> Image.Image:
    # Cap dimensions before grabCut — natural-photo datasets (e.g. PlantDoc)
    # contain 4k+ images that overflow cv2's allocator on systems with small pagefiles.
    img = image.convert("RGB")
    w, h = img.size
    scale = max_side / max(w, h)
    if scale < 1.0:
        img = img.resize((int(w * scale), int(h * scale)), Image.BILINEAR)
    arr = np.array(img)
    bgr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
    mask = np.zeros(bgr.shape[:2], np.uint8)

    bg_model = np.zeros((1, 65), np.float64)
    fg_model = np.zeros((1, 65), np.float64)
    h, w = bgr.shape[:2]
    rect = (max(1, w // 20), max(1, h // 20), max(1, w - (w // 10)), max(1, h - (h // 10)))

    cv2.grabCut(bgr, mask, rect, bg_model, fg_model, iter_count, cv2.GC_INIT_WITH_RECT)
    mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype("uint8")

    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    out = rgb * mask2[:, :, np.newaxis]
    white_bg = np.full_like(out, 255)
    merged = np.where(mask2[:, :, np.newaxis] == 1, out, white_bg)
    return Image.fromarray(merged)


def foreground_ratio(image: Image.Image) -> float:
    arr = np.array(image.convert("RGB"))
    bg_like = np.all(arr > 245, axis=-1)
    fg = (~bg_like).sum()
    total = arr.shape[0] * arr.shape[1]
    return float(fg / max(total, 1))


def segment_leaf(
    image: Image.Image,
    min_fg_ratio: float = 0.03,
    use_rembg: bool = True,
) -> Tuple[Image.Image, str]:
    if use_rembg:
        rembg_out = segment_with_rembg(image)
        if rembg_out is not None and foreground_ratio(rembg_out) >= min_fg_ratio:
            return rembg_out, "rembg"

    grabcut_out = segment_with_grabcut(image)
    return grabcut_out, "grabcut"
