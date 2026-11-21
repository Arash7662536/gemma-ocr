"""Page rendering and image preparation (no model calls)."""
from __future__ import annotations

import base64
import io
from pathlib import Path

import pymupdf
from PIL import Image, ImageFilter, ImageOps

from .config import Config

SUPPORTED = (".pdf", ".jpg", ".jpeg", ".png")
ROTATE = {90: Image.ROTATE_90, 180: Image.ROTATE_180, 270: Image.ROTATE_270}


def load_pages(path: Path, dpi: int = 300) -> list[dict]:
    """[{'page': n, 'img': PIL.Image, 'text_layer': str}] for a PDF (rendered at `dpi`) or an image file."""
    if path.suffix.lower() == ".pdf":
        doc = pymupdf.open(path)
        out = []
        for i, pg in enumerate(doc):
            pix = pg.get_pixmap(dpi=dpi)
            img = Image.open(io.BytesIO(pix.tobytes("png"))).convert("RGB")
            out.append({"page": i + 1, "img": img, "text_layer": pg.get_text()})
        return out
    img = ImageOps.exif_transpose(Image.open(path)).convert("RGB")
    return [{"page": 1, "img": img, "text_layer": ""}]


def rotate(img: Image.Image, degrees_ccw: int) -> Image.Image:
    return img.transpose(ROTATE[degrees_ccw]) if degrees_ccw else img


def crop_margins(img: Image.Image, threshold: int = 200, pad: int = 24) -> Image.Image:
    """Trim the white / scanner border so the document fills the frame."""
    g = img.convert("L").filter(ImageFilter.MedianFilter(5))
    mask = g.point(lambda v: 255 if v < threshold else 0)
    box = mask.getbbox()
    if not box:
        return img
    x0, y0, x1, y1 = box
    x0, y0 = max(0, x0 - pad), max(0, y0 - pad)
    x1, y1 = min(img.width, x1 + pad), min(img.height, y1 + pad)
    if (x1 - x0) * (y1 - y0) < 0.25 * img.width * img.height:
        return img  # suspicious crop (mostly blank page) -> keep as is
    return img.crop((x0, y0, x1, y1))


def fit(img: Image.Image, max_px: int) -> Image.Image:
    """Downscale (never upscale) to at most max_px pixels."""
    if img.width * img.height <= max_px:
        return img
    s = (max_px / (img.width * img.height)) ** 0.5
    return img.resize((int(img.width * s), int(img.height * s)), Image.LANCZOS)


def upscale_small(img: Image.Image, below: int, width: int, max_scale: float) -> Image.Image:
    """Small scans/screenshots: upscale so strips give Gemma more pixels per character."""
    if img.width >= below:
        return img
    s = min(max_scale, width / img.width)
    return img.resize((int(img.width * s), int(img.height * s)), Image.LANCZOS)


def strips(img: Image.Image, max_px: int, overlap: float = 0.18, max_n: int = 6, min_n: int = 0) -> list[Image.Image]:
    """Full-width horizontal strips, each <= max_px so Gemma sees it at native resolution.

    A page that already fits one image (<= 1.15 x max_px) gets no strips unless min_n > 0."""
    W, H = img.size
    if W * H <= max_px * 1.15:
        if min_n < 2:
            return []
        n = min_n
        h = int(H / (n - (n - 1) * overlap))
    else:
        h = max(400, int(max_px / W))
        step = int(h * (1 - overlap))
        n = 1 + max(0, -(-(H - h) // step))
        if n > max_n:  # page too tall for native res: use max_n strips, Gemma downsizes each a bit
            n = max_n
            h = int(H / (n - (n - 1) * overlap))
    step = int(h * (1 - overlap))
    out = []
    for i in range(n):
        y0 = min(i * step, H - h)
        out.append(img.crop((0, max(0, y0), W, min(H, y0 + h))))
    return out


def page_images(img: Image.Image, cfg: Config, use_strips: bool = True) -> tuple[list[Image.Image], int]:
    """[overview] + strips. The overview is the whole page downscaled to the per-image budget."""
    ss = []
    if use_strips:
        big = upscale_small(img, cfg.upscale_below_width, cfg.upscale_width, cfg.upscale_max)
        ss = strips(big, cfg.px_per_image, cfg.strip_overlap, cfg.max_strips, cfg.min_strips)
    return [fit(img, cfg.px_per_image)] + ss, len(ss)


def jpeg_url(img: Image.Image, q: int = 92) -> str:
    b = io.BytesIO()
    img.save(b, "JPEG", quality=q)
    return "data:image/jpeg;base64," + base64.b64encode(b.getvalue()).decode()


def image_parts(imgs: list[Image.Image], q: int = 92) -> list[dict]:
    return [{"type": "image_url", "image_url": {"url": jpeg_url(i, q)}} for i in imgs]
