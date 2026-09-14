"""Shared helpers for the receipt pipeline: font caching, text metrics, noise fields.

Everything here is deliberately free of I/O side effects beyond reading font
files, so it can be imported cheaply at Cloud Run cold start (Phase 5 caches
fonts and overlays in-process via the lru_caches below).
"""

from __future__ import annotations

import math
from functools import lru_cache
from pathlib import Path
from typing import Iterable, Sequence

import cv2
import numpy as np
from PIL import Image, ImageFont

PROJECT_ROOT = Path(__file__).resolve().parents[2]
ASSETS_DIR = PROJECT_ROOT / "assets"
FONTS_DIR = ASSETS_DIR / "fonts"
MASKS_DIR = ASSETS_DIR / "masks"
TEMPLATES_DIR = ASSETS_DIR / "templates"

# Logical font name -> file in assets/fonts. Callers may also pass a raw path.
FONT_ALIASES = {
    "mono": "DejaVuSansMono.ttf",
    "mono-bold": "DejaVuSansMono-Bold.ttf",
    "thermal": "LiberationMono-Regular.ttf",
    "thermal-bold": "LiberationMono-Bold.ttf",
}

# Relative type scale, multiplied against the template's base_size.
SIZE_SCALE = {
    "xs": 0.70,
    "sm": 0.85,
    "md": 1.00,
    "lg": 1.30,
    "xl": 1.70,
    "xxl": 2.20,
}


def resolve_font_path(name: str) -> Path:
    """Map a logical font name (or path) to a concrete .ttf on disk."""
    if name in FONT_ALIASES:
        path = FONTS_DIR / FONT_ALIASES[name]
        if path.exists():
            return path
    candidate = Path(name)
    if candidate.is_file():
        return candidate
    bundled = FONTS_DIR / name
    if bundled.is_file():
        return bundled
    raise FileNotFoundError(
        f"Font {name!r} not found. Known aliases: {sorted(FONT_ALIASES)}; "
        f"or drop a .ttf into {FONTS_DIR}"
    )


@lru_cache(maxsize=256)
def load_font(name: str, size: int) -> ImageFont.FreeTypeFont:
    """Load (and cache) a TrueType font at a pixel size."""
    return ImageFont.truetype(str(resolve_font_path(name)), size)


def scaled_size(base_size: int, size: str | int | float) -> int:
    """Resolve 'lg' / 1.4 / 26 against the template base size."""
    if isinstance(size, str):
        if size not in SIZE_SCALE:
            raise ValueError(f"Unknown size keyword {size!r}; use {sorted(SIZE_SCALE)} or a number")
        return max(6, round(base_size * SIZE_SCALE[size]))
    if isinstance(size, float):
        return max(6, round(base_size * size))
    return max(6, int(size))


def text_width(font: ImageFont.FreeTypeFont, text: str) -> int:
    return int(round(font.getlength(text)))


def line_height(font: ImageFont.FreeTypeFont) -> int:
    ascent, descent = font.getmetrics()
    return ascent + descent


@lru_cache(maxsize=64)
def char_width(font_name: str, size: int) -> float:
    """Advance width of one character - receipts are laid out in column units."""
    return load_font(font_name, size).getlength("0")


def wrap_text(text: str, font: ImageFont.FreeTypeFont, max_width: int) -> list[str]:
    """Greedy word wrap, falling back to hard character breaks for long tokens.

    Fit tests use the exact float advance rather than `text_width`; rounding to
    int there lets a line overshoot `max_width` by up to half a pixel, which is
    enough to clip the last glyph against a tight column.
    """
    if not text:
        return [""]

    fits = lambda s: font.getlength(s) <= max_width

    lines: list[str] = []
    for paragraph in text.split("\n"):
        if not paragraph:
            lines.append("")
            continue
        current = ""
        for word in paragraph.split(" "):
            probe = f"{current} {word}".strip()
            if fits(probe) or not current:
                current = probe
            else:
                lines.append(current)
                current = word
            while not fits(current) and len(current) > 1:
                cut = len(current)
                while cut > 1 and not fits(current[:cut]):
                    cut -= 1
                lines.append(current[:cut])
                current = current[cut:]
        lines.append(current)
    return lines


def align_x(box: tuple[int, int], width: int, align: str) -> int:
    """Left edge for a run of `width` px inside `box` under the given alignment."""
    x0, x1 = box
    if align == "center":
        return x0 + max(0, (x1 - x0 - width) // 2)
    if align == "right":
        return x1 - width
    return x0


# --------------------------------------------------------------------------
# Array conversion
# --------------------------------------------------------------------------

def to_float(arr: np.ndarray) -> np.ndarray:
    """uint8 [0,255] -> float32 [0,1]."""
    return arr.astype(np.float32) / 255.0


def to_uint8(arr: np.ndarray) -> np.ndarray:
    """float32 [0,1] -> uint8 [0,255], clipped."""
    return np.clip(arr * 255.0 + 0.5, 0, 255).astype(np.uint8)


def pil_to_array(img: Image.Image) -> np.ndarray:
    return np.asarray(img)


def array_to_pil(arr: np.ndarray, mode: str | None = None) -> Image.Image:
    return Image.fromarray(arr, mode=mode) if mode else Image.fromarray(arr)


# --------------------------------------------------------------------------
# Procedural noise - used by every degradation stage so no binary mask assets
# are required to get a realistic result (real overlays can still be layered on).
# --------------------------------------------------------------------------

def value_noise(height: int, width: int, cells: int, rng: np.random.Generator) -> np.ndarray:
    """One octave of smooth value noise in [0,1], `cells` cells across the width."""
    gw = max(2, int(cells))
    gh = max(2, int(round(gw * height / max(1, width))))
    grid = rng.random((gh, gw)).astype(np.float32)
    return cv2.resize(grid, (width, height), interpolation=cv2.INTER_CUBIC)


def fractal_noise(
    height: int,
    width: int,
    rng: np.random.Generator,
    octaves: int = 4,
    base_cells: int = 3,
    persistence: float = 0.55,
    scale: float = 0.5,
) -> np.ndarray:
    """Sum of value-noise octaves, normalised to [0,1].

    `scale` builds the field at a fraction of the target resolution and upsamples
    once at the end. The octaves are band-limited anyway, so for the low-frequency
    fields (fade, crumple, stains) this is visually free and ~4x cheaper. Pass
    scale=1.0 when the finest octave approaches pixel scale, e.g. paper grain.
    """
    finest = base_cells * (2 ** (octaves - 1))
    # Don't downsample below ~4 samples per cell of the finest octave.
    if scale < 1.0 and finest * 4 > width * scale:
        scale = 1.0

    work_w = max(8, int(width * scale))
    work_h = max(8, int(height * scale))

    out = np.zeros((work_h, work_w), np.float32)
    amplitude, total = 1.0, 0.0
    for octave in range(octaves):
        out += amplitude * value_noise(work_h, work_w, base_cells * (2 ** octave), rng)
        total += amplitude
        amplitude *= persistence
    out /= max(total, 1e-6)

    if (work_h, work_w) != (height, width):
        out = cv2.resize(out, (width, height), interpolation=cv2.INTER_CUBIC)
    return normalize(out)


def normalize(arr: np.ndarray) -> np.ndarray:
    lo, hi = float(arr.min()), float(arr.max())
    if hi - lo < 1e-6:
        return np.zeros_like(arr)
    return (arr - lo) / (hi - lo)


def smoothstep(edge0: float, edge1: float, x: np.ndarray | float):
    t = np.clip((x - edge0) / max(edge1 - edge0, 1e-6), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def gradient_ramp(height: int, width: int, direction: str) -> np.ndarray:
    """Linear 0->1 ramp across the canvas in the requested direction."""
    ys = np.linspace(0.0, 1.0, height, dtype=np.float32)[:, None]
    xs = np.linspace(0.0, 1.0, width, dtype=np.float32)[None, :]
    ramps = {
        "top": 1.0 - ys + 0.0 * xs,
        "bottom": ys + 0.0 * xs,
        "left": 1.0 - xs + 0.0 * ys,
        "right": xs + 0.0 * ys,
    }
    if direction not in ramps:
        raise ValueError(f"Unknown direction {direction!r}; use {sorted(ramps)}")
    return np.broadcast_to(ramps[direction], (height, width)).astype(np.float32).copy()


def radial_falloff(height: int, width: int, cx: float, cy: float, radius: float) -> np.ndarray:
    """1 at the centre, 0 beyond `radius` (all args in normalised 0-1 units)."""
    ys = (np.linspace(0.0, 1.0, height, dtype=np.float32)[:, None] - cy)
    xs = (np.linspace(0.0, 1.0, width, dtype=np.float32)[None, :] - cx)
    aspect = width / max(height, 1)
    dist = np.sqrt((xs) ** 2 + (ys * aspect) ** 2)
    return 1.0 - smoothstep(radius * 0.35, radius, dist)


def lerp(a, b, t):
    return a + (b - a) * t


def clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


def rand_range(rng: np.random.Generator, lo: float, hi: float) -> float:
    return float(rng.uniform(lo, hi))


def pick(rng: np.random.Generator, options: Sequence):
    return options[int(rng.integers(0, len(options)))]
