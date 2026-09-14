"""Physical wear + capture-artifact pipeline for thermal receipts.

Takes the clean ink map from `renderer` and pushes it through stages that
roughly follow the real physical chain:

    print defects  ->  paper substrate  ->  handling/deformation  ->  capture

Every stage is individually scalable, and the whole thing is driven by one
seeded RNG so a given (spec, seed) pair always reproduces the same image.
All masks are generated procedurally, so no binary overlay assets are needed -
real scanned overlays can still be layered in via `DegradationConfig.overlays`.
"""

from __future__ import annotations

from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any, Sequence

import cv2
import numpy as np
from PIL import Image

from .utils import (
    ASSETS_DIR,
    fractal_noise,
    gradient_ramp,
    normalize,
    pick,
    rand_range,
    smoothstep,
    to_float,
    to_uint8,
    value_noise,
)

PAPER_WHITE = (250, 248, 243)
THERMAL_INK = (34, 32, 38)


@dataclass
class DegradationConfig:
    """Per-effect strengths, all roughly 0..1 unless noted."""

    seed: int | None = None
    intensity: float = 1.0          # master multiplier over every effect below

    # --- print defects
    ink_bleed: float = 0.45         # thermal dot spread / smearing
    ink_starvation: float = 0.35    # patchy under-burn, ink looks thin
    dropout: float = 0.30           # random dead pixels in the print
    dead_columns: int = 2           # burnt-out printhead elements -> white stripes
    streaks: float = 0.35           # faint horizontal drag lines

    # --- paper + ageing
    thermal_fade: float = 0.55      # heat/light fade, strongest at one end
    fade_direction: str = "random"  # top | bottom | left | right | random
    paper_texture: float = 0.40     # fibre grain
    yellowing: float = 0.45         # age tint
    stains: int = 2                 # number of stain blobs
    stain_strength: float = 0.55

    # --- handling
    crumple: float = 0.45           # micro-wrinkle shading + displacement
    folds: int = 1                  # hard horizontal fold creases
    vignette: float = 0.45          # edge darkening / curl shadow
    edge_tear: float = 0.60         # ragged tear at top & bottom

    # --- capture
    rotation: float = 0.55          # scaled to +/- max_rotation_deg
    max_rotation_deg: float = 2.2
    perspective: float = 0.35
    blur: float = 0.45
    noise: float = 0.35             # sensor noise
    jpeg_quality: int = 0           # 0 disables the recompression pass

    # --- colour
    paper_color: tuple[int, int, int] = PAPER_WHITE
    ink_color: tuple[int, int, int] = THERMAL_INK

    # --- optional real scanned overlays (paths under assets/, multiply-blended)
    overlays: Sequence[str] = field(default_factory=tuple)
    overlay_strength: float = 0.35

    @classmethod
    def from_spec(cls, spec: dict[str, Any] | None) -> "DegradationConfig":
        raw = dict(spec or {})
        if raw.pop("preset", None) is not None:
            raise ValueError("resolve presets with `preset_config()` before building a config")
        known = {f.name for f in fields(cls)}
        unknown = set(raw) - known
        if unknown:
            raise ValueError(f"Unknown degradation keys: {sorted(unknown)}")
        for key in ("paper_color", "ink_color"):
            if key in raw:
                raw[key] = tuple(raw[key])
        return cls(**raw)


# Named starting points; override individual fields on top of any of these.
PRESETS: dict[str, dict[str, Any]] = {
    "pristine": dict(intensity=0.0, edge_tear=0.35, rotation=0.2, blur=0.15, noise=0.1),
    "light": dict(intensity=0.45, stains=1, folds=0, dead_columns=1),
    "worn": dict(intensity=1.0),
    "heavy": dict(intensity=1.5, stains=4, folds=2, dead_columns=3,
                  thermal_fade=0.75, crumple=0.7, edge_tear=0.8),
    "faded_receipt": dict(intensity=1.2, thermal_fade=0.9, ink_starvation=0.7,
                          yellowing=0.7, stains=1, crumple=0.35),
    "pocket": dict(intensity=1.3, crumple=0.85, folds=3, edge_tear=0.75,
                   stains=2, thermal_fade=0.5),
}


def preset_config(name: str | None = None, **overrides: Any) -> DegradationConfig:
    """Build a config from a preset name plus explicit field overrides."""
    base = dict(PRESETS.get(name or "worn", {}))
    if name and name not in PRESETS:
        raise ValueError(f"Unknown preset {name!r}; available: {sorted(PRESETS)}")
    base.update({k: v for k, v in overrides.items() if v is not None})
    return DegradationConfig.from_spec(base)


class Degrader:
    """Applies the wear pipeline to a clean ink map."""

    def __init__(self, config: DegradationConfig | None = None) -> None:
        self.cfg = config or DegradationConfig()
        self.rng = np.random.default_rng(self.cfg.seed)

    def amount(self, value: float) -> float:
        """Scale one effect by the master intensity, clamped to a sane ceiling."""
        return float(np.clip(value * self.cfg.intensity, 0.0, 2.0))

    # -- public API --------------------------------------------------------

    def apply(self, receipt: Image.Image) -> Image.Image:
        """Clean grayscale ink map -> worn RGBA receipt with a torn alpha edge."""
        if receipt.mode != "L":
            receipt = receipt.convert("L")

        ink = 1.0 - to_float(np.asarray(receipt))  # 1.0 = full ink coverage

        # 1. print defects, applied to ink coverage before it hits the paper
        ink = self._ink_bleed(ink)
        ink = self._ink_starvation(ink)
        ink = self._printhead_defects(ink)
        ink = self._thermal_fade(ink)

        # 2. paper substrate
        rgb = self._compose_paper(ink)
        rgb = self._paper_texture(rgb)
        rgb = self._yellowing(rgb)
        rgb = self._stains(rgb)
        rgb = self._overlays(rgb)

        # 3. handling
        rgb = self._folds(rgb)
        rgb = self._crumple(rgb)
        rgb = self._vignette(rgb)

        # 4. capture
        alpha = self._edge_tear(rgb.shape[:2])
        rgba = np.dstack([rgb, alpha])
        rgba = self._geometry(rgba)
        rgba = self._capture(rgba)

        return Image.fromarray(to_uint8(rgba), mode="RGBA")

    # -- 1. print defects --------------------------------------------------

    def _ink_bleed(self, ink: np.ndarray) -> np.ndarray:
        """Thermal dots bloom slightly into neighbouring paper."""
        strength = self.amount(self.cfg.ink_bleed)
        if strength <= 0.01:
            return ink
        bloom = cv2.GaussianBlur(ink, (0, 0), 0.6 + 0.9 * strength)
        return np.clip(ink + bloom * 0.55 * strength, 0.0, 1.0)

    def _ink_starvation(self, ink: np.ndarray) -> np.ndarray:
        """Uneven burn: mid-frequency patches where the print came out thin."""
        strength = self.amount(self.cfg.ink_starvation)
        if strength <= 0.01:
            return ink
        h, w = ink.shape
        patchiness = fractal_noise(h, w, self.rng, octaves=3, base_cells=6)
        return ink * (1.0 - strength * 0.55 * patchiness)

    def _printhead_defects(self, ink: np.ndarray) -> np.ndarray:
        """Dead heating elements, random dot dropouts and drag streaks."""
        h, w = ink.shape
        cfg = self.cfg

        dropout = self.amount(cfg.dropout)
        if dropout > 0.01:
            keep = self.rng.random((h, w)) > (0.05 * dropout)
            ink = ink * keep

        # Burnt-out elements print as full-height white stripes - the single most
        # recognisable thermal defect.
        for _ in range(int(cfg.dead_columns)):
            x = int(self.rng.integers(0, w))
            width = int(self.rng.integers(1, 3))
            y0 = int(self.rng.integers(0, max(1, h // 3)))
            y1 = int(self.rng.integers(h // 2, h))
            ink[y0:y1, x:x + width] *= rand_range(self.rng, 0.0, 0.25)

        streaks = self.amount(cfg.streaks)
        if streaks > 0.01:
            for _ in range(int(self.rng.integers(2, 6))):
                y = int(self.rng.integers(0, h))
                thickness = int(self.rng.integers(1, 4))
                x0 = int(self.rng.integers(0, max(1, w // 2)))
                x1 = int(self.rng.integers(w // 2, w))
                ink[y:y + thickness, x0:x1] *= 1.0 - rand_range(self.rng, 0.3, 0.9) * streaks

        return np.clip(ink, 0.0, 1.0)

    def _thermal_fade(self, ink: np.ndarray) -> np.ndarray:
        """Heat/light exposure eats the print, strongest toward one end."""
        strength = self.amount(self.cfg.thermal_fade)
        if strength <= 0.01:
            return ink
        h, w = ink.shape
        direction = self.cfg.fade_direction
        if direction == "random":
            direction = pick(self.rng, ("top", "bottom", "left", "right"))

        ramp = gradient_ramp(h, w, direction) ** rand_range(self.rng, 1.3, 2.4)
        blotches = fractal_noise(h, w, self.rng, octaves=4, base_cells=3)
        field = np.clip(0.65 * ramp + 0.55 * blotches, 0.0, 1.0)
        return ink * (1.0 - np.clip(strength, 0.0, 0.95) * field)

    # -- 2. paper ----------------------------------------------------------

    def _compose_paper(self, ink: np.ndarray) -> np.ndarray:
        paper = np.array(self.cfg.paper_color, np.float32) / 255.0
        pigment = np.array(self.cfg.ink_color, np.float32) / 255.0
        coverage = ink[..., None]
        return paper * (1.0 - coverage) + pigment * coverage

    def _paper_texture(self, rgb: np.ndarray) -> np.ndarray:
        """Fine fibre grain in the paper stock."""
        strength = self.amount(self.cfg.paper_texture)
        if strength <= 0.01:
            return rgb
        h, w = rgb.shape[:2]
        fibres = fractal_noise(h, w, self.rng, octaves=3, base_cells=40, persistence=0.7, scale=1.0)
        grain = 1.0 + (fibres - 0.5) * 0.10 * strength
        return np.clip(rgb * grain[..., None], 0.0, 1.0)

    def _yellowing(self, rgb: np.ndarray) -> np.ndarray:
        """Age tint - warmer and dirtier toward the edges."""
        strength = self.amount(self.cfg.yellowing)
        if strength <= 0.01:
            return rgb
        h, w = rgb.shape[:2]
        aged = np.array([1.0, 0.955, 0.855], np.float32)
        edge = 1.0 - self._edge_distance(h, w)
        blotch = fractal_noise(h, w, self.rng, octaves=3, base_cells=4)
        weight = np.clip(strength * (0.45 + 0.35 * edge + 0.35 * blotch), 0.0, 1.0)[..., None]
        return np.clip(rgb * (1.0 - weight) + rgb * aged * weight, 0.0, 1.0)

    def _stains(self, rgb: np.ndarray) -> np.ndarray:
        """Coffee rings, oil patches and water marks, multiply-blended."""
        count = int(self.cfg.stains)
        strength = self.amount(self.cfg.stain_strength)
        if count <= 0 or strength <= 0.01:
            return rgb

        h, w = rgb.shape[:2]
        for _ in range(count):
            kind = pick(self.rng, ("coffee_ring", "oil", "water"))
            mask = self._stain_mask(h, w, kind)
            tint = {
                "coffee_ring": np.array([0.60, 0.44, 0.28], np.float32),
                "oil": np.array([0.82, 0.78, 0.66], np.float32),
                "water": np.array([0.88, 0.87, 0.80], np.float32),
            }[kind]
            weight = np.clip(mask * strength * rand_range(self.rng, 0.35, 0.8), 0.0, 0.9)[..., None]
            rgb = rgb * (1.0 - weight) + rgb * tint * weight
        return np.clip(rgb, 0.0, 1.0)

    def _stain_mask(self, h: int, w: int, kind: str) -> np.ndarray:
        """Irregular blob mask; coffee gets a dark ring at the dried edge."""
        cx, cy = rand_range(self.rng, -0.1, 1.1), rand_range(self.rng, 0.0, 1.0)
        radius = rand_range(self.rng, 0.12, 0.42)

        ys = np.linspace(0.0, 1.0, h, dtype=np.float32)[:, None] - cy
        xs = np.linspace(0.0, 1.0, w, dtype=np.float32)[None, :] - cx
        aspect = w / max(h, 1)
        dist = np.sqrt(xs ** 2 + (ys * aspect) ** 2)

        # Wobble the boundary so it never reads as a clean circle.
        dist = dist * (0.75 + 0.5 * fractal_noise(h, w, self.rng, octaves=3, base_cells=5))

        if kind == "coffee_ring":
            ring = np.exp(-(((dist - radius) / (radius * 0.16 + 1e-6)) ** 2))
            fill = (1.0 - smoothstep(radius * 0.7, radius, dist)) * 0.35
            return np.clip(ring + fill, 0.0, 1.0)
        if kind == "oil":
            blob = 1.0 - smoothstep(radius * 0.2, radius, dist)
            return cv2.GaussianBlur(blob.astype(np.float32), (0, 0), max(2.0, radius * w * 0.06))
        edge = np.exp(-(((dist - radius) / (radius * 0.28 + 1e-6)) ** 2)) * 0.8
        return np.clip(edge + (1.0 - smoothstep(radius * 0.5, radius, dist)) * 0.18, 0.0, 1.0)

    def _overlays(self, rgb: np.ndarray) -> np.ndarray:
        """Multiply-blend optional real scanned textures from assets/masks."""
        if not self.cfg.overlays or self.cfg.overlay_strength <= 0.01:
            return rgb
        h, w = rgb.shape[:2]
        strength = self.amount(self.cfg.overlay_strength)
        for name in self.cfg.overlays:
            path = Path(name)
            if not path.is_absolute():
                path = ASSETS_DIR / path
            if not path.exists():
                raise FileNotFoundError(f"Overlay texture not found: {path}")
            texture = to_float(np.asarray(Image.open(path).convert("L").resize((w, h))))
            rgb = rgb * (1.0 - strength * (1.0 - texture))[..., None]
        return np.clip(rgb, 0.0, 1.0)

    # -- 3. handling -------------------------------------------------------

    def _folds(self, rgb: np.ndarray) -> np.ndarray:
        """Hard horizontal creases from folding the receipt to pocket it."""
        count = int(self.cfg.folds)
        if count <= 0 or self.cfg.intensity <= 0.01:
            return rgb
        h, w = rgb.shape[:2]
        shading = np.ones((h, w), np.float32)
        ys = np.arange(h, dtype=np.float32)[:, None]

        for _ in range(count):
            y = rand_range(self.rng, 0.18, 0.82) * h
            # A crease is a thin dark trough with a bright highlight beside it.
            wobble = (value_noise(h, w, 6, self.rng) - 0.5) * h * 0.01
            offset = ys - y + wobble
            width = rand_range(self.rng, 2.0, 5.0)
            trough = np.exp(-((offset / width) ** 2)) * rand_range(self.rng, 0.10, 0.24)
            highlight = np.exp(-(((offset - width * 2.2) / (width * 1.8)) ** 2)) * 0.07
            shading *= 1.0 - trough + highlight
        return np.clip(rgb * shading[..., None], 0.0, 1.0)

    def _crumple(self, rgb: np.ndarray) -> np.ndarray:
        """Micro-wrinkles: directional shading plus a matching UV displacement."""
        strength = self.amount(self.cfg.crumple)
        if strength <= 0.01:
            return rgb
        h, w = rgb.shape[:2]

        height_field = fractal_noise(h, w, self.rng, octaves=3, base_cells=3, persistence=0.5)
        height_field = cv2.GaussianBlur(height_field, (0, 0), max(1.5, w * 0.006))
        gx = cv2.Sobel(height_field, cv2.CV_32F, 1, 0, ksize=5)
        gy = cv2.Sobel(height_field, cv2.CV_32F, 0, 1, ksize=5)

        # Fake a single light source raking across the crumpled surface.
        angle = rand_range(self.rng, 0.0, 2.0 * np.pi)
        lighting = gx * np.cos(angle) + gy * np.sin(angle)
        lighting = normalize(lighting) * 2.0 - 1.0
        rgb = np.clip(rgb * (1.0 + lighting * 0.16 * strength)[..., None], 0.0, 1.0)

        amplitude = 2.0 + 5.0 * strength
        dx = (fractal_noise(h, w, self.rng, octaves=3, base_cells=4) - 0.5) * 2 * amplitude
        dy = (fractal_noise(h, w, self.rng, octaves=3, base_cells=4) - 0.5) * 2 * amplitude
        grid_x, grid_y = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
        return cv2.remap(
            rgb, grid_x + dx, grid_y + dy,
            interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE,
        )

    def _vignette(self, rgb: np.ndarray) -> np.ndarray:
        """Shadow where the paper curls away from the flat surface."""
        strength = self.amount(self.cfg.vignette)
        if strength <= 0.01:
            return rgb
        h, w = rgb.shape[:2]
        falloff = self._edge_distance(h, w)
        shade = 1.0 - (1.0 - falloff) * 0.30 * strength
        return np.clip(rgb * shade[..., None], 0.0, 1.0)

    @staticmethod
    def _edge_distance(h: int, w: int) -> np.ndarray:
        """1 in the middle, easing to 0 at the nearest edge."""
        fx = smoothstep(0.0, 0.18, np.minimum(
            np.linspace(0.0, 1.0, w, dtype=np.float32),
            1.0 - np.linspace(0.0, 1.0, w, dtype=np.float32),
        ))[None, :]
        fy = smoothstep(0.0, 0.06, np.minimum(
            np.linspace(0.0, 1.0, h, dtype=np.float32),
            1.0 - np.linspace(0.0, 1.0, h, dtype=np.float32),
        ))[:, None]
        return np.clip(fx * fy, 0.0, 1.0)

    def _edge_tear(self, shape: tuple[int, int]) -> np.ndarray:
        """Alpha mask with a ragged tear-off edge top and bottom."""
        h, w = shape
        alpha = np.ones((h, w), np.float32)
        strength = self.amount(self.cfg.edge_tear)
        if strength <= 0.01:
            return alpha

        ys = np.arange(h, dtype=np.float32)[:, None]
        for edge in ("top", "bottom"):
            teeth = int(self.rng.integers(6, 14))
            phase = rand_range(self.rng, 0.0, 1.0)
            sawtooth = np.abs(((np.arange(w, dtype=np.float32) / w * teeth + phase) % 1.0) - 0.5) * 2.0
            jag = 0.55 * sawtooth
            jag += 0.45 * value_noise(1, w, int(self.rng.integers(10, 30)), self.rng)[0]
            jag += 0.25 * value_noise(1, w, int(self.rng.integers(40, 90)), self.rng)[0]
            profile = normalize(jag) * (5.0 + 24.0 * strength) + 1.0
            boundary = profile[None, :] if edge == "top" else (h - 1 - profile)[None, :]
            alpha *= smoothstep(0, 1.5, ys - boundary) if edge == "top" else smoothstep(0, 1.5, boundary - ys)

        # Slight fraying on the long sides too.
        side = value_noise(h, 1, 30, self.rng)[:, 0] * 3.0 * strength
        xs = np.arange(w, dtype=np.float32)[None, :]
        alpha *= smoothstep(0, 1.2, xs - side[:, None])
        alpha *= smoothstep(0, 1.2, (w - 1 - side[:, None]) - xs)
        return np.clip(alpha, 0.0, 1.0)

    # -- 4. capture --------------------------------------------------------

    def _geometry(self, rgba: np.ndarray) -> np.ndarray:
        """Micro-rotation and a touch of perspective, as if photographed."""
        h, w = rgba.shape[:2]
        rotation = self.amount(self.cfg.rotation)
        perspective = self.amount(self.cfg.perspective)
        if rotation <= 0.01 and perspective <= 0.01:
            return rgba

        pad = int(max(h, w) * 0.02) + 6
        rgba = cv2.copyMakeBorder(rgba, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=(0, 0, 0, 0))
        h, w = rgba.shape[:2]

        if perspective > 0.01:
            jitter = perspective * min(w, h) * 0.02
            src = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
            dst = src + self.rng.uniform(-jitter, jitter, src.shape).astype(np.float32)
            rgba = cv2.warpPerspective(
                rgba, cv2.getPerspectiveTransform(src, dst), (w, h),
                flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0, 0),
            )

        if rotation > 0.01:
            angle = rand_range(self.rng, -1.0, 1.0) * self.cfg.max_rotation_deg * min(rotation, 1.5)
            matrix = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
            rgba = cv2.warpAffine(
                rgba, matrix, (w, h),
                flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0, 0),
            )
        return np.clip(rgba, 0.0, 1.0)

    def _capture(self, rgba: np.ndarray) -> np.ndarray:
        """Lens softness, sensor noise and optional JPEG recompression."""
        rgb, alpha = rgba[..., :3], rgba[..., 3]

        blur = self.amount(self.cfg.blur)
        if blur > 0.01:
            sigma = 0.35 + 1.1 * blur
            rgb = cv2.GaussianBlur(rgb, (0, 0), sigma)
            alpha = cv2.GaussianBlur(alpha, (0, 0), max(sigma * 0.6, 0.3))

        noise = self.amount(self.cfg.noise)
        if noise > 0.01:
            rgb = rgb + self.rng.normal(0.0, 0.012 * noise, rgb.shape).astype(np.float32)
            rgb = np.clip(rgb, 0.0, 1.0)

        quality = int(self.cfg.jpeg_quality)
        if quality > 0:
            ok, buffer = cv2.imencode(
                ".jpg", to_uint8(rgb)[:, :, ::-1], [int(cv2.IMWRITE_JPEG_QUALITY), quality]
            )
            if not ok:
                raise RuntimeError("JPEG recompression pass failed")
            rgb = to_float(cv2.imdecode(buffer, cv2.IMREAD_COLOR)[:, :, ::-1])

        return np.dstack([rgb, np.clip(alpha, 0.0, 1.0)])


def flatten(rgba: Image.Image, background: tuple[int, int, int] = (228, 226, 221)) -> Image.Image:
    """Composite an RGBA receipt onto a flat surface, for JPEG output."""
    backdrop = Image.new("RGB", rgba.size, background)
    backdrop.paste(rgba, (0, 0), rgba)
    return backdrop
