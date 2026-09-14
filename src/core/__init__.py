"""Image processing pipeline: clean layout -> physical wear -> final image."""

from typing import Any

from PIL import Image

from .degradation import (
    PRESETS,
    DegradationConfig,
    Degrader,
    flatten,
    preset_config,
)
from .renderer import ReceiptRenderer, RenderStyle

__all__ = [
    "DegradationConfig",
    "Degrader",
    "PRESETS",
    "ReceiptRenderer",
    "RenderStyle",
    "flatten",
    "preset_config",
    "render_receipt",
    "build_degradation_config",
]


def build_degradation_config(
    spec: dict[str, Any] | None,
    *,
    preset: str | None = None,
    seed: int | None = None,
) -> DegradationConfig:
    """Merge a template's `degradation` block with CLI/API overrides.

    Precedence: explicit args > template fields > preset defaults.
    """
    raw = dict(spec or {})
    chosen = preset or raw.pop("preset", None)
    raw.pop("preset", None)
    if seed is not None:
        raw["seed"] = seed
    return preset_config(chosen, **raw)


def render_receipt(
    spec: dict[str, Any],
    *,
    preset: str | None = None,
    seed: int | None = None,
    degrade: bool = True,
) -> Image.Image:
    """Render a template spec end to end.

    Returns RGBA when degraded (the tear edge lives in the alpha channel),
    or the clean grayscale ink map when `degrade` is False.
    """
    clean = ReceiptRenderer.from_spec(spec).render(spec)
    if not degrade:
        return clean
    config = build_degradation_config(spec.get("degradation"), preset=preset, seed=seed)
    return Degrader(config).apply(clean)
