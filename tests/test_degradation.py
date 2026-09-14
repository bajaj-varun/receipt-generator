"""Wear-pipeline tests.

The effects are stochastic, so these assert on invariants and measurable
direction of change (more fade -> less ink, tear -> non-uniform alpha) rather
than exact pixel values, plus strict reproducibility for a fixed seed.
"""

import numpy as np
import pytest
from PIL import Image

from src.core import build_degradation_config, render_receipt
from src.core.degradation import (
    PRESETS,
    DegradationConfig,
    Degrader,
    flatten,
    preset_config,
)
from src.core.renderer import ReceiptRenderer

SPEC = {
    "style": {"width": 384},
    "elements": [
        {"type": "text", "value": "TOTAL DUE", "align": "center", "bold": True},
        {"type": "rule", "char": "-"},
        {"type": "kv", "key": "AMOUNT", "value": "1234.00"},
    ],
}


@pytest.fixture(scope="module")
def clean() -> Image.Image:
    return ReceiptRenderer.from_spec(SPEC).render(SPEC)


# Geometry pads and warps the canvas; disable it when a test needs the output
# pixel-aligned with the clean input.
NO_GEOMETRY = dict(rotation=0.0, perspective=0.0)


def luminance(img: Image.Image) -> np.ndarray:
    arr = np.asarray(img).astype(np.float32) / 255.0
    return arr[..., :3].mean(axis=2) if arr.ndim == 3 else arr


def ink_contrast(clean: Image.Image, out: Image.Image) -> float:
    """Paper luminance minus ink luminance, measured through the clean ink mask.

    Mean darkness alone is useless here - stains, vignette and crumple shading
    all darken the paper, so a heavily worn receipt can be 'darker' overall
    while its print is far less legible. Requires `out` to be geometry-aligned.
    """
    assert luminance(out).shape == luminance(clean).shape, "output must be geometry-aligned"
    mask = np.asarray(clean) < 128
    lum = luminance(out)
    return float(lum[~mask].mean() - lum[mask].mean())


class TestOutputContract:
    def test_returns_rgba_same_or_larger_than_input(self, clean):
        out = Degrader(DegradationConfig(seed=1)).apply(clean)
        assert out.mode == "RGBA"
        # _geometry pads before warping, so the canvas grows slightly.
        assert out.width >= clean.width and out.height >= clean.height

    def test_accepts_non_grayscale_input(self, clean):
        assert Degrader(DegradationConfig(seed=1)).apply(clean.convert("RGB")).mode == "RGBA"

    def test_no_nans_or_out_of_range_values(self, clean):
        arr = np.asarray(Degrader(DegradationConfig(seed=3, intensity=1.8)).apply(clean))
        assert arr.dtype == np.uint8 and np.isfinite(arr).all()


class TestDeterminism:
    def test_same_seed_is_bit_identical(self, clean):
        a = Degrader(DegradationConfig(seed=99)).apply(clean)
        b = Degrader(DegradationConfig(seed=99)).apply(clean)
        assert np.array_equal(np.asarray(a), np.asarray(b))

    def test_different_seeds_diverge(self, clean):
        a = np.asarray(Degrader(DegradationConfig(seed=1)).apply(clean))
        b = np.asarray(Degrader(DegradationConfig(seed=2)).apply(clean))
        assert a.shape != b.shape or not np.array_equal(a, b)

    def test_unseeded_runs_diverge(self, clean):
        a = np.asarray(Degrader(DegradationConfig()).apply(clean))
        b = np.asarray(Degrader(DegradationConfig()).apply(clean))
        assert a.shape != b.shape or not np.array_equal(a, b)

    def test_render_receipt_is_reproducible(self):
        a = render_receipt(SPEC, seed=11)
        b = render_receipt(SPEC, seed=11)
        assert np.array_equal(np.asarray(a), np.asarray(b))


class TestEffectDirection:
    def test_fade_reduces_print_contrast(self, clean):
        low = Degrader(DegradationConfig(seed=5, thermal_fade=0.05, **NO_GEOMETRY)).apply(clean)
        high = Degrader(DegradationConfig(seed=5, thermal_fade=0.95, **NO_GEOMETRY)).apply(clean)
        assert ink_contrast(clean, high) < ink_contrast(clean, low)

    def test_wear_presets_are_ordered_by_legibility(self, clean):
        def contrast(preset):
            cfg = preset_config(preset, seed=5, **NO_GEOMETRY)
            return ink_contrast(clean, Degrader(cfg).apply(clean))

        assert contrast("pristine") > contrast("light") > contrast("heavy")

    def test_intensity_zero_leaves_the_ink_map_intact(self, clean):
        cfg = DegradationConfig(seed=5, intensity=0.0, stains=0, folds=0, dead_columns=0)
        out = Degrader(cfg).apply(clean)
        assert out.size == clean.size  # nothing enabled, so no geometry padding
        correlation = np.corrcoef(luminance(clean).ravel(), luminance(out).ravel())[0, 1]
        assert correlation > 0.99

    def test_dead_columns_punch_vertical_gaps(self, clean):
        quiet = dict(seed=4, intensity=1.0, dropout=0, streaks=0, ink_bleed=0,
                     ink_starvation=0, thermal_fade=0)
        d0 = Degrader(DegradationConfig(dead_columns=0, **quiet))
        d8 = Degrader(DegradationConfig(dead_columns=8, **quiet))
        ink = 1.0 - np.asarray(clean).astype(np.float32) / 255.0
        assert d8._printhead_defects(ink.copy()).sum() < d0._printhead_defects(ink.copy()).sum()

    def test_edge_tear_makes_alpha_ragged(self):
        d = Degrader(DegradationConfig(seed=8, edge_tear=0.8))
        alpha = d._edge_tear((400, 200))
        first_opaque = np.argmax(alpha > 0.5, axis=0)
        assert first_opaque.max() - first_opaque.min() > 3   # not a straight cut
        # The long sides fray by a few px, so check the interior of a middle row.
        assert alpha[alpha.shape[0] // 2, 5:-5].min() > 0.9  # middle stays intact

    def test_edge_tear_disabled_leaves_a_clean_rectangle(self):
        alpha = Degrader(DegradationConfig(seed=8, edge_tear=0.0)).apply(
            ReceiptRenderer.from_spec(SPEC).render(SPEC)
        )
        assert np.asarray(alpha)[..., 3].max() == 255

    def test_jpeg_pass_runs_and_stays_in_range(self, clean):
        out = Degrader(DegradationConfig(seed=2, jpeg_quality=25)).apply(clean)
        assert out.mode == "RGBA" and np.asarray(out).max() <= 255


class TestConfig:
    def test_every_preset_builds_and_renders(self, clean):
        for name in PRESETS:
            out = Degrader(preset_config(name, seed=1)).apply(clean)
            assert out.mode == "RGBA"

    def test_unknown_preset_lists_valid_names(self):
        with pytest.raises(ValueError, match="Unknown preset"):
            preset_config("soggy")

    def test_unknown_config_key_rejected(self):
        with pytest.raises(ValueError, match="Unknown degradation keys"):
            DegradationConfig.from_spec({"sparkles": 1})

    def test_preset_must_be_resolved_before_from_spec(self):
        with pytest.raises(ValueError, match="resolve presets"):
            DegradationConfig.from_spec({"preset": "worn"})

    def test_explicit_args_beat_template_fields(self):
        cfg = build_degradation_config(
            {"preset": "light", "thermal_fade": 0.2, "seed": 1}, preset="heavy", seed=42
        )
        assert cfg.seed == 42                # CLI arg wins over the template
        assert cfg.thermal_fade == 0.2       # template field wins over the preset
        assert cfg.stains == PRESETS["heavy"]["stains"]  # preset fills the rest

    def test_colour_lists_from_json_become_tuples(self):
        cfg = DegradationConfig.from_spec({"paper_color": [10, 20, 30]})
        assert cfg.paper_color == (10, 20, 30)

    def test_missing_overlay_file_is_reported(self, clean):
        cfg = DegradationConfig(seed=1, overlays=("masks/does_not_exist.png",))
        with pytest.raises(FileNotFoundError, match="does_not_exist"):
            Degrader(cfg).apply(clean)


class TestFlatten:
    def test_composites_onto_an_opaque_background(self, clean):
        rgba = Degrader(DegradationConfig(seed=1)).apply(clean)
        flat = flatten(rgba, (10, 20, 30))
        assert flat.mode == "RGB" and flat.size == rgba.size
        corner = np.asarray(flat)[0, 0]
        assert tuple(corner) == (10, 20, 30)  # fully transparent corner shows the surface


class TestRenderReceipt:
    def test_degrade_false_returns_the_clean_ink_map(self):
        out = render_receipt(SPEC, degrade=False)
        assert out.mode == "L" and out.width == 384

    def test_template_degradation_block_reaches_the_degrader(self):
        # paper_color is an unambiguous fingerprint: if the block were ignored the
        # paper would come out near-white instead of saturated red.
        spec = {**SPEC, "degradation": {
            "intensity": 0.0, "stains": 0, "folds": 0, "dead_columns": 0,
            "paper_color": [255, 0, 0],
        }}
        corner = np.asarray(render_receipt(spec, seed=1))[2, 2]
        assert corner[0] > 200 and corner[1] < 60 and corner[2] < 60

    def test_cli_preset_overrides_the_template_block(self):
        spec = {**SPEC, "degradation": {"preset": "pristine"}}
        clean = ReceiptRenderer.from_spec(SPEC).render(SPEC)
        as_written = render_receipt(spec, seed=1)
        overridden = render_receipt(spec, preset="heavy", seed=1)
        assert as_written.size != overridden.size or not np.array_equal(
            np.asarray(as_written), np.asarray(overridden)
        )
