"""Layout engine tests - all deterministic, no randomness involved."""

import numpy as np
import pytest
from PIL import Image

from src.core.renderer import (
    CODE39,
    PAPER_WIDTHS,
    ReceiptRenderer,
    RenderStyle,
    encode_code39,
)
from src.core.utils import load_font, wrap_text


def ink_fraction(img: Image.Image) -> float:
    """Share of pixels carrying ink."""
    return float((np.asarray(img) < 200).mean())


def simple_spec(**overrides):
    spec = {"style": {"width": 384}, "elements": [{"type": "text", "value": "HELLO"}]}
    spec.update(overrides)
    return spec


class TestRenderBasics:
    def test_returns_grayscale_at_requested_width(self):
        img = ReceiptRenderer.from_spec(simple_spec()).render(simple_spec())
        assert img.mode == "L"
        assert img.width == 384

    def test_named_paper_width_resolves(self):
        spec = simple_spec(style={"width": "80mm"})
        assert ReceiptRenderer.from_spec(spec).render(spec).width == PAPER_WIDTHS["80mm"]

    def test_unknown_paper_width_rejected(self):
        with pytest.raises(ValueError, match="Unknown paper width"):
            RenderStyle.from_spec({"style": {"width": "13mm"}})

    def test_unknown_style_key_rejected(self):
        with pytest.raises(ValueError, match="Unknown style keys"):
            RenderStyle.from_spec({"style": {"colour": "red"}})

    def test_unknown_element_type_reports_index(self):
        spec = simple_spec(elements=[{"type": "text", "value": "a"}, {"type": "nope"}])
        with pytest.raises(ValueError, match=r"element\[1\].*unknown type 'nope'"):
            ReceiptRenderer.from_spec(spec).render(spec)

    def test_height_grows_with_content(self):
        one = simple_spec(elements=[{"type": "text", "value": "A"}])
        many = simple_spec(elements=[{"type": "text", "value": "A"}] * 6)
        r = ReceiptRenderer.from_spec(one)
        assert r.render(many).height > r.render(one).height

    def test_empty_receipt_still_renders(self):
        spec = simple_spec(elements=[])
        img = ReceiptRenderer.from_spec(spec).render(spec)
        assert img.height > 0 and ink_fraction(img) == 0.0


class TestAlignment:
    @staticmethod
    def _ink_centroid_x(spec) -> float:
        arr = (np.asarray(ReceiptRenderer.from_spec(spec).render(spec)) < 200).astype(float)
        xs = np.arange(arr.shape[1])
        return float((arr.sum(axis=0) * xs).sum() / max(arr.sum(), 1))

    def test_alignment_shifts_ink_horizontally(self):
        def spec(align):
            return simple_spec(elements=[{"type": "text", "value": "HI", "align": align}])

        left = self._ink_centroid_x(spec("left"))
        centre = self._ink_centroid_x(spec("center"))
        right = self._ink_centroid_x(spec("right"))
        assert left < centre < right

    def test_tracking_widens_a_line(self):
        base = simple_spec(elements=[{"type": "text", "value": "WIDE", "align": "left"}])
        tracked = simple_spec(
            elements=[{"type": "text", "value": "WIDE", "align": "left", "tracking": 6}]
        )

        def extent(spec):
            arr = np.asarray(ReceiptRenderer.from_spec(spec).render(spec)) < 200
            cols = np.where(arr.any(axis=0))[0]
            return cols.max() - cols.min()

        assert extent(tracked) > extent(base)


class TestElements:
    def test_kv_puts_value_flush_right(self):
        spec = simple_spec(elements=[{"type": "kv", "key": "TOTAL", "value": "99"}])
        arr = np.asarray(ReceiptRenderer.from_spec(spec).render(spec)) < 200
        cols = np.where(arr.any(axis=0))[0]
        style = RenderStyle.from_spec(spec)
        assert cols.min() == pytest.approx(style.margin_x, abs=6)
        assert cols.max() == pytest.approx(spec["style"]["width"] - style.margin_x, abs=8)

    def test_leader_fills_the_gap(self):
        plain = simple_spec(elements=[{"type": "kv", "key": "A", "value": "B"}])
        led = simple_spec(elements=[{"type": "kv", "key": "A", "value": "B", "leader": "."}])
        r = ReceiptRenderer.from_spec(plain)
        assert ink_fraction(r.render(led)) > ink_fraction(r.render(plain))

    def test_items_requires_columns(self):
        spec = simple_spec(elements=[{"type": "items", "rows": [{"a": 1}]}])
        with pytest.raises(ValueError, match="requires 'columns'"):
            ReceiptRenderer.from_spec(spec).render(spec)

    def test_column_widths_exactly_fill_content_width(self):
        cols = [{"key": "a", "width": 0.5}, {"key": "b", "width": 0.3}, {"key": "c", "width": 0.2}]
        assert sum(ReceiptRenderer._column_widths(cols, 517)) == 517

    def test_column_widths_default_to_equal_split(self):
        widths = ReceiptRenderer._column_widths([{"key": "a"}, {"key": "b"}], 400)
        assert widths == [200, 200]

    def test_wrapped_row_consumes_extra_lines(self):
        def spec(item):
            return simple_spec(elements=[{
                "type": "items", "wrap": True,
                "columns": [{"key": "i", "width": 0.4}, {"key": "amt", "width": 0.6}],
                "rows": [{"i": item, "amt": "1.00"}],
            }])

        r = ReceiptRenderer.from_spec(spec("X"))
        assert r.render(spec("A VERY LONG PRODUCT NAME HERE")).height > r.render(spec("X")).height

    def test_rule_char_spans_content_width(self):
        spec = simple_spec(elements=[{"type": "rule", "char": "-"}])
        arr = np.asarray(ReceiptRenderer.from_spec(spec).render(spec)) < 200
        cols = np.where(arr.any(axis=0))[0]
        assert cols.max() - cols.min() > spec["style"]["width"] * 0.85

    def test_zero_width_rule_char_rejected(self):
        spec = simple_spec(elements=[{"type": "rule", "char": "​"}])
        with pytest.raises(ValueError, match="zero width"):
            ReceiptRenderer.from_spec(spec).render(spec)

    def test_spacer_adds_height_without_ink(self):
        none = simple_spec(elements=[{"type": "spacer", "height": 0}])
        tall = simple_spec(elements=[{"type": "spacer", "height": 80}])
        r = ReceiptRenderer.from_spec(none)
        assert r.render(tall).height - r.render(none).height == 80
        assert ink_fraction(r.render(tall)) == 0.0


class TestSideText:
    def test_side_text_narrows_the_body_and_marks_the_edges(self):
        plain = simple_spec(elements=[{"type": "rule", "char": "-"}])
        sided = simple_spec(
            elements=[{"type": "rule", "char": "-"}],
            side_text={"left": "PINE LABS", "right": "AMEX"},
        )

        def body_extent(spec):
            arr = np.asarray(ReceiptRenderer.from_spec(spec).render(spec)) < 200
            cols = np.where(arr.any(axis=0))[0]
            return cols.min(), cols.max()

        p_lo, p_hi = body_extent(plain)
        s_lo, s_hi = body_extent(sided)
        assert s_lo < p_lo and s_hi > p_hi  # side strips sit outside the body column

    def test_string_shorthand_matches_dict_form(self):
        a = simple_spec(side_text={"left": "PINE LABS"})
        b = simple_spec(side_text={"left": {"text": "PINE LABS"}})
        r = ReceiptRenderer.from_spec(a)
        assert np.array_equal(np.asarray(r.render(a)), np.asarray(r.render(b)))


class TestCode39:
    def test_pattern_table_is_well_formed(self):
        for char, pattern in CODE39.items():
            assert len(pattern) == 9, char
            assert pattern.count("w") == 3, char  # exactly 3 wide elements per symbol

    def test_encoding_is_guarded_and_correctly_sized(self):
        encoded = encode_code39("A1")
        assert encoded.startswith(CODE39["*"]) and encoded.endswith(CODE39["*"])
        # 4 symbols (start, A, 1, stop) of 9 elements + 3 inter-character gaps
        assert len(encoded) == 4 * 9 + 3

    def test_lowercase_is_accepted(self):
        assert encode_code39("ab") == encode_code39("AB")

    def test_unencodable_characters_are_listed(self):
        with pytest.raises(ValueError, match=r"\['!', '@'\]"):
            encode_code39("A@B!")

    def test_star_is_reserved_for_guards(self):
        with pytest.raises(ValueError):
            encode_code39("A*B")

    def test_barcode_shrinks_to_fit_rather_than_overflowing(self):
        spec = simple_spec(
            elements=[{"type": "barcode", "value": "0123456789ABCDEFGH", "narrow": 4}]
        )
        arr = np.asarray(ReceiptRenderer.from_spec(spec).render(spec)) < 200
        assert np.where(arr.any(axis=0))[0].max() < spec["style"]["width"]


class TestTextWrapping:
    def test_wraps_on_word_boundaries(self):
        font = load_font("mono", 20)
        lines = wrap_text("alpha beta gamma delta", font, 120)
        assert len(lines) > 1
        assert all(font.getlength(line) <= 120 for line in lines)

    def test_hard_breaks_an_unbreakable_token(self):
        font = load_font("mono", 20)
        lines = wrap_text("X" * 60, font, 100)
        assert len(lines) > 1
        assert all(font.getlength(line) <= 100 for line in lines)

    def test_preserves_explicit_newlines(self):
        assert wrap_text("a\nb", load_font("mono", 20), 500) == ["a", "b"]


class TestBarcodeRoundTrip:
    """Decode the rendered pixels back to the source string.

    Encoding correctly is not the same as rendering scannably - this reads the
    bar/space run lengths straight off the canvas, which catches off-by-one
    widths and bar/space phase errors that a table-only test would miss.
    """

    @staticmethod
    def _decode(img: Image.Image) -> str:
        arr = np.asarray(img) < 128
        rows = np.where(arr.any(axis=1))[0]
        scan = arr[(rows.min() + rows.max()) // 2]  # midline through the bars

        cols = np.where(scan)[0]
        scan = scan[cols.min():cols.max() + 1]  # trim to the barcode extent

        # Run-length encode; the first run is a bar because we trimmed to ink.
        boundaries = np.where(np.diff(scan))[0] + 1
        runs = np.diff([0, *boundaries, len(scan)])

        threshold = (runs.min() + runs.max()) / 2
        widths = "".join("w" if r > threshold else "n" for r in runs)

        lookup = {v: k for k, v in CODE39.items()}
        chars = [lookup[widths[i:i + 9]] for i in range(0, len(widths), 10)]
        assert chars[0] == "*" and chars[-1] == "*", "guard characters missing"
        return "".join(chars[1:-1])

    def test_rendered_barcode_decodes_to_its_value(self):
        value = "PL8842217-0087"
        spec = simple_spec(
            style={"width": 576},
            elements=[{"type": "barcode", "value": value, "caption": False}],
        )
        assert self._decode(ReceiptRenderer.from_spec(spec).render(spec)) == value

    def test_decodes_after_shrink_to_fit(self):
        value = "0123456789ABCDEFGH"
        spec = simple_spec(
            style={"width": 576},
            elements=[{"type": "barcode", "value": value, "narrow": 4, "caption": False}],
        )
        assert self._decode(ReceiptRenderer.from_spec(spec).render(spec)) == value

    def test_all_symbols_are_uniquely_decodable(self):
        assert len(set(CODE39.values())) == len(CODE39)
