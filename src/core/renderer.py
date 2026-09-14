"""Pillow-based POS receipt composition engine.

Renders a template spec (plain dicts, so it maps 1:1 onto the JSON template
schema Phase 2 will store in Firestore) into a clean grayscale "ink map":
mode "L", 255 = bare paper, 0 = full ink. The degradation pipeline consumes
that map; keeping the two stages separate means layout is deterministic and
unit-testable without any of the random wear effects.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from PIL import Image, ImageDraw, ImageFont

from .utils import (
    align_x,
    load_font,
    line_height,
    scaled_size,
    text_width,
    wrap_text,
)

# Common thermal paper widths at 203 dpi.
PAPER_WIDTHS = {"58mm": 384, "80mm": 576}

# Scratch canvas height; the receipt is cropped to its real content afterwards.
MAX_CANVAS_HEIGHT = 20000

WHITE = 255
BLACK = 0


@dataclass
class RenderStyle:
    """Typography + geometry for one receipt."""

    width: int = 576
    margin_x: int = 22
    margin_top: int = 28
    margin_bottom: int = 46
    base_size: int = 20
    line_spacing: int = 5
    font_regular: str = "mono"
    font_bold: str = "mono-bold"

    @classmethod
    def from_spec(cls, spec: dict[str, Any]) -> "RenderStyle":
        raw = dict(spec.get("style", {}))
        width = raw.pop("width", spec.get("width", cls.width))
        if isinstance(width, str):
            if width not in PAPER_WIDTHS:
                raise ValueError(f"Unknown paper width {width!r}; use {sorted(PAPER_WIDTHS)} or px")
            width = PAPER_WIDTHS[width]
        known = {f.name for f in cls.__dataclass_fields__.values()}
        unknown = set(raw) - known
        if unknown:
            raise ValueError(f"Unknown style keys: {sorted(unknown)}")
        return cls(width=int(width), **raw)

    def font(self, bold: bool = False, size: str | int | float = "md") -> ImageFont.FreeTypeFont:
        name = self.font_bold if bold else self.font_regular
        return load_font(name, scaled_size(self.base_size, size))


@dataclass
class _Cursor:
    """Mutable draw state threaded through the element handlers."""

    y: int
    box: tuple[int, int]  # (x0, x1) of the printable column

    @property
    def content_width(self) -> int:
        return self.box[1] - self.box[0]


class ReceiptRenderer:
    """Turns a template spec into a clean ink map."""

    def __init__(self, style: RenderStyle | None = None) -> None:
        self.style = style or RenderStyle()
        self._handlers: dict[str, Callable[..., None]] = {
            "text": self._draw_text,
            "lines": self._draw_lines,
            "kv": self._draw_kv,
            "items": self._draw_items,
            "rule": self._draw_rule,
            "spacer": self._draw_spacer,
            "image": self._draw_image,
            "barcode": self._draw_barcode,
        }

    # -- public API --------------------------------------------------------

    @classmethod
    def from_spec(cls, spec: dict[str, Any]) -> "ReceiptRenderer":
        return cls(RenderStyle.from_spec(spec))

    def render(self, spec: dict[str, Any]) -> Image.Image:
        """Render `spec` and return the cropped grayscale ink map."""
        style = self.style
        gutter_l, gutter_r = self._gutters(spec)

        canvas = Image.new("L", (style.width, MAX_CANVAS_HEIGHT), WHITE)
        draw = ImageDraw.Draw(canvas)
        cursor = _Cursor(
            y=style.margin_top,
            box=(style.margin_x + gutter_l, style.width - style.margin_x - gutter_r),
        )

        for index, element in enumerate(spec.get("elements", [])):
            kind = element.get("type")
            handler = self._handlers.get(kind)
            if handler is None:
                raise ValueError(
                    f"element[{index}]: unknown type {kind!r}; "
                    f"supported: {sorted(self._handlers)}"
                )
            if cursor.y > MAX_CANVAS_HEIGHT - 200:
                raise ValueError(f"Receipt exceeds max canvas height ({MAX_CANVAS_HEIGHT}px)")
            handler(draw, canvas, element, cursor)

        height = min(cursor.y + style.margin_bottom, MAX_CANVAS_HEIGHT)
        receipt = canvas.crop((0, 0, style.width, height))
        self._draw_side_text(receipt, spec)
        return receipt

    # -- element handlers --------------------------------------------------

    def _draw_text(self, draw, canvas, el, cur) -> None:
        value = str(el.get("value", ""))
        if el.get("uppercase"):
            value = value.upper()
        font = self.style.font(bold=el.get("bold", False), size=el.get("size", "md"))
        align = el.get("align", "left")
        tracking = int(el.get("tracking", 0))

        lines = wrap_text(value, font, cur.content_width) if el.get("wrap", True) else [value]
        lh = line_height(font) + self.style.line_spacing + int(el.get("extra_spacing", 0))

        for line in lines:
            if tracking:
                self._draw_tracked(draw, line, font, cur, align, tracking)
            else:
                x = align_x(cur.box, text_width(font, line), align)
                draw.text((x, cur.y), line, font=font, fill=BLACK)
            cur.y += lh
        cur.y += int(el.get("margin_bottom", 0))

    def _draw_tracked(self, draw, line, font, cur, align, tracking) -> None:
        """Letter-spaced text - POS headers are often printed expanded."""
        total = sum(text_width(font, ch) + tracking for ch in line) - tracking
        x = align_x(cur.box, total, align)
        for ch in line:
            draw.text((x, cur.y), ch, font=font, fill=BLACK)
            x += text_width(font, ch) + tracking

    def _draw_lines(self, draw, canvas, el, cur) -> None:
        shared = {k: v for k, v in el.items() if k not in ("type", "values")}
        for value in el.get("values", []):
            self._draw_text(draw, canvas, {**shared, "type": "text", "value": value}, cur)

    def _draw_kv(self, draw, canvas, el, cur) -> None:
        """Key on the left, value flush right, optional dot leader between."""
        font = self.style.font(bold=el.get("bold", False), size=el.get("size", "md"))
        key = str(el.get("key", ""))
        value = str(el.get("value", ""))
        leader = el.get("leader", "")

        x0, x1 = cur.box
        key_w, val_w = text_width(font, key), text_width(font, value)
        draw.text((x0, cur.y), key, font=font, fill=BLACK)
        draw.text((x1 - val_w, cur.y), value, font=font, fill=BLACK)

        if leader:
            gap = (x1 - val_w) - (x0 + key_w) - 8
            unit = text_width(font, leader)
            if gap > unit > 0:
                fill_text = leader * int(gap // unit)
                draw.text((x0 + key_w + 4, cur.y), fill_text, font=font, fill=BLACK)

        cur.y += line_height(font) + self.style.line_spacing + int(el.get("margin_bottom", 0))

    def _draw_items(self, draw, canvas, el, cur) -> None:
        """Line-item table. Column widths are fractions of the content width."""
        columns = el.get("columns") or []
        rows = el.get("rows") or []
        if not columns:
            raise ValueError("items element requires 'columns'")

        font = self.style.font(bold=False, size=el.get("size", "md"))
        head_font = self.style.font(bold=True, size=el.get("size", "md"))
        widths = self._column_widths(columns, cur.content_width)
        lh = line_height(font) + self.style.line_spacing

        if el.get("header", True):
            self._draw_row(
                draw, cur, head_font, widths, columns,
                {c["key"]: str(c.get("label", c["key"])).upper() for c in columns}, wrap=False,
            )
            cur.y += lh

        for row in rows:
            used = self._draw_row(draw, cur, font, widths, columns, row, wrap=el.get("wrap", True))
            cur.y += lh * used
        cur.y += int(el.get("margin_bottom", 0))

    def _draw_row(self, draw, cur, font, widths, columns, row, wrap: bool) -> int:
        """Draw one table row; returns the number of text lines it consumed."""
        cells: list[list[str]] = []
        for column, width in zip(columns, widths):
            text = "" if row.get(column["key"]) is None else str(row.get(column["key"]))
            cells.append(wrap_text(text, font, width) if wrap else [text])

        used = max(len(c) for c in cells)
        lh = line_height(font) + self.style.line_spacing
        x = cur.box[0]
        for column, width, lines in zip(columns, widths, cells):
            align = column.get("align", "left")
            for i, line in enumerate(lines):
                draw.text(
                    (align_x((x, x + width), text_width(font, line), align), cur.y + i * lh),
                    line, font=font, fill=BLACK,
                )
            x += width
        return used

    @staticmethod
    def _column_widths(columns: list[dict], content_width: int) -> list[int]:
        weights = [float(c.get("width", 0)) for c in columns]
        if not any(weights):
            weights = [1.0] * len(columns)
        total = sum(weights)
        widths = [max(1, int(content_width * w / total)) for w in weights]
        widths[-1] += content_width - sum(widths)  # absorb rounding drift
        return widths

    def _draw_rule(self, draw, canvas, el, cur) -> None:
        """Separator: a repeated character (the POS classic) or a solid line."""
        char = el.get("char", "-")
        pad = int(el.get("padding", 4))
        cur.y += pad
        x0, x1 = cur.box

        if el.get("solid") or not char:
            thickness = int(el.get("thickness", 2))
            draw.rectangle([x0, cur.y, x1, cur.y + thickness - 1], fill=BLACK)
            cur.y += thickness + pad
            return

        font = self.style.font(bold=el.get("bold", False), size=el.get("size", "md"))
        unit = text_width(font, char)
        if unit <= 0:
            raise ValueError(f"rule char {char!r} has zero width")
        draw.text((x0, cur.y), char * int((x1 - x0) // unit), font=font, fill=BLACK)
        cur.y += line_height(font) + pad

    def _draw_spacer(self, draw, canvas, el, cur) -> None:
        cur.y += int(el.get("height", 12))

    def _draw_image(self, draw, canvas, el, cur) -> None:
        """Paste a logo. Thermal printers are 1-bit, so dither by default."""
        path = Path(el["path"])
        if not path.is_absolute():
            from .utils import ASSETS_DIR
            path = ASSETS_DIR / path
        logo = Image.open(path).convert("L")

        max_w = int(el.get("max_width", cur.content_width))
        max_w = min(max_w, cur.content_width)
        if logo.width > max_w:
            logo = logo.resize((max_w, round(logo.height * max_w / logo.width)), Image.LANCZOS)
        if el.get("dither", True):
            logo = logo.convert("1", dither=Image.FLOYDSTEINBERG).convert("L")

        x = align_x(cur.box, logo.width, el.get("align", "center"))
        canvas.paste(logo, (x, cur.y))
        cur.y += logo.height + int(el.get("margin_bottom", 8))

    def _draw_barcode(self, draw, canvas, el, cur) -> None:
        """Code 39 barcode - the format most POS terminals print for a bill ref."""
        value = str(el.get("value", "")).upper()
        height = int(el.get("height", 70))
        narrow = max(1, int(el.get("narrow", 2)))
        wide = narrow * int(el.get("ratio", 3))

        widths = [wide if e == "w" else narrow for e in encode_code39(value)]
        total = sum(widths)
        if total > cur.content_width:  # shrink to fit rather than overflow
            narrow = max(1, int(narrow * cur.content_width / total))
            wide = max(narrow + 1, narrow * int(el.get("ratio", 3)))
            widths = [wide if e == "w" else narrow for e in encode_code39(value)]
            total = sum(widths)

        x = align_x(cur.box, total, el.get("align", "center"))
        for i, width in enumerate(widths):
            if i % 2 == 0:  # even index = bar, odd = space
                draw.rectangle([x, cur.y, x + width - 1, cur.y + height - 1], fill=BLACK)
            x += width
        cur.y += height + 6

        if el.get("caption", True):
            font = self.style.font(size=el.get("caption_size", "sm"))
            label = el.get("caption_text", value)
            draw.text((align_x(cur.box, text_width(font, label), "center"), cur.y),
                      label, font=font, fill=BLACK)
            cur.y += line_height(font) + self.style.line_spacing

    # -- vertical edge text ------------------------------------------------

    def _gutters(self, spec: dict[str, Any]) -> tuple[int, int]:
        """Horizontal space reserved for the rotated side-text strips."""
        side = spec.get("side_text") or {}
        out = []
        for edge in ("left", "right"):
            cfg = side.get(edge)
            if not cfg:
                out.append(0)
                continue
            cfg = {"text": cfg} if isinstance(cfg, str) else cfg
            font = self.style.font(bold=cfg.get("bold", False), size=cfg.get("size", "xs"))
            out.append(line_height(font) + int(cfg.get("padding", 6)))
        return tuple(out)  # type: ignore[return-value]

    def _draw_side_text(self, receipt: Image.Image, spec: dict[str, Any]) -> None:
        """Repeat a label vertically down the left/right edge, e.g. 'PINE LABS'."""
        side = spec.get("side_text") or {}
        for edge in ("left", "right"):
            cfg = side.get(edge)
            if not cfg:
                continue
            cfg = {"text": cfg} if isinstance(cfg, str) else cfg
            font = self.style.font(bold=cfg.get("bold", False), size=cfg.get("size", "xs"))
            text = str(cfg.get("text", ""))
            if not text:
                continue

            separator = cfg.get("separator", "   ")
            unit_w = text_width(font, text + separator)
            band_h = line_height(font)
            repeats = max(1, -(-receipt.height // max(unit_w, 1))) if cfg.get("repeat", True) else 1

            strip = Image.new("L", (max(unit_w * repeats, receipt.height), band_h), WHITE)
            ImageDraw.Draw(strip).text((0, 0), (text + separator) * repeats, font=font, fill=BLACK)

            # Read bottom-to-top on the left edge, top-to-bottom on the right.
            rotated = strip.rotate(90 if edge == "left" else -90, expand=True)
            top = 0 if edge == "left" else max(0, receipt.height - rotated.height)
            rotated = rotated.crop((0, max(0, rotated.height - receipt.height) if edge == "left" else 0,
                                   rotated.width, rotated.height if edge == "left" else receipt.height))
            x = int(cfg.get("padding", 6)) // 2 if edge == "left" else receipt.width - rotated.width - 3
            receipt.paste(rotated, (max(0, x), 0))


# --------------------------------------------------------------------------
# Code 39 (each character is 5 bars + 4 spaces; 'n' narrow, 'w' wide)
# --------------------------------------------------------------------------

CODE39 = {
    "0": "nnnwwnwnn", "1": "wnnwnnnnw", "2": "nnwwnnnnw", "3": "wnwwnnnnn",
    "4": "nnnwwnnnw", "5": "wnnwwnnnn", "6": "nnwwwnnnn", "7": "nnnwnnwnw",
    "8": "wnnwnnwnn", "9": "nnwwnnwnn", "A": "wnnnnwnnw", "B": "nnwnnwnnw",
    "C": "wnwnnwnnn", "D": "nnnnwwnnw", "E": "wnnnwwnnn", "F": "nnwnwwnnn",
    "G": "nnnnnwwnw", "H": "wnnnnwwnn", "I": "nnwnnwwnn", "J": "nnnnwwwnn",
    "K": "wnnnnnnww", "L": "nnwnnnnww", "M": "wnwnnnnwn", "N": "nnnnwnnww",
    "O": "wnnnwnnwn", "P": "nnwnwnnwn", "Q": "nnnnnnwww", "R": "wnnnnnwwn",
    "S": "nnwnnnwwn", "T": "nnnnwnwwn", "U": "wwnnnnnnw", "V": "nwwnnnnnw",
    "W": "wwwnnnnnn", "X": "nwnnwnnnw", "Y": "wwnnwnnnn", "Z": "nwwnwnnnn",
    "-": "nwnnnnwnw", ".": "wwnnnnwnn", " ": "nwwnnnnwn", "$": "nwnwnwnnn",
    "/": "nwnwnnnwn", "+": "nwnnnwnwn", "%": "nnnwnwnwn", "*": "nwnnwnwnn",
}


def encode_code39(value: str) -> str:
    """Return the bar/space width pattern ('n'/'w') including start/stop guards."""
    value = value.upper()
    invalid = sorted({c for c in value if c not in CODE39 or c == "*"})
    if invalid:
        raise ValueError(f"Characters not encodable in Code 39: {invalid}")
    parts = [CODE39["*"]] + [CODE39[c] for c in value] + [CODE39["*"]]
    return "n".join(parts)  # narrow inter-character space between symbols
