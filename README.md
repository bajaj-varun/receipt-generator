# Thermal Receipt Generator

Generates realistic worn thermal POS receipts from JSON templates — a clean
Pillow layout pass followed by a physically-ordered degradation pipeline
(print defects → paper substrate → handling → capture).

Built for synthetic training/test data: OCR and receipt-parsing models need
worn, faded, creased examples, and real ones are hard to collect at volume.
Every render is reproducible from a `(template, seed)` pair.

**Status: Phase 1 complete** (local generation core). See [Roadmap](#roadmap).

---

## Quick start

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

python generate_receipt.py --data examples/pine_labs_pos.json --seed 42
# -> output/pine_labs_pos.png
```

### CLI

```bash
python generate_receipt.py --data TEMPLATE.json [options]
```

| Option | Description |
| --- | --- |
| `--data PATH` | Receipt template JSON (required) |
| `--out PATH` | Output file, or directory when `--count > 1`. Default `output/<name>.<ext>` |
| `--preset NAME` | `pristine`, `light`, `worn`, `heavy`, `faded_receipt`, `pocket` |
| `--seed N` | Seed for reproducible wear; a batch uses `N, N+1, …` |
| `--count N` | Generate N seeded variants from one template |
| `--format png\|jpeg` | PNG keeps the torn edge transparent; JPEG flattens onto `--background` |
| `--background R,G,B` | Surface colour behind JPEG output (default `228,226,221`) |
| `--quality N` | JPEG quality (default 92) |
| `--no-degrade` | Emit the clean grayscale ink map — useful as an OCR ground-truth pair |
| `--quiet` | Suppress per-file logging |

Generating a labelled dataset — a worn image plus its clean counterpart:

```bash
python generate_receipt.py --data examples/retail_itemised.json --count 200 --seed 0 --out data/worn
python generate_receipt.py --data examples/retail_itemised.json --no-degrade --out data/clean.png
```

### Python API

```python
from src.core import render_receipt

image = render_receipt(spec, preset="worn", seed=42)   # RGBA, torn edge in alpha
clean = render_receipt(spec, degrade=False)            # mode "L" ink map
```

---

## Template schema

A template is a JSON object with four optional top-level keys.

```json
{
  "style":       { "width": "80mm", "base_size": 20 },
  "side_text":   { "left": "PINE LABS", "right": "AMERICAN EXPRESS CARDS" },
  "degradation": { "preset": "worn", "fade_direction": "bottom" },
  "elements":    [ { "type": "text", "value": "CAFE MONTMARTRE" } ]
}
```

### `style`

| Key | Default | Notes |
| --- | --- | --- |
| `width` | `576` | px, or `"58mm"` (384) / `"80mm"` (576) at 203 dpi |
| `base_size` | `20` | Base font size; the `size` keywords scale off this |
| `margin_x` / `margin_top` / `margin_bottom` | `22` / `28` / `46` | px |
| `line_spacing` | `5` | Extra px between baselines |
| `font_regular` / `font_bold` | `mono` / `mono-bold` | Alias or a `.ttf` path |

Fonts live in `assets/fonts`. Aliases: `mono`, `mono-bold` (DejaVu Sans Mono),
`thermal`, `thermal-bold` (Liberation Mono).

Size keywords accepted anywhere a `size` is taken: `xs` `sm` `md` `lg` `xl`
`xxl`, or a float multiplier, or an integer pixel size.

### `elements`

Rendered top to bottom; the canvas is cropped to the content.

| Type | Keys |
| --- | --- |
| `text` | `value`, `align` (`left`/`center`/`right`), `size`, `bold`, `wrap`, `uppercase`, `tracking` (letter-spacing px), `margin_bottom` |
| `lines` | `values` (array of strings) + any `text` key, applied to each |
| `kv` | `key`, `value` (flush right), `leader` (e.g. `"."` for a dot leader), `size`, `bold` |
| `items` | `columns` (`key`, `label`, `width` as a fraction, `align`), `rows`, `header`, `wrap` |
| `rule` | `char` (repeated, e.g. `"-"` `"="` `"_"`) or `solid: true` + `thickness`; `padding` |
| `spacer` | `height` px |
| `image` | `path` (relative to `assets/`), `max_width`, `align`, `dither` (default true — thermal printers are 1-bit) |
| `barcode` | `value` (Code 39), `height`, `narrow`, `ratio`, `caption`, `caption_text` |

### `side_text`

Vertical text repeated down the left and/or right edge — the carrier branding
printed along the border of a charge slip. Either a plain string or an object:

```json
"side_text": {
  "left":  { "text": "PINE LABS", "size": "xs", "repeat": true, "separator": "  ·  " },
  "right": "AMERICAN EXPRESS CARDS"
}
```

Left reads bottom-to-top, right reads top-to-bottom. The body column is
narrowed automatically to make room.

### `degradation`

Start from a `preset`, then override any individual field. Precedence is
**CLI argument > template field > preset default**.

Effects run in physical order. All values are roughly `0..1` and are scaled by
the master `intensity`.

| Stage | Fields |
| --- | --- |
| Print defects | `ink_bleed`, `ink_starvation`, `dropout`, `dead_columns` (int), `streaks` |
| Paper & ageing | `thermal_fade`, `fade_direction` (`top`/`bottom`/`left`/`right`/`random`), `paper_texture`, `yellowing`, `stains` (int), `stain_strength` |
| Handling | `crumple`, `folds` (int), `vignette`, `edge_tear` |
| Capture | `rotation`, `max_rotation_deg`, `perspective`, `blur`, `noise`, `jpeg_quality` |
| Colour | `paper_color`, `ink_color` (`[r, g, b]`) |
| Overlays | `overlays` (paths under `assets/`), `overlay_strength` |

`intensity: 0` disables everything and returns the clean print on tinted paper.

`dead_columns` models burnt-out printhead elements — the full-height white
stripes that are the most recognisable thermal artifact. `edge_tear` cuts a
serrated profile into the alpha channel, matching a receipt torn against the
printer's cutter.

All masks are procedural, so no binary assets are required. Real scanned
textures can be layered in via `overlays` — see `assets/masks/README.md`.

---

## Layout

```
assets/fonts      bundled monospace TTFs
assets/masks      optional scanned overlay textures (procedural by default)
examples/         sample templates
src/core/         renderer.py (layout) · degradation.py (wear) · utils.py
src/api|db|telemetry   Phase 2-3 placeholders
tests/            pytest suite
```

`renderer.py` is deterministic and emits a clean mode-`L` ink map
(255 = paper, 0 = ink). `degradation.py` consumes that map and owns all
randomness, seeded from one RNG. Keeping the split means layout is testable
without any stochastic effects, and the same layout can be re-worn N times.

## Tests

```bash
pytest -q      # 57 passing
```

## Performance

~540 ms for an 80 mm receipt once fonts are cached (first call is slower while
the font cache warms). Low-frequency noise fields are built at half resolution
and upsampled; paper grain is built at full resolution because it is
near-pixel-scale. Further cold-start work is Phase 5.

---

## Roadmap

- [x] **Phase 1** — Layout engine, degradation pipeline, CLI
- [ ] **Phase 2** — Template schema in Firestore, FastAPI `/render` + `/templates`, minimal UI
- [ ] **Phase 3** — OpenTelemetry spans/logs to Cloud Trace & Cloud Logging
- [ ] **Phase 4** — Dockerfile, GCS bucket, Artifact Registry, Cloud Run deploy
- [ ] **Phase 5** — Calibration against real samples, cold-start tuning

The `elements` / `style` / `degradation` structures above are already the
wire format Phase 2 will mirror as Pydantic models in `src/api/schemas.py` and
store in Firestore, so templates written now carry forward unchanged.
