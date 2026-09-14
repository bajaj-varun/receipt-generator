#!/usr/bin/env python3
"""CLI for the thermal receipt generator (Phase 1 deliverable).

    python generate_receipt.py --data examples/pine_labs_pos.json
    python generate_receipt.py --data examples/pine_labs_pos.json --preset heavy --seed 7
    python generate_receipt.py --data examples/pine_labs_pos.json --count 20 --format jpeg

Use --count to emit a seeded batch of variants from one template - each variant
gets seed, seed+1, ... so the whole batch is reproducible.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from PIL import Image

from src.core import PRESETS, flatten, render_receipt

DEFAULT_OUTPUT_DIR = Path("output")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="generate_receipt.py",
        description="Render realistic worn thermal receipts from a JSON template.",
    )
    parser.add_argument("--data", required=True, type=Path,
                        help="Path to the receipt template JSON")
    parser.add_argument("--out", type=Path, default=None,
                        help="Output file (single) or directory (with --count). "
                             f"Default: {DEFAULT_OUTPUT_DIR}/<template-name>.<ext>")
    parser.add_argument("--preset", choices=sorted(PRESETS), default=None,
                        help="Wear preset; overrides the template's degradation.preset")
    parser.add_argument("--seed", type=int, default=None,
                        help="RNG seed for reproducible wear")
    parser.add_argument("--count", type=int, default=1,
                        help="Number of variants to generate (default: 1)")
    parser.add_argument("--format", choices=("png", "jpeg"), default="png",
                        help="png keeps the torn edge transparent; jpeg flattens it")
    parser.add_argument("--background", default="228,226,221",
                        help="R,G,B surface colour composited behind JPEG output")
    parser.add_argument("--quality", type=int, default=92, help="JPEG quality (default: 92)")
    parser.add_argument("--no-degrade", action="store_true",
                        help="Skip the wear pipeline and emit the clean ink map")
    parser.add_argument("--quiet", action="store_true", help="Suppress per-file logging")
    return parser.parse_args(argv)


def parse_background(value: str) -> tuple[int, int, int]:
    parts = value.split(",")
    if len(parts) != 3:
        raise SystemExit(f"--background expects 'R,G,B', got {value!r}")
    try:
        rgb = tuple(int(p) for p in parts)
    except ValueError:
        raise SystemExit(f"--background expects integers, got {value!r}") from None
    if not all(0 <= c <= 255 for c in rgb):
        raise SystemExit(f"--background channels must be 0-255, got {value!r}")
    return rgb  # type: ignore[return-value]


def load_spec(path: Path) -> dict:
    if not path.is_file():
        raise SystemExit(f"Template not found: {path}")
    try:
        spec = json.loads(path.read_text())
    except json.JSONDecodeError as exc:
        raise SystemExit(f"{path}: invalid JSON - {exc}") from None
    if not isinstance(spec, dict):
        raise SystemExit(f"{path}: template must be a JSON object")
    return spec


def resolve_paths(args: argparse.Namespace, stem: str) -> list[Path]:
    ext = "png" if args.format == "png" else "jpg"
    if args.count > 1:
        directory = args.out or DEFAULT_OUTPUT_DIR
        if directory.suffix:
            raise SystemExit("--out must be a directory when --count > 1")
        directory.mkdir(parents=True, exist_ok=True)
        return [directory / f"{stem}_{i:03d}.{ext}" for i in range(args.count)]

    target = args.out or (DEFAULT_OUTPUT_DIR / f"{stem}.{ext}")
    target.parent.mkdir(parents=True, exist_ok=True)
    return [target]


def save(image: Image.Image, path: Path, args: argparse.Namespace) -> None:
    if args.format == "jpeg":
        if image.mode == "RGBA":
            image = flatten(image, parse_background(args.background))
        image.convert("RGB").save(path, "JPEG", quality=args.quality, subsampling=0)
    else:
        image.save(path, "PNG", optimize=True)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.count < 1:
        raise SystemExit("--count must be >= 1")

    spec = load_spec(args.data)
    targets = resolve_paths(args, args.data.stem)

    for index, target in enumerate(targets):
        seed = None if args.seed is None else args.seed + index
        started = time.perf_counter()
        try:
            image = render_receipt(
                spec, preset=args.preset, seed=seed, degrade=not args.no_degrade
            )
        except (ValueError, FileNotFoundError) as exc:
            raise SystemExit(f"{args.data}: {exc}") from None
        save(image, target, args)
        if not args.quiet:
            elapsed = (time.perf_counter() - started) * 1000
            print(f"{target}  {image.width}x{image.height}  {elapsed:.0f}ms"
                  + (f"  seed={seed}" if seed is not None else ""))
    return 0


if __name__ == "__main__":
    sys.exit(main())
