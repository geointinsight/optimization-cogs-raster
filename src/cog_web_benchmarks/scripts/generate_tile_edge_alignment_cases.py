#!/usr/bin/env python3
"""Generate same-size BBOX cases positioned on selected COG tile boundaries.

The generated BBOXs use the source raster transform and are validated through
the same integer-window logic used by the S3 benchmark.  This isolates tile
grid alignment: every case has the same pixel dimensions, but may intersect a
different number of blocks depending on block size.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import rasterio
from rasterio.windows import Window, bounds

from cog_s3_bbox_benchmark import BenchmarkError, bbox_window_direct, tiles_touched


BLOCKS = (128, 256, 512)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-cog",
        type=Path,
        default=Path("benchmark_output/cogs/6a3bfdd89e3416fefefdf90b.tif"),
        help="Local reference COG used only to derive the raster grid.",
    )
    parser.add_argument("--output-csv", type=Path, default=Path("tile_edge_alignment_cases.csv"))
    parser.add_argument("--output-geojson", type=Path, default=Path("tile_edge_alignment_cases.geojson"))
    parser.add_argument("--validation-csv", type=Path, default=Path("tile_edge_alignment_validation.csv"))
    parser.add_argument("--width-px", type=int, default=24, help="Common BBOX width in raster pixels.")
    parser.add_argument("--height-px", type=int, default=18, help="Common BBOX height in raster pixels.")
    parser.add_argument(
        "--anchor-col",
        type=int,
        help="Pixel column aligned to a 512 px boundary; defaults near raster centre.",
    )
    parser.add_argument(
        "--anchor-row",
        type=int,
        help="Pixel row aligned to a 512 px boundary; defaults near raster centre.",
    )
    return parser.parse_args()


def aligned_anchor(value: int) -> int:
    return value // 512 * 512


def assert_even_dimensions(width: int, height: int) -> None:
    if width <= 0 or height <= 0 or width % 2 or height % 2:
        raise BenchmarkError("--width-px and --height-px must be positive even integers")


def case_definitions(anchor_col: int, anchor_row: int) -> list[tuple[str, str, int, int]]:
    """Return name, description, centre column, centre row for every case."""
    interior_col, interior_row = anchor_col + 64, anchor_row + 64
    return [
        ("interior", "Inside a 128 px tile; no grid boundary crossed", interior_col, interior_row),
        ("cross_128_edge", "Crosses a 128 px vertical boundary only", anchor_col + 128, interior_row),
        ("cross_256_edge", "Crosses a 256 px vertical boundary (also a 128 px boundary)", anchor_col + 256, interior_row),
        ("cross_512_edge", "Crosses a 512 px vertical boundary shared by all tested grids", anchor_col + 512, interior_row),
        ("cross_128_corner", "Crosses a 128 px grid corner only", anchor_col + 128, anchor_row + 128),
        ("cross_256_corner", "Crosses a 256 px grid corner (also a 128 px corner)", anchor_col + 256, anchor_row + 256),
        ("cross_512_corner", "Crosses a 512 px grid corner shared by all tested grids", anchor_col + 512, anchor_row + 512),
    ]


def bbox_for_center(src: rasterio.DatasetReader, centre_col: int, centre_row: int, width: int, height: int) -> tuple[float, float, float, float]:
    window = Window(centre_col - width // 2, centre_row - height // 2, width, height)
    if window.col_off < 0 or window.row_off < 0 or window.col_off + window.width > src.width or window.row_off + window.height > src.height:
        raise BenchmarkError(f"Generated window falls outside raster: {window}")
    return tuple(float(value) for value in bounds(window, src.transform))


def expected_tile_counts(src: rasterio.DatasetReader, bbox: tuple[float, float, float, float]) -> dict[int, int]:
    window = bbox_window_direct(src, bbox)
    return {block: tiles_touched(src, window) if src.block_shapes[0] == (block, block) else tile_count(window, block) for block in BLOCKS}


def tile_count(window: Window, block: int) -> int:
    col_start = int(window.col_off) // block
    row_start = int(window.row_off) // block
    col_end = int(window.col_off + window.width - 1) // block
    row_end = int(window.row_off + window.height - 1) // block
    return (col_end - col_start + 1) * (row_end - row_start + 1)


def feature(row: dict[str, Any]) -> dict[str, Any]:
    min_x, min_y, max_x, max_y = (float(row[name]) for name in ("min_x", "min_y", "max_x", "max_y"))
    properties = {key: value for key, value in row.items() if key not in {"min_x", "min_y", "max_x", "max_y"}}
    return {
        "type": "Feature",
        "properties": properties,
        "geometry": {
            "type": "Polygon",
            "coordinates": [[[min_x, min_y], [max_x, min_y], [max_x, max_y], [min_x, max_y], [min_x, min_y]]],
        },
    }


def main() -> int:
    args = parse_args()
    assert_even_dimensions(args.width_px, args.height_px)
    if not args.input_cog.is_file():
        raise BenchmarkError(f"Reference COG does not exist: {args.input_cog}")

    with rasterio.open(args.input_cog) as src:
        if src.crs is None:
            raise BenchmarkError("Reference COG must have a CRS")
        anchor_col = args.anchor_col if args.anchor_col is not None else aligned_anchor(src.width // 2)
        anchor_row = args.anchor_row if args.anchor_row is not None else aligned_anchor(src.height // 2)
        if anchor_col % 512 or anchor_row % 512:
            raise BenchmarkError("--anchor-col and --anchor-row must both align to 512 px")
        rows: list[dict[str, Any]] = []
        validation: list[dict[str, Any]] = []
        for scenario, description, centre_col, centre_row in case_definitions(anchor_col, anchor_row):
            bbox = bbox_for_center(src, centre_col, centre_row, args.width_px, args.height_px)
            window = bbox_window_direct(src, bbox)
            expected = {block: tile_count(window, block) for block in BLOCKS}
            row = {
                "scenario": scenario,
                "description": description,
                "min_x": f"{bbox[0]:.15f}",
                "min_y": f"{bbox[1]:.15f}",
                "max_x": f"{bbox[2]:.15f}",
                "max_y": f"{bbox[3]:.15f}",
                "relative_to_reference": "same-size alignment control",
                "width_px": args.width_px,
                "height_px": args.height_px,
                "centre_col_px": centre_col,
                "centre_row_px": centre_row,
                "expected_tiles_128": expected[128],
                "expected_tiles_256": expected[256],
                "expected_tiles_512": expected[512],
            }
            rows.append(row)
            for block, count in expected.items():
                validation.append({
                    "scenario": scenario,
                    "blocksize": block,
                    "window_col_off": int(window.col_off),
                    "window_row_off": int(window.row_off),
                    "window_width_px": int(window.width),
                    "window_height_px": int(window.height),
                    "expected_tiles_touched": count,
                })

    fieldnames = list(rows[0])
    for path in (args.output_csv, args.validation_csv, args.output_geojson):
        path.parent.mkdir(parents=True, exist_ok=True)
    with args.output_csv.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    with args.validation_csv.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(validation[0]))
        writer.writeheader()
        writer.writerows(validation)
    args.output_geojson.write_text(json.dumps({"type": "FeatureCollection", "features": [feature(row) for row in rows]}, indent=2), encoding="utf-8")
    print(f"Wrote {args.output_csv} ({len(rows)} cases)")
    print(f"Wrote {args.validation_csv} ({len(validation)} expected counts)")
    print(f"Wrote {args.output_geojson}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except BenchmarkError as error:
        print(f"ERROR: {error}")
        raise SystemExit(2)
