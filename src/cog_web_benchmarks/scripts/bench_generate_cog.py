#!/usr/bin/env python3
"""Benchmark COG creation and BBOX reads across compression/blocksize variants.

The benchmark uses one BBOX derived from a parcel file for every variant. It
creates COGs with the GDAL command-line tools, measures cold/reopen and
warm/reuse reads with Rasterio, and writes CSV/PNG results.

Example:
    python cog_benchmark_cli.py \
        --input-raster download-4/6a3bfdd89e3416fefefdf90b.tif \
        --parcel data_plot/plot.geojson \
        --output-dir benchmark_output

To benchmark pre-existing COGs instead of creating them, provide nine
``--raster variant=path`` arguments and use ``--skip-create --mode bbox``.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import platform
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from statistics import mean
from typing import Any, Iterable

import fiona
import numpy as np
import rasterio
from rasterio.crs import CRS
from rasterio.warp import transform_bounds
from rasterio.windows import Window, from_bounds


VARIANTS: tuple[dict[str, Any], ...] = (
    {"variant": "lzw_128", "compression": "LZW", "level": None, "blocksize": 128},
    {"variant": "lzw_256", "compression": "LZW", "level": None, "blocksize": 256},
    {"variant": "lzw_512", "compression": "LZW", "level": None, "blocksize": 512},
    {"variant": "deflate_128", "compression": "DEFLATE", "level": 6, "blocksize": 128},
    {"variant": "deflate_256", "compression": "DEFLATE", "level": 6, "blocksize": 256},
    {"variant": "deflate_512", "compression": "DEFLATE", "level": 6, "blocksize": 512},
    {"variant": "zstd_128", "compression": "ZSTD", "level": 22, "blocksize": 128},
    {"variant": "zstd_256", "compression": "ZSTD", "level": 22, "blocksize": 256},
    {"variant": "zstd_512", "compression": "ZSTD", "level": 22, "blocksize": 512},
)
VARIANT_BY_NAME = {item["variant"]: item for item in VARIANTS}

RAW_FIELDS = [
    "variant",
    "compression",
    "blocksize",
    "run",
    "mode",
    "file_size_mb",
    "window_width_px",
    "window_height_px",
    "tiles_touched",
    "compressed_bytes_read",
    "open_ms",
    "window_ms",
    "read_ms",
    "total_ms",
]
CREATION_FIELDS = [
    "variant",
    "compression",
    "level",
    "blocksize",
    "creation_seconds",
    "file_size_bytes",
    "file_size_mb",
    "width",
    "height",
    "dtype",
    "crs",
    "overview_count",
    "is_tiled",
]
SUMMARY_FIELDS = [
    "variant",
    "compression",
    "blocksize",
    "mode",
    "file_size_mb",
    "tiles_touched",
    "compressed_bytes_read",
    "total_p90_ms",
    "total_p95_ms",
    "total_average_ms",
    "read_p90_ms",
    "latency_reduction_pct",
    "speedup_x",
    "storage_reduction_pct",
    "creation_seconds",
]


class BenchmarkError(RuntimeError):
    """Raised for an invalid benchmark setup or failed benchmark operation."""


@dataclass(frozen=True)
class RasterInfo:
    width: int
    height: int
    dtype: str
    crs: str
    bounds: tuple[float, float, float, float]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create COG variants and benchmark identical parcel BBOX reads."
    )
    parser.add_argument(
        "--input-raster",
        type=Path,
        default=Path("download-4/6a3bfdd89e3416fefefdf90b.tif"),
        help="Source raster used to create COGs",
    )
    parser.add_argument(
        "--parcel",
        type=Path,
        default=Path("data_plot/plot.geojson"),
        help="Parcel GeoPackage or GeoJSON used to derive the BBOX",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("benchmark_output"),
        help="Directory for COGs, CSV files, and plots",
    )
    parser.add_argument(
        "--mode",
        choices=("create", "bbox", "both"),
        default="both",
        help="Benchmark creation, BBOX reads, or both",
    )
    parser.add_argument(
        "--read-mode",
        choices=("cold", "warm", "both"),
        default="both",
        help="Read mode(s) for --mode bbox/both",
    )
    parser.add_argument(
        "--reuse-open",
        action="store_true",
        help="Compatibility shortcut for --read-mode warm",
    )
    parser.add_argument("--runs", type=int, default=20, help="Measured reads per variant")
    parser.add_argument("--warmup", type=int, default=5, help="Warm-up reads per variant")
    parser.add_argument(
        "--baseline",
        choices=tuple(VARIANT_BY_NAME),
        default="lzw_512",
        help="Variant used for storage/latency comparisons",
    )
    parser.add_argument(
        "--skip-create",
        action="store_true",
        help="Do not call GDAL; use existing COGs or --raster mappings",
    )
    parser.add_argument(
        "--keep-cogs",
        action="store_true",
        help="Keep generated COG files after the benchmark",
    )
    parser.add_argument(
        "--raster",
        action="append",
        default=[],
        metavar="VARIANT=PATH",
        help="Existing COG mapping; may be repeated for the nine variants",
    )
    return parser.parse_args()


def write_csv(path: Path, rows: Iterable[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def parse_raster_mappings(values: list[str]) -> dict[str, Path]:
    mappings: dict[str, Path] = {}
    for value in values:
        if "=" not in value:
            raise BenchmarkError(f"Invalid --raster value {value!r}; expected VARIANT=PATH")
        variant, path = value.split("=", 1)
        if variant not in VARIANT_BY_NAME:
            raise BenchmarkError(f"Unknown variant in --raster: {variant}")
        if variant in mappings:
            raise BenchmarkError(f"Duplicate --raster mapping: {variant}")
        mappings[variant] = Path(path)
    return mappings


def require_file(path: Path, description: str) -> None:
    if not path.is_file():
        raise BenchmarkError(f"{description} does not exist: {path}")


def raster_info(path: Path) -> RasterInfo:
    require_file(path, "Input raster")
    with rasterio.open(path) as src:
        if src.count != 1:
            raise BenchmarkError(f"Expected one band in {path}, found {src.count}")
        if src.crs is None:
            raise BenchmarkError(f"Raster has no CRS: {path}")
        return RasterInfo(
            width=src.width,
            height=src.height,
            dtype=src.dtypes[0],
            crs=src.crs.to_string(),
            bounds=(src.bounds.left, src.bounds.bottom, src.bounds.right, src.bounds.top),
        )


def coordinate_bounds(coordinates: Any) -> tuple[float, float, float, float] | None:
    """Find bounds in arbitrarily nested GeoJSON coordinate arrays."""
    if not isinstance(coordinates, (list, tuple)) or not coordinates:
        return None
    first = coordinates[0]
    if isinstance(first, (int, float)) and len(coordinates) >= 2:
        x, y = float(coordinates[0]), float(coordinates[1])
        return x, y, x, y

    bounds: list[tuple[float, float, float, float]] = []
    for child in coordinates:
        child_bounds = coordinate_bounds(child)
        if child_bounds:
            bounds.append(child_bounds)
    if not bounds:
        return None
    return (
        min(item[0] for item in bounds),
        min(item[1] for item in bounds),
        max(item[2] for item in bounds),
        max(item[3] for item in bounds),
    )


def parcel_bbox(parcel_path: Path, target_crs: CRS) -> tuple[float, float, float, float]:
    require_file(parcel_path, "Parcel file")
    all_bounds: list[tuple[float, float, float, float]] = []
    with fiona.open(parcel_path) as source:
        if not source.crs and not source.crs_wkt:
            raise BenchmarkError(f"Parcel has no CRS: {parcel_path}")
        parcel_crs = CRS.from_user_input(source.crs_wkt or source.crs)
        for feature in source:
            geometry = feature.get("geometry")
            if geometry and geometry.get("coordinates"):
                bounds = coordinate_bounds(geometry["coordinates"])
                if bounds:
                    all_bounds.append(bounds)
    if not all_bounds:
        raise BenchmarkError(f"Parcel contains no coordinate geometry: {parcel_path}")
    bounds = (
        min(item[0] for item in all_bounds),
        min(item[1] for item in all_bounds),
        max(item[2] for item in all_bounds),
        max(item[3] for item in all_bounds),
    )
    if parcel_crs != target_crs:
        bounds = transform_bounds(parcel_crs, target_crs, *bounds, densify_pts=21)
    return tuple(float(value) for value in bounds)


def get_gdal_version() -> str:
    """Return the GDAL CLI version, or raise a useful setup error."""
    translate = shutil.which("gdal_translate")
    info = shutil.which("gdalinfo")
    if translate is None or info is None:
        raise BenchmarkError(
            "gdal_translate and gdalinfo are required; install GDAL CLI or use --skip-create"
        )
    result = subprocess.run(
        [translate, "--version"], capture_output=True, text=True, check=True
    )
    return result.stdout.strip()


def creation_command(input_path: Path, output_path: Path, config: dict[str, Any]) -> list[str]:
    options = [
        "COMPRESS=" + config["compression"],
        "BLOCKSIZE=" + str(config["blocksize"]),
        "PREDICTOR=FLOATING_POINT",
        "OVERVIEWS=AUTO",
        "OVERVIEW_RESAMPLING=AVERAGE",
    ]
    if config["level"] is not None:
        options.append("LEVEL=" + str(config["level"]))
    command = ["gdal_translate", "-of", "COG"]
    for option in options:
        command.extend(["-co", option])
    command.extend([str(input_path), str(output_path)])
    return command


def inspect_cog(path: Path, config: dict[str, Any]) -> dict[str, Any]:
    require_file(path, "COG output")
    with rasterio.open(path) as src:
        if src.count != 1:
            raise BenchmarkError(f"{config['variant']} does not have one band")
        if src.crs is None:
            raise BenchmarkError(f"{config['variant']} has no CRS")
        expected_compression = config["compression"].lower()
        actual_compression = src.compression.value.lower() if src.compression else "none"
        if actual_compression != expected_compression.lower():
            raise BenchmarkError(
                f"{config['variant']} compression is {actual_compression}, "
                f"expected {expected_compression.lower()}"
            )
        block_width, block_height = src.block_shapes[0]
        if block_width != config["blocksize"] or block_height != config["blocksize"]:
            raise BenchmarkError(
                f"{config['variant']} block shape is {block_width}x{block_height}, "
                f"expected {config['blocksize']}x{config['blocksize']}"
            )
        overviews = src.overviews(1)
        if not src.profile.get("tiled") or not overviews:
            raise BenchmarkError(f"{config['variant']} is not tiled or has no overviews")
        return {
            "width": src.width,
            "height": src.height,
            "dtype": src.dtypes[0],
            "crs": src.crs.to_string(),
            "overview_count": len(overviews),
            "is_tiled": bool(src.profile.get("tiled")),
        }


def create_variants(
    input_path: Path, output_dir: Path, keep_cogs: bool
) -> tuple[dict[str, Path], list[dict[str, Any]]]:
    if shutil.which("gdal_translate") is None or shutil.which("gdalinfo") is None:
        raise BenchmarkError(
            "gdal_translate and gdalinfo are required; install GDAL CLI or use --skip-create"
        )
    cog_dir = output_dir / "cogs"
    cog_dir.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}
    rows: list[dict[str, Any]] = []
    for config in VARIANTS:
        output_path = cog_dir / f"{config['variant']}.tif"
        if output_path.exists():
            if not output_path.is_file():
                raise BenchmarkError(f"Output path is not a file: {output_path}")
            output_path.unlink()
        command = creation_command(input_path, output_path, config)
        started = time.perf_counter()
        try:
            subprocess.run(command, check=True, capture_output=True, text=True)
        except subprocess.CalledProcessError as error:
            detail = (error.stderr or error.stdout or "").strip()
            raise BenchmarkError(
                f"Failed to create {config['variant']}: {detail}"
            ) from error
        creation_seconds = time.perf_counter() - started
        metadata = inspect_cog(output_path, config)
        file_size_bytes = output_path.stat().st_size
        rows.append(
            {
                **config,
                "creation_seconds": creation_seconds,
                "file_size_bytes": file_size_bytes,
                "file_size_mb": file_size_bytes / (1024 * 1024),
                **metadata,
            }
        )
        paths[config["variant"]] = output_path
        if not keep_cogs:
            output_path.unlink()
    return paths, rows


def window_for_bbox(src: rasterio.DatasetReader, bbox: tuple[float, float, float, float]) -> Window:
    window = from_bounds(*bbox, transform=src.transform)
    window = window.round_offsets().round_lengths()
    if window.width <= 0 or window.height <= 0:
        raise BenchmarkError("BBOX produces an empty raster window")
    if (
        window.col_off < 0
        or window.row_off < 0
        or window.col_off + window.width > src.width
        or window.row_off + window.height > src.height
    ):
        raise BenchmarkError("BBOX window falls outside the raster")
    return window


def tiles_touched(src: rasterio.DatasetReader, window: Window) -> int:
    block_height, block_width = src.block_shapes[0]
    first_col = math.floor(window.col_off / block_width)
    last_col = math.ceil((window.col_off + window.width) / block_width) - 1
    first_row = math.floor(window.row_off / block_height)
    last_row = math.ceil((window.row_off + window.height) / block_height) - 1
    columns = max(0, last_col - first_col + 1)
    rows = max(0, last_row - first_row + 1)
    return columns * rows


def compressed_tile_bytes(path: Path, window: Window) -> int:
    """Sum compressed byte counts for unique full-resolution TIFF tiles.

    This is the compressed payload represented by tiles touched by the read,
    not an OS-level measurement of bytes physically fetched from storage.
    """
    try:
        import tifffile
    except ImportError as error:
        raise BenchmarkError(
            "tifffile is required to calculate compressed_bytes_read"
        ) from error
    with tifffile.TiffFile(path) as tif:
        page = tif.pages[0]
        tile_width = int(page.tilewidth)
        tile_height = int(page.tilelength)
        bytecounts = list(page.databytecounts)
        tiles_across = math.ceil(page.imagewidth / tile_width)
        first_col = math.floor(window.col_off / tile_width)
        last_col = math.ceil((window.col_off + window.width) / tile_width) - 1
        first_row = math.floor(window.row_off / tile_height)
        last_row = math.ceil((window.row_off + window.height) / tile_height) - 1
        total = 0
        for tile_row in range(first_row, last_row + 1):
            for tile_col in range(first_col, last_col + 1):
                index = tile_row * tiles_across + tile_col
                if index < 0 or index >= len(bytecounts):
                    raise BenchmarkError(f"Invalid TIFF tile index for {path}: {index}")
                total += int(bytecounts[index])
        return total


def compressed_bytes_for_bbox(
    path: Path, bbox: tuple[float, float, float, float]
) -> int:
    with rasterio.open(path) as src:
        return compressed_tile_bytes(path, window_for_bbox(src, bbox))


def one_read(
    path: Path,
    bbox: tuple[float, float, float, float],
    variant: str,
    mode: str,
    run: int,
    file_size_mb: float,
    compressed_bytes: int,
) -> dict[str, Any]:
    total_started = time.perf_counter()
    open_started = time.perf_counter()
    with rasterio.open(path) as src:
        open_ms = (time.perf_counter() - open_started) * 1000
        window_started = time.perf_counter()
        window = window_for_bbox(src, bbox)
        window_ms = (time.perf_counter() - window_started) * 1000
        read_started = time.perf_counter()
        array = src.read(1, window=window)
        read_ms = (time.perf_counter() - read_started) * 1000
        block_count = tiles_touched(src, window)
    if array.size == 0:
        raise BenchmarkError(f"Empty read result from {path}")
    total_ms = (time.perf_counter() - total_started) * 1000
    config = VARIANT_BY_NAME[variant]
    return {
        "variant": config["variant"],
        "compression": config["compression"],
        "blocksize": config["blocksize"],
        "run": run,
        "mode": mode,
        "file_size_mb": file_size_mb,
        "window_width_px": int(window.width),
        "window_height_px": int(window.height),
        "tiles_touched": block_count,
        "compressed_bytes_read": compressed_bytes,
        "open_ms": open_ms,
        "window_ms": window_ms,
        "read_ms": read_ms,
        "total_ms": total_ms,
    }


def warm_read(
    src: rasterio.DatasetReader,
    path: Path,
    bbox: tuple[float, float, float, float],
    variant: str,
    run: int,
    file_size_mb: float,
    compressed_bytes: int,
) -> dict[str, Any]:
    total_started = time.perf_counter()
    window_started = time.perf_counter()
    window = window_for_bbox(src, bbox)
    window_ms = (time.perf_counter() - window_started) * 1000
    read_started = time.perf_counter()
    array = src.read(1, window=window)
    read_ms = (time.perf_counter() - read_started) * 1000
    block_count = tiles_touched(src, window)
    if array.size == 0:
        raise BenchmarkError(f"Empty read result from {path}")
    total_ms = (time.perf_counter() - total_started) * 1000
    config = VARIANT_BY_NAME[variant]
    return {
        "variant": config["variant"],
        "compression": config["compression"],
        "blocksize": config["blocksize"],
        "run": run,
        "mode": "warm",
        "file_size_mb": file_size_mb,
        "window_width_px": int(window.width),
        "window_height_px": int(window.height),
        "tiles_touched": block_count,
        "compressed_bytes_read": compressed_bytes,
        "open_ms": 0.0,
        "window_ms": window_ms,
        "read_ms": read_ms,
        "total_ms": total_ms,
    }


def run_reads(
    paths: dict[str, Path],
    bbox: tuple[float, float, float, float],
    output_dir: Path,
    runs: int,
    warmup: int,
    read_mode: str,
) -> list[dict[str, Any]]:
    raw_rows: list[dict[str, Any]] = []
    modes = [read_mode] if read_mode != "both" else ["cold", "warm"]
    for config in VARIANTS:
        path = paths[config["variant"]]
        file_size_mb = path.stat().st_size / (1024 * 1024)
        compressed_bytes = compressed_bytes_for_bbox(path, bbox)
        for mode in modes:
            if mode == "cold":
                for _ in range(warmup):
                    one_read(
                        path,
                        bbox,
                        config["variant"],
                        mode,
                        0,
                        file_size_mb,
                        compressed_bytes,
                    )
                for run in range(1, runs + 1):
                    raw_rows.append(
                        one_read(
                            path,
                            bbox,
                            config["variant"],
                            mode,
                            run,
                            file_size_mb,
                            compressed_bytes,
                        )
                    )
            else:
                # Keep one dataset handle open for the complete warm/reuse series.
                with rasterio.open(path) as src:
                    for _ in range(warmup):
                        warm_read(
                            src,
                            path,
                            bbox,
                            config["variant"],
                            0,
                            file_size_mb,
                            compressed_bytes,
                        )
                    for run in range(1, runs + 1):
                        raw_rows.append(
                            warm_read(
                                src,
                                path,
                                bbox,
                                config["variant"],
                                run,
                                file_size_mb,
                                compressed_bytes,
                            )
                        )
    write_csv(output_dir / "bbox_read_raw.csv", raw_rows, RAW_FIELDS)
    return raw_rows


def percentile(values: list[float], percent: float) -> float:
    return float(np.percentile(np.asarray(values, dtype=float), percent))


def summarize(
    raw_rows: list[dict[str, Any]],
    creation_rows: list[dict[str, Any]],
    baseline: str,
    output_dir: Path,
) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in raw_rows:
        grouped.setdefault((row["variant"], row["mode"]), []).append(row)
    creation_by_variant = {row["variant"]: row for row in creation_rows}
    summaries: list[dict[str, Any]] = []
    for (variant, mode), rows in grouped.items():
        baseline_rows = grouped.get((baseline, mode))
        if not baseline_rows:
            raise BenchmarkError(f"Baseline {baseline} has no {mode} results")
        totals = [float(row["total_ms"]) for row in rows]
        reads = [float(row["read_ms"]) for row in rows]
        baseline_p90 = percentile([float(row["total_ms"]) for row in baseline_rows], 90)
        current_p90 = percentile(totals, 90)
        current_size = float(rows[0]["file_size_mb"])
        baseline_size = float(baseline_rows[0]["file_size_mb"])
        creation = creation_by_variant.get(variant, {})
        summaries.append(
            {
                "variant": variant,
                "compression": rows[0]["compression"],
                "blocksize": rows[0]["blocksize"],
                "mode": mode,
                "file_size_mb": current_size,
                "tiles_touched": rows[0]["tiles_touched"],
                "compressed_bytes_read": rows[0]["compressed_bytes_read"],
                "total_p90_ms": current_p90,
                "total_p95_ms": percentile(totals, 95),
                "total_average_ms": mean(totals),
                "read_p90_ms": percentile(reads, 90),
                "latency_reduction_pct": (baseline_p90 - current_p90) / baseline_p90 * 100,
                "speedup_x": baseline_p90 / current_p90,
                "storage_reduction_pct": (baseline_size - current_size) / baseline_size * 100,
                "creation_seconds": creation.get("creation_seconds", ""),
            }
        )
    summaries.sort(key=lambda row: (row["mode"], row["variant"]))
    write_csv(output_dir / "bbox_read_summary.csv", summaries, SUMMARY_FIELDS)
    return summaries


def make_plots(
    creation_rows: list[dict[str, Any]], summaries: list[dict[str, Any]], output_dir: Path
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labels = [config["variant"] for config in VARIANTS]
    x = np.arange(len(labels))

    def finish_plot(name: str, title: str, ylabel: str) -> None:
        plt.xticks(x, labels, rotation=45, ha="right")
        plt.title(title)
        plt.ylabel(ylabel)
        plt.xlabel("compression + blocksize")
        plt.grid(axis="y", alpha=0.25)
        plt.tight_layout()
        plt.savefig(output_dir / name, dpi=150)
        plt.close()

    creation_by_variant = {row["variant"]: row for row in creation_rows}
    plt.figure(figsize=(11, 6))
    plt.bar(
        x,
        [float(creation_by_variant[label]["file_size_mb"]) for label in labels],
    )
    finish_plot("file_size.png", "COG file size", "File size (MB)")

    creation_times = [
        float(creation_by_variant[label]["creation_seconds"]) for label in labels
    ]
    if all(value != "" for value in creation_times):
        figure, axis = plt.subplots(figsize=(12, 6))
        colors = [
            {"LZW": "#4C78A8", "DEFLATE": "#F58518", "ZSTD": "#54A24B"}[creation_by_variant[label]["compression"]]
            for label in labels
        ]
        bars = axis.barh(labels, creation_times, color=colors)
        axis.set_xscale("log")
        axis.set_xlabel("Creation time (seconds, log scale)")
        axis.set_title("COG creation time")
        axis.grid(axis="x", which="both", alpha=0.25)
        axis.bar_label(bars, fmt="%.2f s", padding=4, fontsize=9)
        figure.tight_layout()
        figure.savefig(output_dir / "creation_time.png", dpi=150)
        plt.close(figure)

    for mode in sorted({row["mode"] for row in summaries}):
        mode_rows = {row["variant"]: row for row in summaries if row["mode"] == mode}
        p90 = [float(mode_rows[label]["total_p90_ms"]) for label in labels]
        p95 = [float(mode_rows[label]["total_p95_ms"]) for label in labels]
        offset = -0.18 if mode == "cold" else 0.18
        plt.bar(x + offset, p90, width=0.35, label=f"{mode} p90")
        plt.bar(x - offset, p95, width=0.35, label=f"{mode} p95")
    plt.legend()
    finish_plot("bbox_latency.png", "BBOX read latency", "Latency (ms)")

    plt.figure(figsize=(11, 6))
    for mode in sorted({row["mode"] for row in summaries}):
        mode_rows = {row["variant"]: row for row in summaries if row["mode"] == mode}
        plt.plot(
            x,
            [float(mode_rows[label]["compressed_bytes_read"]) for label in labels],
            marker="o",
            label=mode,
        )
    plt.legend()
    finish_plot("compressed_bytes.png", "Compressed tile bytes read", "Bytes")


def existing_paths(
    output_dir: Path, mappings: dict[str, Path], skip_create: bool
) -> dict[str, Path]:
    if mappings and len(mappings) != len(VARIANTS) and skip_create:
        missing = sorted(set(VARIANT_BY_NAME) - set(mappings))
        raise BenchmarkError("Missing --raster mappings: " + ", ".join(missing))
    paths: dict[str, Path] = {}
    for config in VARIANTS:
        variant = config["variant"]
        path = mappings.get(variant, output_dir / "cogs" / f"{variant}.tif")
        require_file(path, f"COG for {variant}")
        inspect_cog(path, config)
        paths[variant] = path
    return paths


def write_environment(
    output_dir: Path,
    input_path: Path,
    parcel_path: Path,
    bbox: tuple[float, float, float, float],
    gdal_version: str | None,
) -> None:
    payload = {
        "input_raster": str(input_path),
        "parcel": str(parcel_path),
        "bbox": list(bbox),
        "python": sys.version,
        "platform": platform.platform(),
        "processor": platform.processor(),
        "gdal_version": gdal_version,
        "rasterio_version": rasterio.__version__,
        "numpy_version": np.__version__,
    }
    (output_dir / "environment.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )


def main() -> int:
    args = parse_args()
    if args.runs <= 0 or args.warmup < 0:
        raise BenchmarkError("--runs must be positive and --warmup cannot be negative")
    if args.reuse_open:
        args.read_mode = "warm"
    if args.skip_create and args.mode in ("create", "both"):
        raise BenchmarkError("--skip-create cannot be used with --mode create/both")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    input_metadata = raster_info(args.input_raster)
    bbox = parcel_bbox(args.parcel, CRS.from_user_input(input_metadata.crs))
    mappings = parse_raster_mappings(args.raster)
    gdal_cli_version = None
    creation_rows: list[dict[str, Any]] = []

    if args.mode in ("create", "both"):
        gdal_cli_version = get_gdal_version()
        paths, creation_rows = create_variants(
            args.input_raster,
            args.output_dir,
            args.keep_cogs or args.mode == "both",
        )
        write_csv(args.output_dir / "cog_creation_results.csv", creation_rows, CREATION_FIELDS)
        if mappings:
            paths.update(mappings)
    else:
        paths = existing_paths(args.output_dir, mappings, args.skip_create)

    if args.mode in ("bbox", "both"):
        # If creation was skipped, generate creation metadata from the existing COGs.
        if not creation_rows:
            for config in VARIANTS:
                path = paths[config["variant"]]
                metadata = inspect_cog(path, config)
                size = path.stat().st_size
                creation_rows.append(
                    {
                        **config,
                        "creation_seconds": "",
                        "file_size_bytes": size,
                        "file_size_mb": size / (1024 * 1024),
                        **metadata,
                    }
                )
        raw_rows = run_reads(
            paths,
            bbox,
            args.output_dir,
            args.runs,
            args.warmup,
            args.read_mode,
        )
        summaries = summarize(raw_rows, creation_rows, args.baseline, args.output_dir)
        make_plots(creation_rows, summaries, args.output_dir)

    write_environment(
        args.output_dir, args.input_raster, args.parcel, bbox, gdal_cli_version
    )
    print(f"BBOX: {bbox}")
    print(f"Output: {args.output_dir}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except BenchmarkError as error:
        print(f"ERROR: {error}", file=sys.stderr)
        raise SystemExit(2)
