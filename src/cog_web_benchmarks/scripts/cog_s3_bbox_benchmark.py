#!/usr/bin/env python3
"""Benchmark direct BBOX reads from COG files stored in S3-compatible storage.

The S3 object layout follows the existing NDVI workflow::

    s3://{usr_id}/coverage/{collection_id}/cog/{filename}

The only remote operation performed by this script is ``src.read`` on the
window calculated from the supplied BBOX. It does not resample, mask, merge,
download, or calculate raster statistics. No S3 connection is made when this
module is imported. Use ``--dry-run`` to validate the object URIs and BBOX
without opening any remote raster.

Example:
    python cog_s3_bbox_benchmark.py \
        --usr-id 644b92f9fbf1689fb3497525 \
        --collection-id 69006a3bbe4e87ecda5c176e \
        --s3-file lzw_128=6a3bfdd89e3416fefefdf90b_lzw_128.tif \
        --s3-file lzw_256=6a3bfdd89e3416fefefdf90b_lzw_256.tif \
        --s3-file lzw_512=6a3bfdd89e3416fefefdf90b_lzw_512.tif \
        --s3-file deflate_128=6a3bfdd89e3416fefefdf90b_deflate_128.tif \
        --s3-file deflate_256=6a3bfdd89e3416fefefdf90b_deflate_256.tif \
        --s3-file deflate_512=6a3bfdd89e3416fefefdf90b_deflate_512.tif \
        --s3-file zstd_128=6a3bfdd89e3416fefefdf90b_zstd_128.tif \
        --s3-file zstd_256=6a3bfdd89e3416fefefdf90b_zstd_256.tif \
        --s3-file zstd_512=6a3bfdd89e3416fefefdf90b_zstd_512.tif \
        --parcel data_plot/plot.geojson \
        --output-dir s3_benchmark_output
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import platform
import sys
import time
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Any

import numpy as np
import rasterio
from rasterio.crs import CRS
from rasterio.session import AWSSession
from rasterio.windows import Window, from_bounds

from cog_benchmark_cli import (
    VARIANTS,
    VARIANT_BY_NAME,
    BenchmarkError,
    parcel_bbox,
    tiles_touched,
    write_csv,
)

from dotenv import load_dotenv
load_dotenv()

RAW_FIELDS = [
    "variant",
    "compression",
    "blocksize",
    "run",
    "mode",
    "s3_uri",
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark identical BBOX reads from COGs in S3 storage."
    )
    parser.add_argument("--usr-id", required=True, help="S3 bucket/tenant ID")
    parser.add_argument("--collection-id", required=True)
    parser.add_argument(
        "--s3-file",
        action="append",
        default=[],
        metavar="VARIANT=FILENAME",
        help="S3 filename mapping; repeat once for each COG variant",
    )
    parser.add_argument(
        "--filename-template",
        default="{variant}.tif",
        help="Fallback template when --s3-file is omitted",
    )
    bbox_source = parser.add_mutually_exclusive_group(required=True)
    bbox_source.add_argument(
        "--parcel",
        type=Path,
        help="Parcel file used to derive the BBOX before the S3 read",
    )
    bbox_source.add_argument(
        "--bbox",
        type=float,
        nargs=4,
        metavar=("MIN_X", "MIN_Y", "MAX_X", "MAX_Y"),
        help="Explicit BBOX passed to the direct S3 window read",
    )
    parser.add_argument(
        "--target-crs",
        default="EPSG:4326",
        help="CRS for parcel-derived BBOX (default: EPSG:4326)",
    )
    parser.add_argument(
        "--env-file",
        type=Path,
        default=Path(".env"),
        help="dotenv file containing S3/GDAL configuration",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("s3_benchmark_output"),
    )
    parser.add_argument(
        "--read-mode",
        choices=("cold", "warm", "both"),
        default="both",
        help="Cold/reopen, warm/reuse, or both",
    )
    parser.add_argument("--runs", type=int, default=20)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument(
        "--baseline",
        choices=tuple(VARIANT_BY_NAME),
        default="lzw_512",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print BBOX and S3 URIs without opening S3 or writing results",
    )
    return parser.parse_args()


def load_dotenv_file(path: Path) -> None:
    if not path.is_file():
        return
    try:
        from dotenv import load_dotenv
    except ImportError as error:
        raise BenchmarkError(
            "python-dotenv is required when --env-file is used"
        ) from error
    load_dotenv(path, override=False)


def s3_config() -> dict[str, str]:
    """Build non-secret GDAL S3 configuration.

    Rasterio 1.4 rejects AWS credentials passed as GDAL options. Credentials
    are supplied separately through ``rasterio.session.AWSSession``.
    """
    config: dict[str, str] = {}
    aliases = {
        "AWS_DEFAULT_REGION": "AWS_DEFAULT_REGION",
        "AWS_REGION": "AWS_REGION",
        "AWS_S3_ENDPOINT": "AWS_S3_ENDPOINT",
        "AWS_S3_SIGNATURE_VERSION": "AWS_S3_SIGNATURE_VERSION",
        "AWS_VIRTUAL_HOSTING": "AWS_VIRTUAL_HOSTING",
        "AWS_HTTPS": "AWS_HTTPS",
    }
    defaults = {
        "AWS_DEFAULT_REGION": "ap-southeast-1",
        "AWS_S3_SIGNATURE_VERSION": "s3v4",
        "AWS_VIRTUAL_HOSTING": "FALSE",
        "AWS_HTTPS": "NO",
    }
    for key, env_key in aliases.items():
        value = os.getenv(env_key, defaults.get(key, ""))
        if value:
            config[key] = value
    return config


def aws_session() -> AWSSession:
    """Create the Rasterio AWS session used for S3 authentication."""
    try:
        import boto3
    except ImportError as error:
        raise BenchmarkError(
            "boto3 is required for S3 reads; install cog_benchmark_requirements.txt"
        ) from error

    region = os.getenv("AWS_DEFAULT_REGION") or os.getenv("AWS_REGION")
    endpoint = os.getenv("AWS_S3_ENDPOINT") or None
    session = boto3.Session(region_name=region)
    unsigned = os.getenv("AWS_NO_SIGN_REQUEST", "").upper() in {"YES", "TRUE", "1"}
    return AWSSession(
        session=session,
        aws_unsigned=unsigned,
        endpoint_url=endpoint,
    )


def parse_s3_files(values: list[str], template: str) -> dict[str, str]:
    files: dict[str, str] = {}
    for value in values:
        if "=" not in value:
            raise BenchmarkError(
                f"Invalid --s3-file {value!r}; expected VARIANT=FILENAME"
            )
        variant, filename = value.split("=", 1)
        if variant not in VARIANT_BY_NAME:
            raise BenchmarkError(f"Unknown S3 variant: {variant}")
        if not filename:
            raise BenchmarkError(f"Empty S3 filename for {variant}")
        if variant in files:
            raise BenchmarkError(f"Duplicate S3 mapping: {variant}")
        files[variant] = filename
    for config in VARIANTS:
        files.setdefault(
            config["variant"], template.format(**config)
        )
    return files


def make_s3_uris(usr_id: str, collection_id: str, files: dict[str, str]) -> dict[str, str]:
    prefix = f"s3://{usr_id}/coverage/{collection_id}/cog"
    return {variant: f"{prefix}/{filename}" for variant, filename in files.items()}


def explicit_or_parcel_bbox(args: argparse.Namespace) -> tuple[float, float, float, float]:
    if args.bbox is not None:
        min_x, min_y, max_x, max_y = args.bbox
        if min_x >= max_x or min_y >= max_y:
            raise BenchmarkError("BBOX must satisfy min_x < max_x and min_y < max_y")
        return tuple(args.bbox)
    if args.parcel is None or not args.parcel.is_file():
        raise BenchmarkError(f"Parcel file does not exist: {args.parcel}")
    return parcel_bbox(args.parcel, CRS.from_user_input(args.target_crs))


def remote_file_size_mb(src: rasterio.DatasetReader) -> float | None:
    value = getattr(src, "filesize", None)
    if value is None or value <= 0:
        return None
    return float(value) / (1024 * 1024)


def bbox_window_direct(
    src: rasterio.DatasetReader,
    bbox: tuple[float, float, float, float],
) -> Window:
    """Build the exact integer window used by the S3 workflow example."""
    window = from_bounds(*bbox, transform=src.transform)
    row_start = math.floor(window.row_off)
    col_start = math.floor(window.col_off)
    row_stop = math.ceil(window.row_off + window.height)
    col_stop = math.ceil(window.col_off + window.width)
    final_window = Window(
        col_off=col_start,
        row_off=row_start,
        width=col_stop - col_start,
        height=row_stop - row_start,
    )
    if final_window.width <= 0 or final_window.height <= 0:
        raise BenchmarkError("BBOX produces an empty S3 raster window")
    if (
        final_window.col_off < 0
        or final_window.row_off < 0
        or final_window.col_off + final_window.width > src.width
        or final_window.row_off + final_window.height > src.height
    ):
        raise BenchmarkError("BBOX window falls outside the S3 raster")
    return final_window


def read_bbox_from_dataset(
    src: rasterio.DatasetReader,
    bbox: tuple[float, float, float, float],
) -> tuple[np.ndarray, Any, Any, Window, int]:
    """Read only the integer window corresponding to ``bbox``.

    This is intentionally equivalent to ``clip_from_s3_direct`` from the
    production workflow: no polygon mask, resampling, merge, or full-raster
    download is performed.
    """
    final_window = bbox_window_direct(src, bbox)
    data = src.read(1, window=final_window)
    transform = rasterio.windows.transform(final_window, src.transform)
    return data, transform, src.crs, final_window, tiles_touched(src, final_window)


def clip_from_s3_direct(
    usr_id: str,
    collection_id: str,
    params: dict[str, Any] | tuple[float, float, float, float],
    filename: str | None = None,
) -> tuple[np.ndarray, Any, Any]:
    """Read one S3 COG using only the BBOX supplied in ``params``.

    ``params`` may be the workflow shape ``param1`` through ``param4`` or a
    four-value ``(min_x, min_y, max_x, max_y)`` tuple. The caller must already
    be inside the configured ``rasterio.Env`` context.
    """
    s3_filename = filename or f"{collection_id}.tif"
    s3_url = f"s3://{usr_id}/coverage/{collection_id}/cog/{s3_filename}"
    if isinstance(params, dict):
        bbox = tuple(
            float(params[key])
            for key in ("param1", "param2", "param3", "param4")
        )
    else:
        bbox = tuple(float(value) for value in params)
    with rasterio.open(s3_url) as src:
        data, transform, crs, _, _ = read_bbox_from_dataset(src, bbox)
    return data, transform, crs


def read_one_cold(
    uri: str,
    variant: str,
    bbox: tuple[float, float, float, float],
    run: int,
) -> dict[str, Any]:
    total_started = time.perf_counter()
    open_started = time.perf_counter()
    with rasterio.open(uri) as src:
        open_ms = (time.perf_counter() - open_started) * 1000
        size_mb = remote_file_size_mb(src)
        window_started = time.perf_counter()
        window = bbox_window_direct(src, bbox)
        window_ms = (time.perf_counter() - window_started) * 1000
        read_started = time.perf_counter()
        array = src.read(1, window=window)
        read_ms = (time.perf_counter() - read_started) * 1000
        touched = tiles_touched(src, window)
    if array.size == 0:
        raise BenchmarkError(f"Empty S3 read result: {uri}")
    return make_raw_row(
        uri,
        variant,
        "cold",
        run,
        size_mb,
        window,
        touched,
        open_ms,
        window_ms,
        read_ms,
        (time.perf_counter() - total_started) * 1000,
    )


def read_one_warm(
    src: rasterio.DatasetReader,
    uri: str,
    variant: str,
    bbox: tuple[float, float, float, float],
    run: int,
    size_mb: float | None,
) -> dict[str, Any]:
    total_started = time.perf_counter()
    window_started = time.perf_counter()
    window = bbox_window_direct(src, bbox)
    window_ms = (time.perf_counter() - window_started) * 1000
    read_started = time.perf_counter()
    array = src.read(1, window=window)
    read_ms = (time.perf_counter() - read_started) * 1000
    touched = tiles_touched(src, window)
    if array.size == 0:
        raise BenchmarkError(f"Empty S3 read result: {uri}")
    return make_raw_row(
        uri,
        variant,
        "warm",
        run,
        size_mb,
        window,
        touched,
        0.0,
        window_ms,
        read_ms,
        (time.perf_counter() - total_started) * 1000,
    )


def make_raw_row(
    uri: str,
    variant: str,
    mode: str,
    run: int,
    size_mb: float | None,
    window: Window,
    touched: int,
    open_ms: float,
    window_ms: float,
    read_ms: float,
    total_ms: float,
) -> dict[str, Any]:
    config = VARIANT_BY_NAME[variant]
    return {
        "variant": variant,
        "compression": config["compression"],
        "blocksize": config["blocksize"],
        "run": run,
        "mode": mode,
        "s3_uri": uri,
        "file_size_mb": "" if size_mb is None else size_mb,
        "window_width_px": int(window.width),
        "window_height_px": int(window.height),
        "tiles_touched": touched,
        # Exact network bytes are not exposed by Rasterio/GDAL read timing.
        "compressed_bytes_read": "",
        "open_ms": open_ms,
        "window_ms": window_ms,
        "read_ms": read_ms,
        "total_ms": total_ms,
    }


def run_reads(
    uris: dict[str, str],
    bbox: tuple[float, float, float, float],
    config: dict[str, str],
    output_dir: Path,
    runs: int,
    warmup: int,
    read_mode: str,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    modes = [read_mode] if read_mode != "both" else ["cold", "warm"]
    with rasterio.Env(session=aws_session(), **config):
        for item in VARIANTS:
            variant = item["variant"]
            uri = uris[variant]
            for mode in modes:
                if mode == "cold":
                    for _ in range(warmup):
                        read_one_cold(uri, variant, bbox, 0)
                    for run in range(1, runs + 1):
                        rows.append(read_one_cold(uri, variant, bbox, run))
                else:
                    with rasterio.open(uri) as src:
                        size_mb = remote_file_size_mb(src)
                        for _ in range(warmup):
                            read_one_warm(src, uri, variant, bbox, 0, size_mb)
                        for run in range(1, runs + 1):
                            rows.append(
                                read_one_warm(
                                    src, uri, variant, bbox, run, size_mb
                                )
                            )
    write_csv(output_dir / "bbox_read_raw.csv", rows, RAW_FIELDS)
    return rows


def percentile(values: list[float], value: float) -> float:
    return float(np.percentile(np.asarray(values, dtype=float), value))


def summarize(
    rows: list[dict[str, Any]], baseline: str, output_dir: Path
) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(row["variant"], row["mode"])].append(row)

    summaries: list[dict[str, Any]] = []
    for (variant, mode), current_rows in grouped.items():
        baseline_rows = grouped.get((baseline, mode))
        if not baseline_rows:
            raise BenchmarkError(f"Baseline {baseline} has no {mode} results")
        totals = [float(row["total_ms"]) for row in current_rows]
        reads = [float(row["read_ms"]) for row in current_rows]
        baseline_p90 = percentile(
            [float(row["total_ms"]) for row in baseline_rows], 90
        )
        current_p90 = percentile(totals, 90)
        current_size = current_rows[0]["file_size_mb"]
        baseline_size = baseline_rows[0]["file_size_mb"]
        storage_reduction = ""
        if current_size != "" and baseline_size != "":
            storage_reduction = (
                (float(baseline_size) - float(current_size))
                / float(baseline_size)
                * 100
            )
        summaries.append(
            {
                "variant": variant,
                "compression": current_rows[0]["compression"],
                "blocksize": current_rows[0]["blocksize"],
                "mode": mode,
                "file_size_mb": current_size,
                "tiles_touched": current_rows[0]["tiles_touched"],
                "compressed_bytes_read": "",
                "total_p90_ms": current_p90,
                "total_p95_ms": percentile(totals, 95),
                "total_average_ms": mean(totals),
                "read_p90_ms": percentile(reads, 90),
                "latency_reduction_pct": (baseline_p90 - current_p90)
                / baseline_p90
                * 100,
                "speedup_x": baseline_p90 / current_p90,
                "storage_reduction_pct": storage_reduction,
                "creation_seconds": "",
            }
        )
    summaries.sort(key=lambda row: (row["mode"], row["variant"]))
    write_csv(output_dir / "bbox_read_summary.csv", summaries, SUMMARY_FIELDS)
    return summaries


def write_latency_plot(summaries: list[dict[str, Any]], output_dir: Path) -> None:
    """Write one readable P90/P95 chart per compression and read mode."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    modes = [mode for mode in ("cold", "warm") if any(row["mode"] == mode for row in summaries)]
    for mode in modes:
        for compression in ("DEFLATE", "LZW", "ZSTD"):
            rows = sorted(
                (
                    row
                    for row in summaries
                    if row["mode"] == mode and row["compression"] == compression
                ),
                key=lambda row: int(row["blocksize"]),
            )
            if not rows:
                continue
            x = np.arange(len(rows))
            p90 = [float(row["total_p90_ms"]) for row in rows]
            p95 = [float(row["total_p95_ms"]) for row in rows]
            figure, axis = plt.subplots(figsize=(8, 4.8))
            width = 0.36
            axis.bar(x - width / 2, p90, width=width, label="P90", color="#2563eb")
            axis.bar(x + width / 2, p95, width=width, label="P95", color="#7c3aed")
            axis.set_xticks(x, [f"{row['blocksize']} px" for row in rows])
            axis.set_xlabel("COG block size")
            axis.set_ylabel("Latency (ms)")
            axis.set_title(f"S3 COG BBOX latency — {mode.capitalize()} · {compression}")
            axis.grid(axis="y", alpha=0.25)
            axis.legend()
            for index, value in enumerate(p90):
                axis.annotate(f"{value:.1f}", (index - width / 2, value), textcoords="offset points", xytext=(0, 5), ha="center", fontsize=8)
            figure.tight_layout()
            figure.savefig(output_dir / f"bbox_latency_{mode}_{compression.lower()}.png", dpi=150)
            plt.close(figure)


def write_environment(
    output_dir: Path,
    args: argparse.Namespace,
    bbox: tuple[float, float, float, float],
    uris: dict[str, str],
    config: dict[str, str],
) -> None:
    safe_config = {
        key: value
        for key, value in config.items()
        if key not in {"AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY", "AWS_SESSION_TOKEN"}
    }
    payload = {
        "usr_id": args.usr_id,
        "collection_id": args.collection_id,
        "bbox": list(bbox),
        "read_mode": args.read_mode,
        "runs": args.runs,
        "warmup": args.warmup,
        "s3_uris": uris,
        "gdal_config_non_secret": safe_config,
        "python": sys.version,
        "platform": platform.platform(),
        "rasterio_version": rasterio.__version__,
        "numpy_version": np.__version__,
        "compressed_bytes_read": "not measured for remote reads",
    }
    (output_dir / "environment.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )


def main() -> int:
    args = parse_args()
    if args.runs <= 0 or args.warmup < 0:
        raise BenchmarkError("--runs must be positive and --warmup cannot be negative")
    load_dotenv_file(args.env_file)
    bbox = explicit_or_parcel_bbox(args)
    files = parse_s3_files(args.s3_file, args.filename_template)
    uris = make_s3_uris(args.usr_id, args.collection_id, files)
    config = s3_config()

    print(f"BBOX: {bbox}")
    for variant in (item["variant"] for item in VARIANTS):
        print(f"{variant}: {uris[variant]}")
    if args.dry_run:
        print("Dry run: no S3 raster was opened and no results were written.")
        return 0

    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = run_reads(
        uris,
        bbox,
        config,
        args.output_dir,
        args.runs,
        args.warmup,
        args.read_mode,
    )
    summaries = summarize(rows, args.baseline, args.output_dir)
    write_latency_plot(summaries, args.output_dir)
    write_environment(args.output_dir, args, bbox, uris, config)
    print(f"Output: {args.output_dir}")
    return 0

if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except BenchmarkError as error:
        print(f"ERROR: {error}", file=sys.stderr)
        raise SystemExit(2)
