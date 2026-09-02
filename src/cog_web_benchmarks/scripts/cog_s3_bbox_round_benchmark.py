#!/usr/bin/env python3
"""Run repeated direct-BBOX S3 COG benchmarks without changing the old script.

This script keeps ``cog_s3_bbox_benchmark.py`` unchanged and adds independent
round-based testing. Each round runs every variant, records the execution
sequence, and writes both per-round and overall p90/p95 summaries.

Example:
    python cog_s3_bbox_round_benchmark.py \
        --usr-id gis-asset \
        --collection-id 6a3bfdd89e3416fefefdf90b \
        --bbox 100.07069955101645 14.457056206978876 \
               100.0751394018327 14.460237143938642 \
        --filename-template '{variant}.tif' \
        --rounds 10 --runs 20 --warmup 5 \
        --output-dir s3_cold_round_benchmark_output

Use ``--dry-run`` to print the BBOX and object URIs without opening S3.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import platform
import random
import sys
import time
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Any

import numpy as np
import rasterio

from cog_s3_bbox_benchmark import (
    VARIANTS,
    VARIANT_BY_NAME,
    BenchmarkError,
    aws_session,
    bbox_window_direct,
    explicit_or_parcel_bbox,
    load_dotenv_file,
    make_s3_uris,
    parse_s3_files,
    s3_config,
    tiles_touched,
    write_csv,
)


ROUND_RAW_FIELDS = [
    "round",
    "sequence",
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
    "open_ms",
    "window_ms",
    "read_ms",
    "total_ms",
]
ROUND_SUMMARY_FIELDS = [
    "round",
    "variant",
    "compression",
    "blocksize",
    "mode",
    "file_size_mb",
    "tiles_touched",
    "total_p90_ms",
    "total_p95_ms",
    "total_average_ms",
    "read_p90_ms",
    "latency_reduction_pct",
    "speedup_x",
]
OVERALL_SUMMARY_FIELDS = [
    "variant",
    "compression",
    "blocksize",
    "mode",
    "file_size_mb",
    "tiles_touched",
    "overall_p90_ms",
    "overall_p95_ms",
    "overall_average_ms",
    "read_p90_ms",
    "round_p90_average_ms",
    "round_p90_min_ms",
    "round_p90_max_ms",
    "best_round",
    "worst_round",
    "latency_reduction_pct",
    "speedup_x",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run repeated direct-BBOX S3 COG benchmarks."
    )
    parser.add_argument("--usr-id", required=True)
    parser.add_argument("--collection-id", required=True)
    parser.add_argument(
        "--s3-file",
        action="append",
        default=[],
        metavar="VARIANT=FILENAME",
    )
    parser.add_argument("--filename-template", default="{variant}.tif")
    bbox_source = parser.add_mutually_exclusive_group(required=True)
    bbox_source.add_argument("--parcel", type=Path)
    bbox_source.add_argument(
        "--bbox",
        type=float,
        nargs=4,
        metavar=("MIN_X", "MIN_Y", "MAX_X", "MAX_Y"),
    )
    parser.add_argument("--target-crs", default="EPSG:4326")
    parser.add_argument("--env-file", type=Path, default=Path(".env"))
    parser.add_argument(
        "--output-dir", type=Path, default=Path("s3_cold_round_benchmark_output")
    )
    parser.add_argument("--rounds", type=int, default=10)
    parser.add_argument("--runs", type=int, default=20)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument(
        "--order",
        choices=("fixed", "shuffle"),
        default="shuffle",
        help=(
            "Measured-request order per round; shuffle interleaves variants "
            "to reduce time-based sequence bias"
        ),
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--baseline", choices=tuple(VARIANT_BY_NAME), default="lzw_512")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def percentile(values: list[float], level: float) -> float:
    return float(np.percentile(np.asarray(values, dtype=float), level))


def remote_size_mb(src: rasterio.DatasetReader) -> float | None:
    value = getattr(src, "filesize", None)
    if value is None or value <= 0:
        return None
    return float(value) / (1024 * 1024)


def raw_row(
    *,
    round_id: int,
    sequence: int,
    run: int,
    mode: str,
    variant: str,
    uri: str,
    size_mb: float | None,
    window: rasterio.windows.Window,
    touched: int,
    open_ms: float,
    window_ms: float,
    read_ms: float,
    total_ms: float,
) -> dict[str, Any]:
    config = VARIANT_BY_NAME[variant]
    return {
        "round": round_id,
        "sequence": sequence,
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
        "open_ms": open_ms,
        "window_ms": window_ms,
        "read_ms": read_ms,
        "total_ms": total_ms,
    }


def cold_read(
    uri: str,
    variant: str,
    bbox: tuple[float, float, float, float],
    round_id: int,
    sequence: int,
    run: int,
) -> dict[str, Any]:
    total_started = time.perf_counter()
    open_started = time.perf_counter()
    with rasterio.open(uri) as src:
        open_ms = (time.perf_counter() - open_started) * 1000
        size_mb = remote_size_mb(src)
        window_started = time.perf_counter()
        window = bbox_window_direct(src, bbox)
        window_ms = (time.perf_counter() - window_started) * 1000
        read_started = time.perf_counter()
        data = src.read(1, window=window)
        read_ms = (time.perf_counter() - read_started) * 1000
        touched = tiles_touched(src, window)
    if data.size == 0:
        raise BenchmarkError(f"Empty S3 read result: {uri}")
    return raw_row(
        round_id=round_id,
        sequence=sequence,
        run=run,
        mode="cold",
        variant=variant,
        uri=uri,
        size_mb=size_mb,
        window=window,
        touched=touched,
        open_ms=open_ms,
        window_ms=window_ms,
        read_ms=read_ms,
        total_ms=(time.perf_counter() - total_started) * 1000,
    )


def run_rounds(
    uris: dict[str, str],
    bbox: tuple[float, float, float, float],
    config: dict[str, str],
    rounds: int,
    runs: int,
    warmup: int,
    order: str,
    seed: int,
) -> list[dict[str, Any]]:
    randomizer = random.Random(seed)
    rows: list[dict[str, Any]] = []

    with rasterio.Env(session=aws_session(), **config):
        for round_id in range(1, rounds + 1):
            # Warm-ups are excluded from the results.  Keep each variant's
            # warm-up contiguous so every object reaches the same starting
            # state before the measured requests begin.
            round_variants = list(VARIANTS)
            if order == "shuffle":
                randomizer.shuffle(round_variants)
            for item in round_variants:
                variant = item["variant"]
                uri = uris[variant]
                for _ in range(warmup):
                    cold_read(uri, variant, bbox, round_id, 0, 0)

            # Do not measure all runs of one variant consecutively: an S3 or
            # network slowdown during that interval would otherwise be
            # incorrectly attributed to that variant.  Shuffle the complete
            # request schedule, while keeping one measured run per variant
            # for each run number.
            measured_requests = [
                (item, run)
                for run in range(1, runs + 1)
                for item in VARIANTS
            ]
            if order == "shuffle":
                randomizer.shuffle(measured_requests)

            for sequence, (item, run) in enumerate(measured_requests, start=1):
                variant = item["variant"]
                rows.append(
                    cold_read(
                        uris[variant],
                        variant,
                        bbox,
                        round_id,
                        sequence,
                        run,
                    )
                )
    return rows


def summarize_rounds(
    rows: list[dict[str, Any]], baseline: str, output_dir: Path
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    grouped: dict[tuple[int, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(int(row["round"]), row["variant"], row["mode"])].append(row)

    round_rows: list[dict[str, Any]] = []
    for (round_id, variant, mode), current in grouped.items():
        baseline_rows = grouped.get((round_id, baseline, mode))
        if not baseline_rows:
            raise BenchmarkError(f"Baseline {baseline} missing in round {round_id} ({mode})")
        totals = [float(row["total_ms"]) for row in current]
        reads = [float(row["read_ms"]) for row in current]
        baseline_p90 = percentile([float(row["total_ms"]) for row in baseline_rows], 90)
        current_p90 = percentile(totals, 90)
        round_rows.append(
            {
                "round": round_id,
                "variant": variant,
                "compression": current[0]["compression"],
                "blocksize": current[0]["blocksize"],
                "mode": mode,
                "file_size_mb": current[0]["file_size_mb"],
                "tiles_touched": current[0]["tiles_touched"],
                "total_p90_ms": current_p90,
                "total_p95_ms": percentile(totals, 95),
                "total_average_ms": mean(totals),
                "read_p90_ms": percentile(reads, 90),
                "latency_reduction_pct": (baseline_p90 - current_p90)
                / baseline_p90
                * 100,
                "speedup_x": baseline_p90 / current_p90,
            }
        )

    round_rows.sort(key=lambda row: (row["round"], row["mode"], row["variant"]))
    write_csv(output_dir / "bbox_read_round_summary.csv", round_rows, ROUND_SUMMARY_FIELDS)

    raw_grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    round_grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        raw_grouped[(row["variant"], row["mode"])].append(row)
    for row in round_rows:
        round_grouped[(row["variant"], row["mode"])].append(row)

    overall_rows: list[dict[str, Any]] = []
    for key, current in raw_grouped.items():
        variant, mode = key
        baseline_raw = raw_grouped[(baseline, mode)]
        baseline_p90 = percentile([float(row["total_ms"]) for row in baseline_raw], 90)
        totals = [float(row["total_ms"]) for row in current]
        reads = [float(row["read_ms"]) for row in current]
        current_p90 = percentile(totals, 90)
        current_rounds = round_grouped[key]
        round_p90s = [float(row["total_p90_ms"]) for row in current_rounds]
        best = min(current_rounds, key=lambda row: float(row["total_p90_ms"]))
        worst = max(current_rounds, key=lambda row: float(row["total_p90_ms"]))
        overall_rows.append(
            {
                "variant": variant,
                "compression": current[0]["compression"],
                "blocksize": current[0]["blocksize"],
                "mode": mode,
                "file_size_mb": current[0]["file_size_mb"],
                "tiles_touched": current[0]["tiles_touched"],
                "overall_p90_ms": current_p90,
                "overall_p95_ms": percentile(totals, 95),
                "overall_average_ms": mean(totals),
                "read_p90_ms": percentile(reads, 90),
                "round_p90_average_ms": mean(round_p90s),
                "round_p90_min_ms": min(round_p90s),
                "round_p90_max_ms": max(round_p90s),
                "best_round": best["round"],
                "worst_round": worst["round"],
                "latency_reduction_pct": (baseline_p90 - current_p90)
                / baseline_p90
                * 100,
                "speedup_x": baseline_p90 / current_p90,
            }
        )
    overall_rows.sort(key=lambda row: (row["mode"], row["variant"]))
    write_csv(output_dir / "bbox_read_summary.csv", overall_rows, OVERALL_SUMMARY_FIELDS)
    return round_rows, overall_rows


def write_round_plot(round_rows: list[dict[str, Any]], output_dir: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labels = [item["variant"] for item in VARIANTS]
    x = np.arange(len(labels))
    modes = [mode for mode in ("cold", "warm") if any(row["mode"] == mode for row in round_rows)]
    figure, axes = plt.subplots(
        len(modes),
        1,
        figsize=(12, 5 * len(modes)),
        squeeze=False,
    )
    axes = axes[:, 0]

    for axis, mode in zip(axes, modes):
        mode_rows = [row for row in round_rows if row["mode"] == mode]
        for round_id in sorted({int(row["round"]) for row in mode_rows}):
            by_variant = {
                row["variant"]: row
                for row in mode_rows
                if int(row["round"]) == round_id
            }
            axis.plot(
                x,
                [float(by_variant[label]["total_p90_ms"]) for label in labels],
                marker="o",
                linewidth=1.2,
                alpha=0.7,
                label=f"round {round_id}",
            )
        axis.set_title(f"{mode.capitalize()} read: p90 by round")
        axis.set_ylabel("p90 latency (ms)")
        axis.grid(axis="y", alpha=0.25)
        axis.legend(ncol=2, fontsize=8)

    axes[-1].set_xticks(x, labels, rotation=45, ha="right")
    axes[-1].set_xlabel("Compression + blocksize")
    figure.suptitle("S3 COG BBOX latency across benchmark rounds", fontsize=16)
    figure.tight_layout(rect=(0, 0, 1, 0.97))
    figure.savefig(output_dir / "bbox_latency_rounds.png", dpi=150)
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
        "rounds": args.rounds,
        "runs_per_round": args.runs,
        "warmup_per_round": args.warmup,
        "read_mode": "cold",
        "variant_order": args.order,
        "seed": args.seed,
        "s3_uris": uris,
        "gdal_config_non_secret": safe_config,
        "python": sys.version,
        "platform": platform.platform(),
        "rasterio_version": rasterio.__version__,
        "numpy_version": np.__version__,
    }
    (output_dir / "environment.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )


def main() -> int:
    args = parse_args()
    if args.rounds <= 0 or args.runs <= 0 or args.warmup < 0:
        raise BenchmarkError("--rounds and --runs must be positive; --warmup cannot be negative")
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
    rows = run_rounds(
        uris,
        bbox,
        config,
        args.rounds,
        args.runs,
        args.warmup,
        args.order,
        args.seed,
    )
    round_rows, overall_rows = summarize_rounds(rows, args.baseline, args.output_dir)
    write_round_plot(round_rows, args.output_dir)
    write_environment(args.output_dir, args, bbox, uris, config)
    write_csv(args.output_dir / "bbox_read_raw.csv", rows, ROUND_RAW_FIELDS)
    print(f"Output: {args.output_dir}")
    print(f"Raw rows: {len(rows)}")
    print(f"Round summary rows: {len(round_rows)}")
    print(f"Overall summary rows: {len(overall_rows)}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except BenchmarkError as error:
        print(f"ERROR: {error}", file=sys.stderr)
        raise SystemExit(2)
