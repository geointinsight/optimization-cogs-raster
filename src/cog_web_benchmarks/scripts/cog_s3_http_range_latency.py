#!/usr/bin/env python3
"""Measure explicit S3 HTTP Range GET latency for tiles intersecting BBOX cases.

This complements the Rasterio end-to-end benchmark.  It first derives the
exact compressed TIFF tile byte ranges from the COG layout, then repeatedly
fetches those ranges directly through the S3 API.  It writes request-level,
BBOX-level, and percentile summaries without storing credentials or response
bodies.

``fresh-client`` creates a client for every GET; ``reused-client`` reuses one
client and its HTTP connection pool.  These modes do *not* claim to clear an
object-store cache; they only make client/connection reuse explicit.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import platform
import random
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from statistics import mean
from typing import Any
from urllib.parse import urlparse

import boto3
import numpy as np
import rasterio
from botocore import UNSIGNED
from botocore.config import Config

from cog_s3_bbox_benchmark import (
    VARIANTS,
    VARIANT_BY_NAME,
    BenchmarkError,
    aws_session,
    load_dotenv_file,
    make_s3_uris,
    parse_s3_files,
    s3_config,
)
from cog_s3_cog_tile_range_analysis import S3TiffReader, load_cases, ranges_for_bbox


REQUEST_FIELDS = [
    "round", "sequence", "run", "mode", "concurrency", "scenario", "variant",
    "compression", "blocksize", "tile_row", "tile_col", "tile_index",
    "range_start", "range_end", "expected_bytes", "http_status", "content_range",
    "content_length", "response_body_bytes", "headers_ms", "ttfb_ms",
    "download_ms", "total_ms", "valid_response", "error",
]
BBOX_FIELDS = [
    "round", "sequence", "run", "mode", "concurrency", "scenario", "variant",
    "compression", "blocksize", "tiles_touched", "expected_bytes", "completed_gets",
    "response_body_bytes", "bbox_wall_ms", "request_total_ms_sum", "all_valid",
]
SUMMARY_FIELDS = [
    "mode", "concurrency", "scenario", "variant", "compression", "blocksize",
    "samples", "tiles_touched", "expected_bytes_per_bbox", "bbox_p50_ms", "bbox_p90_ms",
    "bbox_p95_ms", "bbox_average_ms", "request_ttfb_p50_ms", "request_ttfb_p90_ms",
    "request_total_p50_ms", "request_total_p90_ms", "request_total_p95_ms",
    "request_download_p90_ms", "valid_get_ratio",
]


@dataclass(frozen=True)
class TileRequest:
    scenario: str
    variant: str
    bbox: tuple[float, float, float, float]
    tile_row: int
    tile_col: int
    tile_index: int
    range_start: int
    expected_bytes: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--usr-id", required=True)
    parser.add_argument("--collection-id", required=True)
    parser.add_argument("--case-file", type=Path, default=Path("bbox_size_test_cases.csv"))
    parser.add_argument("--s3-file", action="append", default=[], metavar="VARIANT=FILENAME")
    parser.add_argument("--filename-template", default="{variant}.tif")
    parser.add_argument("--env-file", type=Path, default=Path(".env"))
    parser.add_argument("--output-dir", type=Path, default=Path("s3_http_range_latency_output"))
    parser.add_argument("--rounds", type=int, default=5)
    parser.add_argument("--runs", type=int, default=10)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--mode", choices=("fresh-client", "reused-client", "both"), default="both")
    parser.add_argument("--concurrency", type=int, default=1)
    parser.add_argument("--order", choices=("fixed", "shuffle"), default="shuffle")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def percentile(values: list[float], level: float) -> float:
    return float(np.percentile(np.asarray(values, dtype=float), level))


def make_client(concurrency: int) -> Any:
    endpoint = os.getenv("AWS_S3_ENDPOINT", "").strip()
    use_https = os.getenv("AWS_HTTPS", "NO").upper() not in {"NO", "FALSE", "0"}
    if endpoint and not endpoint.startswith(("http://", "https://")):
        endpoint = f"{'https' if use_https else 'http'}://{endpoint}"
    config: dict[str, Any] = {
        "s3": {"addressing_style": "path"},
        "max_pool_connections": max(10, concurrency),
    }
    if os.getenv("AWS_NO_SIGN_REQUEST", "").upper() in {"YES", "TRUE", "1"}:
        config["signature_version"] = UNSIGNED
    return boto3.session.Session(
        region_name=os.getenv("AWS_DEFAULT_REGION") or os.getenv("AWS_REGION")
    ).client("s3", endpoint_url=endpoint or None, config=Config(**config))


def build_requests(
    cases: list[tuple[str, tuple[float, float, float, float]]],
    uris: dict[str, str],
    concurrency: int,
) -> dict[tuple[str, str], list[TileRequest]]:
    """Read COG layout once, outside the timed request phase."""
    client = make_client(concurrency)
    selected: dict[tuple[str, str], list[TileRequest]] = {}
    with rasterio.Env(session=aws_session(), **s3_config()):
        for item in VARIANTS:
            variant = item["variant"]
            uri = uris[variant]
            with rasterio.open(uri) as src:
                ifd = S3TiffReader(client, uri).image_ifd(src.width, src.height)
                for scenario, bbox in cases:
                    selected[(scenario, variant)] = [
                        TileRequest(scenario, variant, bbox, row, col, index, offset, count)
                        for row, col, index, offset, count in ranges_for_bbox(src, bbox, ifd)
                    ]
    return selected


def fetch_one(
    request: TileRequest,
    uri: str,
    client: Any,
    mode: str,
    concurrency: int,
    round_id: int,
    sequence: int,
    run: int,
) -> dict[str, Any]:
    metadata = VARIANT_BY_NAME[request.variant]
    start = time.perf_counter()
    response: Any | None = None
    try:
        active_client = make_client(concurrency) if mode == "fresh-client" else client
        parsed = urlparse(uri)
        response = active_client.get_object(
            Bucket=parsed.netloc,
            Key=parsed.path.lstrip("/"),
            Range=f"bytes={request.range_start}-{request.range_start + request.expected_bytes - 1}",
        )
        headers_ms = (time.perf_counter() - start) * 1000
        body = response["Body"]
        first = body.read(1)
        ttfb_ms = (time.perf_counter() - start) * 1000
        body_bytes = len(first)
        download_start = time.perf_counter()
        while chunk := body.read(1024 * 1024):
            body_bytes += len(chunk)
        download_ms = (time.perf_counter() - download_start) * 1000
        headers = response.get("ResponseMetadata", {}).get("HTTPHeaders", {})
        status = int(response.get("ResponseMetadata", {}).get("HTTPStatusCode", 0))
        content_length = int(response.get("ContentLength", headers.get("content-length", 0)))
        content_range = response.get("ContentRange", headers.get("content-range", ""))
        valid = status == 206 and content_length == request.expected_bytes and body_bytes == request.expected_bytes
        error = "" if valid else "unexpected HTTP status or response length"
        return {
            "round": round_id, "sequence": sequence, "run": run, "mode": mode,
            "concurrency": concurrency, "scenario": request.scenario, "variant": request.variant,
            "compression": metadata["compression"], "blocksize": metadata["blocksize"],
            "tile_row": request.tile_row, "tile_col": request.tile_col, "tile_index": request.tile_index,
            "range_start": request.range_start, "range_end": request.range_start + request.expected_bytes - 1,
            "expected_bytes": request.expected_bytes, "http_status": status, "content_range": content_range,
            "content_length": content_length, "response_body_bytes": body_bytes,
            "headers_ms": headers_ms, "ttfb_ms": ttfb_ms, "download_ms": download_ms,
            "total_ms": (time.perf_counter() - start) * 1000, "valid_response": valid, "error": error,
        }
    except Exception as error:  # retain failed attempts for diagnosis
        return {
            "round": round_id, "sequence": sequence, "run": run, "mode": mode,
            "concurrency": concurrency, "scenario": request.scenario, "variant": request.variant,
            "compression": metadata["compression"], "blocksize": metadata["blocksize"],
            "tile_row": request.tile_row, "tile_col": request.tile_col, "tile_index": request.tile_index,
            "range_start": request.range_start, "range_end": request.range_start + request.expected_bytes - 1,
            "expected_bytes": request.expected_bytes, "http_status": "", "content_range": "",
            "content_length": "", "response_body_bytes": "", "headers_ms": "", "ttfb_ms": "",
            "download_ms": "", "total_ms": (time.perf_counter() - start) * 1000,
            "valid_response": False, "error": f"{type(error).__name__}: {error}",
        }
    finally:
        if response is not None:
            response["Body"].close()


def fetch_group(
    requests: list[TileRequest], uri: str, client: Any, mode: str, concurrency: int,
    round_id: int, sequence: int, run: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    started = time.perf_counter()
    if concurrency == 1:
        rows = [fetch_one(item, uri, client, mode, concurrency, round_id, sequence, run) for item in requests]
    else:
        with ThreadPoolExecutor(max_workers=concurrency) as executor:
            futures = [executor.submit(fetch_one, item, uri, client, mode, concurrency, round_id, sequence, run) for item in requests]
            rows = [future.result() for future in as_completed(futures)]
    metadata = VARIANT_BY_NAME[requests[0].variant]
    body_bytes = sum(int(row["response_body_bytes"] or 0) for row in rows)
    bbox_row = {
        "round": round_id, "sequence": sequence, "run": run, "mode": mode,
        "concurrency": concurrency, "scenario": requests[0].scenario, "variant": requests[0].variant,
        "compression": metadata["compression"], "blocksize": metadata["blocksize"],
        "tiles_touched": len(requests), "expected_bytes": sum(item.expected_bytes for item in requests),
        "completed_gets": len(rows), "response_body_bytes": body_bytes,
        "bbox_wall_ms": (time.perf_counter() - started) * 1000,
        "request_total_ms_sum": sum(float(row["total_ms"]) for row in rows),
        "all_valid": all(row["valid_response"] is True for row in rows),
    }
    return rows, bbox_row


def summarize(request_rows: list[dict[str, Any]], bbox_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped_bbox: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    grouped_requests: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    for row in bbox_rows:
        key = tuple(row[field] for field in ("mode", "concurrency", "scenario", "variant", "compression", "blocksize"))
        grouped_bbox.setdefault(key, []).append(row)
    for row in request_rows:
        key = tuple(row[field] for field in ("mode", "concurrency", "scenario", "variant", "compression", "blocksize"))
        grouped_requests.setdefault(key, []).append(row)
    summaries: list[dict[str, Any]] = []
    for key, boxes in grouped_bbox.items():
        valid_requests = [row for row in grouped_requests[key] if row["valid_response"] is True]
        box_values = [float(row["bbox_wall_ms"]) for row in boxes]
        total_values = [float(row["total_ms"]) for row in valid_requests]
        ttfb_values = [float(row["ttfb_ms"]) for row in valid_requests]
        download_values = [float(row["download_ms"]) for row in valid_requests]
        summaries.append({
            "mode": key[0], "concurrency": key[1], "scenario": key[2], "variant": key[3],
            "compression": key[4], "blocksize": key[5], "samples": len(boxes),
            "tiles_touched": boxes[0]["tiles_touched"], "expected_bytes_per_bbox": boxes[0]["expected_bytes"],
            "bbox_p50_ms": percentile(box_values, 50), "bbox_p90_ms": percentile(box_values, 90),
            "bbox_p95_ms": percentile(box_values, 95), "bbox_average_ms": mean(box_values),
            "request_ttfb_p50_ms": percentile(ttfb_values, 50) if ttfb_values else "",
            "request_ttfb_p90_ms": percentile(ttfb_values, 90) if ttfb_values else "",
            "request_total_p50_ms": percentile(total_values, 50) if total_values else "",
            "request_total_p90_ms": percentile(total_values, 90) if total_values else "",
            "request_total_p95_ms": percentile(total_values, 95) if total_values else "",
            "request_download_p90_ms": percentile(download_values, 90) if download_values else "",
            "valid_get_ratio": len(valid_requests) / len(grouped_requests[key]),
        })
    return sorted(summaries, key=lambda row: (row["mode"], row["concurrency"], row["scenario"], row["variant"]))


def main() -> int:
    args = parse_args()
    if args.rounds <= 0 or args.runs <= 0 or args.warmup < 0 or args.concurrency <= 0:
        raise BenchmarkError("--rounds, --runs, and --concurrency must be positive; --warmup cannot be negative")
    load_dotenv_file(args.env_file)
    cases = load_cases(args.case_file)
    uris = make_s3_uris(args.usr_id, args.collection_id, parse_s3_files(args.s3_file, args.filename_template))
    if args.dry_run:
        print(f"Cases: {len(cases)}; variants: {len(uris)}; modes: {args.mode}; concurrency: {args.concurrency}")
        print("Dry run: no S3 objects were opened and no output was written.")
        return 0
    selected = build_requests(cases, uris, args.concurrency)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    modes = [args.mode] if args.mode != "both" else ["fresh-client", "reused-client"]
    rng = random.Random(args.seed)
    request_rows: list[dict[str, Any]] = []
    bbox_rows: list[dict[str, Any]] = []
    groups = [(scenario, variant, rows) for (scenario, variant), rows in selected.items()]
    for mode in modes:
        client = make_client(args.concurrency)
        for round_id in range(1, args.rounds + 1):
            schedule = list(groups)
            if args.order == "shuffle":
                rng.shuffle(schedule)
            for scenario, variant, rows in schedule:
                for _ in range(args.warmup):
                    fetch_group(rows, uris[variant], client, mode, args.concurrency, round_id, 0, 0)
            measured = [(scenario, variant, rows, run) for run in range(1, args.runs + 1) for scenario, variant, rows in schedule]
            if args.order == "shuffle":
                rng.shuffle(measured)
            for sequence, (_, variant, rows, run) in enumerate(measured, start=1):
                current_requests, current_bbox = fetch_group(rows, uris[variant], client, mode, args.concurrency, round_id, sequence, run)
                request_rows.extend(current_requests)
                bbox_rows.append(current_bbox)
        print(f"Completed {mode}: {len([row for row in bbox_rows if row['mode'] == mode])} BBOX samples")
    write_csv(args.output_dir / "http_range_request_raw.csv", request_rows, REQUEST_FIELDS)
    write_csv(args.output_dir / "http_range_bbox_raw.csv", bbox_rows, BBOX_FIELDS)
    write_csv(args.output_dir / "http_range_latency_summary.csv", summarize(request_rows, bbox_rows), SUMMARY_FIELDS)
    safe_config = {key: value for key, value in s3_config().items() if "KEY" not in key and "TOKEN" not in key}
    (args.output_dir / "environment.json").write_text(json.dumps({
        "case_file": str(args.case_file), "rounds": args.rounds, "runs": args.runs,
        "warmup": args.warmup, "modes": modes, "concurrency": args.concurrency,
        "order": args.order, "seed": args.seed, "gdal_config_non_secret": safe_config,
        "python": sys.version, "platform": platform.platform(), "rasterio_version": rasterio.__version__,
    }, indent=2), encoding="utf-8")
    failures = sum(row["valid_response"] is not True for row in request_rows)
    print(f"Output: {args.output_dir}")
    print(f"HTTP Range GET samples: {len(request_rows)}; failed validation: {failures}")
    return 2 if failures else 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except BenchmarkError as error:
        print(f"ERROR: {error}", file=sys.stderr)
        raise SystemExit(2)
