#!/usr/bin/env python3
"""Calculate COG tile byte ranges needed by BBOX reads.

This is a deterministic COG-layout analysis, not a GDAL HTTP trace. It reads
only TIFF headers/IFD arrays with S3 Range GETs, then reports the compressed
tile byte intervals that intersect each requested BBOX.
"""

from __future__ import annotations

import argparse
import csv
import os
import struct
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import boto3
import rasterio
from botocore import UNSIGNED
from botocore.config import Config

from cog_s3_bbox_benchmark import (
    VARIANTS,
    VARIANT_BY_NAME,
    BenchmarkError,
    bbox_window_direct,
    load_dotenv_file,
    make_s3_uris,
    parse_s3_files,
)


RANGE_FIELDS = [
    "scenario", "variant", "compression", "blocksize", "tile_row", "tile_col",
    "tile_index", "range_start", "range_end", "compressed_tile_bytes",
    "http_status", "content_range", "content_length", "response_body_bytes",
]
SUMMARY_FIELDS = [
    "scenario", "variant", "compression", "blocksize", "tiles_touched",
    "compressed_tile_bytes", "min_tile_bytes", "max_tile_bytes",
    "http_response_bytes",
]
TYPE_FORMATS = {1: ("B", 1), 3: ("H", 2), 4: ("I", 4), 16: ("Q", 8)}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Calculate COG tile byte ranges for BBOX cases.")
    parser.add_argument("--usr-id", required=True)
    parser.add_argument("--collection-id", required=True)
    parser.add_argument("--case-file", type=Path, default=Path("bbox_size_test_cases.csv"))
    parser.add_argument("--s3-file", action="append", default=[], metavar="VARIANT=FILENAME")
    parser.add_argument("--filename-template", default="{variant}.tif")
    parser.add_argument("--env-file", type=Path, default=Path(".env"))
    parser.add_argument("--output-dir", type=Path, default=Path("s3_cog_tile_range_output"))
    parser.add_argument(
        "--verify-http-response",
        action="store_true",
        help="Issue one MinIO GetObject Range request per selected COG tile and record response headers/bytes.",
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def load_cases(path: Path) -> list[tuple[str, tuple[float, float, float, float]]]:
    required = {"scenario", "min_x", "min_y", "max_x", "max_y"}
    if not path.is_file():
        raise BenchmarkError(f"BBOX case file does not exist: {path}")
    with path.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        if not reader.fieldnames or not required.issubset(reader.fieldnames):
            raise BenchmarkError("Case CSV must contain scenario,min_x,min_y,max_x,max_y")
        result = []
        for row in reader:
            name = (row.get("scenario") or "").strip()
            try:
                bbox = tuple(float(row[key]) for key in ("min_x", "min_y", "max_x", "max_y"))
            except (KeyError, TypeError, ValueError) as error:
                raise BenchmarkError(f"Invalid BBOX case {name!r}") from error
            if not name or bbox[0] >= bbox[2] or bbox[1] >= bbox[3]:
                raise BenchmarkError(f"Invalid BBOX case {name!r}")
            result.append((name, bbox))
    return result


def s3_client() -> Any:
    endpoint = os.getenv("AWS_S3_ENDPOINT", "").strip()
    use_https = os.getenv("AWS_HTTPS", "NO").upper() not in {"NO", "FALSE", "0"}
    if endpoint and not endpoint.startswith(("http://", "https://")):
        endpoint = f"{'https' if use_https else 'http'}://{endpoint}"
    config: dict[str, Any] = {"s3": {"addressing_style": "path"}}
    if os.getenv("AWS_NO_SIGN_REQUEST", "").upper() in {"YES", "TRUE", "1"}:
        config["signature_version"] = UNSIGNED
    return boto3.session.Session(
        region_name=os.getenv("AWS_DEFAULT_REGION") or os.getenv("AWS_REGION")
    ).client("s3", endpoint_url=endpoint or None, config=Config(**config))


@dataclass
class Ifd:
    width: int
    height: int
    tile_width: int
    tile_height: int
    offsets: list[int]
    bytecounts: list[int]


class S3TiffReader:
    def __init__(self, client: Any, uri: str) -> None:
        parsed = urlparse(uri)
        self.client, self.bucket, self.key = client, parsed.netloc, parsed.path.lstrip("/")
        self.cache: dict[tuple[int, int], bytes] = {}
        self.endian, self.big_tiff, self.first_ifd = self._header()

    def read_at(self, offset: int, size: int) -> bytes:
        key = (offset, size)
        if key not in self.cache:
            response = self.client.get_object(
                Bucket=self.bucket, Key=self.key, Range=f"bytes={offset}-{offset + size - 1}"
            )
            self.cache[key] = response["Body"].read()
        data = self.cache[key]
        if len(data) != size:
            raise BenchmarkError(f"Short S3 Range GET at byte {offset}: expected {size}, got {len(data)}")
        return data

    def _header(self) -> tuple[str, bool, int]:
        header = self.read_at(0, 16)
        if header[:2] == b"II":
            endian = "<"
        elif header[:2] == b"MM":
            endian = ">"
        else:
            raise BenchmarkError("Object is not a TIFF (invalid byte order)")
        magic = struct.unpack(endian + "H", header[2:4])[0]
        if magic == 42:
            return endian, False, struct.unpack(endian + "I", header[4:8])[0]
        if magic == 43:
            if struct.unpack(endian + "H", header[4:6])[0] != 8:
                raise BenchmarkError("Unsupported BigTIFF offset size")
            return endian, True, struct.unpack(endian + "Q", header[8:16])[0]
        raise BenchmarkError("Object is not a TIFF (invalid magic value)")

    def _values(self, type_id: int, count: int, value_field: bytes) -> list[int]:
        if type_id not in TYPE_FORMATS:
            raise BenchmarkError(f"Unsupported TIFF tag type {type_id}")
        fmt, width = TYPE_FORMATS[type_id]
        size = width * count
        data = value_field[:size] if size <= len(value_field) else self.read_at(
            struct.unpack(self.endian + ("Q" if self.big_tiff else "I"), value_field)[0], size
        )
        return list(struct.unpack(self.endian + fmt * count, data))

    def _read_ifd(self, offset: int) -> tuple[dict[int, list[int]], int]:
        count_size, entry_size, value_size = (8, 20, 8) if self.big_tiff else (2, 12, 4)
        count = struct.unpack(self.endian + ("Q" if self.big_tiff else "H"), self.read_at(offset, count_size))[0]
        raw = self.read_at(offset + count_size, count * entry_size + value_size)
        tags: dict[int, list[int]] = {}
        for index in range(count):
            entry = raw[index * entry_size:(index + 1) * entry_size]
            tag, type_id = struct.unpack(self.endian + "HH", entry[:4])
            value_count = struct.unpack(self.endian + ("Q" if self.big_tiff else "I"), entry[4:4 + value_size])[0]
            try:
                tags[tag] = self._values(type_id, value_count, entry[4 + value_size:])
            except BenchmarkError:
                if tag in {256, 257, 322, 323, 324, 325}:
                    raise
        next_offset = struct.unpack(self.endian + ("Q" if self.big_tiff else "I"), raw[count * entry_size:count * entry_size + value_size])[0]
        return tags, next_offset

    def image_ifd(self, width: int, height: int) -> Ifd:
        offset = self.first_ifd
        while offset:
            tags, offset = self._read_ifd(offset)
            if tags.get(256, [None])[0] != width or tags.get(257, [None])[0] != height:
                continue
            required = (322, 323, 324, 325)
            if not all(tag in tags for tag in required):
                raise BenchmarkError("TIFF main image is not tiled")
            return Ifd(width, height, tags[322][0], tags[323][0], tags[324], tags[325])
        raise BenchmarkError(f"No TIFF IFD matches raster dimensions {width} x {height}")


def ranges_for_bbox(src: rasterio.DatasetReader, bbox: tuple[float, float, float, float], ifd: Ifd) -> list[tuple[int, int, int, int, int]]:
    window = bbox_window_direct(src, bbox)
    col_start, row_start = int(window.col_off) // ifd.tile_width, int(window.row_off) // ifd.tile_height
    col_end = int(window.col_off + window.width - 1) // ifd.tile_width
    row_end = int(window.row_off + window.height - 1) // ifd.tile_height
    tiles_per_row = (ifd.width + ifd.tile_width - 1) // ifd.tile_width
    result = []
    for row in range(row_start, row_end + 1):
        for col in range(col_start, col_end + 1):
            index = row * tiles_per_row + col
            count = ifd.bytecounts[index]
            result.append((row, col, index, ifd.offsets[index], count))
    return result


def fetch_range_response(client: Any, uri: str, start: int, count: int) -> dict[str, Any]:
    """Issue one explicit MinIO S3 Range GET and return safe response metrics."""
    parsed = urlparse(uri)
    response = client.get_object(
        Bucket=parsed.netloc,
        Key=parsed.path.lstrip("/"),
        Range=f"bytes={start}-{start + count - 1}",
    )
    body_bytes = 0
    while chunk := response["Body"].read(1024 * 1024):
        body_bytes += len(chunk)
    headers = response.get("ResponseMetadata", {}).get("HTTPHeaders", {})
    content_length = int(response.get("ContentLength", headers.get("content-length", 0)))
    return {
        "http_status": int(response.get("ResponseMetadata", {}).get("HTTPStatusCode", 0)),
        "content_range": response.get("ContentRange", headers.get("content-range", "")),
        "content_length": content_length,
        "response_body_bytes": body_bytes,
    }


def main() -> int:
    args = parse_args()
    load_dotenv_file(args.env_file)
    cases, files = load_cases(args.case_file), parse_s3_files(args.s3_file, args.filename_template)
    uris = make_s3_uris(args.usr_id, args.collection_id, files)
    if args.dry_run:
        print(f"Cases: {len(cases)}; variants: {len(uris)}")
        return 0
    client = s3_client()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    range_rows, summary_rows = [], []
    for variant, uri in uris.items():
        with rasterio.open(uri) as src:
            ifd = S3TiffReader(client, uri).image_ifd(src.width, src.height)
            for scenario, bbox in cases:
                tiles = ranges_for_bbox(src, bbox, ifd)
                metadata = VARIANT_BY_NAME[variant]
                counts = [tile[4] for tile in tiles]
                current_rows = []
                for row, col, index, offset, count in tiles:
                    response_metrics = (
                        fetch_range_response(client, uri, offset, count)
                        if args.verify_http_response
                        else {"http_status": "", "content_range": "", "content_length": "", "response_body_bytes": ""}
                    )
                    if args.verify_http_response and (
                        response_metrics["http_status"] != 206
                        or response_metrics["content_length"] != count
                        or response_metrics["response_body_bytes"] != count
                    ):
                        raise BenchmarkError(
                            f"Unexpected Range response for {variant} tile {index}: {response_metrics}"
                        )
                    current_rows.append({"scenario": scenario, "variant": variant, "compression": metadata["compression"], "blocksize": metadata["blocksize"], "tile_row": row, "tile_col": col, "tile_index": index, "range_start": offset, "range_end": offset + count - 1, "compressed_tile_bytes": count, **response_metrics})
                range_rows.extend(current_rows)
                summary_rows.append({"scenario": scenario, "variant": variant, "compression": metadata["compression"], "blocksize": metadata["blocksize"], "tiles_touched": len(tiles), "compressed_tile_bytes": sum(counts), "min_tile_bytes": min(counts), "max_tile_bytes": max(counts), "http_response_bytes": sum(int(item["response_body_bytes"] or 0) for item in current_rows)})
    write_csv(args.output_dir / "cog_tile_ranges.csv", range_rows, RANGE_FIELDS)
    write_csv(args.output_dir / "cog_tile_range_summary.csv", summary_rows, SUMMARY_FIELDS)
    print(f"Output: {args.output_dir}")
    print(f"Tile ranges: {len(range_rows)}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except BenchmarkError as error:
        print(f"ERROR: {error}", file=sys.stderr)
        raise SystemExit(2)
