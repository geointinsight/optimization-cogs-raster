#!/usr/bin/env python3
"""Create presentation-ready trend charts from BBOX-size and tile-edge latency runs.

The output contains both views: charts faceted by compression and charts
faceted by block size, including BBOX read latency and Range payload.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from statistics import mean
from typing import Any

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.ticker import FuncFormatter, MultipleLocator


COMPRESSIONS = ("DEFLATE", "LZW", "ZSTD")
BLOCKS = (128, 256, 512)
BLOCK_COLORS = {128: "#93c5fd", 256: "#3b82f6", 512: "#1d4ed8"}
BLOCK_MARKERS = {128: "o", 256: "s", 512: "D"}
COMPRESSION_COLORS = {"DEFLATE": "#2563eb", "LZW": "#f97316", "ZSTD": "#16a34a"}
COMPRESSION_MARKERS = {"DEFLATE": "o", "LZW": "s", "ZSTD": "D"}
EDGE_ORDER = (
    "interior", "cross_128_edge", "cross_256_edge", "cross_512_edge",
    "cross_128_corner", "cross_256_corner", "cross_512_corner",
)
EDGE_LABELS = (
    "Interior", "128 edge", "256 edge", "512 edge",
    "128 corner", "256 corner", "512 corner",
)
BBOX_REQUEST_SIZES = {
    "tiny_12x9px": ("Point Inspection", "12 × 9 px"),
    "small_24x18px": ("Parcel View", "24 × 18 px"),
    "reference_49x36px": ("Reference Area", "49 × 36 px"),
    "viewport_98x72px": ("Map View", "98 × 72 px"),
    "viewport_196x144px": ("Wide View", "196 × 144 px"),
    "large_aoi_392x288px": ("Analysis Area", "392 × 288 px"),
}
BOUNDARY_GROUPS = {
    128: ("cross_128_edge", "cross_128_corner"),
    256: ("cross_256_edge", "cross_256_corner"),
    512: ("cross_512_edge", "cross_512_corner"),
}
HTTP_STACK_COMPONENTS = ("TTFB", "Download")
HTTP_STACK_COLORS = {"TTFB": "#2563eb", "Download": "#38bdf8"}
E2E_STACK_COMPONENTS = ("Open / metadata", "Window", "Raster read", "Other")
E2E_STACK_COLORS = {"Open / metadata": "#7c3aed", "Window": "#f59e0b", "Raster read": "#16a34a", "Other": "#cbd5e1"}
COMPRESSION_HATCHES = {"DEFLATE": "", "LZW": "//", "ZSTD": "xx"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bbox-output", type=Path, default=Path("s3_latency_bbox_size_output"))
    parser.add_argument("--edge-output", type=Path, default=Path("s3_latency_tile_edge_output"))
    parser.add_argument("--case-file", type=Path, default=Path("bbox_size_test_cases.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("s3_latency_trend_charts"))
    return parser.parse_args()


def csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def load_e2e(root: Path) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for path in sorted((root / "end_to_end").glob("*/bbox_read_summary.csv")):
        for row in csv_rows(path):
            row["scenario"] = path.parent.name
            rows.append(row)
    if not rows:
        raise FileNotFoundError(f"No E2E summaries below {root / 'end_to_end'}")
    return rows


def load_http(root: Path) -> list[dict[str, str]]:
    paths = sorted((root / "http_range").glob("concurrency_*/http_range_latency_summary.csv"))
    if not paths:
        raise FileNotFoundError(f"No HTTP Range summaries below {root / 'http_range'}")
    rows: list[dict[str, str]] = []
    for path in paths:
        rows.extend(csv_rows(path))
    return rows


def index_rows(rows: list[dict[str, str]], *keys: str) -> dict[tuple[str, ...], dict[str, str]]:
    return {tuple(row[key] for key in keys): row for row in rows}


def format_bytes(value: float) -> str:
    return f"{value / 1048576:.2f} MiB" if value >= 1048576 else f"{value / 1024:.0f} KiB"


def bbox_size_labels(case_rows: list[dict[str, str]]) -> list[str]:
    """Use the same BBOX request-size names shown in the web animation."""
    labels: list[str] = []
    for row in case_rows:
        name, _ = BBOX_REQUEST_SIZES[row["scenario"]]
        labels.append(name)
    return labels


def save_bbox_read_latency_chart_csvs(
    bbox_http: list[dict[str, str]], case_rows: list[dict[str, str]], output_dir: Path,
) -> None:
    """Export the exact source rows used by each 08 block-size line chart.

    The chart uses the reused-client, concurrency-1 HTTP Range summary and
    plots ``bbox_p90_ms``.  Keeping one CSV beside every PNG makes the plotted
    values and benchmark conditions independently reviewable.
    """
    scenarios = [row["scenario"] for row in case_rows]
    label_by_scenario = {
        row["scenario"]: BBOX_REQUEST_SIZES[row["scenario"]]
        for row in case_rows
    }
    rows_by_key = index_rows(
        [row for row in bbox_http if row["mode"] == "reused-client" and row["concurrency"] == "1"],
        "scenario", "compression", "blocksize",
    )
    fieldnames = [
        "bbox_request_size", "bbox_pixels", "scenario", "compression", "blocksize_px",
        "mode", "concurrency", "samples", "tiles_touched", "bbox_p90_ms", "bbox_average_ms",
        "expected_bytes_per_bbox", "valid_get_ratio", "source_csv",
    ]
    source_csv = "s3_latency_bbox_size_output/http_range/concurrency_1/http_range_latency_summary.csv"

    for block in BLOCKS:
        output_path = output_dir / f"08_bbox_size_read_latency_block_{block}px.csv"
        with output_path.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=fieldnames)
            writer.writeheader()
            for scenario in scenarios:
                request_size, pixels = label_by_scenario[scenario]
                for compression in COMPRESSIONS:
                    row = rows_by_key[(scenario, compression, str(block))]
                    writer.writerow({
                        "bbox_request_size": request_size,
                        "bbox_pixels": pixels,
                        "scenario": scenario,
                        "compression": compression,
                        "blocksize_px": block,
                        "mode": row["mode"],
                        "concurrency": row["concurrency"],
                        "samples": row["samples"],
                        "tiles_touched": row["tiles_touched"],
                        "bbox_p90_ms": row["bbox_p90_ms"],
                        "bbox_average_ms": row["bbox_average_ms"],
                        "expected_bytes_per_bbox": row["expected_bytes_per_bbox"],
                        "valid_get_ratio": row["valid_get_ratio"],
                        "source_csv": source_csv,
                    })


def format_axis_bytes(value: float, _: float) -> str:
    """Keep linear payload axes readable without scientific notation."""
    return format_bytes(value)


def format_axis_number(value: float, _: float) -> str:
    """Render whole-number latency ticks with thousands separators."""
    return f"{value:,.0f}"


def set_linear_latency_axis(axis: plt.Axes) -> None:
    axis.yaxis.set_major_formatter(FuncFormatter(format_axis_number))


def set_linear_payload_axis(axis: plt.Axes) -> None:
    axis.yaxis.set_major_formatter(FuncFormatter(format_axis_bytes))


def shared_ceiling(component_rows: dict[tuple[str, str, str], dict[str, float]], step: int = 500) -> float:
    return step * np.ceil(max(sum(values.values()) for values in component_rows.values()) / step)


def load_http_stack_components(root: Path) -> dict[tuple[str, str, str], dict[str, float]]:
    """Mean additive HTTP read-time components for reused-client, concurrency 1."""
    request_rows = csv_rows(root / "http_range" / "concurrency_1" / "http_range_request_raw.csv")
    bbox_rows = csv_rows(root / "http_range" / "concurrency_1" / "http_range_bbox_raw.csv")
    request_groups: dict[tuple[str, ...], list[dict[str, str]]] = {}
    for row in request_rows:
        if row["mode"] != "reused-client" or row["concurrency"] != "1" or row["valid_response"] != "True":
            continue
        key = tuple(row[field] for field in ("round", "sequence", "run", "scenario", "variant"))
        request_groups.setdefault(key, []).append(row)
    grouped: dict[tuple[str, str, str], list[dict[str, float]]] = {}
    for row in bbox_rows:
        if row["mode"] != "reused-client" or row["concurrency"] != "1" or row["all_valid"] != "True":
            continue
        request_key = tuple(row[field] for field in ("round", "sequence", "run", "scenario", "variant"))
        requests = request_groups.get(request_key, [])
        ttfb = sum(float(request["ttfb_ms"]) for request in requests)
        download = sum(float(request["download_ms"]) for request in requests)
        values = {"TTFB": ttfb, "Download": download}
        key = (row["scenario"], row["compression"], row["blocksize"])
        grouped.setdefault(key, []).append(values)
    if not grouped:
        raise FileNotFoundError("No valid reused-client HTTP component samples were found")
    return {
        key: {component: mean(sample[component] for sample in samples) for component in HTTP_STACK_COMPONENTS}
        for key, samples in grouped.items()
    }


def load_e2e_stack_components(root: Path) -> dict[tuple[str, str, str], dict[str, float]]:
    """Mean additive cold E2E timing components for the tile-edge experiment."""
    grouped: dict[tuple[str, str, str], list[dict[str, float]]] = {}
    for path in sorted((root / "end_to_end").glob("*/bbox_read_raw.csv")):
        scenario = path.parent.name
        for row in csv_rows(path):
            if row["mode"] != "cold":
                continue
            opened = float(row["open_ms"])
            window = float(row["window_ms"])
            read = float(row["read_ms"])
            total = float(row["total_ms"])
            values = {
                "Open / metadata": opened,
                "Window": window,
                "Raster read": read,
                "Other": max(0.0, total - opened - window - read),
            }
            key = (scenario, row["compression"], row["blocksize"])
            grouped.setdefault(key, []).append(values)
    if not grouped:
        raise FileNotFoundError("No cold E2E component samples were found")
    return {
        key: {component: mean(sample[component] for sample in samples) for component in E2E_STACK_COMPONENTS}
        for key, samples in grouped.items()
    }


def save_bbox_e2e_charts(
    e2e: list[dict[str, str]], case_rows: list[dict[str, str]], output_dir: Path
) -> None:
    by_key = index_rows(e2e, "scenario", "compression", "blocksize", "mode")
    scenarios = [row["scenario"] for row in case_rows]
    labels = bbox_size_labels(case_rows)
    x = np.arange(len(scenarios))
    for compression in COMPRESSIONS:
        figure, axis = plt.subplots(figsize=(10.5, 5.4))
        for block in BLOCKS:
            values = [float(by_key[(scenario, compression, str(block), "cold")]["total_p90_ms"]) for scenario in scenarios]
            tiles = [int(by_key[(scenario, compression, str(block), "cold")]["tiles_touched"]) for scenario in scenarios]
            axis.plot(x, values, marker=BLOCK_MARKERS[block], markersize=6, linewidth=2.2, color=BLOCK_COLORS[block], label=f"{block} px")
            label_offset = {128: 8, 256: -15, 512: 8}[block]
            for index, (value, count) in enumerate(zip(values, tiles)):
                axis.annotate(f"{count}t", (index, value), textcoords="offset points", xytext=(0, label_offset), ha="center", fontsize=8, color=BLOCK_COLORS[block], fontweight="bold")
        set_linear_latency_axis(axis)
        axis.set_xticks(x, labels)
        axis.set_xlabel("BBOX request size")
        axis.set_ylabel("Cold E2E P90 latency (ms)")
        axis.set_title(f"BBOX size trend — {compression}")
        axis.grid(axis="y", alpha=0.28, which="both")
        axis.legend(title="Block size", ncol=3)
        axis.text(0.01, -0.19, "Point labels show COG tiles touched.", transform=axis.transAxes, fontsize=9, color="#475569")
        figure.tight_layout()
        figure.savefig(output_dir / f"01_bbox_size_cold_e2e_{compression.lower()}.png", dpi=170)
        plt.close(figure)


def save_bbox_payload_charts(
    http: list[dict[str, str]], case_rows: list[dict[str, str]], output_dir: Path
) -> None:
    rows = [row for row in http if row["mode"] == "reused-client" and row["concurrency"] == "1"]
    by_key = index_rows(rows, "scenario", "compression", "blocksize")
    scenarios = [row["scenario"] for row in case_rows]
    labels = bbox_size_labels(case_rows)
    x = np.arange(len(scenarios))
    for compression in COMPRESSIONS:
        figure, axis = plt.subplots(figsize=(10.5, 5.4))
        for block in BLOCKS:
            payloads = [float(by_key[(scenario, compression, str(block))]["expected_bytes_per_bbox"]) for scenario in scenarios]
            tiles = [int(by_key[(scenario, compression, str(block))]["tiles_touched"]) for scenario in scenarios]
            axis.plot(x, payloads, marker=BLOCK_MARKERS[block], markersize=6, linewidth=2.2, color=BLOCK_COLORS[block], label=f"{block} px")
            label_offset = {128: 8, 256: -15, 512: 8}[block]
            for index, (value, count) in enumerate(zip(payloads, tiles)):
                axis.annotate(f"{count}t", (index, value), textcoords="offset points", xytext=(0, label_offset), ha="center", fontsize=8, color=BLOCK_COLORS[block], fontweight="bold")
        set_linear_payload_axis(axis)
        axis.set_xticks(x, labels)
        axis.set_xlabel("BBOX request size")
        axis.set_ylabel("Verified HTTP Range payload per BBOX")
        axis.set_title(f"BBOX size → Range payload — {compression}")
        axis.grid(axis="y", alpha=0.28, which="both")
        axis.legend(title="Block size", ncol=3)
        axis.text(0.01, -0.19, "Reused-client, concurrency 1. Point labels show COG tiles touched.", transform=axis.transAxes, fontsize=9, color="#475569")
        figure.tight_layout()
        figure.savefig(output_dir / f"02_bbox_size_range_payload_{compression.lower()}.png", dpi=170)
        plt.close(figure)


def save_edge_heatmaps(e2e: list[dict[str, str]], http: list[dict[str, str]], output_dir: Path) -> None:
    e2e_index = index_rows(e2e, "scenario", "compression", "blocksize", "mode")
    http_index = index_rows(
        [row for row in http if row["mode"] == "reused-client" and row["concurrency"] == "1"],
        "scenario", "compression", "blocksize",
    )
    for compression in COMPRESSIONS:
        values = np.array([
            [float(e2e_index[(scenario, compression, str(block), "cold")]["total_p90_ms"]) for block in BLOCKS]
            for scenario in EDGE_ORDER
        ])
        figure, axis = plt.subplots(figsize=(8.2, 6.5))
        image = axis.imshow(values, cmap="YlOrRd", vmin=values.min(), vmax=values.max(), aspect="auto")
        for row_index, scenario in enumerate(EDGE_ORDER):
            for col_index, block in enumerate(BLOCKS):
                e_row = e2e_index[(scenario, compression, str(block), "cold")]
                h_row = http_index[(scenario, compression, str(block))]
                value = float(e_row["total_p90_ms"])
                text_color = "white" if value > (values.min() + values.max()) / 2 else "#172033"
                axis.text(col_index, row_index, f"{value:.0f} ms\n{e_row['tiles_touched']}t · {format_bytes(float(h_row['expected_bytes_per_bbox']))}", ha="center", va="center", fontsize=8, color=text_color, fontweight="bold")
        axis.set_xticks(range(len(BLOCKS)), [f"{block} px" for block in BLOCKS])
        axis.set_yticks(range(len(EDGE_LABELS)), EDGE_LABELS)
        axis.set_xlabel("COG block size")
        axis.set_ylabel("Same-size BBOX placement")
        axis.set_title(f"Tile-boundary penalty — {compression}\nCold E2E P90; labels: tiles · Range payload")
        colorbar = figure.colorbar(image, ax=axis, pad=0.02)
        colorbar.set_label("Cold E2E P90 latency (ms)")
        colorbar.ax.yaxis.set_major_formatter(FuncFormatter(format_axis_number))
        figure.tight_layout()
        figure.savefig(output_dir / f"03_tile_edge_heatmap_{compression.lower()}.png", dpi=170)
        plt.close(figure)


def save_edge_penalty_scatter(e2e: list[dict[str, str]], http: list[dict[str, str]], output_dir: Path) -> None:
    e2e_index = index_rows(e2e, "scenario", "compression", "blocksize", "mode")
    http_index = index_rows(
        [row for row in http if row["mode"] == "reused-client" and row["concurrency"] == "1"],
        "scenario", "compression", "blocksize",
    )
    for compression in COMPRESSIONS:
        figure, axis = plt.subplots(figsize=(8.4, 5.6))
        for block in BLOCKS:
            interior_e2e = float(e2e_index[("interior", compression, str(block), "cold")]["total_p90_ms"])
            interior_payload = float(http_index[("interior", compression, str(block))]["expected_bytes_per_bbox"])
            xs, ys = [], []
            for scenario in EDGE_ORDER:
                xs.append(float(http_index[(scenario, compression, str(block))]["expected_bytes_per_bbox"]) / interior_payload)
                ys.append(float(e2e_index[(scenario, compression, str(block), "cold")]["total_p90_ms"]) / interior_e2e)
            axis.scatter(xs, ys, s=68, marker=BLOCK_MARKERS[block], color=BLOCK_COLORS[block], edgecolor="white", linewidth=0.7, label=f"{block} px", zorder=3)
            corner = EDGE_ORDER.index("cross_512_corner")
            axis.annotate(f"512 corner · {block}px", (xs[corner], ys[corner]), textcoords="offset points", xytext=(6, 5), fontsize=8, color=BLOCK_COLORS[block])
        axis.axvline(1, color="#64748b", linewidth=1, linestyle="--")
        axis.axhline(1, color="#64748b", linewidth=1, linestyle="--")
        axis.set_xlabel("Range payload multiplier vs Interior (×)")
        axis.set_ylabel("Cold E2E P90 multiplier vs Interior (×)")
        axis.set_title(f"Boundary penalty relationship — {compression}")
        axis.grid(alpha=0.25, which="both")
        axis.legend(title="Block size")
        figure.tight_layout()
        figure.savefig(output_dir / f"04_tile_edge_penalty_vs_payload_{compression.lower()}.png", dpi=170)
        plt.close(figure)


def save_blocksize_facets(
    bbox_e2e: list[dict[str, str]], bbox_http: list[dict[str, str]],
    edge_e2e: list[dict[str, str]], case_rows: list[dict[str, str]], output_dir: Path,
) -> None:
    """Add one chart per block size, comparing compression methods."""
    bbox_e2e_index = index_rows(bbox_e2e, "scenario", "compression", "blocksize", "mode")
    bbox_http_index = index_rows(
        [row for row in bbox_http if row["mode"] == "reused-client" and row["concurrency"] == "1"],
        "scenario", "compression", "blocksize",
    )
    edge_index = index_rows(edge_e2e, "scenario", "compression", "blocksize", "mode")
    scenarios = [row["scenario"] for row in case_rows]
    size_labels = bbox_size_labels(case_rows)
    x_size = np.arange(len(scenarios))
    read_latency_ceiling = 500 * np.ceil(max(
        float(bbox_http_index[(scenario, compression, str(block))]["bbox_p90_ms"])
        for scenario in scenarios
        for compression in COMPRESSIONS
        for block in BLOCKS
    ) / 500)
    edge_latency_ceiling = 500 * np.ceil(max(
        float(edge_index[(scenario, compression, str(block), "cold")]["total_p90_ms"])
        for scenario in EDGE_ORDER
        for compression in COMPRESSIONS
        for block in BLOCKS
    ) / 500)
    edge_latency_ticks = [0, 250, 500, *range(1000, int(edge_latency_ceiling) + 1, 500)]

    for block in BLOCKS:
        figure, axis = plt.subplots(figsize=(10.5, 5.4))
        for compression in COMPRESSIONS:
            values = [float(bbox_e2e_index[(scenario, compression, str(block), "cold")]["total_p90_ms"]) for scenario in scenarios]
            axis.plot(x_size, values, marker=COMPRESSION_MARKERS[compression], markersize=6, linewidth=2.2, color=COMPRESSION_COLORS[compression], label=compression)
        set_linear_latency_axis(axis)
        axis.set_xticks(x_size, size_labels)
        axis.set_xlabel("BBOX request size")
        axis.set_ylabel("Cold E2E P90 latency (ms)")
        axis.set_title(f"BBOX size trend — {block} px blocks")
        axis.grid(axis="y", alpha=0.28, which="both")
        axis.legend(title="Compression", ncol=3)
        figure.tight_layout()
        figure.savefig(output_dir / f"05_bbox_size_cold_e2e_block_{block}px.png", dpi=170)
        plt.close(figure)

        figure, axis = plt.subplots(figsize=(10.5, 5.4))
        for compression in COMPRESSIONS:
            values = [float(bbox_http_index[(scenario, compression, str(block))]["expected_bytes_per_bbox"]) for scenario in scenarios]
            axis.plot(x_size, values, marker=COMPRESSION_MARKERS[compression], markersize=6, linewidth=2.2, color=COMPRESSION_COLORS[compression], label=compression)
        set_linear_payload_axis(axis)
        axis.set_xticks(x_size, size_labels)
        axis.set_xlabel("BBOX request size")
        axis.set_ylabel("Verified HTTP Range payload per BBOX")
        axis.set_title(f"BBOX size → Range payload — {block} px blocks")
        axis.grid(axis="y", alpha=0.28, which="both")
        axis.legend(title="Compression", ncol=3)
        axis.text(0.01, -0.19, "Reused-client, concurrency 1.", transform=axis.transAxes, fontsize=9, color="#475569")
        figure.tight_layout()
        figure.savefig(output_dir / f"06_bbox_size_range_payload_block_{block}px.png", dpi=170)
        plt.close(figure)

        figure, axis = plt.subplots(figsize=(10.5, 5.4))
        compression_order = ("LZW", "DEFLATE", "ZSTD")
        bar_width = 0.22
        bar_offsets = np.linspace(-bar_width, bar_width, len(compression_order))
        for compression, offset in zip(compression_order, bar_offsets):
            values = [float(bbox_http_index[(scenario, compression, str(block))]["bbox_p90_ms"]) for scenario in scenarios]
            is_zstd = compression == "ZSTD"
            axis.bar(
                x_size + offset, values, width=bar_width,
                color=COMPRESSION_COLORS[compression],
                alpha=0.92 if is_zstd else 0.24,
                edgecolor=COMPRESSION_COLORS[compression], linewidth=0.9,
                label=compression,
            )
        set_linear_latency_axis(axis)
        axis.set_ylim(0, read_latency_ceiling)
        axis.yaxis.set_major_locator(MultipleLocator(500))
        axis.set_xticks(x_size, size_labels)
        axis.set_xlabel("BBOX request size")
        axis.set_ylabel("BBOX read latency P90 (ms)")
        axis.set_title(f"BBOX size → read latency — {block} px blocks")
        axis.grid(axis="y", alpha=0.28, which="both")
        axis.legend(title="Compression", ncol=3)
        axis.text(0.01, -0.19, "Grouped bars show HTTP Range BBOX read latency P90. Reused-client, concurrency 1.", transform=axis.transAxes, fontsize=9, color="#475569")
        figure.tight_layout()
        figure.savefig(output_dir / f"08_bbox_size_read_latency_block_{block}px.png", dpi=170)
        plt.close(figure)

        edge_tiles = [edge_index[(scenario, "DEFLATE", str(block), "cold")]["tiles_touched"] for scenario in EDGE_ORDER]
        edge_labels = [
            f"{label}\n{count} {'tile' if count == '1' else 'tiles'}"
            for label, count in zip(EDGE_LABELS, edge_tiles)
        ]
        x_edge = np.arange(len(EDGE_ORDER))
        figure, axis = plt.subplots(figsize=(10.5, 5.4))
        for compression in COMPRESSIONS:
            values = [float(edge_index[(scenario, compression, str(block), "cold")]["total_p90_ms"]) for scenario in EDGE_ORDER]
            axis.plot(x_edge, values, marker=COMPRESSION_MARKERS[compression], markersize=6, linewidth=2.2, color=COMPRESSION_COLORS[compression], label=compression)
        set_linear_latency_axis(axis)
        axis.set_ylim(0, edge_latency_ceiling)
        axis.set_yticks(edge_latency_ticks)
        axis.set_xticks(x_edge, edge_labels, rotation=18, ha="right")
        axis.set_xlabel("Same-size BBOX placement · COG tiles touched")
        axis.set_ylabel("Cold E2E P90 latency (ms)")
        axis.set_title(f"Tile-boundary trend — {block} px blocks · shared scale")
        axis.grid(axis="y", alpha=0.28, which="both")
        axis.legend(title="Compression", ncol=3)
        axis.text(0.01, -0.29, "Same BBOX dimensions; only its alignment to the COG tile grid changes.", transform=axis.transAxes, fontsize=9, color="#475569")
        figure.tight_layout()
        figure.subplots_adjust(bottom=0.28)
        figure.savefig(output_dir / f"07_tile_edge_cold_e2e_block_{block}px.png", dpi=170)
        plt.close(figure)


def save_combined_bbox_read_latency(
    bbox_http: list[dict[str, str]], case_rows: list[dict[str, str]], output_dir: Path,
) -> None:
    """Create one shared-scale view of BBOX read latency across all block sizes."""
    rows = [row for row in bbox_http if row["mode"] == "reused-client" and row["concurrency"] == "1"]
    by_key = index_rows(rows, "scenario", "compression", "blocksize")
    scenarios = [row["scenario"] for row in case_rows]
    x_size = np.arange(len(scenarios))
    latency_ceiling = 500 * np.ceil(max(
        float(by_key[(scenario, compression, str(block))]["bbox_p90_ms"])
        for scenario in scenarios
        for compression in COMPRESSIONS
        for block in BLOCKS
    ) / 500)

    figure, axes = plt.subplots(1, len(COMPRESSIONS), figsize=(17.5, 5.8), sharey=True)
    handles = []
    for axis, compression in zip(axes, COMPRESSIONS):
        for block in BLOCKS:
            values = [float(by_key[(scenario, compression, str(block))]["bbox_p90_ms"]) for scenario in scenarios]
            line, = axis.plot(
                x_size, values, marker=BLOCK_MARKERS[block], markersize=5.5,
                linewidth=2.1, color=BLOCK_COLORS[block], label=f"{block} px",
            )
            if compression == COMPRESSIONS[0]:
                handles.append(line)
        set_linear_latency_axis(axis)
        axis.set_ylim(0, latency_ceiling)
        axis.yaxis.set_major_locator(MultipleLocator(500))
        axis.set_xticks(x_size, bbox_size_labels(case_rows), rotation=22, ha="right", fontsize=8.5)
        axis.set_xlabel("BBOX request size")
        axis.set_title(compression)
        axis.grid(axis="y", alpha=0.28)
    axes[0].set_ylabel("BBOX read latency P90 (ms)")
    figure.suptitle("BBOX size → read latency — all block sizes", fontsize=17, y=0.99)
    figure.legend(handles=handles, title="Block size", ncol=3, loc="upper center", bbox_to_anchor=(0.5, 0.93))
    figure.text(0.5, 0.015, "HTTP Range benchmark; reused-client, concurrency 1. Shared Y-axis: 0–4,500 ms.", ha="center", fontsize=10, color="#475569")
    figure.subplots_adjust(left=0.065, right=0.99, top=0.80, bottom=0.23, wspace=0.13)
    figure.savefig(output_dir / "08_bbox_size_read_latency_all_blocks.png", dpi=170)
    plt.close(figure)


def save_compression_bbox_read_latency(
    bbox_http: list[dict[str, str]], case_rows: list[dict[str, str]], output_dir: Path,
) -> None:
    """Create one shared-scale BBOX read-latency chart for each compression."""
    rows = [row for row in bbox_http if row["mode"] == "reused-client" and row["concurrency"] == "1"]
    by_key = index_rows(rows, "scenario", "compression", "blocksize")
    scenarios = [row["scenario"] for row in case_rows]
    x_size = np.arange(len(scenarios))
    latency_ceiling = 500 * np.ceil(max(
        float(by_key[(scenario, compression, str(block))]["bbox_p90_ms"])
        for scenario in scenarios
        for compression in COMPRESSIONS
        for block in BLOCKS
    ) / 500)

    for compression in COMPRESSIONS:
        figure, axis = plt.subplots(figsize=(10.5, 5.5))
        for block in BLOCKS:
            values = [float(by_key[(scenario, compression, str(block))]["bbox_p90_ms"]) for scenario in scenarios]
            axis.plot(
                x_size, values, marker=BLOCK_MARKERS[block], markersize=6,
                linewidth=2.2, color=BLOCK_COLORS[block], label=f"{block} px",
            )
        set_linear_latency_axis(axis)
        axis.set_ylim(0, latency_ceiling)
        axis.yaxis.set_major_locator(MultipleLocator(500))
        axis.set_xticks(x_size, bbox_size_labels(case_rows))
        axis.set_xlabel("BBOX request size")
        axis.set_ylabel("BBOX read latency P90 (ms)")
        axis.set_title(f"BBOX size → read latency — {compression}")
        axis.grid(axis="y", alpha=0.28)
        axis.legend(title="COG block size", ncol=3)
        axis.text(0.01, -0.19, "HTTP Range benchmark; reused-client, concurrency 1. Shared Y-axis: 0–4,500 ms.", transform=axis.transAxes, fontsize=9, color="#475569")
        figure.tight_layout()
        figure.savefig(output_dir / f"08_bbox_size_read_latency_{compression.lower()}.png", dpi=170)
        plt.close(figure)


def save_bbox_http_stackbar_charts(
    components: dict[tuple[str, str, str], dict[str, float]],
    case_rows: list[dict[str, str]], output_dir: Path,
) -> None:
    """Stack additive HTTP read-time components for each BBOX-size benchmark."""
    scenarios = [row["scenario"] for row in case_rows]
    x = np.arange(len(scenarios))
    offsets = {"DEFLATE": -0.25, "LZW": 0.0, "ZSTD": 0.25}
    ceiling = shared_ceiling(components)
    component_handles = [Patch(facecolor=HTTP_STACK_COLORS[item], label=item) for item in HTTP_STACK_COMPONENTS]
    compression_handles = [
        Patch(facecolor="#ffffff", edgecolor=COMPRESSION_COLORS[item], hatch=COMPRESSION_HATCHES[item], label=item)
        for item in COMPRESSIONS
    ]

    for block in BLOCKS:
        figure, axis = plt.subplots(figsize=(11.2, 5.9))
        for compression in COMPRESSIONS:
            bottoms = np.zeros(len(scenarios))
            for component in HTTP_STACK_COMPONENTS:
                values = np.array([components[(scenario, compression, str(block))][component] for scenario in scenarios])
                axis.bar(
                    x + offsets[compression], values, width=0.23, bottom=bottoms,
                    color=HTTP_STACK_COLORS[component], edgecolor=COMPRESSION_COLORS[compression],
                    linewidth=0.8, hatch=COMPRESSION_HATCHES[compression], alpha=1.0 if compression == "ZSTD" else 0.74,
                )
                bottoms += values
        axis.set_ylim(0, ceiling)
        axis.yaxis.set_major_locator(MultipleLocator(500))
        set_linear_latency_axis(axis)
        axis.set_xticks(x, bbox_size_labels(case_rows))
        axis.set_xlabel("BBOX request size")
        axis.set_ylabel("Mean summed HTTP Range time (ms)")
        axis.set_title(f"HTTP Range read-time composition — {block} px blocks")
        axis.grid(axis="y", alpha=0.28)
        component_legend = axis.legend(handles=component_handles, title="Additive time component", loc="upper left", ncol=3)
        axis.add_artist(component_legend)
        axis.legend(handles=compression_handles, title="Compression / hatch", loc="upper right", ncol=3)
        figure.text(
            0.5, 0.015,
            "Each bar is one compression: DEFLATE · LZW · ZSTD (left to right). Total bar height = mean summed HTTP Range time; reused-client, concurrency 1.",
            ha="center", fontsize=9, color="#475569",
        )
        figure.subplots_adjust(left=0.09, right=0.99, top=0.90, bottom=0.16)
        figure.savefig(output_dir / f"08_bbox_size_read_latency_stackbar_block_{block}px.png", dpi=170)
        plt.close(figure)


def save_edge_e2e_stackbar_charts(
    components: dict[tuple[str, str, str], dict[str, float]], output_dir: Path,
) -> None:
    """Show total cold E2E time by block size for interior, edge, and corner placement."""
    ceiling = shared_ceiling(components)
    offsets = {128: -0.25, 256: 0.0, 512: 0.25}
    block_handles = [Patch(facecolor=BLOCK_COLORS[item], edgecolor=BLOCK_COLORS[item], label=f"{item} px") for item in BLOCKS]

    for boundary, (edge, corner) in BOUNDARY_GROUPS.items():
        scenarios = ("interior", edge, corner)
        x = np.arange(len(scenarios))
        labels = ("Interior", f"{boundary} edge", f"{boundary} corner")
        figure, axes = plt.subplots(1, len(COMPRESSIONS), figsize=(17.0, 6.1), sharey=True)
        for axis, compression in zip(axes, COMPRESSIONS):
            for block in BLOCKS:
                values = np.array([
                    sum(components[(scenario, compression, str(block))].values())
                    for scenario in scenarios
                ])
                axis.bar(
                    x + offsets[block], values, width=0.23,
                    color=BLOCK_COLORS[block], edgecolor=BLOCK_COLORS[block], linewidth=0.9,
                )
            axis.set_ylim(0, ceiling)
            axis.set_yticks([0, 250, 500, *range(1000, int(ceiling) + 1, 500)])
            set_linear_latency_axis(axis)
            axis.set_xticks(x, labels)
            axis.set_xlabel("Same-size BBOX placement")
            axis.set_title(compression)
            axis.grid(axis="y", alpha=0.28)
        axes[0].set_ylabel("Mean cold E2E time (ms)")
        figure.suptitle(f"Cold E2E time — {boundary} px boundary", fontsize=16, y=0.98)
        figure.legend(handles=block_handles, title="COG block size", loc="upper left", bbox_to_anchor=(0.065, 0.945), ncol=3)
        figure.text(
            0.5, 0.06,
            f"Interior = within a tile · {boundary} edge = crosses one tile edge · {boundary} corner = crosses a tile corner",
            ha="center", fontsize=9.5, color="#475569",
        )
        figure.text(
            0.5, 0.015,
            "Each placement group has 128 px · 256 px · 512 px bars (left to right). Bar height = mean of 10 cold E2E reads.",
            ha="center", fontsize=9, color="#475569",
        )
        figure.subplots_adjust(left=0.065, right=0.99, top=0.80, bottom=0.20, wspace=0.13)
        figure.savefig(output_dir / f"09_tile_edge_boundary_{boundary}px_stackbar.png", dpi=170)
        plt.close(figure)


def save_boundary_group_charts(edge_e2e: list[dict[str, str]], output_dir: Path) -> None:
    """Group the same numbered edge and corner placements with Interior as baseline."""
    by_key = index_rows(edge_e2e, "scenario", "compression", "blocksize", "mode")
    for boundary, (edge, corner) in BOUNDARY_GROUPS.items():
        scenarios = ("interior", edge, corner)
        boundary_max = max(
            float(by_key[(scenario, compression, str(block), "cold")]["total_p90_ms"])
            for scenario in scenarios
            for compression in COMPRESSIONS
            for block in BLOCKS
        )
        tick_step = 250 if boundary_max <= 1500 else 500
        latency_ceiling = max(tick_step * 2, tick_step * np.ceil(boundary_max * 1.12 / tick_step))
        latency_ticks = list(range(0, int(latency_ceiling) + 1, tick_step))
        x = np.arange(len(scenarios))
        labels = ("Interior", f"{boundary} edge", f"{boundary} corner")
        figure, axes = plt.subplots(1, len(COMPRESSIONS), figsize=(16.8, 6.2), sharey=True)
        bar_width = 0.23
        bar_offsets = {128: -bar_width, 256: 0.0, 512: bar_width}
        handles = [Patch(facecolor=BLOCK_COLORS[block], edgecolor=BLOCK_COLORS[block], label=f"{block} px") for block in BLOCKS]
        for axis, compression in zip(axes, COMPRESSIONS):
            for block in BLOCKS:
                values = [float(by_key[(scenario, compression, str(block), "cold")]["total_p90_ms"]) for scenario in scenarios]
                axis.bar(
                    x + bar_offsets[block], values, width=bar_width,
                    color=BLOCK_COLORS[block], edgecolor=BLOCK_COLORS[block], linewidth=0.9,
                )
            set_linear_latency_axis(axis)
            axis.set_ylim(0, latency_ceiling)
            axis.set_yticks(latency_ticks)
            axis.set_xticks(x, labels)
            axis.set_xlabel("Same-size BBOX placement")
            axis.set_title(compression)
            axis.grid(axis="y", alpha=0.28)
        axes[0].set_ylabel("Cold E2E P90 latency (ms)")
        figure.suptitle(f"Same-size BBOX — {boundary} px boundary alignment", fontsize=17, y=0.99)
        figure.legend(handles=handles, title="COG block size", ncol=3, loc="upper center", bbox_to_anchor=(0.5, 0.945))
        figure.text(
            0.5, 0.065,
            f"Interior = BBOX stays within a tile · {boundary} edge = crosses one {boundary} px tile edge · "
            f"{boundary} corner = crosses the {boundary} px tile corner",
            ha="center", fontsize=9.5, color="#475569",
        )
        tile_counts = []
        for block in BLOCKS:
            counts = "/".join(by_key[(scenario, "DEFLATE", str(block), "cold")]["tiles_touched"] for scenario in scenarios)
            tile_counts.append(f"{block} px: {counts} tiles")
        figure.text(
            0.5, 0.015,
            "Interior / edge / corner tiles touched — " + " · ".join(tile_counts),
            ha="center", fontsize=10, color="#475569",
        )
        figure.subplots_adjust(left=0.065, right=0.99, top=0.75, bottom=0.22, wspace=0.13)
        figure.savefig(output_dir / f"09_tile_edge_boundary_{boundary}px.png", dpi=170)
        plt.close(figure)


def save_512_corner_comparison(edge_e2e: list[dict[str, str]], output_dir: Path) -> None:
    """Compare all compressions and COG block sizes for the 512 px corner case."""
    by_key = index_rows(edge_e2e, "scenario", "compression", "blocksize", "mode")
    scenario = "cross_512_corner"
    values = {
        (compression, block): float(by_key[(scenario, compression, str(block), "cold")]["total_p90_ms"])
        for compression in COMPRESSIONS
        for block in BLOCKS
    }
    ceiling = 500 * np.ceil(max(values.values()) * 1.12 / 500)
    x = np.arange(len(COMPRESSIONS))
    bar_width = 0.23
    offsets = {128: -bar_width, 256: 0.0, 512: bar_width}
    figure, axis = plt.subplots(figsize=(10.5, 5.8))
    for block in BLOCKS:
        bar_values = [values[(compression, block)] for compression in COMPRESSIONS]
        axis.bar(
            x + offsets[block], bar_values, width=bar_width,
            color=BLOCK_COLORS[block], edgecolor=BLOCK_COLORS[block], linewidth=0.9,
            label=f"{block} px",
        )
    set_linear_latency_axis(axis)
    axis.set_ylim(0, ceiling)
    axis.set_yticks(list(range(0, int(ceiling) + 1, 500)))
    axis.set_xticks(x, COMPRESSIONS)
    axis.set_xlabel("Compression")
    axis.set_ylabel("Cold E2E P90 latency (ms)")
    axis.set_title("512 px corner crossing")
    axis.grid(axis="y", alpha=0.28)
    axis.legend(title="COG block size", ncol=3, loc="upper left")
    figure.text(
        0.5, 0.015,
        "Same-size BBOX crosses a 512 px tile corner. Lower bar = faster P90 response.",
        ha="center", fontsize=9.5, color="#475569",
    )
    figure.subplots_adjust(left=0.10, right=0.98, top=0.90, bottom=0.16)
    figure.savefig(output_dir / "10_512_corner_by_compression.png", dpi=170)
    plt.close(figure)


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    bbox_e2e, edge_e2e = load_e2e(args.bbox_output), load_e2e(args.edge_output)
    bbox_http, edge_http = load_http(args.bbox_output), load_http(args.edge_output)
    case_rows = csv_rows(args.case_file)
    save_bbox_read_latency_chart_csvs(bbox_http, case_rows, args.output_dir)
    save_bbox_e2e_charts(bbox_e2e, case_rows, args.output_dir)
    save_bbox_payload_charts(bbox_http, case_rows, args.output_dir)
    save_edge_heatmaps(edge_e2e, edge_http, args.output_dir)
    save_edge_penalty_scatter(edge_e2e, edge_http, args.output_dir)
    save_blocksize_facets(bbox_e2e, bbox_http, edge_e2e, case_rows, args.output_dir)
    save_combined_bbox_read_latency(bbox_http, case_rows, args.output_dir)
    save_compression_bbox_read_latency(bbox_http, case_rows, args.output_dir)
    save_boundary_group_charts(edge_e2e, args.output_dir)
    save_512_corner_comparison(edge_e2e, args.output_dir)
    save_bbox_http_stackbar_charts(load_http_stack_components(args.bbox_output), case_rows, args.output_dir)
    print(f"Output: {args.output_dir}")
    print("Charts: 35")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
