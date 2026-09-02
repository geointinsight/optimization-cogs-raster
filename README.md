# COG S3 Benchmark and Optimization

This package contains reproducible Python benchmarks for choosing a Cloud Optimized GeoTIFF (COG) compression and block size for S3-compatible object storage. It measures both user-visible read latency and the underlying HTTP Range GET work needed to service each BBOX request.

The included result folders are immutable reference runs. Write new experiments to a separate timestamped directory so that the supplied baseline remains comparable.

## What is measured

| Question | Measurement | Primary script |
| --- | --- | --- |
| How long does a raster BBOX read take? | Cold/warm end-to-end latency, including open, window construction, and raster read | `scripts/cog_s3_bbox_benchmark.py` |
| What does an explicit object-store read cost? | HTTP `Range` GET TTFB, download time, response bytes, and BBOX wall time | `scripts/cog_s3_http_range_latency.py` |
| Which compressed COG tiles intersect a BBOX? | Tile count and compressed byte ranges, optionally verified with HTTP 206 responses | `scripts/cog_s3_cog_tile_range_analysis.py` |
| What changes when a BBOX crosses a tile edge or corner? | Same-size BBOX placement against 128, 256, and 512 px tile grids | `scripts/generate_tile_edge_alignment_cases.py` and `scripts/cog_s3_bbox_round_benchmark.py` |
| How do the results compare? | Presentation-ready trend charts derived from the raw and summary CSVs | `scripts/plot_latency_trends.py` |

## Directory layout

```text
cases/                         Reproducible BBOX test inputs
scripts/                       Benchmark, range-analysis, and chart scripts
results/
  s3_latency_bbox_size_output/ BBOX-size diagnostics: E2E + HTTP Range data
  s3_latency_tile_edge_output/ Tile-boundary diagnostics: E2E + HTTP Range data
  s3_cog_tile_http_range_output/ Verified COG tile-range payload reference data
  s3_tile_edge_alignment_benchmark_output/ Multi-round tile-edge latency reference data
  s3_tile_edge_alignment_range_output/ Verified tile-edge Range GET payload data
  s3_latency_trend_charts/     Derived comparison charts and chart-source CSVs
```

## Test data contract

- BBOX CSV files use WGS 84 longitude/latitude order: `min_x,min_y,max_x,max_y`.
- Every test compares the same nine COG variants: `LZW`, `DEFLATE`, and `ZSTD`, each with 128, 256, and 512 px blocks.
- The source raster geometry is held constant; compression and block size are the variables under test.
- Result `environment.json` files retain non-secret runtime configuration and the tested object identifiers. Credentials and response bodies are not stored.

## Requirements

Use Python 3.11+ with packages compatible with the local GDAL/Rasterio build:

```bash
python3 -m pip install boto3 numpy rasterio python-dotenv matplotlib
```

Provide S3-compatible access through environment variables or a local `.env` file. Do not commit credentials.

```bash
AWS_DEFAULT_REGION=ap-southeast-1
AWS_S3_ENDPOINT=host:port
AWS_S3_SIGNATURE_VERSION=s3v4
AWS_VIRTUAL_HOSTING=FALSE
AWS_HTTPS=NO
AWS_ACCESS_KEY_ID=...
AWS_SECRET_ACCESS_KEY=...
```

The scripts accept `USR_ID` and `COLLECTION_ID` to select the S3 object layout:

```text
s3://<USR_ID>/coverage/<COLLECTION_ID>/cog/<variant>.tif
```

## Recommended optimization workflow

1. Establish a BBOX-size baseline with end-to-end and explicit HTTP Range measurements.
2. Run the same-size tile-edge cases to quantify how many extra COG tiles are read when a request crosses an edge or corner.
3. Compare P90 latency, tile fan-out, verified response bytes, and file size together. Do not decide from compression ratio alone.
4. Use cold E2E P90 for a user-facing latency decision; use HTTP Range data to explain request fan-out and object-store transfer behavior.
5. Repeat at the expected client concurrency before selecting a production default.

## Run BBOX-size diagnostics

From this directory, write a fresh run outside the supplied reference folders:

```bash
CASE_FILE=cases/bbox_size_test_cases.csv \
OUTPUT_ROOT=results/runs/bbox_size_$(date +%Y%m%d_%H%M%S) \
USR_ID=gis-asset \
COLLECTION_ID=<collection-id> \
E2E_RUNS=30 E2E_WARMUP=5 \
HTTP_ROUNDS=5 HTTP_RUNS=10 HTTP_WARMUP=2 \
CONCURRENCIES='1 4 8' \
bash scripts/run_latency_diagnostics.sh
```

For a quick pilot, use `E2E_RUNS=10`, `E2E_WARMUP=3`, `HTTP_ROUNDS=3`, `HTTP_RUNS=5`, and `HTTP_WARMUP=1`.

## Run tile-edge diagnostics

Use the same request footprint in `tile_edge_alignment_cases.csv`; only its location changes from interior to edge or corner.

```bash
CASE_FILE=cases/tile_edge_alignment_cases.csv \
OUTPUT_ROOT=results/runs/tile_edge_$(date +%Y%m%d_%H%M%S) \
USR_ID=gis-asset \
COLLECTION_ID=<collection-id> \
E2E_RUNS=30 E2E_WARMUP=5 \
HTTP_ROUNDS=5 HTTP_RUNS=10 HTTP_WARMUP=2 \
CONCURRENCIES='1 4 8' \
bash scripts/run_latency_diagnostics.sh
```

For the multi-round reopen benchmark and one verified payload read per tile:

```bash
CASE_FILE=cases/tile_edge_alignment_cases.csv \
OUTPUT_ROOT=results/runs/tile_edge_rounds_$(date +%Y%m%d_%H%M%S) \
ROUNDS=5 RUNS=10 WARMUP=5 \
bash scripts/run_tile_edge_alignment_benchmark.sh

CASE_FILE=cases/tile_edge_alignment_cases.csv \
OUTPUT_DIR=results/runs/tile_edge_ranges_$(date +%Y%m%d_%H%M%S) \
bash scripts/run_tile_edge_alignment_range_analysis.sh
```

## Generate comparison charts

Point this command at two completed `run_latency_diagnostics.sh` outputs:

```bash
python3 scripts/plot_latency_trends.py \
  --bbox-output results/s3_latency_bbox_size_output \
  --edge-output results/s3_latency_tile_edge_output \
  --case-file cases/bbox_size_test_cases.csv \
  --output-dir results/runs/trend_charts_$(date +%Y%m%d_%H%M%S)
```

## How to interpret the outputs

- `bbox_read_raw.csv` and `http_range_request_raw.csv` are the audit-level records. Use them for investigation and custom aggregation.
- `bbox_read_summary.csv` and `http_range_latency_summary.csv` contain P50, P90, P95, mean, tile count, and byte summaries by variant and test condition.
- `tiles_touched` is the key fan-out measure: a BBOX that crosses a block boundary can require multiple independent compressed tile reads.
- `expected_bytes_per_bbox` and verified response bytes describe transfer work; they are diagnostic metrics, not substitutes for end-to-end latency.
- `cold` reflects first-open behavior in the client process. `warm` reflects reuse within that process. `fresh-client` and `reused-client` make HTTP client reuse explicit, but do not prove object-store cache state.

## Selection guidance

Choose the COG configuration against the access pattern:

- Favor smaller blocks when the dominant workload is many small, localized BBOX requests and measured P90 remains acceptable.
- Favor larger blocks when requests commonly cover a larger contiguous area and fewer tiles reduce request fan-out.
- Treat ZSTD file-size savings and read-latency results as separate measurements. A smaller object is not automatically the fastest remote read.
- Validate a candidate at interior, edge, and corner placement. A configuration that looks fast within one tile can degrade when the BBOX crosses grid boundaries.

Record the exact test case file, runs, warmups, concurrency, client mode, object version, and result directory whenever comparing configurations.
