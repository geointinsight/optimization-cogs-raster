#!/usr/bin/env bash
# Benchmark BBOXs deliberately positioned on COG tile boundaries.
#
# Generate/validate the case file first:
#   python3 generate_tile_edge_alignment_cases.py
#
# Validate command construction without opening S3:
#   bash run_tile_edge_alignment_benchmark.sh --dry-run
#
# Presentation-grade latency run:
#   ROUNDS=5 RUNS=10 WARMUP=5 bash run_tile_edge_alignment_benchmark.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CASE_FILE="${CASE_FILE:-${SCRIPT_DIR}/tile_edge_alignment_cases.csv}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${SCRIPT_DIR}/s3_tile_edge_alignment_benchmark_output}"
USR_ID="${USR_ID:-gis-asset}"
COLLECTION_ID="${COLLECTION_ID:-6a3bfdd89e3416fefefdf90b}"
ROUNDS="${ROUNDS:-5}"
RUNS="${RUNS:-10}"
WARMUP="${WARMUP:-5}"

if [[ ! -f "${CASE_FILE}" ]]; then
  echo "Case file not found: ${CASE_FILE}" >&2
  exit 2
fi

while IFS=, read -r scenario description min_x min_y max_x max_y relative_size _; do
  [[ "${scenario}" == "scenario" || -z "${scenario}" ]] && continue
  echo "Running ${scenario} (${description}; ${relative_size})"
  command=(
    python3 "${SCRIPT_DIR}/cog_s3_bbox_round_benchmark.py"
    --usr-id "${USR_ID}"
    --collection-id "${COLLECTION_ID}"
    --bbox "${min_x}" "${min_y}" "${max_x}" "${max_y}"
    --filename-template '{variant}.tif'
    --rounds "${ROUNDS}"
    --runs "${RUNS}"
    --warmup "${WARMUP}"
    --output-dir "${OUTPUT_ROOT}/${scenario}"
  )
  if (( $# )); then
    command+=("$@")
  fi
  "${command[@]}"
done < "${CASE_FILE}"
