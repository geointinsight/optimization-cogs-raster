#!/usr/bin/env bash
# Verify explicit MinIO HTTP 206 Range GET payloads for the tile-edge cases.
#
# Quick deterministic layout check (no object-body GETs):
#   bash run_tile_edge_alignment_range_analysis.sh --dry-run
#
# Full payload verification:
#   bash run_tile_edge_alignment_range_analysis.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CASE_FILE="${CASE_FILE:-${SCRIPT_DIR}/tile_edge_alignment_cases.csv}"
OUTPUT_DIR="${OUTPUT_DIR:-${SCRIPT_DIR}/s3_tile_edge_alignment_range_output}"
USR_ID="${USR_ID:-gis-asset}"
COLLECTION_ID="${COLLECTION_ID:-6a3bfdd89e3416fefefdf90b}"

if [[ ! -f "${CASE_FILE}" ]]; then
  echo "Case file not found: ${CASE_FILE}" >&2
  exit 2
fi

command=(
  python3 "${SCRIPT_DIR}/cog_s3_cog_tile_range_analysis.py"
  --usr-id "${USR_ID}"
  --collection-id "${COLLECTION_ID}"
  --case-file "${CASE_FILE}"
  --filename-template '{variant}.tif'
  --verify-http-response
  --output-dir "${OUTPUT_DIR}"
)
if (( $# )); then
  command+=("$@")
fi
"${command[@]}"
