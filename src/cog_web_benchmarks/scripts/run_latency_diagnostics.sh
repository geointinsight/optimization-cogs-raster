#!/usr/bin/env bash
# Run comparable end-to-end and explicit HTTP Range GET latency diagnostics.
#
# Default BBOX cases cover increasing request sizes.  For the same-size,
# tile-boundary experiment, set CASE_FILE=tile_edge_alignment_cases.csv.
#
# Pilot:
#   E2E_RUNS=10 E2E_WARMUP=3 HTTP_ROUNDS=3 HTTP_RUNS=5 HTTP_WARMUP=1 \
#     CONCURRENCIES='1 4 8' bash run_latency_diagnostics.sh
#
# Full:
#   E2E_RUNS=30 E2E_WARMUP=5 HTTP_ROUNDS=5 HTTP_RUNS=10 HTTP_WARMUP=2 \
#     CONCURRENCIES='1 4 8' bash run_latency_diagnostics.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CASE_FILE="${CASE_FILE:-${SCRIPT_DIR}/bbox_size_test_cases.csv}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${SCRIPT_DIR}/s3_latency_diagnostics_output}"
USR_ID="${USR_ID:-gis-asset}"
COLLECTION_ID="${COLLECTION_ID:-6a3bfdd89e3416fefefdf90b}"
E2E_RUNS="${E2E_RUNS:-30}"
E2E_WARMUP="${E2E_WARMUP:-5}"
HTTP_ROUNDS="${HTTP_ROUNDS:-5}"
HTTP_RUNS="${HTTP_RUNS:-10}"
HTTP_WARMUP="${HTTP_WARMUP:-2}"
CONCURRENCIES="${CONCURRENCIES:-1 4 8}"

if [[ ! -f "${CASE_FILE}" ]]; then
  echo "Case file not found: ${CASE_FILE}" >&2
  exit 2
fi

mkdir -p "${OUTPUT_ROOT}/end_to_end"
while IFS=, read -r scenario description min_x min_y max_x max_y _; do
  [[ "${scenario}" == "scenario" || -z "${scenario}" ]] && continue
  echo "End-to-end cold/warm: ${scenario} (${description})"
  python3 "${SCRIPT_DIR}/cog_s3_bbox_benchmark.py" \
    --usr-id "${USR_ID}" \
    --collection-id "${COLLECTION_ID}" \
    --bbox "${min_x}" "${min_y}" "${max_x}" "${max_y}" \
    --filename-template '{variant}.tif' \
    --read-mode both \
    --runs "${E2E_RUNS}" \
    --warmup "${E2E_WARMUP}" \
    --output-dir "${OUTPUT_ROOT}/end_to_end/${scenario}"
done < "${CASE_FILE}"

for concurrency in ${CONCURRENCIES}; do
  echo "HTTP Range GET timing: concurrency=${concurrency}"
  python3 "${SCRIPT_DIR}/cog_s3_http_range_latency.py" \
    --usr-id "${USR_ID}" \
    --collection-id "${COLLECTION_ID}" \
    --case-file "${CASE_FILE}" \
    --filename-template '{variant}.tif' \
    --mode both \
    --concurrency "${concurrency}" \
    --rounds "${HTTP_ROUNDS}" \
    --runs "${HTTP_RUNS}" \
    --warmup "${HTTP_WARMUP}" \
    --output-dir "${OUTPUT_ROOT}/http_range/concurrency_${concurrency}"
done

echo "Output: ${OUTPUT_ROOT}"
