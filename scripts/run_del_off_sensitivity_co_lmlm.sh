#!/usr/bin/env bash
# Standard audit under both retrieval-disabled controls.
#
# Optional: OUTPUT_DIR, STANDARD_CLOSURE, DEL_OFF_MODES (space-separated
#           subset of "null-retrieval forbid-token"). The audit suite passes
#           DEL_OFF_MODES to skip the arm its own phase 1 already covers.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BASE_OUTPUT_DIR="${OUTPUT_DIR:-$REPO_ROOT/outputs/trex_del_off_sensitivity}"
STANDARD_CLOSURE="${STANDARD_CLOSURE:-geometric,value}"
DEL_OFF_MODES="${DEL_OFF_MODES:-null-retrieval forbid-token}"

for mode in $DEL_OFF_MODES; do
    echo "=== DEL-OFF sensitivity: $mode ==="
    OUTPUT_DIR="$BASE_OUTPUT_DIR/$mode" \
    "$REPO_ROOT/scripts/run_audit_co_lmlm.sh" \
        --closure "$STANDARD_CLOSURE" \
        --co-lmlm-del-off-mode "$mode" \
        "$@"
done
