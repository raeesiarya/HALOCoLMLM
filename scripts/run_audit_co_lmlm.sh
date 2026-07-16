#!/usr/bin/env bash
# Run the HALO audit on Co-LMLM.
#
# Invoke from the HALO repo, but it executes inside the public Co-LMLM
# environment (Co-LMLM ships its own `lmlm` package, so it must run there):
# it cd's into the checkout and puts HALO's src on PYTHONPATH.
#
# Required:  COLMLM_DIR=/path/to/Co-LMLM   (the public checkout)
# Optional:  INDEX_DIR, PROMPTS, OUTPUT_DIR
# Extra flags (e.g. --closure, --radius-grid, --adversarial) are passed through:
#   ./scripts/run_audit_co_lmlm.sh --closure geometric,semantic --radius-grid 0.95:0.70:0.05
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
: "${COLMLM_DIR:?set COLMLM_DIR to the public Co-LMLM checkout}"

INDEX_DIR="${INDEX_DIR:-$REPO_ROOT/data/co-lmlm-wiki-index}"
PROMPTS="${PROMPTS:-$REPO_ROOT/data/prompts.jsonl}"
OUTPUT_DIR="${OUTPUT_DIR:-$REPO_ROOT/outputs/popqa}"

cd "$COLMLM_DIR"
PYTHONPATH="$REPO_ROOT/src:src${PYTHONPATH:+:$PYTHONPATH}" \
uv run python -m halo.run_audit \
    --backend co-lmlm \
    --index-path "$INDEX_DIR" \
    --prompt-files "$PROMPTS" \
    --bootstrap-oracle-from-full \
    --wandb-activation on \
    --output-dir "$OUTPUT_DIR" \
    "$@"
