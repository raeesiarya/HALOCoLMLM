"""Isolate why a sweep reuse canary failed: nondeterminism vs. search-depth.

The reuse fast paths in ``run_entanglement_sweep`` rest on two claims about
the backend: greedy decoding is deterministic, and a deletion manifest that
catches nothing the FULL pass retrieved cannot change the generation. The
second claim is only sound if the index's top-1 is independent of the
requested ``k`` — FULL searches at ``k=1`` while DEL-ON searches at
``k = 1 + |entry_ids|`` (halo.interventions.filtering).

This script separates the two. Arm A runs the same prompt repeatedly in the
same state; any disagreement is nondeterminism. Arm B runs DEL-ON with
*inert* exclusion sets of growing size — ids that match nothing, so the
filter removes no candidate and only ``search_k`` varies. Any disagreement
there is k-dependence, which falsifies ``full_row_unaffected``.

Run it from the Co-LMLM checkout, in that environment:

    PYTHONPATH=/path/to/HALO/src:src uv run python \\
        /path/to/HALO/scripts/diagnose_reuse_canary.py \\
        --index-path /path/to/HALO/data/co-lmlm-wiki-index \\
        --prompts /path/to/HALO/data/prompts_trex.jsonl \\
        --subject-key fa3aada7-7a6e-4980-b303-4159aa3566dc
"""

from __future__ import annotations

import argparse
import dataclasses
from pathlib import Path

from halo.core.examples import DeletionManifest
from halo.core.states import DatabaseState


def _observe(backend, example, state, max_new_tokens):
    observation = backend.generate(example, state, max_new_tokens=max_new_tokens)
    trace = dict(observation.retrieval_trace or {})
    selected = trace.get("selected_candidate") or {}
    events = trace.get("retrieval_events") or []
    return {
        "output": observation.model_output,
        "selected": selected.get("entry_id"),
        "value": trace.get("selected_value"),
        "searched_top_k": [event.get("searched_top_k") for event in events],
        "candidates": trace.get("all_candidates_count"),
        "retrievals": trace.get("num_retrievals"),
        "failed": trace.get("failed_retrievals"),
    }


def _report(label, observation, baseline=None):
    flag = ""
    if baseline is not None and observation["output"] != baseline["output"]:
        flag = "   <-- DIVERGED"
    print(
        f"  {label:<24} k={str(observation['searched_top_k']):<10} "
        f"cands={str(observation['candidates']):<6} "
        f"nret={observation['retrievals']} fail={observation['failed']} "
        f"sel={observation['selected']}"
    )
    print(f"  {'':<24} -> {observation['output']!r}{flag}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index-path", type=Path, required=True)
    parser.add_argument("--prompts", type=Path, required=True)
    parser.add_argument("--subject-key", required=True)
    parser.add_argument("--max-new-tokens", type=int, default=12)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument(
        "--k-grid",
        default="1,2,4,8,32,128",
        help="Inert exclusion-set sizes; search_k becomes 1 + each value.",
    )
    args = parser.parse_args()

    from halo.cli.runner import _load_examples
    from models.co_lmlm import MODEL, NPROBE, SIMILARITY_THRESHOLD, SOURCE_PATH
    from models.co_lmlm.backend import CoLMLMAuditBackend

    backend = CoLMLMAuditBackend.from_public_release(
        model_path=MODEL,
        index_path=args.index_path,
        db_path=args.index_path / "entries.db",
        source_path=SOURCE_PATH,
        similarity_threshold=SIMILARITY_THRESHOLD,
        nprobe=NPROBE,
        max_new_tokens=args.max_new_tokens,
    )

    examples = _load_examples(args.prompts, None)
    example = examples.get(args.subject_key)
    if example is None:
        raise SystemExit(
            f"No example keyed {args.subject_key!r} in {args.prompts} "
            f"({len(examples)} loaded)."
        )
    print(f"Subject {args.subject_key}\nPrompt: {example.prompt!r}\n")

    print("Arm A - determinism (same state, same manifest, repeated):")
    baseline = None
    for repeat in range(args.repeats):
        observation = _observe(
            backend, example, DatabaseState.FULL, args.max_new_tokens
        )
        _report(f"FULL #{repeat + 1}", observation, baseline)
        baseline = baseline or observation
    full_baseline = baseline

    print("\nArm B - search depth (inert exclusions, nothing is filtered):")
    for size in [int(token) for token in args.k_grid.split(",") if token.strip()]:
        manifest = DeletionManifest(
            entry_ids=tuple(f"__inert_no_such_entry_{n}__" for n in range(size)),
            strategy="diagnostic-inert",
        )
        subject = dataclasses.replace(example, deletion_manifest=manifest)
        observation = _observe(
            backend, subject, DatabaseState.DEL_ON, args.max_new_tokens
        )
        _report(f"DEL-ON |ids|={size}", observation, full_baseline)

    print(
        "\nArm A disagreement  => decoding is nondeterministic; the fingerprint "
        "hook's precondition fails.\n"
        "Arm B disagreement  => the index's top-1 depends on k; "
        "full_row_unaffected is unsound.\n"
        "Neither             => the divergence came from the exclusion set "
        "itself; re-check the manifest."
    )


if __name__ == "__main__":
    main()
