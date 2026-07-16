from __future__ import annotations

import argparse
import csv
import os
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

# CSV basenames each backend may emit; merged with a `backend` column so a
# rel-vs-Co-LMLM comparison is a single join.
COMPARABLE_CSVS = (
    "cross_state_metrics.csv",
    "per_state_metrics.csv",
    "entanglement_gaps.csv",
    "entanglement_curves.csv",
    "evasion.csv",
    "margins.csv",
)

# Evaluation flags forwarded verbatim to both backends.
_SHARED_VALUE_FLAGS = (
    "--max-new-tokens",
    "--limit",
    "--closure",
    "--closure-radius",
    "--closure-envelope-k",
    "--closure-max-size",
    "--radius-grid",
    "--neighbor-mode",
    "--neighbor-ball",
    "--neighbor-cap",
    "--adversarial-epsilons",
    "--adversarial-templates",
    "--adversarial-topology",
    "--adversarial-count",
    "--adversarial-seed",
)
_SHARED_STORE_TRUE_FLAGS = ("--adversarial",)


@dataclass(frozen=True)
class LegPlan:
    name: str
    command: list[str]
    env: dict[str, str]
    output_dir: Path


@dataclass
class LegResult:
    name: str
    returncode: int
    output_dir: Path
    skipped: bool = False


def _repo_src_dir() -> Path:
    return Path(__file__).resolve().parents[2]


def _leg_interpreter(python: str | None, env_dir: str | None) -> list[str]:
    if python:
        return [python, "-m", "lmlm_audit.cli.run_audit"]
    project = env_dir or "."
    return [
        "uv",
        "run",
        "--project",
        project,
        "python",
        "-m",
        "lmlm_audit.cli.run_audit",
    ]


def _forward_shared(args: argparse.Namespace) -> list[str]:
    forwarded: list[str] = []
    for flag in _SHARED_VALUE_FLAGS:
        attr = flag.lstrip("-").replace("-", "_")
        value = getattr(args, attr, None)
        if value is not None:
            forwarded += [flag, str(value)]
    for flag in _SHARED_STORE_TRUE_FLAGS:
        attr = flag.lstrip("-").replace("-", "_")
        if getattr(args, attr, False):
            forwarded.append(flag)
    if args.states:
        forwarded += ["--states", *args.states]
    return forwarded


def build_leg_plans(args: argparse.Namespace) -> list[LegPlan]:
    prompt_args = ["--prompt-files", *[str(path) for path in args.prompt_files]]
    shared = _forward_shared(args)
    src_dir = str(_repo_src_dir())
    plans: list[LegPlan] = []

    if not args.skip_rel:
        rel_out = args.output_dir / "rel"
        rel_cmd = _leg_interpreter(args.rel_python, args.rel_env) + [
            "--backend",
            "rel-lmlm",
            *prompt_args,
            "--output-dir",
            str(rel_out),
            "--model-name",
            args.rel_model_name,
            *shared,
        ]
        if args.rel_database_path is not None:
            rel_cmd += ["--database-path", str(args.rel_database_path)]
        rel_env = dict(os.environ)
        rel_env["PYTHONPATH"] = os.pathsep.join(
            [src_dir, rel_env.get("PYTHONPATH", "")]
        ).rstrip(os.pathsep)
        plans.append(LegPlan("rel-lmlm", rel_cmd, rel_env, rel_out))

    if not args.skip_colmlm:
        colmlm_out = args.output_dir / "colmlm"
        colmlm_cmd = _leg_interpreter(
            args.colmlm_python, args.colmlm_env
        ) + [
            "--backend",
            "colmlm",
            *prompt_args,
            "--output-dir",
            str(colmlm_out),
            *shared,
        ]
        for flag, value in (
            ("--colmlm-source-path", args.colmlm_source_path),
            ("--colmlm-model-path", args.colmlm_model_path),
            ("--index-path", args.index_path),
            ("--entries-db-path", args.entries_db_path),
            ("--device", args.device),
            ("--attn-implementation", args.attn_implementation),
            ("--nprobe", args.nprobe),
        ):
            if value is not None:
                colmlm_cmd += [flag, str(value)]
        if args.bootstrap_oracle_from_full:
            colmlm_cmd.append("--bootstrap-oracle-from-full")
        if args.faiss_mmap:
            colmlm_cmd.append("--faiss-mmap")
        # lmlm_audit is not installed in the Co-LMLM env; expose our src.
        colmlm_env = dict(os.environ)
        colmlm_env["PYTHONPATH"] = os.pathsep.join(
            [src_dir, colmlm_env.get("PYTHONPATH", "")]
        ).rstrip(os.pathsep)
        plans.append(
            LegPlan("colmlm", colmlm_cmd, colmlm_env, colmlm_out)
        )

    return plans


def _run_legs(plans: Sequence[LegPlan]) -> list[LegResult]:
    """Launch every leg concurrently and wait for all of them."""
    processes: list[tuple[LegPlan, subprocess.Popen]] = []
    for plan in plans:
        plan.output_dir.mkdir(parents=True, exist_ok=True)
        print(f"[{plan.name}] launching: {' '.join(plan.command)}")
        processes.append(
            (
                plan,
                subprocess.Popen(plan.command, env=plan.env),
            )
        )
    results: list[LegResult] = []
    for plan, process in processes:
        returncode = process.wait()
        status = "ok" if returncode == 0 else f"FAILED ({returncode})"
        print(f"[{plan.name}] finished: {status}")
        results.append(LegResult(plan.name, returncode, plan.output_dir))
    return results


def merge_comparison(
    legs: Sequence[LegResult],
    comparison_dir: Path,
) -> dict[str, Any]:
    comparison_dir.mkdir(parents=True, exist_ok=True)
    written: list[str] = []
    for basename in COMPARABLE_CSVS:
        merged_rows: list[dict[str, Any]] = []
        fieldnames: list[str] = []
        for leg in legs:
            for path in sorted(leg.output_dir.rglob(basename)):
                rel_path = path.relative_to(leg.output_dir)
                with path.open("r", encoding="utf-8", newline="") as handle:
                    for row in csv.DictReader(handle):
                        tagged = {
                            "backend": leg.name,
                            "source": str(rel_path),
                            **row,
                        }
                        for key in tagged:
                            if key not in fieldnames:
                                fieldnames.append(key)
                        merged_rows.append(tagged)
        if not merged_rows:
            continue
        out_path = comparison_dir / basename
        with out_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(merged_rows)
        written.append(basename)

    summary = _gap_summary(comparison_dir / "entanglement_gaps.csv")
    return {"written": written, "gap_summary": summary}


def _gap_summary(gaps_csv: Path) -> dict[str, dict[str, float]]:
    if not gaps_csv.is_file():
        return {}
    per_backend: dict[str, list[float]] = {}
    with gaps_csv.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            try:
                gap = float(row["gap"])
            except (KeyError, ValueError):
                continue
            per_backend.setdefault(row["backend"], []).append(gap)
    return {
        backend: {
            "facts": len(gaps),
            "gap_mean": sum(gaps) / len(gaps),
            "gap_min": min(gaps),
            "gap_max": max(gaps),
        }
        for backend, gaps in per_backend.items()
        if gaps
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run the rel-LMLM and Co-LMLM audits as one comparison. Each "
            "backend runs in its own environment (their `lmlm` packages "
            "cannot share a process); outputs are merged into a combined "
            "report."
        )
    )
    parser.add_argument(
        "--prompt-files", nargs="+", type=Path, required=True
    )
    parser.add_argument("--output-dir", type=Path, required=True)

    parser.add_argument(
        "--rel-env",
        default=".",
        help="uv project dir for the rel-LMLM environment.",
    )
    parser.add_argument(
        "--rel-python",
        default=None,
        help="Python executable for the rel leg (overrides --rel-env).",
    )
    parser.add_argument("--rel-database-path", type=Path, default=None)
    parser.add_argument(
        "--rel-model-name", default="kilian-group/LMLM-llama2-382M"
    )

    parser.add_argument(
        "--colmlm-env",
        default=None,
        help="uv project dir for the Co-LMLM environment.",
    )
    parser.add_argument(
        "--colmlm-python",
        default=None,
        help="Python executable for the Co-LMLM leg (overrides --colmlm-env).",
    )
    parser.add_argument("--colmlm-source-path", type=Path, default=None)
    parser.add_argument("--colmlm-model-path", type=Path, default=None)
    parser.add_argument("--index-path", type=Path, default=None)
    parser.add_argument("--entries-db-path", type=Path, default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--attn-implementation", default=None)
    parser.add_argument("--nprobe", type=int, default=None)
    parser.add_argument("--faiss-mmap", action="store_true")
    parser.add_argument(
        "--bootstrap-oracle-from-full", action="store_true"
    )

    parser.add_argument("--max-new-tokens", type=int, default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--closure", default=None)
    parser.add_argument("--closure-radius", default=None)
    parser.add_argument("--closure-envelope-k", default=None)
    parser.add_argument("--closure-max-size", default=None)
    parser.add_argument("--radius-grid", default=None)
    parser.add_argument("--neighbor-mode", default=None)
    parser.add_argument("--neighbor-ball", default=None)
    parser.add_argument("--neighbor-cap", default=None)
    parser.add_argument("--adversarial", action="store_true")
    parser.add_argument("--adversarial-epsilons", default=None)
    parser.add_argument("--adversarial-templates", default=None)
    parser.add_argument("--adversarial-topology", default=None)
    parser.add_argument("--adversarial-count", default=None)
    parser.add_argument("--adversarial-seed", default=None)
    parser.add_argument("--states", nargs="+", default=None)

    parser.add_argument("--skip-rel", action="store_true")
    parser.add_argument("--skip-colmlm", action="store_true")

    args = parser.parse_args(argv)
    if args.skip_rel and args.skip_colmlm:
        parser.error("Nothing to run: both legs are skipped.")
    if not args.skip_colmlm and args.colmlm_env is None and args.colmlm_python is None:
        parser.error(
            "The Co-LMLM leg needs --colmlm-env or --colmlm-python (its "
            "`lmlm` package must live in a separate environment). Pass "
            "--skip-colmlm to run only rel-LMLM."
        )
    return args


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    plans = build_leg_plans(args)
    results = _run_legs(plans)

    comparison_dir = args.output_dir / "comparison"
    report = merge_comparison(results, comparison_dir)

    print("\n=== Comparison ===")
    if report["written"]:
        print(
            f"Merged {', '.join(report['written'])} into {comparison_dir}"
        )
    else:
        print(f"No comparable CSVs were produced under {args.output_dir}.")
    for backend, stats in report["gap_summary"].items():
        print(
            f"  G(f) [{backend}]: mean {stats['gap_mean']:.3f}, "
            f"min {stats['gap_min']:.3f}, max {stats['gap_max']:.3f} "
            f"over {stats['facts']} facts"
        )

    failures = [leg.name for leg in results if leg.returncode != 0]
    if failures:
        print(f"\nWARNING: leg(s) failed: {', '.join(failures)}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
