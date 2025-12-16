"""
Summarize OOD evaluation results produced by experiments/evaluate_ood.py.

This script scans a directory (default: ``experiments/evaluation_results``)
for ``full_results.json`` files and prints a compact table that includes
per-model ID accuracy, calibration error, and OOD metrics (AUROC, AUPR,
FPR@95, Cohen's d). It supports multiple evaluation runs stored either
directly in the root directory or in its subdirectories.

Usage examples:

    python experiments/summarize_ood_results.py
    python experiments/summarize_ood_results.py --results-root ./experiments/eval_runs
    python experiments/summarize_ood_results.py --dataset svhn --model edl --json
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


@dataclass
class OODRecord:
    """Flattened evaluation result for a single model/dataset pair."""

    run_name: str
    path: Path
    model: str
    dataset: str
    id_accuracy: Optional[float]
    ece: Optional[float]
    auroc: Optional[float]
    aupr: Optional[float]
    fpr95: Optional[float]
    cohens_d: Optional[float]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Summarize metrics emitted by evaluate_ood.py."
    )
    parser.add_argument(
        "--results-root",
        type=str,
        default="experiments/evaluation_results",
        help="Directory containing evaluation outputs (full_results.json, etc.).",
    )
    parser.add_argument(
        "--run-filter",
        type=str,
        default=None,
        help="Only include runs whose directory name contains this substring.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Optional model filter (case-insensitive).",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default=None,
        help="Optional dataset filter (e.g., svhn).",
    )
    parser.add_argument(
        "--sort-by",
        type=str,
        default="auroc",
        choices=["auroc", "aupr", "fpr95", "cohens_d", "id_accuracy", "ece"],
        help="Metric used for sorting rows (default: auroc).",
    )
    parser.add_argument(
        "--ascending",
        action="store_true",
        help="Sort in ascending order instead of descending.",
    )
    parser.add_argument(
        "--top",
        type=int,
        default=None,
        help="Show only the top-N rows after sorting.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output raw JSON instead of a formatted table.",
    )
    return parser.parse_args()


def find_runs(root: Path) -> List[Tuple[str, Path]]:
    """
    Returns (run_name, path) pairs for every directory that contains full_results.json.
    The root directory itself is included if it has the file directly.
    """
    runs: List[Tuple[str, Path]] = []
    root_full = root / "full_results.json"
    if root_full.exists():
        runs.append((root.name or "evaluation_results", root))

    for entry in sorted(root.iterdir()):
        if entry.is_dir() and (entry / "full_results.json").exists():
            runs.append((entry.name, entry))

    return runs


def load_full_results(path: Path) -> Optional[List[Dict[str, Any]]]:
    try:
        with (path / "full_results.json").open("r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        return None
    except json.JSONDecodeError as exc:
        print(f"Warning: Failed to parse {path / 'full_results.json'}: {exc}")
        return None


def flatten_results(run_name: str, run_path: Path, payload: List[Dict[str, Any]]) -> List[OODRecord]:
    records: List[OODRecord] = []
    for entry in payload:
        model = str(entry.get("model", "unknown"))
        id_accuracy = entry.get("id_accuracy")
        ece = entry.get("ece")
        ood_results = entry.get("ood", {})

        for dataset, metrics in ood_results.items():
            records.append(
                OODRecord(
                    run_name=run_name,
                    path=run_path,
                    model=model,
                    dataset=dataset,
                    id_accuracy=float(id_accuracy) if id_accuracy is not None else None,
                    ece=float(ece) if ece is not None else None,
                    auroc=_safe_float(metrics.get("auroc")),
                    aupr=_safe_float(metrics.get("aupr")),
                    fpr95=_safe_float(metrics.get("fpr95")),
                    cohens_d=_safe_float(metrics.get("cohens_d")),
                )
            )
    return records


def _safe_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def metric_value(record: OODRecord, key: str, ascending: bool) -> float:
    value = getattr(record, key, None)
    if value is None:
        return float("inf") if ascending else float("-inf")
    return float(value)


def format_float(value: Optional[float], digits: int = 4) -> str:
    if value is None:
        return "-"
    return f"{value:.{digits}f}"


def print_table(records: List[OODRecord]) -> None:
    headers = [
        "Run",
        "Model",
        "Dataset",
        "AUROC",
        "AUPR",
        "FPR95",
        "Cohen's d",
        "ID Acc",
        "ECE",
    ]

    rows = [
        [
            rec.run_name,
            rec.model,
            rec.dataset,
            format_float(rec.auroc),
            format_float(rec.aupr),
            format_float(rec.fpr95),
            format_float(rec.cohens_d),
            format_float(rec.id_accuracy, digits=2),
            format_float(rec.ece, digits=4),
        ]
        for rec in records
    ]

    widths = [
        max(len(header), *(len(row[idx]) for row in rows) if rows else [len(header)])
        for idx, header in enumerate(headers)
    ]

    def print_row(values: List[str]) -> None:
        print(" | ".join(value.ljust(widths[i]) for i, value in enumerate(values)))

    print_row(headers)
    print("-+-".join("-" * width for width in widths))
    for row in rows:
        print_row(row)


def main() -> None:
    args = parse_args()
    root = Path(args.results_root).expanduser()
    if not root.exists():
        raise SystemExit(f"Results directory not found: {root}")

    run_entries = find_runs(root)
    if args.run_filter:
        run_entries = [r for r in run_entries if args.run_filter in r[0]]

    if not run_entries:
        print("No evaluation runs found.")
        return

    all_records: List[OODRecord] = []
    for run_name, run_path in run_entries:
        payload = load_full_results(run_path)
        if not payload:
            continue
        all_records.extend(flatten_results(run_name, run_path, payload))

    if not all_records:
        print("No valid OOD metrics found.")
        return

    if args.model:
        model_lower = args.model.lower()
        all_records = [r for r in all_records if r.model.lower() == model_lower]
    if args.dataset:
        dataset_lower = args.dataset.lower()
        all_records = [r for r in all_records if r.dataset.lower() == dataset_lower]

    if not all_records:
        print("No records matched the provided filters.")
        return

    sort_key = args.sort_by
    all_records.sort(
        key=lambda rec: metric_value(rec, sort_key, args.ascending),
        reverse=not args.ascending,
    )

    if args.top is not None and args.top > 0:
        all_records = all_records[: args.top]

    if args.json:
        payload = [
            {
                "run": rec.run_name,
                "path": str(rec.path),
                "model": rec.model,
                "dataset": rec.dataset,
                "auroc": rec.auroc,
                "aupr": rec.aupr,
                "fpr95": rec.fpr95,
                "cohens_d": rec.cohens_d,
                "id_accuracy": rec.id_accuracy,
                "ece": rec.ece,
            }
            for rec in all_records
        ]
        print(json.dumps(payload, indent=2))
    else:
        print(
            f"Found {len(all_records)} evaluation entries from {len(run_entries)} run(s) under {root}.\n"
        )
        print_table(all_records)


if __name__ == "__main__":
    main()
