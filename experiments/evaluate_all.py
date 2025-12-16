"""Utility for summarizing validation metrics from all saved experiments.

이 스크립트는 train.py로 학습한 각 실험 디렉터리(예: ``experiments/checkpoints``
하위)에 저장된 ``config.json``과 ``history.json``을 읽어서 검증 정확도,
불확실성 등의 핵심 지표를 한 번에 볼 수 있도록 표 형태로 출력합니다.

사용 예시:

    python experiments/evaluate_all.py \
        --checkpoints_dir experiments/checkpoints --model edl

``--json`` 옵션을 사용하면 요약 정보를 JSON으로 바로 확인할 수 있습니다.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional


@dataclass
class ExperimentSummary:
    """간단한 실험 요약."""

    name: str
    model: str
    epochs: int
    best_val_acc: Optional[float]
    best_epoch: Optional[int]
    final_val_acc: Optional[float]
    final_train_acc: Optional[float]
    final_val_uncertainty: Optional[float]
    final_val_confidence: Optional[float]
    config: Dict[str, Any]

    def hparam_note(self) -> str:
        model = (self.model or "").lower()
        if model == "edl":
            kl = self.config.get("edl_lambda")
            return f"λ_KL={kl}" if kl is not None else "λ_KL=?"
        if model in {"mbpe", "mbee"}:
            ncl = self.config.get("lambda_ncl")
            or_ = self.config.get("lambda_or")
            fdl = self.config.get("lambda_fdl")
            branches = self.config.get("num_branches")
            parts = [
                f"λ_NCL={ncl}" if ncl is not None else None,
                f"λ_OR={or_}" if or_ is not None else None,
                f"λ_FDL={fdl}" if fdl is not None else None,
                f"branches={branches}" if branches is not None else None,
            ]
            return ", ".join([p for p in parts if p]) or "(no hyper info)"
        return ""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Summarize validation results stored under checkpoints directories."
    )
    parser.add_argument(
        "--checkpoints_dir",
        type=str,
        default="experiments/checkpoints",
        help="Path to the root directory that contains experiment subdirectories.",
    )
    parser.add_argument(
        "--model",
        type=str,
        choices=["mbpe", "edl"],
        default=None,
        help="Optional filter to show only a specific model type.",
    )
    parser.add_argument(
        "--name-contains",
        type=str,
        default=None,
        help="Filter experiments whose directory name contains this substring.",
    )
    parser.add_argument(
        "--sort-by",
        type=str,
        default="best_val_acc",
        choices=[
            "best_val_acc",
            "final_val_acc",
            "final_train_acc",
            "final_val_uncertainty",
            "final_val_confidence",
        ],
        help="Metric used to sort experiments (defaults to best validation accuracy).",
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
        help="Show only the top-N entries after sorting.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print the raw summaries as JSON instead of a table.",
    )
    return parser.parse_args()


def load_json(path: Path) -> Any:
    try:
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        return None
    except json.JSONDecodeError:
        print(f"Warning: Failed to parse JSON file {path}. Skipping.")
        return None


def summarize_experiment(exp_dir: Path) -> Optional[ExperimentSummary]:
    history_path = exp_dir / "history.json"
    config_path = exp_dir / "config.json"

    if not history_path.exists():
        return None

    history = load_json(history_path)
    if not history:
        return None

    config = load_json(config_path) or {}

    def safe_metric(entry: Dict[str, Any], key: str) -> Optional[float]:
        value = entry.get(key)
        return float(value) if value is not None else None

    best_entry = max(history, key=lambda x: x.get("val_acc", float("-inf")))
    final_entry = history[-1]

    return ExperimentSummary(
        name=exp_dir.name,
        model=str(config.get("model", "unknown")),
        epochs=int(config.get("epochs", len(history))),
        best_val_acc=safe_metric(best_entry, "val_acc"),
        best_epoch=best_entry.get("epoch"),
        final_val_acc=safe_metric(final_entry, "val_acc"),
        final_train_acc=safe_metric(final_entry, "train_acc"),
        final_val_uncertainty=safe_metric(final_entry, "val_uncertainty"),
        final_val_confidence=safe_metric(final_entry, "val_confidence"),
        config=config,
    )


def find_experiments(base_dir: Path) -> List[ExperimentSummary]:
    summaries: List[ExperimentSummary] = []
    for path in sorted(base_dir.iterdir()):
        if not path.is_dir():
            continue
        summary = summarize_experiment(path)
        if summary is not None:
            summaries.append(summary)
    return summaries


def metric_value(summary: ExperimentSummary, sort_key: str, ascending: bool) -> float:
    value = getattr(summary, sort_key, None)
    if value is None:
        return float("inf") if ascending else float("-inf")
    return float(value)


def format_float(value: Optional[float], digits: int = 2) -> str:
    if value is None:
        return "-"
    return f"{value:.{digits}f}"


def print_table(summaries: List[ExperimentSummary]) -> None:
    headers = [
        "Experiment",
        "Model",
        "Epochs",
        "Best Val",
        "Best@",
        "Final Val",
        "Train Acc",
        "Val Unc",
        "Val Conf",
        "Notes",
    ]

    rows = []
    for s in summaries:
        rows.append(
            [
                s.name,
                s.model,
                str(s.epochs),
                format_float(s.best_val_acc),
                str(s.best_epoch or "-"),
                format_float(s.final_val_acc),
                format_float(s.final_train_acc),
                format_float(s.final_val_uncertainty, digits=4),
                format_float(s.final_val_confidence, digits=4),
                s.hparam_note(),
            ]
        )

    widths = [max(len(header), *(len(row[i]) for row in rows) if rows else [len(header)]) for i, header in enumerate(headers)]

    def print_row(values: List[str]) -> None:
        print(" | ".join(value.ljust(widths[idx]) for idx, value in enumerate(values)))

    print_row(headers)
    print("-+-".join("-" * width for width in widths))
    for row in rows:
        print_row(row)


def main() -> None:
    args = parse_args()
    checkpoints_dir = Path(args.checkpoints_dir).expanduser()
    if not checkpoints_dir.exists():
        raise SystemExit(f"Checkpoint directory not found: {checkpoints_dir}")

    summaries = find_experiments(checkpoints_dir)

    if args.model:
        summaries = [s for s in summaries if s.model.lower() == args.model]
    if args.name_contains:
        summaries = [s for s in summaries if args.name_contains in s.name]

    if not summaries:
        print("No experiments found with the provided filters.")
        return

    sort_key = args.sort_by
    sort_attr = sort_key
    summaries.sort(
        key=lambda s: metric_value(s, sort_attr, args.ascending),
        reverse=not args.ascending,
    )

    if args.top is not None and args.top > 0:
        summaries = summaries[: args.top]

    if args.json:
        json_payload = [
            {
                "name": s.name,
                "model": s.model,
                "epochs": s.epochs,
                "best_val_acc": s.best_val_acc,
                "best_epoch": s.best_epoch,
                "final_val_acc": s.final_val_acc,
                "final_train_acc": s.final_train_acc,
                "final_val_uncertainty": s.final_val_uncertainty,
                "final_val_confidence": s.final_val_confidence,
                "hyperparams": s.hparam_note(),
            }
            for s in summaries
        ]
        print(json.dumps(json_payload, indent=2))
    else:
        print(f"Summaries extracted from {checkpoints_dir} (n={len(summaries)}):\n")
        print_table(summaries)


if __name__ == "__main__":
    main()
