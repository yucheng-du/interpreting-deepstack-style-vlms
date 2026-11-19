#!/usr/bin/env python3
"""Summarize complete, paired eight-mask DeepStack pilot results (stdlib only).

Usage: python scripts/analyze_pilot.py results.jsonl --output-dir analysis
Multiple input files may be supplied for disjoint shards of the same run.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path
import statistics
from typing import Any


MASKS = ("111", "110", "101", "011", "100", "010", "001", "000")
BASELINE = "111"
REQUIRED = {
    "id", "task", "mask", "scores", "answer_label", "prediction",
    "correct", "margin", "elapsed_seconds",
}


def _finite_number(value: Any, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field} must be a finite number")
    if not math.isfinite(value):
        raise ValueError(f"{field} must be finite")
    return float(value)


def _reject_nonfinite(value: Any, field: str = "row") -> None:
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError(f"{field} contains a non-finite number")
    if isinstance(value, dict):
        for key, child in value.items():
            _reject_nonfinite(child, f"{field}.{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            _reject_nonfinite(child, f"{field}[{index}]")


def validate_row(row: Any) -> dict[str, Any]:
    if not isinstance(row, dict):
        raise ValueError("each JSONL record must be an object")
    _reject_nonfinite(row)
    missing = REQUIRED - row.keys()
    if missing:
        raise ValueError(f"missing required fields: {', '.join(sorted(missing))}")
    if isinstance(row["id"], bool) or not isinstance(row["id"], (str, int)):
        raise ValueError("id must be a nonempty string or an integer")
    if row["id"] == "":
        raise ValueError("id cannot be empty")
    for field in ("task", "answer_label", "prediction"):
        if not isinstance(row[field], str) or not row[field]:
            raise ValueError(f"{field} must be a nonempty string")
    if not isinstance(row["mask"], str) or row["mask"] not in MASKS:
        raise ValueError(f"mask must be one of {MASKS} (a three-character string)")
    scores = row["scores"]
    if not isinstance(scores, dict) or len(scores) < 2:
        raise ValueError("scores must contain at least two option labels")
    if not all(isinstance(label, str) and label for label in scores):
        raise ValueError("scores keys must be nonempty option labels")
    for label, score in scores.items():
        _finite_number(score, f"scores[{label!r}]")
    answer, prediction = row["answer_label"], row["prediction"]
    if answer not in scores or prediction not in scores:
        raise ValueError("answer_label and prediction must both appear in scores")
    if scores[prediction] != max(scores.values()):
        raise ValueError("prediction must be a maximum-scoring option (ties allowed)")
    if not isinstance(row["correct"], bool) or row["correct"] != (prediction == answer):
        raise ValueError("correct must be a boolean consistent with prediction and answer_label")
    margin = _finite_number(row["margin"], "margin")
    expected = scores[answer] - max(score for label, score in scores.items() if label != answer)
    if not math.isclose(margin, expected, rel_tol=1e-6, abs_tol=1e-6):
        raise ValueError(f"margin {margin} does not match score-derived margin {expected}")
    if _finite_number(row["elapsed_seconds"], "elapsed_seconds") < 0:
        raise ValueError("elapsed_seconds cannot be negative")
    return row


def load_rows(paths: list[Path]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    unique: dict[tuple[str, str, str], dict[str, Any]] = {}
    origins: dict[tuple[str, str, str], str] = {}
    total_rows = duplicate_rows = 0
    sources = []
    for path in paths:
        content = path.read_bytes()
        sources.append({"path": str(path.resolve()), "sha256": hashlib.sha256(content).hexdigest()})
        for line_number, line in enumerate(content.decode("utf-8").splitlines(), 1):
            if not line.strip():
                continue
            origin = f"{path}:{line_number}"
            try:
                row = validate_row(json.loads(line))
            except (ValueError, TypeError) as exc:
                raise ValueError(f"{origin}: {exc}") from exc
            total_rows += 1
            key = (row["task"], str(row["id"]), row["mask"])
            if key in unique:
                if row != unique[key]:
                    raise ValueError(f"{origin}: conflicting duplicate for {key}; first seen at {origins[key]}")
                duplicate_rows += 1
            else:
                unique[key] = row
                origins[key] = origin
    return list(unique.values()), {
        "sources": sources,
        "input_rows": total_rows,
        "identical_duplicate_rows_ignored": duplicate_rows,
        "unique_rows": len(unique),
    }


def mean_se(values: list[float]) -> dict[str, Any]:
    return {
        "n": len(values),
        "mean": statistics.fmean(values) if values else None,
        "se": statistics.stdev(values) / math.sqrt(len(values)) if len(values) > 1 else None,
    }


def top_score_tie(row: dict[str, Any]) -> dict[str, Any]:
    """Exact maximum-score ties in the recorded score-label order."""
    scores = row["scores"]
    highest = max(scores.values())
    labels = [label for label, score in scores.items() if score == highest]
    tied = len(labels) > 1
    return {"top_score_tied": tied, "tied_top_labels": labels if tied else []}


def minimal_preserving_masks(preserving: list[str]) -> list[str]:
    """Inclusion-minimal sets; preserve nonmonotonic behavior, including 000."""
    return [mask for mask in MASKS if mask in preserving and not any(
        other != mask and all(a == "0" or b == "1" for a, b in zip(other, mask))
        for other in preserving
    )]


def summarize_pairs(pairs: list[dict[str, dict[str, Any]]]) -> dict[str, Any]:
    conditions = {}
    n = len(pairs)
    for mask in MASKS:
        flips = sum(pair[mask]["prediction"] != pair[BASELINE]["prediction"] for pair in pairs)
        ties = sum(top_score_tie(pair[mask])["top_score_tied"] for pair in pairs)
        conditions[mask] = {
            "n": n,
            "accuracy": statistics.fmean(pair[mask]["correct"] for pair in pairs) if n else None,
            "mean_margin": statistics.fmean(pair[mask]["margin"] for pair in pairs) if n else None,
            "paired_margin_change_vs_111": mean_se([
                pair[mask]["margin"] - pair[BASELINE]["margin"] for pair in pairs
            ]),
            "accuracy_change_vs_111": statistics.fmean(
                int(pair[mask]["correct"]) - int(pair[BASELINE]["correct"]) for pair in pairs
            ) if n else None,
            "prediction_flips_vs_111": flips,
            "prediction_flip_rate_vs_111": flips / n if n else None,
            "top_score_tie_count": ties,
            "top_score_tie_rate": ties / n if n else None,
            "correct_to_incorrect": sum(pair[BASELINE]["correct"] and not pair[mask]["correct"] for pair in pairs),
            "incorrect_to_correct": sum(not pair[BASELINE]["correct"] and pair[mask]["correct"] for pair in pairs),
            "wrong_to_different_wrong": sum(
                not pair[BASELINE]["correct"] and not pair[mask]["correct"]
                and pair[mask]["prediction"] != pair[BASELINE]["prediction"] for pair in pairs
            ),
        }
    return {
        "n_items": n, "se_unit": "paired items", "conditions": conditions,
        "top_score_tie_count": sum(row["top_score_tie_count"] for row in conditions.values()),
        "n_items_with_top_score_tie": sum(
            any(top_score_tie(pair[mask])["top_score_tied"] for mask in MASKS) for pair in pairs
        ),
        "baseline_top_score_tie_count": conditions[BASELINE]["top_score_tie_count"],
    }


def analyze(rows: list[dict[str, Any]], provenance: dict[str, Any] | None = None) -> dict[str, Any]:
    # load_rows performs duplicate checks; this public entry point enforces them too.
    grouped: dict[tuple[str, str], dict[str, dict[str, Any]]] = defaultdict(dict)
    for source_row in rows:
        row = validate_row(source_row)
        key, mask = (row["task"], str(row["id"])), row["mask"]
        if mask in grouped[key] and grouped[key][mask] != row:
            raise ValueError(f"conflicting duplicate for {key + (mask,)}")
        grouped[key][mask] = row
    by_task: dict[str, list[dict[str, dict[str, Any]]]] = defaultdict(list)
    incomplete, items = [], []
    for (task, item_id), pair in sorted(grouped.items()):
        first = next(iter(pair.values()))
        for row in pair.values():
            if row["answer_label"] != first["answer_label"] or row["scores"].keys() != first["scores"].keys():
                raise ValueError(f"inconsistent answer label or option set within {(task, item_id)}")
        if set(pair) != set(MASKS):
            incomplete.append({"task": task, "id": item_id, "missing_masks": [mask for mask in MASKS if mask not in pair]})
            continue
        by_task[task].append(pair)
        baseline = pair[BASELINE]
        preserving = [mask for mask in MASKS if pair[mask]["prediction"] == baseline["prediction"]]
        minimal = minimal_preserving_masks(preserving)
        min_count = min(mask.count("1") for mask in preserving)
        conditions = {}
        for mask in MASKS:
            row = pair[mask]
            conditions[mask] = {
                "prediction": row["prediction"], "correct": row["correct"], "margin": row["margin"],
                "margin_change_vs_111": row["margin"] - baseline["margin"],
                "prediction_changed_vs_111": row["prediction"] != baseline["prediction"],
                "elapsed_seconds": row["elapsed_seconds"],
                **top_score_tie(row),
            }
        items.append({
            "task": task, "id": item_id, "answer_label": baseline["answer_label"],
            "baseline_prediction": baseline["prediction"], "baseline_correct": baseline["correct"],
            "baseline_top_score_tied": conditions[BASELINE]["top_score_tied"],
            "baseline_tied_top_labels": conditions[BASELINE]["tied_top_labels"],
            "preserving_masks": preserving, "inclusion_minimal_preserving_masks": minimal,
            "minimum_path_count": min_count,
            "minimum_cardinality_preserving_masks": [mask for mask in preserving if mask.count("1") == min_count],
            "conditions": conditions,
        })
    task_summaries = {task: summarize_pairs(pairs) for task, pairs in sorted(by_task.items())}
    all_pairs = [pair for pairs in by_task.values() for pair in pairs]
    equal_task_conditions = {}
    for mask in MASKS:
        task_conditions = [summary["conditions"][mask] for summary in task_summaries.values()]
        equal_task_conditions[mask] = {
            "n_tasks": len(task_conditions),
            "accuracy": statistics.fmean(row["accuracy"] for row in task_conditions) if task_conditions else None,
            "mean_margin": statistics.fmean(row["mean_margin"] for row in task_conditions) if task_conditions else None,
            "accuracy_change_vs_111": statistics.fmean(row["accuracy_change_vs_111"] for row in task_conditions) if task_conditions else None,
            "prediction_flip_rate_vs_111": statistics.fmean(row["prediction_flip_rate_vs_111"] for row in task_conditions) if task_conditions else None,
            "top_score_tie_count": sum(row["top_score_tie_count"] for row in task_conditions),
            "top_score_tie_rate": statistics.fmean(row["top_score_tie_rate"] for row in task_conditions) if task_conditions else None,
            "task_mean_paired_margin_change_vs_111": mean_se([
                row["paired_margin_change_vs_111"]["mean"] for row in task_conditions
            ]),
        }
    patterns = Counter(" + ".join(item["inclusion_minimal_preserving_masks"]) for item in items)
    return {
        "schema_version": 1,
        "provenance": provenance or {},
        "definitions": {
            "mask": "b0b1b2; 1 retains a supplementary path; 000 still retains the main visual input",
            "baseline": BASELINE,
            "margin": "score(answer_label) - max(score(other options)); summed label log probability",
            "margin_change": "intervention margin minus the same item's 111 margin; negative means a decrease",
            "pairing": "task + id; include only items with all eight masks",
            "prediction_preservation": "matches 111 prediction, whether that prediction is correct or incorrect",
            "minimal_sets": "all inclusion-minimal retaining masks; minimum-cardinality masks also reported; no monotonicity assumed",
            "standard_error": "sample standard deviation divided by sqrt(n); null for n < 2; descriptive only",
            "top_score_tie": "two or more exactly equal maximum scores; no approximate threshold; tied_top_labels follows recorded score-label order and is empty when there is no tie",
            "tie_breaking": "the current runner's argmax chooses the first label in option/score order among tied maxima; analysis preserves recorded predictions, margins, and all complete paired items",
            "tie_counts": "condition counts are raw counts, including in equal-task summaries; top_score_tie_rate follows the summary's weighting; block totals count item-mask records across all eight masks",
        },
        "counts": {
            "unique_items": len(grouped), "complete_items": len(items),
            "excluded_incomplete_items": len(incomplete), "represented_complete_tasks": len(by_task),
            "top_score_tied_conditions": sum(
                condition["top_score_tied"] for item in items for condition in item["conditions"].values()
            ),
            "complete_items_with_top_score_tie": sum(
                any(condition["top_score_tied"] for condition in item["conditions"].values()) for item in items
            ),
            "baseline_top_score_tied_items": sum(item["baseline_top_score_tied"] for item in items),
        },
        "excluded_incomplete_items": incomplete,
        "by_task": task_summaries,
        "overall_question_weighted": summarize_pairs(all_pairs),
        "overall_equal_task_weighted": {
            "n_items": len(items), "n_tasks": len(by_task),
            "se_unit": "task means (between-task variation, not item-level uncertainty)",
            "conditions": equal_task_conditions,
        },
        "minimal_preserving_set_patterns": dict(sorted(patterns.items())),
        "items": items,
        "interpretation_limits": [
            "Exploratory descriptive results; interpret task effects with their sample sizes and uncertainty, and distinguish discovery from new items.",
            "Accuracy non-additivity alone is not evidence of mechanistic synergy; inspect paired margins.",
            "Preserving a baseline prediction does not imply preserving a correct answer.",
            "A path binds encoder source, merger, and decoder injection; effects do not isolate encoder depth.",
            "Only represented complete items/tasks contribute; excluded items can change the evaluated sample.",
        ],
    }


def _number(value: float | None) -> str:
    return "—" if value is None else f"{value:.4f}"


def _md(value: Any) -> str:
    return str(value).replace("|", "\\|").replace("\n", " ")




def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, nargs="+", help="JSONL file(s) from one compatible run")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        rows, provenance = load_rows(args.input)
        summary = analyze(rows, provenance)
    except (OSError, UnicodeError, ValueError) as exc:
        parser.error(str(exc))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = args.output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(f"Complete paired items: {summary['counts']['complete_items']}; excluded incomplete: "
          f"{summary['counts']['excluded_incomplete_items']}")
    print(summary_path.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
