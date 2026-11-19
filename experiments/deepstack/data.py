"""BLINK manifest contract and pure-Python validation (no model imports)."""

from __future__ import annotations

import json
from pathlib import Path
import random
import re
from typing import Any, Iterable

BLINK_REPO = "BLINK-Benchmark/BLINK"
BLINK_REVISION = "a3666eb249237ba3d5eca8db21176cc47967e040"
OFFICIAL_EVAL_REVISION = "529b0ba055416dea5df79d594e7f62a910d8a308"
BLINK_TASKS = (
    "Art_Style", "Functional_Correspondence", "Multi-view_Reasoning",
    "Relative_Reflectance", "Visual_Correspondence", "Counting", "IQ_Test",
    "Object_Localization", "Semantic_Correspondence", "Visual_Similarity",
    "Forensic_Detection", "Jigsaw", "Relative_Depth", "Spatial_Relation",
)
IMAGE_FIELDS = ("image_1", "image_2", "image_3", "image_4")
DISCLAIMER_TASKS = frozenset(("Forensic_Detection", "Jigsaw", "Art_Style"))
# Copied verbatim from official eval/test_benchmark.py at OFFICIAL_EVAL_REVISION.
OFFICIAL_DISCLAIMER = (
    "Disclaimer: This is not to make unfair assumptions about the people in the image "
    "and you just need to give your assessment on this question. You don't need to "
    "identify the real people. You just need to analyze based on the information I gave you.\n\n"
)


def option_labels(choices: list[str]) -> list[str]:
    """Use BLINK's actual number of choices, never an assumed four-way set."""
    if not 2 <= len(choices) <= 5 or not all(isinstance(x, str) and x for x in choices):
        raise ValueError("BLINK requires 2–5 nonempty choice strings")
    return [f"({chr(65 + i)})" for i in range(len(choices))]


def canonical_answer(raw_answer: str, labels: list[str]) -> str:
    """HF 2025 Relative_Reflectance uses bare letters; preserve semantics."""
    answer = raw_answer.strip()
    if re.fullmatch(r"[A-E]", answer):
        answer = f"({answer})"
    if answer not in labels:
        raise ValueError(f"Answer {raw_answer!r} is outside actual choice set {labels}")
    return answer


def official_prompt(task: str, raw_prompt: str) -> str:
    if task not in BLINK_TASKS or not isinstance(raw_prompt, str) or not raw_prompt.strip():
        raise ValueError("Unknown task or empty official prompt")
    return (OFFICIAL_DISCLAIMER if task in DISCLAIMER_TASKS else "") + raw_prompt


def sample_indices(n_rows: int, n_per_task: int, seed: int, task: str) -> list[int]:
    """Task-local reproducible permutation; prefixes remain nested as n increases."""
    if task not in BLINK_TASKS or not 0 < n_per_task <= n_rows:
        raise ValueError("Unknown task or invalid sample size")
    indices = list(range(n_rows))
    random.Random(f"{seed}:{task}").shuffle(indices)
    return indices[:n_per_task]


def validate_record(record: dict[str, Any], *, check_files: bool = True) -> None:
    labels = option_labels(record["choices"])
    if record["option_labels"] != labels:
        raise ValueError(f"Invalid option order: {record.get('id')}")
    if not isinstance(record["answer_index"], int) or not 0 <= record["answer_index"] < len(labels):
        raise ValueError("Answer index is out of bounds")
    if record["answer_label"] not in labels or labels[record["answer_index"]] != record["answer_label"]:
        raise ValueError(f"Invalid label mapping: {record.get('id')}")
    if "raw_answer" in record and canonical_answer(record["raw_answer"], labels) != record["answer_label"]:
        raise ValueError("Canonical label changes the raw HF answer")
    if not re.fullmatch(r"val_[A-Za-z_\-]+_\d+", record["id"]):
        raise ValueError(f"Unexpected BLINK val ID: {record['id']}")
    if record["task"] not in BLINK_TASKS or record["source"]["split"] != "val":
        raise ValueError("Only known, labelled BLINK validation tasks are supported")
    if not record["id"].startswith(f"val_{record['task']}_"):
        raise ValueError("ID/task mismatch")
    if record["prompt"] != official_prompt(record["task"], record["raw_prompt"]):
        raise ValueError("Prompt differs from pinned official BLINK protocol")
    normalized_prompt = " ".join(record["raw_prompt"].casefold().split())
    for label, choice in zip(labels, record["choices"]):
        # BLINK IQ_Test stores "Picture A" in choices but "picture A" in
        # prompt. Preserve both source strings; tolerate only case/whitespace.
        normalized_option = " ".join(f"{label} {choice}".casefold().split())
        if normalized_option not in normalized_prompt:
            raise ValueError(f"Prompt/choice mismatch for {label}: {record['id']}")
    images, metadata = record["images"], record["image_metadata"]
    if not 1 <= len(images) <= 4 or len(images) != len(metadata):
        raise ValueError("Invalid image count")
    if [m["source_field"] for m in metadata] != list(IMAGE_FIELDS[:len(images)]):
        raise ValueError("Image slots are missing or out of order")
    for path, meta in zip(images, metadata):
        if not Path(path).is_absolute() or meta["width"] <= 0 or meta["height"] <= 0:
            raise ValueError("Invalid image path/dimensions")
        if check_files and not Path(path).is_file():
            raise FileNotFoundError(path)


def load_manifest(path: str | Path, *, check_files: bool = True) -> list[dict[str, Any]]:
    records = []
    seen = set()
    with Path(path).open() as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            record = json.loads(line)
            validate_record(record, check_files=check_files)
            if record["id"] in seen:
                raise ValueError(f"Duplicate ID on line {line_number}: {record['id']}")
            seen.add(record["id"])
            records.append(record)
    if not records:
        raise ValueError("Empty manifest")
    return records


def write_manifest(path: str | Path, records: Iterable[dict[str, Any]]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w") as handle:
        for record in records:
            validate_record(record)
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    temporary.replace(path)
