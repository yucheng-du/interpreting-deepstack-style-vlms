#!/usr/bin/env python3
"""Prepare pinned BLINK val parquets as a deterministic sample or full manifest.

This data preparation step does not load a model.
Original image bytes are preserved, with no JPEG re-encoding or resizing.
"""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import io
import json
import os
from pathlib import Path
import re
import sys
import time
import urllib.request

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from deepstack.data import (  # noqa: E402
    BLINK_REPO, BLINK_REVISION, BLINK_TASKS, IMAGE_FIELDS,
    OFFICIAL_EVAL_REVISION, canonical_answer, official_prompt, option_labels, sample_indices,
    load_manifest, validate_record, write_manifest,
)

DEFAULT_OUTPUT = Path(os.environ.get("DS_DATA_ROOT", "work")) / "data/blink"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def download(url: str, destination: Path, expected_sha256: str | None = None,
             expected_size: int | None = None, *, offline: bool = False) -> str:
    if destination.exists():
        if expected_size is not None and destination.stat().st_size != expected_size:
            raise ValueError(f"Existing file has wrong size: {destination}")
        current = sha256_file(destination)
        if expected_sha256 is None or current == expected_sha256:
            return current
        raise ValueError(f"Existing file has wrong hash; inspect before replacing: {destination}")
    if offline:
        raise FileNotFoundError(f"Offline preparation requires cached source: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".partial")
    if expected_size is None:
        with urllib.request.urlopen(url, timeout=60) as response, temporary.open("wb") as handle:
            while chunk := response.read(1024 * 1024):
                handle.write(chunk)
    else:
        # The network may close a long body early without raising an exception.
        # Bounded, checked byte ranges allow safe retries and preserve progress.
        failures = 0
        while (offset := temporary.stat().st_size if temporary.exists() else 0) < expected_size:
            end = min(offset + 8 * 1024 * 1024, expected_size) - 1
            request = urllib.request.Request(url + f"?download=true&range_start={offset}",
                                             headers={"Range": f"bytes={offset}-{end}"})
            try:
                with urllib.request.urlopen(request, timeout=60) as response:
                    content_range = response.headers.get("Content-Range", "")
                    match = re.fullmatch(r"bytes (\d+)-(\d+)/(\d+)", content_range)
                    if not match or tuple(map(int, match.groups())) != (offset, end, expected_size):
                        raise ValueError(f"Server did not honor requested range: {content_range!r}")
                    with temporary.open("ab") as handle:
                        remaining = end - offset + 1
                        while remaining:
                            chunk = response.read(min(1024 * 1024, remaining))
                            if not chunk:
                                raise OSError("Truncated range body")
                            handle.write(chunk)
                            remaining -= len(chunk)
                failures = 0
                print(f"  Downloaded {temporary.stat().st_size / 1e6:.1f}/{expected_size / 1e6:.1f} MB", flush=True)
            except (OSError, TimeoutError) as error:
                failures += 1
                if failures >= 4:
                    raise
                print(f"  Download interrupted; retrying saved offset: {error}", flush=True)
                time.sleep(2)
        if temporary.stat().st_size != expected_size:
            raise ValueError(f"Downloaded size mismatch: {destination}")
    checksum = sha256_file(temporary)
    if expected_sha256 is not None and checksum != expected_sha256:
        raise ValueError(f"Downloaded checksum mismatch: {destination}")
    temporary.replace(destination)
    return checksum


def partition_records(records, previous_records):
    """Keep exploratory membership explicit without changing any record fields."""
    full = {r["id"]: r for r in records}
    previous = {r["id"]: r for r in previous_records}
    if len(full) != len(records) or len(previous) != len(previous_records):
        raise ValueError("Duplicate ID in full or previous manifest")
    for item_id, record in previous.items():
        if item_id not in full:
            raise ValueError(f"Previous ID absent from full manifest: {item_id}")
        if full[item_id] != record:
            raise ValueError(f"Previous record changed in full manifest: {item_id}")
    return ([r for r in records if r["id"] in previous],
            [r for r in records if r["id"] not in previous])


def export_record(row, task, row_index, rank, image_root, revision, parquet_sha, github_answer):
    from PIL import Image
    labels = option_labels(row["choices"])
    answer = canonical_answer(row["answer"], labels)
    image_paths, metadata = [], []
    for field in IMAGE_FIELDS:
        value = row.get(field)
        if not value:
            continue
        content = value.get("bytes")
        if not content:
            raise ValueError(f"Expected embedded image bytes: {row['idx']} {field}")
        checksum = hashlib.sha256(content).hexdigest()
        with Image.open(io.BytesIO(content)) as image:
            image.load()
            suffix = {"JPEG": ".jpg", "PNG": ".png", "WEBP": ".webp"}.get(image.format)
            if suffix is None:
                raise ValueError(f"Unsupported source image format: {image.format}")
            dimensions = image.size
            mode, image_format = image.mode, image.format
        path = image_root / task / f"{row['idx']}_{field}{suffix}"
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists():
            if sha256_file(path) != checksum:
                raise ValueError(f"Existing extracted image changed: {path}")
        else:
            path.write_bytes(content)
        image_paths.append(str(path))
        metadata.append({"source_field": field, "source_path": value.get("path"),
                         "sha256": checksum, "bytes": len(content),
                         "width": dimensions[0], "height": dimensions[1],
                         "mode": mode, "format": image_format})
    record = {
        "schema_version": 1, "id": row["idx"], "task": task,
        "sub_task": row.get("sub_task"), "question": row["question"],
        "raw_prompt": row["prompt"], "prompt": official_prompt(task, row["prompt"]),
        "prompt_protocol": "blink_official_load_prompt_v1",
        "choices": row["choices"], "option_labels": labels,
        "prompt_choice_exact_matches": [f"{label} {choice}" in row["prompt"]
                                          for label, choice in zip(labels, row["choices"])],
        "raw_answer": row["answer"], "answer_label": answer, "answer_index": labels.index(answer),
        "github_answer_label": github_answer, "github_label_agrees": answer == github_answer,
        "images": image_paths, "image_metadata": metadata,
        "sample_rank": rank,
        "source": {"repo_id": BLINK_REPO, "revision": revision, "split": "val",
                   "label_source": "pinned_huggingface_row",
                   "row_index": row_index, "parquet_sha256": parquet_sha,
                   "official_eval_revision": OFFICIAL_EVAL_REVISION},
    }
    validate_record(record)
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--revision", default=BLINK_REVISION)
    parser.add_argument("--seed", type=int, default=20260926)
    parser.add_argument("--per-task", type=int, default=8)
    parser.add_argument("--all", action="store_true", help="Export every validation row in the same seeded order")
    parser.add_argument("--offline", action="store_true", help="Require cached sources; never make network requests")
    parser.add_argument("--prior-manifest", type=Path,
                        help="With --all, verify and separately export previously explored records and their complement")
    parser.add_argument("--tasks", nargs="+", choices=BLINK_TASKS, default=list(BLINK_TASKS))
    args = parser.parse_args()
    root = args.output_dir.expanduser().resolve()
    if len(args.revision) != 40 or any(x not in "0123456789abcdef" for x in args.revision):
        parser.error("Use an exact 40-character dataset revision")
    if args.prior_manifest and not args.all:
        parser.error("--prior-manifest requires --all")
    previous_records = load_manifest(args.prior_manifest) if args.prior_manifest else None
    prior_sha = sha256_file(args.prior_manifest) if args.prior_manifest else None
    import pyarrow.parquet as pq
    from PIL import __version__ as pillow_version
    import pyarrow
    source_root = root / "sources" / args.revision
    source_root.mkdir(parents=True, exist_ok=True)
    tree_url = f"https://huggingface.co/api/datasets/{BLINK_REPO}/tree/{args.revision}?recursive=true&expand=false"
    tree_file = source_root / "hf_tree.json"
    download(tree_url, tree_file, offline=args.offline)
    tree = json.loads(tree_file.read_text())
    files = {x["path"]: x for x in tree if x["type"] == "file"}
    official_root = root / "sources" / f"official_eval_{OFFICIAL_EVAL_REVISION}"
    for filename in ("test_benchmark.py", "val_answers.json"):
        download(f"https://raw.githubusercontent.com/zeyofu/BLINK_Benchmark/{OFFICIAL_EVAL_REVISION}/eval/{filename}",
                 official_root / filename, offline=args.offline)
    official_answers = json.loads((official_root / "val_answers.json").read_text())
    records, task_info = [], {}
    download_order = sorted(args.tasks, key=lambda task: files[f"{task}/val-00000-of-00001.parquet"]["size"])
    for task in download_order:
        relative = f"{task}/val-00000-of-00001.parquet"
        file_info = files[relative]
        parquet_path = source_root / relative
        print(f"Preparing {task}: {file_info['size'] / 1e6:.1f} MB", flush=True)
        checksum = download(f"https://huggingface.co/datasets/{BLINK_REPO}/resolve/{args.revision}/{relative}",
                            parquet_path, file_info["lfs"]["oid"], file_info["size"], offline=args.offline)
        # Read/validate the small text columns for every source row. Decode image
        # bytes only for selected rows (all rows with --all).
        text_table = pq.read_table(parquet_path, columns=["idx", "choices", "answer", "prompt"])
        ids, github_disagreement_ids, normalized_answer_ids = set(), [], []
        for row in text_table.to_pylist():
            if row["idx"] in ids:
                raise ValueError(f"Duplicate source ID: {row['idx']}")
            ids.add(row["idx"])
            labels = option_labels(row["choices"])
            answer = canonical_answer(row["answer"], labels)
            if answer != row["answer"]:
                normalized_answer_ids.append(row["idx"])
            if official_answers[row["idx"]] != answer:
                github_disagreement_ids.append(row["idx"])
        indices = sample_indices(len(text_table), len(text_table) if args.all else args.per_task, args.seed, task)
        table = pq.read_table(parquet_path)
        task_records = [export_record(row, task, index, rank, root / "images" / args.revision,
                                      args.revision, checksum, official_answers[row["idx"]])
                        for rank, (index, row) in enumerate(zip(indices, table.take(indices).to_pylist()))]
        del table
        records.extend(task_records)
        task_info[task] = {"available_val": len(text_table), "sampled": len(task_records),
                           "selected_ids": [r["id"] for r in task_records],
                           "image_count_distribution": dict(Counter(len(r["images"]) for r in task_records)),
                           "choice_count_distribution": dict(Counter(len(r["choices"]) for r in task_records)),
                           "source_bytes": file_info["size"], "parquet_sha256": checksum}
        task_info[task].update({"answer_format_normalized_count": len(normalized_answer_ids),
                               "github_disagreement_count": len(github_disagreement_ids),
                               "github_disagreement_ids": github_disagreement_ids})
        print(f"  Validated {len(text_table)} labels; GitHub disagreements={len(github_disagreement_ids)}; "
              f"exported {len(task_records)} items", flush=True)
    records.sort(key=lambda r: (args.tasks.index(r["task"]), r["sample_rank"]))
    suffix = "" if tuple(args.tasks) == BLINK_TASKS else "_" + "_".join(args.tasks)
    selection = "all" if args.all else f"n{args.per_task}"
    manifest = root / f"val_seed{args.seed}_{selection}{suffix}.jsonl"
    partitions = None
    if previous_records is not None:
        previous, new = partition_records(records, previous_records)
        partitions = {}
        for label, rows in (("exploratory", previous), ("new", new)):
            path = manifest.with_name(f"{manifest.stem}_{label}{len(rows)}.jsonl")
            write_manifest(path, rows)
            partitions[label] = {"manifest": str(path), "sha256": sha256_file(path), "count": len(rows)}
        if sha256_file(args.prior_manifest) != prior_sha:
            raise ValueError("Previous manifest changed during preparation")
    write_manifest(manifest, records)
    smoke = None if args.all else root / f"val_seed{args.seed}_smoke{suffix}.jsonl"
    if smoke is not None:
        write_manifest(smoke, [r for r in records if r["sample_rank"] == 0])
    provenance = {"created_at_utc": datetime.now(timezone.utc).isoformat(),
                  "repo_id": BLINK_REPO, "revision": args.revision,
                  "official_eval_revision": OFFICIAL_EVAL_REVISION,
                  "seed": args.seed, "sampling": "random.Random(f'{seed}:{task}').shuffle(row_indices); " + ("all rows" if args.all else "prefix n"),
                  "sampled_count": len(records), "available_val_count": sum(x["available_val"] for x in task_info.values()),
                  "manifest": str(manifest), "manifest_sha256": sha256_file(manifest),
                  "smoke_manifest": str(smoke) if smoke else None, "smoke_manifest_sha256": sha256_file(smoke) if smoke else None,
                  "offline": args.offline,
                  "prior_manifest": str(args.prior_manifest) if args.prior_manifest else None,
                  "prior_manifest_sha256": prior_sha, "partitions": partitions,
                  "image_bytes_preserved": True,
                  "label_check": "HF val labels checked against actual options; GitHub val_answers disagreement recorded, never overwritten",
                  "libraries": {"python": sys.version, "pyarrow": pyarrow.__version__, "pillow": pillow_version},
                  "tasks": task_info}
    provenance_path = manifest.with_suffix(".provenance.json")
    provenance_path.write_text(json.dumps(provenance, indent=2) + "\n")
    print(json.dumps({"manifest": str(manifest), "smoke": str(smoke) if smoke else None, "provenance": str(provenance_path),
                      "count": len(records)}, indent=2), flush=True)


if __name__ == "__main__":
    main()
