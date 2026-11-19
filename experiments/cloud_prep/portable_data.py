#!/usr/bin/env python3
"""Portable BLINK manifests; standard library only, with no model imports.

Export changes only ``images`` paths. It keeps labels, prompts, image/source
hashes and every other record field. The default export references existing
images: pass their root to load/validate, or transfer that images/ tree beside
manifest.jsonl. ``--copy-images`` creates a self-contained directory explicitly.

Examples (run from the repository):
  python3 -B -m cloud_prep.portable_data validate /path/to/manifest.jsonl
  python3 -B -m cloud_prep.portable_data export /path/to/manifest.jsonl \
      --output-dir work/data/blink14
  python3 -B -m cloud_prep.portable_data validate /bundle/manifest.jsonl \
      --image-root /path/to/blink

Choose the destination and image location with --output-dir and --image-root.
"""

from __future__ import annotations

import argparse
from collections import Counter
import copy
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import shutil
import sys
from typing import Any

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from deepstack.data import validate_record

BUNDLE_SCHEMA = "deepstack_blink_portable_v1"


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json_hash(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                    separators=(",", ":")).encode()).hexdigest()


def records_identity(records: list[dict[str, Any]]) -> str:
    """Ordered content identity, independent only of image filesystem paths.

    The existing image_metadata hashes remain part of this identity. Validation
    separately checks that every image byte stream matches those hashes.
    """
    return _json_hash([{k: v for k, v in row.items() if k != "images"} for row in records])


def _guard_path(path: Path, *, write: bool = False) -> Path:
    return path.expanduser().resolve()


def _read_manifest(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"Invalid JSON on line {line_number}: {path}") from error
            if not isinstance(row, dict) or not isinstance(row.get("id"), str):
                raise ValueError(f"Invalid record on line {line_number}: {path}")
            if row["id"] in seen:
                raise ValueError(f"Duplicate ID on line {line_number}: {row['id']}")
            seen.add(row["id"])
            rows.append(row)
    if not rows:
        raise ValueError(f"Empty manifest: {path}")
    return rows


def _bundle_path(path: Path) -> Path:
    return path.with_suffix(".bundle.json")


def _check_bundle_identity(path: Path, rows: list[dict[str, Any]]) -> dict[str, Any] | None:
    sidecar = _bundle_path(path)
    if not sidecar.exists():
        return None
    info = json.loads(sidecar.read_text(encoding="utf-8"))
    if info.get("schema") != BUNDLE_SCHEMA:
        raise ValueError(f"Unknown bundle schema: {sidecar}")
    checks = {
        "manifest_sha256": sha256_file(path),
        "record_count": len(rows),
        "ordered_ids_sha256": _json_hash([r["id"] for r in rows]),
        "records_identity_sha256": records_identity(rows),
    }
    for key, actual in checks.items():
        if info.get(key) != actual:
            raise ValueError(f"Bundle {key} mismatch: {path}")
    source_copy = path.parent / info["source_manifest"]["file"]
    if sha256_file(source_copy) != info["source_manifest"]["sha256"]:
        raise ValueError("Saved source manifest SHA mismatch")
    source_rows = _read_manifest(source_copy)
    if records_identity(source_rows) != checks["records_identity_sha256"]:
        raise ValueError("Export changed content or omitted/reordered source rows")
    for attachment in info.get("source_attachments", []):
        if sha256_file(path.parent / attachment["file"]) != attachment["sha256"]:
            raise ValueError(f"Source attachment SHA mismatch: {attachment['file']}")
    return info


def load_records(manifest_path: str | Path, image_root: str | Path | None = None) -> list[dict[str, Any]]:
    """Return original record fields with absolute, existing image paths.

    Relative paths resolve under image_root, or beside the manifest if omitted.
    For an old absolute-path manifest, an explicit image_root relocates paths
    relative to its original manifest directory. Export with --source-root first
    if that old manifest lives separately from its images (e.g. run batches).

    This checks the existing BLINK record contract and bundle/source identity,
    but does not reread every image byte. Call validate_manifest before a run or
    after transfer for image SHA verification. There is no sample filtering.
    """
    path = _guard_path(Path(manifest_path))
    root = _guard_path(Path(image_root)) if image_root is not None else path.parent
    rows = _read_manifest(path)
    _check_bundle_identity(path, rows)
    output = []
    for row in rows:
        relocated = copy.deepcopy(row)
        images = []
        for value in row["images"]:
            original = Path(value)
            if original.is_absolute() and image_root is None:
                resolved = _guard_path(original)
            else:
                if original.is_absolute():
                    try:
                        original = original.relative_to(path.parent)
                    except ValueError as error:
                        raise ValueError("Absolute image is outside the source manifest directory; "
                                         "export first with --source-root") from error
                if ".." in original.parts:
                    raise ValueError(f"Image path escapes data root: {value}")
                resolved = _guard_path(root / original)
                if not resolved.is_relative_to(root):
                    raise ValueError(f"Image path escapes data root: {value}")
            images.append(str(resolved))
        relocated["images"] = images
        validate_record(relocated, check_files=True)
        output.append(relocated)
    return output


def validate_manifest(manifest_path: str | Path, image_root: str | Path | None = None,
                      *, expected_count: int | None = None,
                      expected_sha256: str | None = None) -> dict[str, Any]:
    """Validate all rows and image bytes; errors raise instead of dropping rows."""
    path = _guard_path(Path(manifest_path))
    manifest_sha = sha256_file(path)
    if expected_sha256 is not None and manifest_sha != expected_sha256:
        raise ValueError("Manifest SHA differs from requested identity")
    rows = load_records(path, image_root=image_root)
    if expected_count is not None and len(rows) != expected_count:
        raise ValueError(f"Expected {expected_count} rows, found {len(rows)}")
    # The saved provenance is an additional identity anchor when it describes
    # this exact source manifest. It is never rewritten to fit changed data.
    provenance = path.with_suffix(".provenance.json")
    provenance_checked = False
    if provenance.exists():
        info = json.loads(provenance.read_text(encoding="utf-8"))
        if Path(info.get("manifest", "")).name == path.name:
            if info.get("manifest_sha256") != manifest_sha:
                raise ValueError("Source provenance manifest SHA mismatch")
            if info.get("sampled_count") != len(rows):
                raise ValueError("Source provenance record count mismatch")
            provenance_checked = True
    image_cache: dict[str, tuple[int, str]] = {}
    for row in rows:
        for value, meta in zip(row["images"], row["image_metadata"]):
            expected = meta.get("sha256")
            if not isinstance(expected, str) or not re.fullmatch(r"[a-f0-9]{64}", expected):
                raise ValueError(f"Missing/invalid image SHA: {row['id']}")
            if value not in image_cache:
                image_cache[value] = (Path(value).stat().st_size, sha256_file(value))
            size, actual = image_cache[value]
            if meta.get("bytes") != size or expected != actual:
                raise ValueError(f"Image bytes/SHA mismatch: {value}")
    if sha256_file(path) != manifest_sha:
        raise ValueError("Manifest changed during validation")
    return {
        "status": "passed", "manifest": str(path), "manifest_sha256": manifest_sha,
        "record_count": len(rows), "ordered_ids_sha256": _json_hash([r["id"] for r in rows]),
        "records_identity_sha256": records_identity(rows),
        "image_references": sum(len(r["images"]) for r in rows),
        "unique_image_paths": len(image_cache), "image_bytes_checked": sum(v[0] for v in image_cache.values()),
        "tasks": dict(sorted(Counter(r["task"] for r in rows).items())),
        "dataset_revisions": sorted({r["source"]["revision"] for r in rows}),
        "source_provenance_checked": provenance_checked,
        "bundle_identity_checked": _bundle_path(path).exists(),
        "image_sha256_checked": True,
    }


def export_bundle(manifest_path: str | Path, output_dir: str | Path, *,
                  source_root: str | Path | None = None, copy_images: bool = False) -> dict[str, Any]:
    """Export all source rows, refusing to overwrite any existing destination."""
    source = _guard_path(Path(manifest_path))
    root = _guard_path(Path(source_root)) if source_root else source.parent
    destination = _guard_path(Path(output_dir), write=True)
    if destination.exists():
        raise FileExistsError(f"Use a new bundle directory: {destination}")
    raw_rows = _read_manifest(source)
    # Absolute original manifests do not need relocation; portable inputs do.
    relative_input = any(not Path(p).is_absolute() for r in raw_rows for p in r["images"])
    load_root = root if relative_input else None
    validation = validate_manifest(source, image_root=load_root)
    rows = load_records(source, image_root=load_root)
    portable = copy.deepcopy(rows)
    image_map: dict[str, str] = {}
    for row in portable:
        for i, value in enumerate(row["images"]):
            try:
                relative = Path(value).relative_to(root).as_posix()
            except ValueError as error:
                raise ValueError(f"Image is outside --source-root {root}: {value}") from error
            row["images"][i] = relative
            image_map[relative] = value
    # Complete validation and path planning before creating the output.
    destination.mkdir(parents=True, exist_ok=False)
    saved_source = destination / "source_manifest.jsonl"
    shutil.copyfile(source, saved_source)
    attachments = []
    for suffix in (".provenance.json", ".integrity.json"):
        attachment = source.with_suffix(suffix)
        if attachment.exists():
            target = destination / ("source" + suffix)
            shutil.copyfile(attachment, target)
            attachments.append({"file": target.name, "sha256": sha256_file(target),
                                "original_path": str(attachment)})
    if copy_images:
        for relative, value in image_map.items():
            target = destination / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(value, target)
    target_manifest = destination / "manifest.jsonl"
    with target_manifest.open("w", encoding="utf-8") as handle:
        for row in portable:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    info = {
        "schema": BUNDLE_SCHEMA, "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "manifest_sha256": sha256_file(target_manifest), "record_count": len(rows),
        "ordered_ids_sha256": validation["ordered_ids_sha256"],
        "records_identity_sha256": validation["records_identity_sha256"],
        "images_mode": "copied" if copy_images else "reference",
        "source_image_root": str(root), "relative_image_root": ".",
        "source_manifest": {"file": saved_source.name, "original_path": str(source),
                            "sha256": validation["manifest_sha256"]},
        "source_attachments": attachments,
        "source_validation": validation,
        "note": "Only images paths changed. Source labels/prompts/version are preserved; "
                "scientific run parameters remain separately adjustable.",
    }
    _bundle_path(target_manifest).write_text(json.dumps(info, indent=2) + "\n", encoding="utf-8")
    result = validate_manifest(target_manifest, image_root=None if copy_images else root,
                               expected_count=len(rows))
    if result["records_identity_sha256"] != validation["records_identity_sha256"]:
        raise ValueError("Export changed record content")
    result["images_mode"] = info["images_mode"]
    result["validation_image_root"] = str(destination if copy_images else root)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True)
    validate = commands.add_parser("validate", help="Check every record and image SHA")
    validate.add_argument("manifest", type=Path)
    validate.add_argument("--image-root", type=Path)
    validate.add_argument("--expected-count", type=int)
    validate.add_argument("--expected-sha256")
    export = commands.add_parser("export", help="Export portable manifest; image copying is opt-in")
    export.add_argument("manifest", type=Path)
    export.add_argument("--output-dir", type=Path, required=True)
    export.add_argument("--source-root", type=Path, help="Root relative image paths are taken from")
    export.add_argument("--copy-images", action="store_true")
    args = parser.parse_args()
    try:
        if args.command == "validate":
            result = validate_manifest(args.manifest, args.image_root,
                                       expected_count=args.expected_count, expected_sha256=args.expected_sha256)
        else:
            result = export_bundle(args.manifest, args.output_dir, source_root=args.source_root,
                                   copy_images=args.copy_images)
    except (OSError, ValueError, KeyError, TypeError, RuntimeError) as error:
        print(json.dumps({"status": "failed", "error": str(error)}), file=sys.stderr)
        return 1
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
