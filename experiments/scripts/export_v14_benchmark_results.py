#!/usr/bin/env python3
"""Copy the twelve accepted v14 benchmark analyses into plot/table-ready JSON.

This exporter does not score strings, reconstruct observations, estimate effects,
or repeat archive verification. It reads the accepted analysis JSON members and
copies their estimates, named sensitivities, denominators and diagnostics. The
recorded archive hashes are provenance from existing manifests, not a new audit.
"""
from __future__ import annotations

import argparse
import copy
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tarfile
import tempfile


MODELS = ("deepstack7b", "ibm4b", "qwen8b", "qwen32b")
DATASETS = {
    "gqa_attribute": "gqa-attribute-lc",
    "gqa_spatial": "gqa-spatial-lc",
    "docvqa_document_qa": "docvqa",
}
CELLS = ("clean__clean", "clean__degraded", "degraded__clean", "degraded__degraded")
CONTRASTS = (
    "later_recovery_with_main_degraded",
    "later_effect_with_main_clean",
    "main_quality_by_later_quality_interaction",
    "main_recovery_with_later_degraded",
    "joint_degradation_loss",
)
DEFAULT_RUN = (Path(os.environ.get('DS_DATA_ROOT', 'work')) / 'runs/measurement-v14-cloud-20261028')
DEFAULT_OUTPUT = (Path(os.environ.get('DS_DATA_ROOT', 'work')) / 'analysis/v14-integration-20261029')


def json_member(pack: Path, name: str):
    """Read one small JSON member without extracting the archive to disk."""
    found = None
    with subprocess.Popen(["zstd", "-dc", str(pack)], stdout=subprocess.PIPE,
                          stderr=subprocess.PIPE) as process:
        assert process.stdout is not None
        with tarfile.open(fileobj=process.stdout, mode="r|") as archive:
            for member in archive:
                if member.name == name:
                    if found is not None or not member.isfile():
                        raise ValueError(f"Non-unique regular JSON member: {pack}:{name}")
                    stream = archive.extractfile(member)
                    assert stream is not None
                    found = json.load(stream)
        process.stdout.close()
        stderr = process.stderr.read() if process.stderr is not None else b""
        if process.wait():
            raise RuntimeError(f"Cannot read {pack}: {stderr.decode(errors='replace')}")
    if found is None:
        raise ValueError(f"Missing JSON member: {pack}:{name}")
    return found


def recorded_source(packs: Path, stem: str, member: str):
    """Bind a member to the existing manifest; do not rehash accepted packs."""
    manifest_path = packs / f"{stem}.manifest.json"
    manifest = json.loads(manifest_path.read_text())
    entries = [entry for entry in manifest["files"] if entry["path"] == member]
    if len(entries) != 1:
        raise ValueError(f"Expected one manifest entry: {manifest_path}:{member}")
    entry = entries[0]
    return {
        "pack": str(packs / f"{stem}.tar.zst"),
        "member": member,
        "manifest": str(manifest_path),
        "recorded_pack_sha256": manifest.get("pack_sha256"),
        "recorded_member_sha256": entry["sha256"],
        "recorded_member_bytes": entry["bytes"],
    }


def runtime_metadata(packs: Path, job: str):
    stem = f"{job}.attempt-2"
    manifest = json.loads((packs / f"{stem}.manifest.json").read_text())
    member = manifest["plan"]["path"]
    source = recorded_source(packs, stem, member)
    run = json_member(Path(source["pack"]), member)
    generation = run.get("generation_config", {})
    return {
        "source": source,
        "model": run["model"],
        "config": run["config"],
        "query_count": run["query_count"],
        "units": run["units"],
        "status": run["status"],
        "manifest_sha256": run["manifest_sha256"],
        "actual_image_processor": run.get("actual_image_processor"),
        "generation": {key: generation.get(key) for key in (
            "max_new_tokens", "do_sample", "num_beams", "use_cache",
            "eos_token_id", "pad_token_id")},
        "code_provenance": run.get("code_provenance"),
        "libraries": run.get("libraries"),
    }


def atomic_json(path: Path, value):
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w") as handle:
            json.dump(value, handle, indent=2, ensure_ascii=False, allow_nan=False)
            handle.write("\n")
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def export(run_root: Path, output: Path):
    packs = run_root / "packs"
    output.mkdir(parents=True, exist_ok=True)
    datasets = {}
    material_sources = []
    for task, material in DATASETS.items():
        stem = f"materials-{material}.attempt-1"
        member = "attempt-1/materials/import_metadata.json"
        source = recorded_source(packs, stem, member)
        metadata = json_member(Path(source["pack"]), member)
        # Per-unit selection remains in the accepted material archive. Retain
        # counts, dataset/version/split, seed, hashes and degradation settings.
        datasets[task] = {key: value for key, value in metadata.items() if key != "selection"}
        datasets[task]["source"] = source
        material_sources.append(source)

    reports, concise_rows, analysis_sources, runtime_sources = [], [], [], []
    comparisons = 0
    metric_sources = None
    implementation_note = None
    sensitivity_note = None
    for model in MODELS:
        for task, material in DATASETS.items():
            job = f"{model}-{material}-recovery-batched"
            stem = f"analysis-{job}.run-1"
            member = f"run-1/{job}.json"
            source = recorded_source(packs, stem, member)
            original = json_member(Path(source["pack"]), member)
            if len(original["reports"]) != 1:
                raise ValueError(f"Expected one report in {stem}")
            report = original["reports"][0]
            if (report["model"], report["task"]) != (model, task):
                raise ValueError(f"Unexpected report identity in {stem}")
            if report["execution_version"] != "v14-benchmark-recovery-batched-1":
                raise ValueError(f"Unexpected execution version in {stem}")
            if report["primary_policy"] != "invalid_as_zero":
                raise ValueError(f"Unexpected primary policy in {stem}")
            if original["provenance"]["manifest_sha256"] != datasets[task]["manifest_sha256"]:
                raise ValueError(f"Dataset identity differs in {stem}")
            if metric_sources is None:
                metric_sources = original["metric_sources"]
                implementation_note = original.get("implementation_note")
                sensitivity_note = original.get("sensitivity_note")
            elif metric_sources != original["metric_sources"]:
                raise ValueError("Mixed metric source versions")

            item = {key: copy.deepcopy(value) for key, value in report.items() if key != "policies"}
            item.update(job=job, dataset_key=task, source=source,
                        source_report_pointer="/reports/0",
                        analysis_provenance=copy.deepcopy(original["provenance"]))
            item["runtime"] = runtime_metadata(packs, job)
            if item["runtime"]["manifest_sha256"] != datasets[task]["manifest_sha256"]:
                raise ValueError(f"Runtime material identity differs in {job}")
            item["policies"] = {}
            for policy_name, policy in report["policies"].items():
                copied_policy = {key: copy.deepcopy(value) for key, value in policy.items() if key != "metrics"}
                copied_policy["metrics"] = {}
                for metric, metric_report in policy["metrics"].items():
                    copied_metric = {key: copy.deepcopy(value) for key, value in metric_report.items()
                                     if key not in ("unit_values", "query_values")}
                    estimates = copied_metric["estimates"]
                    if not set(CELLS + CONTRASTS).issubset(estimates):
                        raise ValueError(f"Missing saved estimands in {job}/{policy_name}/{metric}")
                    for name, estimate in estimates.items():
                        # Literal equality with the source, including intervals,
                        # n, nulls and sensitivities; no recomputation/clipping.
                        assert estimate == metric_report["estimates"][name]
                        comparisons += 1
                    copied_policy["metrics"][metric] = copied_metric
                    concise_rows.append({
                        "job": job, "model": model, "task": task,
                        "benchmark": report["benchmark"], "policy": policy_name,
                        "metric": metric,
                        "is_primary": policy_name == report["primary_policy"] and metric == report["primary_metric"],
                        "queries": policy["queries"], "clusters": policy["clusters"],
                        "native": copy.deepcopy(estimates["clean__clean"]),
                        "joint_degraded": copy.deepcopy(estimates["degraded__degraded"]),
                        **{name: copy.deepcopy(estimates[name]) for name in CONTRASTS},
                        "source_pointer": f"/reports/0/policies/{policy_name}/metrics/{metric}/estimates",
                        "analysis_pack": source["pack"], "analysis_member": member,
                    })
                item["policies"][policy_name] = copied_policy
            reports.append(item)
            analysis_sources.append({**source, "original_provenance": original["provenance"]})
            runtime_sources.append(item["runtime"]["source"])

    assert len(reports) == 12 and len({(r["model"], r["task"]) for r in reports}) == 12
    created = datetime.now(timezone.utc).isoformat()
    numbers = {
        "schema": "deepstack-v14-benchmark-export-v1", "created_at": created,
        "operation": "Copy accepted estimates and metadata; no rescoring or re-estimation",
        "cell_order": list(CELLS),
        "cell_semantics": "First quality is main, second is later; clean-reference task scores",
        "metric_sources": metric_sources, "implementation_note": implementation_note,
        "sensitivity_note": sensitivity_note, "datasets": datasets,
        "reports": reports, "concise_rows": concise_rows,
    }
    provenance = {
        "schema": "deepstack-v14-benchmark-export-provenance-v1", "created_at": created,
        "script": str(Path(__file__).resolve()),
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "run_root": str(run_root.resolve()),
        "operation": "Read small JSON members and copy their existing estimates; not a new archive audit",
        "analysis_sources": analysis_sources, "material_sources": material_sources,
        "runtime_sources": runtime_sources,
        "omitted_bulk": "Per-query/per-cluster value arrays and full selected-ID lists remain in source packs",
        "copy_check": {"reports": len(reports), "saved_estimate_objects_compared": comparisons,
                       "all_equal_to_source": True, "concise_rows": len(concise_rows)},
        "outputs": [str(output / "benchmark_numbers.json"), str(output / "benchmark_provenance.json")],
    }
    atomic_json(output / "benchmark_numbers.json", numbers)
    atomic_json(output / "benchmark_provenance.json", provenance)
    # Check the written artifact's key counts and exact policy objects once.
    written = json.loads((output / "benchmark_numbers.json").read_text())
    assert written == numbers
    print(json.dumps(provenance["copy_check"], ensure_ascii=False))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, default=DEFAULT_RUN)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    export(args.run_root, args.output)


if __name__ == "__main__":
    main()
