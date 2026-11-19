#!/usr/bin/env python3
"""Exploratory post-hoc referent alignment of completed v10 full-source scores.

Standard library only. Reads exported effects and plans/metadata, never raw model
outputs. Source layouts determine referents before looking at effects. Directions
and the original physical-source matched role contrast are never changed.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
import math
from pathlib import Path
import os as _path_os
DATA_ROOT = Path(_path_os.environ.get("DS_DATA_ROOT", "work")).expanduser().resolve()
import statistics
import sys
from collections import Counter, defaultdict

MODELS = ("ibm4b", "qwen8b", "qwen32b")
DEFAULT_WORK = (DATA_ROOT / 'runs/measurement-v10-cloud-20261014/remote/work')
DEFAULT_OUT = (DATA_ROOT / 'analysis/v10-integration-20261016/referents')
COORDS = ("changing_layout", "fixed_layout", "changing_strength", "fixed_strength", "fixed_other_assignment")
READOUTS = (("direct", "none"), ("label", "identity"), ("label", "cyclic"))
GRID = tuple(itertools.product(("M0", "M1"), ("M0", "M1"), (.08, 1.), (.08, 1.), ("a", "b")))
QUERY_FIELDS = ("id", "scene_id", "intervention_id", "changed_object", "queried_object", "query_type", "query_target", "direction_a_index", "direction_b_index", "referred_object_by_source", "visual_source_signature")


def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def read_json(path):
    return json.loads(path.read_text())


def save_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def save_csv(path, rows):
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def finite(value):
    return value is not None and math.isfinite(value)


def subtract(a, b):
    return a - b if finite(a) and finite(b) else None


def stats(values):
    if not values:
        return dict(n=0, mean=None, se=None, low=None, high=None)
    mean = statistics.fmean(values)
    se = statistics.stdev(values) / math.sqrt(len(values)) if len(values) > 1 else None
    return dict(n=len(values), mean=mean, se=se, low=mean - 1.96 * se if se is not None else None,
                high=mean + 1.96 * se if se is not None else None)


def views(row):
    return ("direct",) if row["response_format"] == "direct" else ("label_pooled", "label_" + row["mapping"])


def summarize(rows, group_fields, measures, scene_ids):
    """Complete scene means; two changed objects, plus both mappings for pooled label."""
    grouped = defaultdict(lambda: defaultdict(list))
    for row in rows:
        for view in views(row):
            group = tuple(row[f] for f in group_fields) + (view,)
            grouped[group][row["scene_id"]].append(row)
    result = []
    for group, scenes in sorted(grouped.items(), key=lambda pair: str(pair[0])):
        view = group[-1]
        expected_repeats = 4 if view == "label_pooled" else 2
        assert set(scenes) == scene_ids, (group, "planned scene coverage")
        assert all(len(repeats) == expected_repeats for repeats in scenes.values()), (group, "repeat coverage")
        for measure in measures:
            complete = [statistics.fmean(r[measure] for r in repeats) for repeats in scenes.values()
                        if all(finite(r[measure]) for r in repeats)]
            result.append(dict(zip(group_fields, group[:-1])) | dict(readout=view, measure=measure,
                planned_scenes=len(scene_ids), complete_scenes=len(complete), planned_repeats=len(scene_ids) * expected_repeats,
                **stats(complete)))
    return result


def input_record(path):
    return dict(path=str(path), bytes=path.stat().st_size, sha256=sha(path))


def load_model(work, campaign, model, provenance):
    name = model + "-full-control-all"
    plan = next(v["plan"] for v in campaign["validated_commands"] if v["job"] == name + "-main")
    queries = {q["id"]: q for q in plan["queries"]}
    conds = {c["name"]: c for c in plan["conditions"]}
    assert len(queries) == 512 and len({q["scene_id"] for q in queries.values()}) == 64
    assert {(r["response_format"], r["mapping"]) for r in plan["readouts"]} == set(READOUTS)
    seen_queries = set()
    score_count = 0
    metadata_files = sorted((work / campaign["run_id"]).glob(name + "-main-s*/metadata.json"))
    for path in metadata_files:
        metadata = read_json(path)
        provenance.append(input_record(path))
        assert metadata["conditions"] == plan["conditions"]
        assert metadata["readouts"] == plan["readouts"]
        for q in metadata["queries"]:
            assert q["id"] not in seen_queries
            seen_queries.add(q["id"])
            assert all(q[f] == queries[q["id"]][f] for f in QUERY_FIELDS), (path, q["id"])
        assert metadata["completed_score_rows"] == metadata["expected_score_rows"]
        score_count += metadata["completed_score_rows"]
    assert seen_queries == set(queries) and score_count == plan["expected_score_rows"]
    directory = work / "analysis" / name
    summary_path = directory / "summary.json"
    summary = read_json(summary_path)
    provenance.append(input_record(summary_path))
    assert summary["coverage"]["planned_pairs"] == len(queries) * len(READOUTS) * len(GRID) * 2
    effects_path = directory / "query_effects.jsonl"
    provenance.append(input_record(effects_path))
    effects = {}
    for line in effects_path.open():
        r = json.loads(line)
        q = queries[r["id"]]
        assert all(r[f] == q[f] for f in QUERY_FIELDS if f not in ("referred_object_by_source", "visual_source_signature"))
        role = r["changed_entry"]
        other = "later" if role == "main" else "main"
        coords = tuple(r[f] for f in COORDS)
        assert coords in GRID
        assert r["changing_layout"] == r[role + "_layout"] and r["fixed_layout"] == r[other + "_layout"]
        assert r["changing_strength"] == r[role + "_strength"] and r["fixed_strength"] == r[other + "_strength"]
        ca, cb = conds[r["a_condition"]], conds[r["b_condition"]]
        assert ca[role + "_assignment"] == "a" and cb[role + "_assignment"] == "b"
        assert ca[other + "_assignment"] == cb[other + "_assignment"] == r["fixed_other_assignment"]
        changing_source_key = "main" if role == "main" else "donor_source"
        fixed_source_key = "donor_source" if role == "main" else "main"
        refs = q["referred_object_by_source"]
        changing_ref, fixed_ref = refs[ca[changing_source_key]], refs[ca[fixed_source_key]]
        assert changing_ref == refs[cb[changing_source_key]] and fixed_ref == refs[cb[fixed_source_key]]
        value = subtract(r["b_score"], r["a_score"])
        assert (value is None and r["effect"] is None) or math.isclose(value, r["effect"], abs_tol=1e-12)
        key = (r["id"], r["response_format"], r["mapping"], *coords, role)
        assert key not in effects
        effects[key] = (r["effect"], changing_ref, fixed_ref, r["a_condition"], r["b_condition"])
    expected = {(q, fmt, mapping, *coords, role) for q, (fmt, mapping), coords, role
                in itertools.product(queries, READOUTS, GRID, ("main", "later"))}
    assert set(effects) == expected
    return plan, queries, effects, summary, dict(metadata_shards=len(metadata_files), queries=len(queries),
        scenes=64, planned_effects=len(expected), present_effects=len(effects), completed_score_rows=score_count)


def analyze_model(work, campaign, model, provenance):
    plan, queries, effects, original_summary, coverage = load_model(work, campaign, model, provenance)
    role_rows = []
    role_index = {}
    for q, (fmt, mapping), coords in itertools.product(queries.values(), READOUTS, GRID):
        k = (q["id"], fmt, mapping, *coords)
        main, later = effects[k + ("main",)], effects[k + ("later",)]
        assert main[1:3] == later[1:3], "Source referents must follow C/F during role exchange"
        cref, fref = main[1:3]
        row = dict(model=model, query_id=q["id"], scene_id=q["scene_id"], intervention_id=q["intervention_id"],
                   changed_object=q["changed_object"], query_type=q["query_type"], m0_query_target=q["query_target"],
                   m0_queried_object=q["queried_object"], direction_a_index=q["direction_a_index"], direction_b_index=q["direction_b_index"],
                   response_format=fmt, mapping=mapping, **dict(zip(COORDS, coords)),
                   changing_referent=cref, fixed_referent=fref,
                   changing_referent_role="changed" if cref == q["changed_object"] else "unchanged",
                   fixed_referent_role="changed" if fref == q["changed_object"] else "unchanged",
                   referent_status="agree" if cref == fref else "conflict",
                   E_main=main[0], E_later=later[0], D_main_minus_later=subtract(main[0], later[0]),
                   main_a_condition=main[3], main_b_condition=main[4], later_a_condition=later[3], later_b_condition=later[4])
        role_rows.append(row)
        role_index[k] = row
    identity_queries = {(q["intervention_id"], q["queried_object"]): q for q in queries.values() if q["query_type"] == "identity"}
    comparisons = []
    for relation in role_rows:
        if relation["query_type"] != "relation":
            continue
        coords = tuple(relation[f] for f in COORDS)
        suffix = (relation["response_format"], relation["mapping"], *coords)
        original_q = identity_queries[(relation["intervention_id"], relation["m0_queried_object"])]
        original = role_index[(original_q["id"], *suffix)]
        anchors = (("common_referent", relation["changing_referent"]),) if relation["referent_status"] == "agree" else (
            ("changing_source_referent", relation["changing_referent"]), ("fixed_source_referent", relation["fixed_referent"]))
        for anchor, obj in anchors:
            aligned_q = identity_queries[(relation["intervention_id"], obj)]
            aligned = role_index[(aligned_q["id"], *suffix)]
            row = dict(model=model, scene_id=relation["scene_id"], intervention_id=relation["intervention_id"],
                       relation_query_id=relation["query_id"], original_identity_query_id=original_q["id"],
                       aligned_identity_query_id=aligned_q["id"], changed_object=relation["changed_object"],
                       m0_query_target=relation["m0_query_target"], anchor=anchor,
                       aligned_referent_role="changed" if obj == relation["changed_object"] else "unchanged",
                       referent_status=relation["referent_status"], changing_referent=relation["changing_referent"], fixed_referent=relation["fixed_referent"],
                       response_format=relation["response_format"], mapping=relation["mapping"], **dict(zip(COORDS, coords)))
            for measure in ("E_main", "E_later", "D_main_minus_later"):
                row[measure + "_relation"] = relation[measure]
                row[measure + "_original_identity"] = original[measure]
                row[measure + "_aligned_identity"] = aligned[measure]
                row[measure + "_raw_query_difference"] = subtract(relation[measure], original[measure])
                row[measure + "_aligned_query_difference"] = subtract(relation[measure], aligned[measure])
                row[measure + "_referent_component"] = subtract(aligned[measure], original[measure])
                if finite(row[measure + "_raw_query_difference"]):
                    assert math.isclose(row[measure + "_raw_query_difference"], row[measure + "_aligned_query_difference"] + row[measure + "_referent_component"], abs_tol=1e-10)
            comparisons.append(row)
    scenes = {q["scene_id"] for q in queries.values()}
    role_groups = ("model", *COORDS, "query_type", "m0_query_target", "changing_referent_role", "fixed_referent_role", "referent_status")
    role_summaries = summarize(role_rows, role_groups, ("E_main", "E_later", "D_main_minus_later"), scenes)
    comp_groups = ("model", *COORDS, "m0_query_target", "anchor", "aligned_referent_role", "referent_status")
    comp_measures = tuple(m + suffix for m in ("E_main", "E_later", "D_main_minus_later")
                          for suffix in ("_relation", "_original_identity", "_aligned_identity", "_raw_query_difference", "_aligned_query_difference", "_referent_component"))
    comp_summaries = summarize(comparisons, comp_groups, comp_measures, scenes)
    # Existing role summaries provide a non-circular arithmetic check of this new join.
    official = {(r["response_format"], r["query_type"], r["query_target"], *(r[f] for f in COORDS)): r
                for r in original_summary["contrast_summaries"] if r["contrast"] == "changed_entry"}
    verified = 0
    for r in role_summaries:
        if r["measure"] != "D_main_minus_later" or r["readout"] not in ("direct", "label_pooled"):
            continue
        fmt = "direct" if r["readout"] == "direct" else "label"
        old = official[(fmt, r["query_type"], r["m0_query_target"], *(r[f] for f in COORDS))]
        assert old["complete_unit_n"] == r["complete_scenes"]
        assert math.isclose(old["statistics"]["mean"], r["mean"], abs_tol=1e-10)
        assert math.isclose(old["statistics"]["normal_95_interval"][0], r["low"], abs_tol=1e-10)
        verified += 1
    coverage |= dict(role_pairs=len(role_rows), relation_identity_comparisons=len(comparisons),
                     official_role_summary_matches=verified, complete_score_effects=sum(finite(v[0]) for v in effects.values()),
                     role_summary_cells=len(role_summaries), comparison_summary_cells=len(comp_summaries))
    examples = []
    scene = sorted(scenes)[0]
    for r in comparisons:
        if r["scene_id"] == scene and r["changed_object"] == "circle" and r["response_format"] == "direct" and r["changing_strength"] == r["fixed_strength"] == 1 and r["fixed_other_assignment"] == "a" and r["m0_query_target"] == "changed":
            examples.append(r)
    return role_rows, comparisons, role_summaries, comp_summaries, coverage, examples


def census(rows):
    return dict(cells=len(rows), positive=sum(r["mean"] > 0 for r in rows), negative=sum(r["mean"] < 0 for r in rows),
                interval_positive=sum(r["low"] > 0 for r in rows), interval_negative=sum(r["high"] < 0 for r in rows),
                minimum=min((r["mean"] for r in rows), default=None), maximum=max((r["mean"] for r in rows), default=None))




def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work", type=Path, default=DEFAULT_WORK)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    campaign_path = args.work / "main_plan.json"
    campaign = read_json(campaign_path)
    script_path = Path(__file__).resolve()
    provenance = [input_record(campaign_path), input_record(script_path)]
    design_notes = script_path.parents[1] / "docs/full_control_v10.md"
    if design_notes.is_file():
        provenance.append(input_record(design_notes))
    all_role, all_comp, coverage, examples = [], [], {}, []
    for model in MODELS:
        roles, comps, rs, cs, cov, ex = analyze_model(args.work, campaign, model, provenance)
        save_csv(args.output / (model + "-role-pairs.csv"), roles)
        save_csv(args.output / (model + "-referent-comparisons.csv"), comps)
        all_role.extend(rs)
        all_comp.extend(cs)
        coverage[model] = cov
        examples.extend(ex)
        print(model, json.dumps(cov), flush=True)
    save_csv(args.output / "role-summaries.csv", all_role)
    save_csv(args.output / "referent-comparison-summaries.csv", all_comp)
    save_json(args.output / "known-pair-examples.json", examples)
    sign_tables = {}
    for model in MODELS:
        role = [r for r in all_role if r["model"] == model and r["measure"] == "D_main_minus_later" and r["readout"] in ("direct", "label_pooled")]
        comp = [r for r in all_comp if r["model"] == model and r["readout"] in ("direct", "label_pooled")]
        sign_tables[model] = {
            "identity_changed_all": census([r for r in role if r["query_type"] == "identity" and r["m0_query_target"] == "changed"]),
            "relation_m0target_changed_equal_strength": census([r for r in role if r["query_type"] == "relation" and r["m0_query_target"] == "changed" and r["changing_strength"] == r["fixed_strength"]]),
            "relation_m0target_changed_full_strength": census([r for r in role if r["query_type"] == "relation" and r["m0_query_target"] == "changed" and r["changing_strength"] == r["fixed_strength"] == 1]),
        }
        for actual in ("changed", "unchanged"):
            sign_tables[model]["same_referent_" + actual + "_role"] = census([r for r in role if r["query_type"] == "relation" and r["referent_status"] == "agree" and r["changing_referent_role"] == actual])
            sign_tables[model]["same_referent_" + actual + "_query_difference"] = census([r for r in comp if r["anchor"] == "common_referent" and r["aligned_referent_role"] == actual and r["measure"] == "D_main_minus_later_aligned_query_difference"])
    result = dict(schema="v10-full-referents-posthoc-20261016-v1", phase="Exploratory post-hoc analysis of archived effects; not held-out prediction or new experiment",
                  direction="Fixed changed-object a/b for all queries and roles, without sign recoding",
                  paired_role="D=E_main(C,F)-E_later(C,F), holding changing/fixed source layout, strength and assignment; C/F exchange physical entry",
                  referent_rule="Source-specific referred_object_by_source in executed query metadata; assignment a/b must not change referent",
                  comparison_rule="Agreeing source referents: relation minus identity for the common object. Conflicting source referents: separate changing-source and fixed-source anchors, neither designated unique hybrid truth",
                  decomposition="Raw relation-minus-M0-identity = aligned relation-minus-identity + identity(aligned referent)-identity(M0 target); algebraic description, not causal mediation",
                  aggregation="Differences before scene means; complete two changed-object repeats and both label mappings when pooled; 64 scenes; normal descriptive 95% intervals; no native-success filtering; no across-model pooling",
                  scope="Score endpoint only. Generation is not reanalyzed here. Original exported E retained in role-pairs.csv; source analysis not modified",
                  coverage=coverage, sign_census=sign_tables, inputs=provenance,
                  reproduce=[sys.executable, str(Path(__file__).resolve()), "--work", str(args.work), "--output", str(args.output)])
    save_json(args.output / "analysis.json", result)
    print(json.dumps(sign_tables, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
