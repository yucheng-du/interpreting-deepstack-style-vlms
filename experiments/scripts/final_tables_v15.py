#!/usr/bin/env python3
"""Final v15 report tables from curves.py output; stdlib only, system python3.

Copies values, intervals and denominators from the curves rows (severity_curves.json/.csv and
capability_calibrated.json/.csv written by cloud_prep/measurement_v15/curves.py), from calibrate.py's
calibration.json, from the benchmark import_metadata.json files and, optionally, from processor-geometry
summaries and benchmark scorer JSONs. Nothing is re-estimated here: no mean, interval or denominator is
computed. The only actual-loss values without an interval are the CC-DD differences that curves.py itself
derives for two-choice (synthetic) inputs; benchmark CC-DD is the scorer's paired estimate (value_source
says which). A quantity that is part of a run's design but has no curves row gets explicit NA cells (CSV
"NA", JSON null) and is named in missing_quantities.

Usage (any cwd):
  python3 scripts/final_tables_v15.py --curves CURVES_DIR --calibration calibration.json \\
      --import-metadata MATERIALS/import_metadata.json [...] [--geometry GEOMETRY.json ...] \\
      [--benchmark-scores MODEL-TASK-...-score.json ...] --output OUT_DIR

  --curves            directory with severity_curves.json (or .csv) and capability_calibrated.json (or .csv)
  --calibration       calibrate.py output (shared grid, per-model r, capability-calibrated entries)
  --import-metadata   benchmark importer metadata (severity.noop_images per label); a directory holding
                      import_metadata.json is accepted too
  --geometry          optional processor-geometry summaries: scripts/geometry_summary_v15.py outputs (groups per
                      model x material_set x task, flat per-source table), run reports with a "geometry" block,
                      bare {model, by_task, by_source} summaries, or any JSON with
                      model/material/task records (flattened field by field). The main table's
                      processor_geometry column joins task-level geometry distributions on model x task and
                      material_set (else split).
  --benchmark-scores  optional scorer JSONs (scripts/score_followup_benchmark_v15.py): adds per-label
                      planned questions/clusters and per-cell missing/invalid/truncated counts to
                      sample_sizes (the curves rows carry only the cluster n for free-form rows)

Output (OUT_DIR must not be the curves directory; curves.py's capability_calibrated.* would be replaced):
  main_table, sample_sizes, sensitivities, capability_calibrated, noop_images, grid, geometry
  (.csv and .json each) and tables_index.md (columns, conventions, inputs with sha256, row counts, warnings).
"""
from __future__ import annotations
import argparse
from collections import defaultdict
import csv
import hashlib
import io
import json
import os
from pathlib import Path
import sys
import tempfile

SCHEMA = 'v15-final-tables-1'
CURVES_SCHEMA = 'v15-severity-curves-1'
CALIBRATION_SCHEMA = 'v15-severity-calibration-1'
IMPORT_SCHEMA = 'followup-benchmark-import-v15'
NA = 'NA'
CALIBRATED_NOTE = 'model-specific severity; a different estimand from the shared physical grid'

MODEL_ORDER = ('qwen8b', 'ibm4b', 'deepstack7b', 'qwen32b')
TASK_ORDER = ('color', 'stripe_orientation', 'relative_position', 'text_reading', 'gqa_attribute', 'gqa_spatial',
              'docvqa_document_qa')
READOUT_ORDER = ('direct/none', 'label/identity', 'label/cyclic', 'freeform/none')
KIND_ORDER = ('score', 'generation')
SPLIT_ORDER = ('evaluation', 'calibration', 'validation')
SOURCE_ORDER = ('analysis', 'benchmark')
ROLE_ORDER = ('primary', 'metric sensitivity', 'policy sensitivity', 'metric and policy sensitivity')

# Primary measures (selected before calibration).
TWO_CHOICE_PRIMARY = 'accuracy_invalid_as_incorrect'
TWO_CHOICE_VALID_ONLY = 'accuracy_valid_only'
CONFLICT_PRIMARY = {'score': 'b_minus_a_log_odds', 'generation': 'answer_b_indicator'}
PRIMARY_POLICY = 'invalid_as_zero'
BENCHMARK_PRIMARY_METRIC = {'gqa_attribute': 'gqa_exact', 'gqa_spatial': 'gqa_exact',
                            'docvqa_document_qa': 'docvqa_anls_organizer_framework'}
METRIC_NOTES = {
    'gqa_exact': 'GQA strict exact equality (official evaluator; primary)',
    'gqa_lower_strip_sensitivity': 'GQA lower_strip: lowercase + strip equality (named sensitivity)',
    'gqa_case_punct_normalized': 'GQA case_punct: lowercase, strip, trailing . ! ? and outer quotes (named sensitivity; '
                                 'the severity-calibration metric)',
    'docvqa_anls_organizer_framework': 'DocVQA ANLS, organiser framework (similarity >= .5 kept; primary)',
    'docvqa_anls_strict_paper_cutoff': 'DocVQA ANLS, strict original-paper cutoff (sensitivity)',
    'docvqa_exact_lower_strip': 'DocVQA exact match after lowercase + strip (sensitivity)',
}
POLICY_NOTES = {
    'invalid_as_zero': 'invalid/empty/truncated output scored 0 (primary policy)',
    'score_returned_strings': 'returned strings scored as given, also when flagged invalid/truncated (sensitivity)',
    'valid_cells_only': 'only questions whose cells are all valid (sensitivity)',
}

# curves quantity -> column prefix, family (recovery: recovery measure; conflict: conflict measure), meaning
QUANTITIES = (
    ('native_clean', 'cc', 'recovery', 'native clean CC (main clean, later clean)'),
    ('joint_degraded', 'dd', 'recovery', 'joint-degraded DD (main and later from the label image)'),
    ('main_clean_later_degraded', 'cd', 'recovery', 'CD (main clean, later from the label image)'),
    ('main_degraded_later_clean', 'dc', 'recovery', 'DC (main from the label image, later clean)'),
    ('realized_native_degradation', 'cc_minus_dd', 'recovery', 'actual native loss CC-DD (the x-axis of the curves)'),
    ('later_recovery', 'dc_minus_dd', 'recovery', 'later recovery DC-DD'),
    ('recovery_interaction', 'recovery_interaction', 'recovery', 'recovery interaction (DC-DD)-(CC-CD)'),
    ('conflict_target_effect', 'conflict_T', 'conflict', 'conflict target effect T (later b-a effect, changed queries) '
                                                         'at this row\'s main state (normal main on reference rows)'),
    ('conflict_control_effect', 'conflict_C', 'conflict', 'conflict control effect C (unchanged/control queries)'),
    ('selectivity_R', 'conflict_R', 'conflict', 'selectivity R (target minus control within interventions)'),
    ('delta_R_vs_normal', 'delta_R', 'conflict', 'dR(L) = R(L) - R(normal) (selectivity interaction)'),
)
RECOVERY_QUANTITIES = tuple(q for q in QUANTITIES if q[2] == 'recovery')
CONTROL_ROLE_PREFERENCE = ('unchanged', 'control')
EXCLUSION_REASONS = ('missing', 'truncated', 'invalid', 'alignment_unknown', 'reference_not_a_or_b')
SCORER_CELLS = {'native_clean': ('clean__clean',), 'joint_degraded': ('degraded__degraded',),
                'main_clean_later_degraded': ('clean__degraded',), 'main_degraded_later_clean': ('degraded__clean',),
                'realized_native_degradation': ('clean__clean', 'degraded__degraded'),
                'later_recovery': ('degraded__clean', 'degraded__degraded'),
                'recovery_interaction': ('clean__clean', 'clean__degraded', 'degraded__clean', 'degraded__degraded')}
GROUP_FIELDS = ('source', 'model', 'model_revision', 'execution_version', 'material_set', 'split', 'task',
                'response_format', 'mapping', 'kind')
REQUIRED_ROW_FIELDS = ('source', 'model', 'task', 'kind', 'measure', 'quantity')
# Typed parsing of the curves CSV fallback (curves.py writes '' for None, JSON for dicts, str() otherwise).
CSV_NUMBER = {'n', 'mean', 'se', 'ci_low', 'ci_high', 'planned', 'used', 'planned_units', 'severity_rank', 'setting_value',
              'calibrated_target', 'calibration_r_mean', 'interpolated_value'}
CSV_BOOL = {'se_zero', 'n_lt_5', 'in_shared_grid', 'primary_measure'}
CSV_JSON = {'excluded'}


# ----------------------------------------------------------------------------------------------- helpers
class Warnings(list):
    def __call__(self, message):
        if message not in self:
            self.append(message)


def order_key(value, sequence):
    return (sequence.index(value), '') if value in sequence else (len(sequence), str(value))


def sha256_bytes(data):
    return hashlib.sha256(data).hexdigest()


def read_json(path):
    data = Path(path).read_bytes()
    return json.loads(data), sha256_bytes(data)


def typed(key, text):
    if text is None or text == '':
        return None
    if key in CSV_BOOL:
        return {'True': True, 'False': False, 'true': True, 'false': False}.get(text, text)
    if key in CSV_NUMBER:
        for cast in (int, float):
            try:
                return cast(text)
            except ValueError:
                pass
        return text
    if key in CSV_JSON:
        try:
            return json.loads(text)
        except ValueError:
            return text
    return text


def get(row, key):
    return None if row is None else row.get(key)


def inference_unit(source, task):
    if source == 'analysis':
        return 'template'
    if str(task).startswith('gqa'):
        return 'image'
    if str(task).startswith('docvqa'):
        return 'document'
    return 'cluster unit'


def split_measure(measure):
    metric, sep, policy = str(measure).partition('|')
    return metric, (policy if sep else None)


def group_key(row):
    return tuple(row.get(k) for k in GROUP_FIELDS)


def group_info(g):
    return dict(zip(GROUP_FIELDS, g))


def label_text(label):
    return 'reference' if label is None else label


# ----------------------------------------------------------------------------------------------- inputs
def read_curves_file(directory, stem, warn, required):
    """(rows, header, provenance) of one curves.py table; JSON preferred, CSV typed as a fallback."""
    jpath, cpath = directory / (stem + '.json'), directory / (stem + '.csv')
    if jpath.exists():
        doc, digest = read_json(jpath)
        if not isinstance(doc, dict) or doc.get('schema') != CURVES_SCHEMA or not isinstance(doc.get('rows'), list):
            raise ValueError(f'{jpath}: not a curves.py output (schema {CURVES_SCHEMA} with a rows list)')
        if cpath.exists():
            with cpath.open(newline='') as f:
                n = sum(1 for _ in csv.DictReader(f))
            if n != len(doc['rows']):
                warn(f'{cpath}: {n} rows, {jpath.name} {len(doc["rows"])}; the JSON rows are used')
        rows, header, prov = doc['rows'], {k: v for k, v in doc.items() if k != 'rows'}, dict(path=str(jpath.resolve()), sha256=digest)
    elif cpath.exists():
        data = cpath.read_bytes()
        rows = [{k: typed(k, v) for k, v in r.items()} for r in csv.DictReader(io.StringIO(data.decode('utf-8'), newline=''))]
        header, prov = {}, dict(path=str(cpath.resolve()), sha256=sha256_bytes(data))
        warn(f'{jpath} not found; rows typed from {cpath.name} (no header: input list and notes unavailable)')
    elif required:
        raise FileNotFoundError(f'{directory}: neither {stem}.json nor {stem}.csv found')
    else:
        return None, {}, None
    for i, r in enumerate(rows):
        if not isinstance(r, dict) or any(k not in r for k in REQUIRED_ROW_FIELDS):
            raise ValueError(f'{prov["path"]}: row {i} lacks one of {REQUIRED_ROW_FIELDS}')
    return rows, header, prov


def read_calibration(path):
    doc, digest = read_json(path)
    if not isinstance(doc, dict) or doc.get('schema') != CALIBRATION_SCHEMA:
        raise ValueError(f'{path}: not a calibrate.py output (schema {CALIBRATION_SCHEMA})')
    return doc, digest


def read_benchmark_scores(paths, warn):
    """{(model, execution_version, material_set, task, label): per-label scorer body} and provenance."""
    index, provenance = {}, []
    for path in paths:
        doc, digest = read_json(path)
        reports = doc.get('reports') if isinstance(doc, dict) else None
        if not isinstance(reports, list):
            raise ValueError(f'{path}: no "reports" list; not a benchmark scorer output')
        provenance.append(dict(role='benchmark_scores', path=str(Path(path).resolve()), sha256=digest))
        for rep in reports:
            if not isinstance(rep, dict):
                continue
            labels = rep.get('per_label') if isinstance(rep.get('per_label'), dict) else {}
            for label, body in labels.items():
                if not isinstance(body, dict):
                    continue
                key = (rep.get('model'), rep.get('execution_version'), str(rep.get('material_set', doc.get('material_set', 'unspecified'))),
                       rep.get('task'), label)
                if key in index:
                    warn(f'{path}: scorer label {key} given twice; the later file is used')
                index[key] = dict(body, _input=str(path))
    return index, provenance


# ----------------------------------------------------------------------------------------------- geometry
GEOMETRY_COLUMNS = ('model', 'model_family', 'material', 'job', 'split', 'task', 'scope', 'image_source', 'field', 'field_value',
                    'count', 'note', 'count_unit', 'execution_version', 'check_run_passed', 'format', 'run_dir', 'manifest', 'input',
                    'input_sha256')


def job_split(job, run_dir):
    text = f'{run_dir or ""}/{job or ""}'
    if '/validation/' in text or str(job or '').startswith('validation'):
        return 'validation'
    for split in ('calibration', 'evaluation'):
        if split in str(job or ''):
            return split
    return None


STATS_KEYS = {'n', 'min', 'median', 'max', 'mean'}
STRUCT_NAMES = {'counts', 'violations', 'checked', 'totals', 'records', 'other_issues', 'examples'}


def is_distribution(name, value):
    """{value text: count} (optionally with a '...' tail note), as the geometry summaries write distributions."""
    return (isinstance(value, dict) and bool(value) and name not in STRUCT_NAMES and not set(value) & STATS_KEYS
            and all((isinstance(v, int) and not isinstance(v, bool)) or (k == '...' and isinstance(v, str)) for k, v in value.items()))


def flatten_fields(value, path=''):
    """(field path, value, count, note): distributions give one row per distinct value, other leaves one row each."""
    name = path.rsplit('.', 1)[-1]
    if is_distribution(name, value):
        for k, v in value.items():
            yield (path, k, None, v) if isinstance(v, str) else (path, k, v, None)
    elif isinstance(value, dict):
        for k, v in value.items():
            yield from flatten_fields(v, f'{path}.{k}' if path else str(k))
    elif isinstance(value, list) and any(isinstance(v, (dict, list)) for v in value):
        for i, v in enumerate(value):
            yield from flatten_fields(v, f'{path}[{i}]')
    else:
        yield path, (json.dumps(value) if isinstance(value, list) else value), None, None


def geometry_records(doc, path, digest, warn):
    """Long geometry rows (GEOMETRY_COLUMNS). Recognised: check_run.py outputs (geometry.by_task / by_source),
    geometry_summary_v15.py outputs (groups per model x material_set x task, flat per-source table), bare
    {model, by_task, by_source} summaries; anything else is flattened field by field. field == 'geometry'
    marks the processor-geometry distribution (value = geometry text, count = blocks)."""
    rows = []
    base_input = dict(input=str(Path(path).resolve()), input_sha256=digest)

    def emit(base, **fields):
        rows.append({k: fields.get(k, base.get(k)) for k in GEOMETRY_COLUMNS})

    def check_run_rows(summary, top, fmt):
        run_dir = top.get('dir') or top.get('run_dir') or top.get('output_dir')
        job = top.get('job') or (Path(run_dir).name if run_dir else None)
        model = summary.get('model') or top.get('model')
        material = top.get('material') or top.get('material_set')
        if material is None and job:
            material = job[len(model) + 1:] if model and job.startswith(model + '-') else job
        by_task = summary.get('by_task') if isinstance(summary.get('by_task'), dict) else {}
        by_source = summary.get('by_source') if isinstance(summary.get('by_source'), dict) else {}
        only_task = next(iter(by_task)) if len(by_task) == 1 else None
        base = dict(base_input, format=fmt, model=model, execution_version=top.get('execution_version'), job=job, material=material,
                    split=top.get('split') or job_split(job, run_dir), run_dir=run_dir, manifest=top.get('manifest'),
                    check_run_passed=top.get('passed'), count_unit=summary.get('unit'))
        for scope, listing in (('task', by_task), ('source', by_source)):
            for key, counts in listing.items():
                where = dict(task=key if scope == 'task' else only_task, scope=scope, image_source=None if scope == 'task' else key, field='geometry')
                if not isinstance(counts, dict):
                    emit(base, **where, field_value=json.dumps(counts, sort_keys=True), note='unrecognised entry')
                    continue
                for text, count in counts.items():
                    numeric = isinstance(count, (int, float)) and not isinstance(count, bool)
                    emit(base, **where, field_value=text, count=count if numeric else None, note=None if numeric else str(count))
        emit(base, task=only_task, scope='run', field='distinct_geometries', field_value=summary.get('distinct'))

    def summary_v15_rows(item):
        table = [t for t in item.get('table') or [] if isinstance(t, dict)]
        for g in item.get('groups') or []:
            if not isinstance(g, dict):
                continue
            versions = g.get('execution_versions')
            base = dict(base_input, format='geometry_summary', model=g.get('model'), model_family=g.get('family'),
                        material=g.get('material_set'), split=g.get('split'), task=g.get('task'), job=item.get('job'),
                        run_dir=item.get('run_dir'), execution_version=','.join(versions) if isinstance(versions, dict) else versions,
                        count_unit='blocks')
            geometry = g.get('geometry') if isinstance(g.get('geometry'), dict) else {}
            for text, count in (geometry.get('distinct') or {}).items():
                numeric = isinstance(count, int) and not isinstance(count, bool)
                emit(base, scope='task', field='geometry', field_value=text, count=count if numeric else None, note=None if numeric else str(count))
            rest = {k: v for k, v in g.items() if k not in ('model', 'family', 'material_set', 'split', 'task', 'sources', 'execution_versions')}
            rest['geometry'] = {k: v for k, v in geometry.items() if k != 'distinct'}
            for field, value, count, note in flatten_fields(rest):
                emit(base, scope='task', field=field, field_value=value, count=count, note=note, count_unit=None)
            mine = [t for t in table if (t.get('model'), t.get('material_set'), t.get('task')) == (g.get('model'), g.get('material_set'), g.get('task'))]
            if mine:
                for t in mine:
                    for field, value in t.items():
                        if field not in ('model', 'family', 'material_set', 'split', 'task', 'source'):
                            emit(base, scope='source', image_source=t.get('source'), field=field, field_value=value, count_unit=None)
            else:
                for source, body in (g.get('sources') or {}).items():
                    for field, value, count, note in flatten_fields(body):
                        emit(base, scope='source', image_source=source, field=field, field_value=value, count=count, note=note, count_unit=None)
        for field, value, count, note in flatten_fields({k: item.get(k) for k in ('totals', 'problems') if item.get(k) is not None}):
            emit(dict(base_input, format='geometry_summary', job=item.get('job'), run_dir=item.get('run_dir')),
                 scope='run', field=field, field_value=value, count=count, note=note)

    def generic_rows(item):
        model = item.get('model') or item.get('model_key')
        task = item.get('task')
        if model is None and task is None:
            warn(f'{path}: geometry record without model/task fields; flattened with NA identifiers')
        base = dict(base_input, format='generic', model=model, execution_version=item.get('execution_version'), job=item.get('job'),
                    material=item.get('material') or item.get('material_set') or item.get('job'),
                    split=item.get('split') or job_split(item.get('job'), item.get('dir')), task=task,
                    run_dir=item.get('dir') or item.get('run_dir'), manifest=item.get('manifest'), scope='record')
        skip = {'model', 'model_key', 'material', 'material_set', 'job', 'task', 'execution_version', 'split', 'dir', 'run_dir', 'manifest'}
        for field, value, count, note in flatten_fields({k: v for k, v in item.items() if k not in skip}):
            emit(base, field=field, field_value=value, count=count, note=note)

    def visit(item, top):
        if not isinstance(item, dict):
            warn(f'{path}: non-object geometry entry skipped')
        elif str(item.get('tool', '')).startswith('v15-check-run') and isinstance(item.get('geometry'), dict):
            check_run_rows(item['geometry'], item, 'check_run')
        elif str(item.get('tool', '')).startswith('v15-geometry-summary') or (
                isinstance(item.get('groups'), list) and any(isinstance(g, dict) and 'geometry' in g for g in item['groups'])):
            summary_v15_rows(item)
        elif isinstance(item.get('by_task'), dict) or isinstance(item.get('by_source'), dict):
            check_run_rows(item, {**top, **item}, 'summary')
        elif isinstance(item.get('geometry'), dict) and isinstance(item['geometry'].get('by_task'), dict):
            check_run_rows(item['geometry'], {**top, **item}, 'summary')
        else:
            generic_rows(item)

    containers = [doc] if isinstance(doc, dict) else doc if isinstance(doc, list) else []
    if not containers:
        warn(f'{path}: geometry input is neither an object nor a list; skipped')
    for item in containers:
        known = isinstance(item, dict) and (isinstance(item.get('geometry'), dict) or str(item.get('tool', '')).startswith('v15-'))
        nested = None if known or not isinstance(item, dict) else next(
            (item[k] for k in ('rows', 'records', 'entries', 'summaries', 'runs', 'results') if isinstance(item.get(k), list)), None)
        if nested is not None:
            top = {k: v for k, v in item.items() if not isinstance(v, (list, dict))}
            for sub in nested:
                visit(sub, top)
        else:
            visit(item, {})
    return rows


def geometry_lookup(rows):
    """(model, task) -> {(split, material, job): [geometry text (count)]} of task-level geometry distributions."""
    found = defaultdict(lambda: defaultdict(list))
    for r in rows:
        if r['scope'] != 'task' or r['field'] != 'geometry':
            continue
        unit = 'blocks' if str(r.get('count_unit') or '').startswith('blocks') else 'n'
        item = f"{r['field_value']} ({r['count']} {unit})" if r.get('count') is not None else f"{r['field_value']}: {r.get('note')}"
        found[(r['model'], r['task'])][(r.get('split'), r.get('material'), r.get('job') or r.get('input'))].append(item)
    return found


def geometry_text(lookup, model, task, split, material_set):
    """Same model and task; summaries of the row's material_set if any, else of the same (or an unknown) split."""
    entries = lookup.get((model, task)) or {}
    exact = {k: v for k, v in entries.items() if k[1] == material_set}
    chosen = exact or {k: v for k, v in entries.items() if k[0] in (split, None)}
    parts = [f'{job}: ' + '; '.join(items) for (_, _, job), items in sorted(chosen.items(), key=lambda kv: str(kv[0]))]
    return ' | '.join(parts) or None


# ----------------------------------------------------------------------------------------------- curves pivot
class CurvesIndex:
    """Curves rows by (group, label, quantity); primary measures per group (model x material x readout x kind)."""

    def __init__(self, rows, warn):
        self.warn = warn
        self.index = defaultdict(lambda: defaultdict(list))
        self.groups = defaultdict(list)
        for r in rows:
            g = group_key(r)
            self.groups[g].append(r)
            self.index[(g, r.get('severity_label'))][r['quantity']].append(r)
        self.primaries = {g: self.resolve_primaries(g, grows) for g, grows in self.groups.items()}

    def resolve_primaries(self, g, rows):
        info = group_info(g)
        ident = '/'.join(str(info[k]) for k in ('model', 'material_set', 'task', 'response_format', 'mapping', 'kind'))
        if info['source'] != 'benchmark':
            return TWO_CHOICE_PRIMARY, CONFLICT_PRIMARY.get(info['kind'])
        metric = BENCHMARK_PRIMARY_METRIC.get(info['task'])
        expected = f'{metric}|{PRIMARY_POLICY}' if metric else None
        measures = {r['measure'] for r in rows}
        flagged = {r['measure'] for r in rows if r.get('primary_measure') is True}
        if expected and expected in measures:
            if flagged and flagged != {expected}:
                self.warn(f'{ident}: scorer primary {sorted(flagged)} differs from the specified primary {expected}; {expected} used')
            return expected, None
        if len(flagged) == 1:
            if expected:
                self.warn(f'{ident}: specified primary {expected} absent from the curves rows; scorer primary {next(iter(flagged))} used')
            return next(iter(flagged)), None
        self.warn(f'{ident}: no primary free-form measure (expected {expected}, scorer-flagged {sorted(flagged)}); recovery columns NA')
        return expected, None

    def pick(self, g, label, quantity, measure):
        found = [r for r in self.index.get((g, label), {}).get(quantity, []) if r.get('measure') == measure]
        if len(found) > 1:
            found.sort(key=lambda r: order_key(r.get('query_role'), CONTROL_ROLE_PREFERENCE))
            info = group_info(g)
            self.warn(f"{info['model']}/{info['task']}/{info['response_format']}/{info['mapping']}/{info['kind']} {label_text(label)} "
                      f"{quantity} {measure}: {len(found)} rows (query roles {[r.get('query_role') for r in found]}); "
                      f"{found[0].get('query_role')} used")
        return found[0] if found else None

    def pairs(self):
        return list(self.index)


def label_properties(cv, pair):
    rows = [r for quantity_rows in cv.index[pair].values() for r in quantity_rows]
    first = rows[0] if rows else {}
    return {k: first.get(k) for k in ('severity_rank', 'family', 'reference_kind', 'setting_value', 'in_shared_grid')}


ID_COLUMNS = ('source', 'model', 'model_revision', 'execution_version', 'material_set', 'split', 'task', 'readout',
              'response_format', 'mapping', 'kind', 'row_type', 'severity_label', 'severity_rank', 'family', 'reference_kind',
              'setting_value', 'in_shared_grid', 'inference_unit')


def quantity_columns(quantities):
    cols = []
    for q, prefix, _, _ in quantities:
        cols += [prefix + '_mean', prefix + '_ci_low', prefix + '_ci_high', prefix + '_n']
        if prefix == 'cc':
            cols.append('cc_scope')
        if prefix == 'cc_minus_dd':
            cols += ['cc_minus_dd_value_source', 'cc_minus_dd_source_estimand']
    return cols


MAIN_COLUMNS = ID_COLUMNS + ('recovery_measure', 'conflict_measure', 'interval_type', 'processor_geometry') + \
    tuple(quantity_columns(QUANTITIES)) + ('missing_quantities',)


def id_fields(g, label, props):
    info = group_info(g)
    return dict(source=info['source'], model=info['model'], model_revision=info['model_revision'],
                execution_version=info['execution_version'], material_set=info['material_set'], split=info['split'],
                task=info['task'], readout=f"{info['response_format']}/{info['mapping']}", response_format=info['response_format'],
                mapping=info['mapping'], kind=info['kind'], row_type='reference' if label is None else 'label',
                severity_label=label, severity_rank=props['severity_rank'], family=props['family'],
                reference_kind=props['reference_kind'], setting_value=props['setting_value'],
                in_shared_grid=props['in_shared_grid'], inference_unit=inference_unit(info['source'], info['task']))


def fill_quantities(row, chosen, quantities):
    for q, prefix, _, _ in quantities:
        x = chosen.get(q)
        row[prefix + '_mean'], row[prefix + '_ci_low'] = get(x, 'mean'), get(x, 'ci_low')
        row[prefix + '_ci_high'], row[prefix + '_n'] = get(x, 'ci_high'), get(x, 'n')
        if q == 'realized_native_degradation':
            row['cc_minus_dd_value_source'], row['cc_minus_dd_source_estimand'] = get(x, 'value_source'), get(x, 'source_estimand')


def choose_rows(cv, g, label, recovery_measure, conflict_measure, quantities=QUANTITIES):
    """{quantity: curves row} for one pair under the given measures, plus the CC scope."""
    chosen, cc_scope = {}, None
    for q, _, family, _ in quantities:
        measure = recovery_measure if family == 'recovery' else conflict_measure
        x = cv.pick(g, label, q, measure) if measure else None
        if q == 'native_clean':
            if x is not None:
                cc_scope = 'reference cell' if label is None else 'label (its own clean cell)'
            elif label is not None and measure:
                x = cv.pick(g, None, q, measure)
                cc_scope = 'shared reference cell (label-free)' if x is not None else None
        if x is not None:
            chosen[q] = x
    return chosen, cc_scope


NATIVE_DESIGN = ('native_clean', 'joint_degraded', 'realized_native_degradation')
FOUR_CELL_DESIGN = ('main_clean_later_degraded', 'main_degraded_later_clean', 'later_recovery', 'recovery_interaction')
CONFLICT_DESIGN = ('conflict_target_effect', 'conflict_control_effect', 'selectivity_R', 'delta_R_vs_normal')
REFERENCE_QUANTITIES = ('native_clean', 'conflict_target_effect', 'conflict_control_effect', 'selectivity_R')


def expected_design(found):
    """Quantities a run design implies from those found anywhere in a group: any recovery quantity implies
    CC, DD and CC-DD; any four-cell quantity all four; any conflict quantity T, C, R and dR."""
    design = set(found)
    if design & set(NATIVE_DESIGN + FOUR_CELL_DESIGN):
        design |= set(NATIVE_DESIGN)
    if design & set(FOUR_CELL_DESIGN):
        design |= set(FOUR_CELL_DESIGN)
    if design & set(CONFLICT_DESIGN):
        design |= set(CONFLICT_DESIGN)
    return design


def build_main(cv, geometry, calibration, warn):
    """Wide rows per (group, label) under the primary measures. Quantities of the group's design without a
    curves row, an absent two-choice reference row and absent shared-grid labels (non-calibration groups)
    are written as NA rows/cells and listed in missing_quantities."""
    rows, chosen_by_pair, present = {}, {}, {}
    grid = calibration.get('severity_labels') or {}
    shared = calibration.get('shared_grid') or {}

    def new_row(g, label, props, chosen, cc_scope):
        rec, conf = cv.primaries[g]
        row = id_fields(g, label, props)
        intervals = sorted({x.get('interval') for x in chosen.values() if x.get('interval')})
        row.update(recovery_measure=rec, conflict_measure=conf, interval_type='/'.join(intervals) or None,
                   processor_geometry=geometry_text(geometry, row['model'], row['task'], row['split'], row['material_set'])
                   if geometry is not None else None)
        fill_quantities(row, chosen, QUANTITIES)
        row['cc_scope'] = cc_scope
        return row

    for pair in cv.pairs():
        g, label = pair
        rec, conf = cv.primaries[g]
        chosen, cc_scope = choose_rows(cv, g, label, rec, conf)
        rows[pair], chosen_by_pair[pair], present[pair] = new_row(g, label, label_properties(cv, pair), chosen, cc_scope), chosen, set(chosen)
    group_found = defaultdict(set)
    for (g, _), found in present.items():
        group_found[g] |= found
    for g in list(cv.groups):
        info = group_info(g)
        absent = []
        if info['source'] == 'analysis' and (g, None) not in rows:
            absent.append((None, dict(severity_rank=0, family=None, reference_kind=None, setting_value=None, in_shared_grid=None)))
        if info['split'] != 'calibration':
            spec = shared.get(info['task']) or {}
            values = dict(zip(spec.get('labels') or [], spec.get('values') or []))
            for label in grid.get(info['task']) or []:
                if (g, label) not in rows:
                    absent.append((label, dict(severity_rank=None, family=spec.get('family'), reference_kind=spec.get('reference_kind'),
                                               setting_value=values.get(label), in_shared_grid=True)))
        for label, props in absent:
            ident = '/'.join(str(info[k]) for k in ('model', 'material_set', 'task', 'response_format', 'mapping', 'kind'))
            warn(f'{ident}: no curves rows for {label_text(label)} (expected: '
                 f'{"two-choice reference row" if label is None else "shared-grid label"}); written as an NA row')
            chosen, cc_scope = ({}, None) if label is None else choose_rows(cv, g, label, *cv.primaries[g])
            rows[(g, label)], chosen_by_pair[(g, label)], present[(g, label)] = new_row(g, label, props, chosen, cc_scope), chosen, set(chosen)
    for (g, label), row in rows.items():
        design = expected_design(group_found[g])
        if label is None:
            design &= set(REFERENCE_QUANTITIES)
        missing = [q for q, _, _, _ in QUANTITIES if q in design - present[(g, label)]]
        row['missing_quantities'] = ';'.join(missing) or None
    return rows, chosen_by_pair


def pair_sort_key(pair, row):
    g, label = pair
    info = group_info(g)
    rank = row.get('severity_rank')
    return (order_key(info['split'], SPLIT_ORDER), order_key(info['task'], TASK_ORDER), order_key(info['source'], SOURCE_ORDER),
            order_key(info['model'], MODEL_ORDER), str(info['material_set']), str(info['model_revision']), str(info['execution_version']),
            order_key(f"{info['response_format']}/{info['mapping']}", READOUT_ORDER), order_key(info['kind'], KIND_ORDER),
            0 if label is None else 1, rank if isinstance(rank, (int, float)) and not isinstance(rank, bool) else 1e9, str(label))


# ----------------------------------------------------------------------------------------------- sample sizes
SAMPLE_COLUMNS = ('source', 'model', 'model_revision', 'execution_version', 'material_set', 'split', 'task', 'readout', 'kind',
                  'row_type', 'severity_label', 'severity_rank', 'in_shared_grid', 'quantity', 'main_table_column', 'measure',
                  'status', 'value_source', 'source_estimand', 'inference_unit', 'n_units', 'planned_units', 'level', 'planned',
                  'used') + tuple('excluded_' + r for r in EXCLUSION_REASONS) + (
                  'excluded_other', 'se_zero', 'n_lt_5', 'invalid_handling', 'valid_only_n_units', 'valid_only_used',
                  'valid_only_excluded_missing', 'valid_only_excluded_truncated', 'valid_only_excluded_invalid',
                  'bench_cell_mode', 'bench_planned_questions', 'bench_planned_clusters', 'bench_questions_all_cells_observed',
                  'bench_clusters_complete', 'bench_incomplete_clusters', 'bench_valid_cell_questions_in_complete_clusters',
                  'bench_policy', 'bench_policy_questions', 'bench_policy_clusters', 'bench_cells', 'bench_cell_present',
                  'bench_cell_missing', 'bench_cell_invalid_any', 'bench_cell_invalid_truncated', 'bench_cell_invalid_reasons',
                  'bench_cell_other_diagnostics')


def exclusion_fields(excluded, prefix):
    out = {}
    for reason in EXCLUSION_REASONS:
        out[prefix + reason] = excluded.get(reason) if isinstance(excluded, dict) else None
    return out


def build_sample_sizes(cv, main_rows, chosen_by_pair, bench):
    out = []
    for pair in sorted(main_rows, key=lambda p: pair_sort_key(p, main_rows[p])):
        g, label = pair
        row, chosen = main_rows[pair], chosen_by_pair[pair]
        rec, conf = cv.primaries[g]
        missing = set((row.get('missing_quantities') or '').split(';')) - {''}
        info = group_info(g)
        scorer = bench.get((info['model'], info['execution_version'], str(info['material_set']), info['task'], label)) \
            if info['source'] == 'benchmark' and label is not None else None
        for q, prefix, family, _ in QUANTITIES:
            x = chosen.get(q)
            scope = str(row.get('cc_scope') or '')
            if q == 'native_clean' and label is not None and not scope.startswith('label') and (
                    scope.startswith('shared') or info['source'] == 'analysis'):
                continue  # the shared clean cell is listed once, on the reference row
            if x is None and q not in missing:
                continue  # not part of this run's design
            measure = get(x, 'measure') or (rec if family == 'recovery' else conf)
            s = {k: row.get(k) for k in ('source', 'model', 'model_revision', 'execution_version', 'material_set', 'split', 'task',
                                          'readout', 'kind', 'row_type', 'severity_label', 'severity_rank', 'in_shared_grid', 'inference_unit')}
            s.update(quantity=q, main_table_column=prefix, measure=measure, status='row' if x is not None else 'missing: no curves row',
                     value_source=get(x, 'value_source'), source_estimand=get(x, 'source_estimand'), n_units=get(x, 'n'),
                     planned_units=get(x, 'planned_units'), level=get(x, 'level'), planned=get(x, 'planned'), used=get(x, 'used'),
                     se_zero=get(x, 'se_zero'), n_lt_5=get(x, 'n_lt_5'))
            excluded = get(x, 'excluded')
            s.update(exclusion_fields(excluded, 'excluded_'))
            other = {k: v for k, v in excluded.items() if k not in EXCLUSION_REASONS} if isinstance(excluded, dict) else {}
            s['excluded_other'] = other or None
            derived = get(x, 'value_source') not in (None, 'copied')
            handling = None
            if x is not None and not derived:
                if info['source'] == 'analysis' and family == 'recovery' and measure == TWO_CHOICE_PRIMARY:
                    handling = 'invalid/truncated scored incorrect and kept in used; their counts: valid_only_excluded_*'
                    v = cv.pick(g, x.get('severity_label'), q, TWO_CHOICE_VALID_ONLY)
                    s.update(valid_only_n_units=get(v, 'n'), valid_only_used=get(v, 'used'))
                    s.update({'valid_only_excluded_' + k: (get(v, 'excluded') or {}).get(k) if v else None
                              for k in ('missing', 'truncated', 'invalid')})
                elif info['source'] == 'analysis':
                    handling = 'invalid/truncated excluded (excluded_*)'
                elif info['source'] == 'benchmark':
                    policy = split_measure(measure)[1]
                    handling = (POLICY_NOTES.get(policy, f'policy {policy}') + '; per-cell counts: bench_cell_* (with --benchmark-scores)')
            s['invalid_handling'] = handling
            if info['source'] == 'benchmark':
                cells = SCORER_CELLS.get(q, ())
                s['bench_cells'] = ','.join(cells) or None
                if scorer is not None:
                    policy = split_measure(measure)[1]
                    pol = (scorer.get('policies') or {}).get(policy) or {}
                    incomplete = scorer.get('incomplete_cluster_ids')
                    s.update(bench_cell_mode=scorer.get('cell_mode'), bench_planned_questions=scorer.get('planned_queries'),
                             bench_planned_clusters=scorer.get('planned_clusters'),
                             bench_questions_all_cells_observed=scorer.get('queries_with_all_cells_observed'),
                             bench_clusters_complete=scorer.get('clusters_with_all_planned_queries_observed'),
                             bench_incomplete_clusters=len(incomplete) if isinstance(incomplete, list) else None,
                             bench_valid_cell_questions_in_complete_clusters=scorer.get('valid_cell_queries_in_complete_clusters'),
                             bench_policy=policy, bench_policy_questions=pol.get('queries'), bench_policy_clusters=pol.get('clusters'))
                    if len(cells) == 1:
                        diag = (scorer.get('per_cell_diagnostics') or {}).get(cells[0])
                        if isinstance(diag, dict):
                            # the scorer counts with a Counter: an absent key of a diagnosed cell is 0
                            zero = 0 if ('missing' in diag or 'present' in diag) else None
                            reasons = {k: v for k, v in diag.items() if k.startswith('invalid_') and k != 'invalid_any'}
                            other = {k: v for k, v in diag.items() if not k.startswith('invalid_') and k not in ('present', 'missing')}
                            s.update(bench_cell_present=diag.get('present', zero), bench_cell_missing=diag.get('missing', zero),
                                     bench_cell_invalid_any=diag.get('invalid_any', zero),
                                     bench_cell_invalid_truncated=diag.get('invalid_truncated', zero),
                                     bench_cell_invalid_reasons=reasons or None, bench_cell_other_diagnostics=other or None)
            out.append({k: s.get(k) for k in SAMPLE_COLUMNS})
    return out


# ----------------------------------------------------------------------------------------------- sensitivities
SENSITIVITY_COLUMNS = ('model', 'model_revision', 'execution_version', 'material_set', 'split', 'task', 'severity_label',
                       'severity_rank', 'setting_value', 'in_shared_grid', 'inference_unit', 'measure', 'metric', 'policy', 'role',
                       'metric_note', 'policy_note', 'scorer_primary_flag', 'interval_type') + \
    tuple(quantity_columns(RECOVERY_QUANTITIES)) + ('primary_measure', 'primary_cc_minus_dd_mean', 'primary_dc_minus_dd_mean',
                                                    'primary_recovery_interaction_mean', 'missing_quantities')


def role_of(measure, primary):
    metric, policy = split_measure(measure)
    pmetric, ppolicy = split_measure(primary) if primary else (None, None)
    if measure == primary:
        return 'primary'
    if policy == ppolicy:
        return 'metric sensitivity'
    if metric == pmetric:
        return 'policy sensitivity'
    return 'metric and policy sensitivity'


def build_sensitivities(cv, main_rows):
    out = []
    for pair in sorted(main_rows, key=lambda p: pair_sort_key(p, main_rows[p])):
        g, label = pair
        info = group_info(g)
        if info['source'] != 'benchmark' or label is None:
            continue
        primary = cv.primaries[g][0]
        main = main_rows[pair]
        measures = sorted({r['measure'] for rows in cv.index[pair].values() for r in rows},
                          key=lambda m: (order_key(role_of(m, primary), ROLE_ORDER), m))
        design = {q for m in measures for q in choose_rows(cv, g, label, m, None, RECOVERY_QUANTITIES)[0]}
        for measure in measures:
            chosen, cc_scope = choose_rows(cv, g, label, measure, None, RECOVERY_QUANTITIES)
            metric, policy = split_measure(measure)
            flags = {r.get('primary_measure') for rows in cv.index[pair].values() for r in rows if r['measure'] == measure}
            row = {k: main.get(k) for k in ('model', 'model_revision', 'execution_version', 'material_set', 'split', 'task',
                                            'severity_label', 'severity_rank', 'setting_value', 'in_shared_grid', 'inference_unit')}
            intervals = sorted({x.get('interval') for x in chosen.values() if x.get('interval')})
            row.update(measure=measure, metric=metric, policy=policy, role=role_of(measure, primary),
                       metric_note=METRIC_NOTES.get(metric), policy_note=POLICY_NOTES.get(policy),
                       scorer_primary_flag=flags.pop() if len(flags) == 1 else None, interval_type='/'.join(intervals) or None)
            fill_quantities(row, chosen, RECOVERY_QUANTITIES)
            row['cc_scope'] = cc_scope
            row.update(primary_measure=primary, primary_cc_minus_dd_mean=main.get('cc_minus_dd_mean'),
                       primary_dc_minus_dd_mean=main.get('dc_minus_dd_mean'),
                       primary_recovery_interaction_mean=main.get('recovery_interaction_mean'),
                       missing_quantities=';'.join(q for q, _, _, _ in RECOVERY_QUANTITIES if q in design - set(chosen)) or None)
            out.append({k: row.get(k) for k in SENSITIVITY_COLUMNS})
    return out


# ----------------------------------------------------------------------------------------------- capability-calibrated
CAPABILITY_COLUMNS = ('estimand', 'calibration_model', 'calibration_task', 'calibrated_label', 'calibrated_value',
                      'calibrated_rank', 'calibration_r_mean', 'calibrated_target', 'interpolated_value',
                      'label_in_shared_grid', 'evaluation_status') + MAIN_COLUMNS


def build_capability(calibration, cap_rows, cv, main_rows, warn):
    entries = (calibration.get('capability_calibrated') or {}).get('entries') or []
    note = (calibration.get('capability_calibrated') or {}).get('note') or CALIBRATED_NOTE
    grid = calibration.get('severity_labels') or {}
    pairs, differing = defaultdict(set), []
    for r in cap_rows:
        pairs[(r.get('model'), r.get('task'), r.get('severity_label'))].add((group_key(r), r.get('severity_label')))
        same = [x for x in cv.index.get((group_key(r), r.get('severity_label')), {}).get(r['quantity'], [])
                if x.get('measure') == r.get('measure') and x.get('query_role') == r.get('query_role')]
        if not same or same[0].get('mean') != r.get('mean'):
            differing.append(f"{r.get('model')}/{r.get('task')}/{r.get('severity_label')}/{r['quantity']}/{r.get('measure')}")
    if differing:
        warn(f'{len(differing)} capability_calibrated rows have no identical severity_curves row (curves files from different runs?), '
             f'e.g. {differing[:3]}; capability rows are taken from severity_curves')
    out = []
    for e in sorted(entries, key=lambda e: (order_key(e.get('task'), TASK_ORDER), order_key(e.get('model'), MODEL_ORDER))):
        key = (e.get('model'), e.get('task'), e.get('label'))
        head = dict(estimand=e.get('note') or note, calibration_model=e.get('model'), calibration_task=e.get('task'),
                    calibrated_label=e.get('label'), calibrated_value=e.get('value'), calibrated_rank=e.get('rank'),
                    calibration_r_mean=e.get('r_mean'), calibrated_target=e.get('target'), interpolated_value=e.get('interpolated_value'),
                    label_in_shared_grid=e.get('label') in (grid.get(e.get('task')) or []))
        found = sorted((p for p in pairs.get(key, ()) if p in main_rows), key=lambda p: pair_sort_key(p, main_rows[p]))
        evaluated = [p for p in found if main_rows[p].get('split') != 'calibration']
        if len(evaluated) < len(found):
            warn(f'capability-calibrated {key}: {len(found) - len(evaluated)} curves group(s) from calibration-split inputs not counted as evaluation rows')
        if not evaluated:
            out.append({k: dict(head, evaluation_status='not evaluated').get(k) for k in CAPABILITY_COLUMNS})
        for p in evaluated:
            out.append({k: {**main_rows[p], **head, 'evaluation_status': 'evaluated'}.get(k) for k in CAPABILITY_COLUMNS})
    keys = {(e.get('model'), e.get('task'), e.get('label')) for e in entries}
    for key in sorted(set(pairs) - keys, key=str):
        warn(f'capability_calibrated rows for {key} have no entry in the given calibration.json')
    return out


# ----------------------------------------------------------------------------------------------- grid
def build_grid(calibration):
    shared = calibration.get('shared_grid') or {}
    medians = calibration.get('label_medians') or []
    curves = calibration.get('model_curves') or []
    warnings = calibration.get('warnings') or []
    models = set()
    for m in medians:
        models |= set((m.get('model_r') or {}).keys())
    for c in curves:
        models.add(c.get('model'))
    models = sorted((m for m in models if m is not None), key=lambda m: order_key(m, MODEL_ORDER))
    task_models = defaultdict(set)
    for c in curves:
        task_models[c.get('task')].add(c.get('model'))
    columns = ('task', 'family', 'reference_kind', 'label', 'value', 'rank', 'in_shared_grid', 'grid_position', 'grid_target',
               'median_r', 'grid_abs_error', 'n_models_in_median', 'models_in_median') + tuple('r_' + m for m in models) + (
               'models_without_r', 'task_total_abs_error', 'task_grid_models', 'task_candidates', 'task_warnings')
    by_label = {(m.get('task'), m.get('label')): m for m in medians}
    tasks = sorted({t for t, _ in by_label} | set(shared), key=lambda t: order_key(t, TASK_ORDER))
    out = []
    for task in tasks:
        grid = shared.get(task) or {}
        labels = list(grid.get('labels') or [])
        task_rows = [m for (t, _), m in by_label.items() if t == task]
        order = sorted(task_rows, key=lambda m: (m.get('rank') if isinstance(m.get('rank'), (int, float)) else 1e9, str(m.get('label'))))
        listed = [m.get('label') for m in order] + [L for L in labels if (task, L) not in by_label]
        task_warn = [w for w in warnings if str(w).startswith((task + ':', task + '/'))]
        for label in listed:
            m = by_label.get((task, label), {})
            position = labels.index(label) if label in labels else None
            pick = lambda key: (grid.get(key) or [None] * len(labels))[position] if position is not None and len(grid.get(key) or []) > position else None
            model_r = m.get('model_r') or {}
            row = dict(task=task, family=grid.get('family'), reference_kind=grid.get('reference_kind'), label=label,
                       value=m.get('value', pick('values')), rank=m.get('rank'), in_shared_grid=position is not None,
                       grid_position=None if position is None else position + 1, grid_target=pick('targets'),
                       median_r=m.get('median_r', pick('median_r')), grid_abs_error=pick('abs_error'), n_models_in_median=m.get('models'),
                       models_in_median=','.join(sorted(model_r, key=lambda x: order_key(x, MODEL_ORDER))) or None,
                       models_without_r=','.join(sorted((x for x in task_models[task] if x not in model_r),
                                                        key=lambda x: order_key(x, MODEL_ORDER))) or None,
                       task_total_abs_error=grid.get('total_abs_error'),
                       task_grid_models=','.join(grid.get('models') or []) or None,
                       task_candidates=','.join(grid.get('candidates') or []) or None,
                       task_warnings=' || '.join(task_warn) or None)
            row.update({'r_' + x: model_r.get(x) for x in models})
            out.append({k: row.get(k) for k in columns})
    return out, columns, models


# ----------------------------------------------------------------------------------------------- no-op images
NOOP_COLUMNS = ('material_set', 'task', 'benchmark', 'v15_split', 'official_split', 'mode', 'label', 'value', 'family',
                'reference_kind', 'noop_images', 'selected_images', 'selected_units', 'unit', 'noop_record', 'in_shared_grid',
                'evaluated_in_curves', 'import_metadata', 'import_metadata_sha256', 'manifest_sha256')


def read_import_metadata(paths, warn):
    docs = []
    for path in paths:
        path = Path(path)
        if path.is_dir():
            path = path / 'import_metadata.json'
        doc, digest = read_json(path)
        if not isinstance(doc, dict):
            raise ValueError(f'{path}: not a JSON object')
        if doc.get('schema') != IMPORT_SCHEMA:
            warn(f'{path}: schema {doc.get("schema")!r}, expected {IMPORT_SCHEMA}; read as far as possible')
        docs.append((path, digest, doc))
    return docs


def build_noop(docs, calibration, curves_labels, warn):
    grid = calibration.get('severity_labels') or {}
    out = []
    for path, digest, doc in sorted(docs, key=lambda d: (order_key(d[2].get('v15_split'), SPLIT_ORDER), order_key(d[2].get('task'), TASK_ORDER),
                                                         str(d[2].get('material_set')))):
        sev = doc.get('severity') if isinstance(doc.get('severity'), dict) else {}
        noop = sev.get('noop_images') if isinstance(sev.get('noop_images'), dict) else None
        if noop is None:
            warn(f'{path}: no severity.noop_images; its labels are written with NA')
            noop = {}
        task, material = doc.get('task'), doc.get('material_set')
        values = dict(zip(sev.get('labels') or [], sev.get('values') or []))
        counts = doc.get('counts') if isinstance(doc.get('counts'), dict) else {}
        selection = doc.get('selection') if isinstance(doc.get('selection'), dict) else {}
        evaluated = curves_labels.get(material, set())
        labels = list(dict.fromkeys(list(noop) + list(sev.get('labels') or []) + list(grid.get(task) or []) + sorted(evaluated)))
        for label in labels:
            recorded = label in noop
            note = 'recorded' if recorded else ('not recorded: label not imported with this material set (added later, '
                                                'e.g. by a level extension)' if label not in values else 'not recorded')
            out.append(dict(material_set=material, task=task, benchmark=doc.get('benchmark'), v15_split=doc.get('v15_split'),
                            official_split=doc.get('split'), mode=doc.get('mode'), label=label, value=values.get(label),
                            family=sev.get('family'), reference_kind=sev.get('reference_kind'),
                            noop_images=noop.get(label) if recorded else None, selected_images=counts.get('selected_images'),
                            selected_units=counts.get('selected_units'), unit=selection.get('unit'), noop_record=note,
                            in_shared_grid=label in (grid.get(task) or []), evaluated_in_curves=label in evaluated,
                            import_metadata=str(path.resolve()), import_metadata_sha256=digest, manifest_sha256=doc.get('manifest_sha256')))
    return out


# ----------------------------------------------------------------------------------------------- output
DESCRIPTIONS = {
    'source': 'analysis = synthetic two-choice (analysis.py); benchmark = free-form scorer (GQA/DocVQA)',
    'model': 'model key', 'model_revision': 'model weights revision', 'execution_version': 'run execution_version',
    'material_set': 'material set of the curves rows (split-specific)', 'split': 'calibration / evaluation (copied)',
    'task': 'task', 'readout': 'response_format/mapping', 'response_format': 'direct, label or freeform',
    'mapping': 'none, identity or cyclic', 'kind': 'score or generation',
    'row_type': 'reference = clean recovery cell and normal conflict main (no label); label = one severity label',
    'severity_label': 'severity label (NA on reference rows)',
    'severity_rank': 'curves.py rank among the task labels present in the curves inputs (1 = mildest, 0 = reference)',
    'family': 'degradation family', 'reference_kind': 'physical reference of the setting',
    'setting_value': 'physical setting of the label (chroma fraction, sigma/period or sigma/dot distance, px)',
    'in_shared_grid': 'label in calibration.json shared physical grid (copied from curves; NA on reference rows)',
    'inference_unit': 'cluster unit of n and intervals: template (synthetic), image (GQA), document (DocVQA)',
    'recovery_measure': 'measure of CC/DD/CD/DC/CC-DD/DC-DD/interaction: accuracy_invalid_as_incorrect (two-choice) or the '
                        'free-form METRIC|POLICY primary (GQA gqa_exact|invalid_as_zero; DocVQA docvqa_anls_organizer_framework|invalid_as_zero)',
    'conflict_measure': 'measure of T/C/R/dR: b_minus_a_log_odds (score) or answer_b_indicator (generation); NA for benchmarks',
    'interval_type': 'interval of the copied CIs: ci95_t (Student-t over units, analysis.py) or ci_low/ci_high (scorer normal 95%)',
    'processor_geometry': 'processor geometry recorded for model x task in --geometry inputs of the same split: '
                          '"job: geometry (count blocks); ..." (NA without --geometry or without a matching summary)',
    'cc_scope': 'which clean cell CC is: the label\'s own clean cell (benchmark: that label\'s complete clusters) or the shared '
                'label-free reference cell (two-choice)',
    'cc_minus_dd_value_source': 'copied = scorer paired estimate with interval (benchmark); derived = difference of the copied CC '
                                'and DD means without an interval (two-choice; curves.py)',
    'cc_minus_dd_source_estimand': 'scorer estimand of a copied CC-DD (joint_degradation_loss); NA when derived',
    'missing_quantities': 'quantities the run design of this model x material x readout x kind implies but without a curves row '
                          '(any recovery quantity implies CC, DD, CC-DD; any four-cell quantity all four cells and contrasts; any '
                          'conflict quantity T, C, R, dR; reference rows: CC, T, C, R); NA cells. A two-choice reference row or a '
                          'shared-grid label (non-calibration groups) without any curves row is written as an all-NA row',
    'quantity': 'curves quantity', 'main_table_column': 'column prefix of this quantity in main_table',
    'measure': 'measure of this curves row (the primary one for its quantity)',
    'status': 'row = copied from a curves row; missing = in the design but no curves row (all values NA)',
    'value_source': 'copied or derived (curves.py)', 'source_estimand': 'analysis/scorer estimand the row was copied from',
    'n_units': 'n: inference units with a value (templates/images/documents)',
    'planned_units': 'planned inference units (analysis.py)', 'level': 'queries or interventions (the planned/used level)',
    'planned': 'planned queries/interventions of this estimand (analysis.py)', 'used': 'queries/interventions used',
    'excluded_missing': 'excluded: missing observation', 'excluded_truncated': 'excluded: truncated generation',
    'excluded_invalid': 'excluded: invalid generation', 'excluded_alignment_unknown': 'excluded: clean-reference alignment unknown',
    'excluded_reference_not_a_or_b': 'excluded: clean reference neither a nor b', 'excluded_other': 'other exclusion reasons (JSON)',
    'se_zero': 'se = 0 (no t interval)', 'n_lt_5': 'fewer than 5 units',
    'invalid_handling': 'how invalid/truncated responses enter this measure',
    'valid_only_n_units': 'accuracy_valid_only counterpart: units', 'valid_only_used': 'accuracy_valid_only counterpart: queries used',
    'valid_only_excluded_missing': 'accuracy_valid_only counterpart: missing',
    'valid_only_excluded_truncated': 'truncated responses (scored incorrect by the primary, excluded by valid-only)',
    'valid_only_excluded_invalid': 'invalid responses (scored incorrect by the primary, excluded by valid-only)',
    'bench_cell_mode': 'scorer cell mode of the label (four_cell / native_only)',
    'bench_planned_questions': 'scorer planned questions', 'bench_planned_clusters': 'scorer planned clusters (images/documents)',
    'bench_questions_all_cells_observed': 'questions with every cell of the label observed',
    'bench_clusters_complete': 'clusters with all planned questions observed in every cell (enter the estimates)',
    'bench_incomplete_clusters': 'number of incomplete cluster ids listed by the scorer',
    'bench_valid_cell_questions_in_complete_clusters': 'questions whose cells are all valid, in complete clusters',
    'bench_policy': 'scorer policy of this row', 'bench_policy_questions': 'questions entering under the policy',
    'bench_policy_clusters': 'clusters entering under the policy', 'bench_cells': 'scorer cells this quantity uses',
    'bench_cell_present': 'single-cell rows: observed questions', 'bench_cell_missing': 'single-cell rows: missing questions',
    'bench_cell_invalid_any': 'single-cell rows: questions with any invalid reason (scorer Counter; absent = 0)',
    'bench_cell_invalid_truncated': 'single-cell rows: truncated generations',
    'bench_cell_invalid_reasons': 'single-cell rows: all invalid_* counts (JSON)',
    'bench_cell_other_diagnostics': 'single-cell rows: other scorer per-cell counts (JSON; e.g. gqa_case_punct_normalized_disagrees_strict, '
                                    'exact_threshold_boundary)',
    'metric': 'scorer metric', 'policy': 'scorer invalid-output policy',
    'role': 'primary / metric sensitivity (same policy) / policy sensitivity (same metric) / metric and policy sensitivity',
    'metric_note': 'what the metric is', 'policy_note': 'what the policy is',
    'scorer_primary_flag': 'curves primary_measure flag (scorer primary metric and policy)',
    'primary_measure': 'the primary measure of this model x task',
    'primary_cc_minus_dd_mean': 'primary measure CC-DD mean, copied beside the sensitivity',
    'primary_dc_minus_dd_mean': 'primary measure DC-DD mean, copied beside the sensitivity',
    'primary_recovery_interaction_mean': 'primary measure interaction mean, copied beside the sensitivity',
    'estimand': 'separately labelled estimand (calibrate.py note)',
    'calibration_model': 'calibration.json entry model', 'calibration_task': 'calibration.json entry task',
    'calibrated_label': 'label whose mean calibration r is nearest the target', 'calibrated_value': 'its physical value',
    'calibrated_rank': 'its rank among the calibration candidates', 'calibration_r_mean': 'mean calibration-split r at that label',
    'calibrated_target': 'calibrated target r', 'interpolated_value': 'linear interpolation of the physical value at the target '
                                                                       '(an untested setting; NA when not bracketed)',
    'label_in_shared_grid': 'calibrated label is also in the shared grid of the task',
    'evaluation_status': 'evaluated = evaluation rows exist in capability_calibrated (curves); not evaluated = none',
    'label': 'severity label', 'value': 'physical value of the label', 'rank': 'calibration rank (1 = mildest candidate)',
    'grid_position': 'position in the shared grid (1 = mildest)', 'grid_target': 'target r of that grid position',
    'median_r': 'cross-model median retained capability r (calibration split)', 'grid_abs_error': '|median r - target|',
    'n_models_in_median': 'models in the median', 'models_in_median': 'models with a defined mean r (native-capable)',
    'models_without_r': 'models calibrated on the task without r at this label (not native-capable or no eligible entry)',
    'task_total_abs_error': 'sum of |median r - target| over the grid', 'task_grid_models': 'models behind the task grid',
    'task_candidates': 'eligible calibration candidates of the task', 'task_warnings': 'calibrate.py task-level warnings (copied)',
    'v15_split': 'calibration or evaluation material', 'official_split': 'official benchmark split', 'mode': 'official or v14_reuse',
    'noop_images': 'source images on which the label is a lowres no-op (clean long side <= target), per importer',
    'selected_images': 'distinct source images (pages for DocVQA) in the material set',
    'selected_units': 'selected inference units (images / documents)', 'unit': 'importer unit definition',
    'noop_record': 'recorded in import_metadata.json or not', 'evaluated_in_curves': 'label has curves rows of this material set',
    'import_metadata': 'import_metadata.json path', 'import_metadata_sha256': 'its sha256', 'manifest_sha256': 'material manifest sha256',
    'material': 'material set (geometry_summary) or the run directory name without the model prefix (check_run)',
    'job': 'run directory name', 'model_family': 'qwen / granite / deepstack (geometry_summary)',
    'scope': 'task = per model x material x task; source = per image source of the run (clean a, counterfactual b, severity '
             'labels); run = whole run; record = generic input record',
    'image_source': 'image source the geometry belongs to (a, b or a severity label); NA on task-level rows',
    'field': 'geometry = processed-input geometry distribution (value = geometry text, count = blocks); other fields are '
             'copied/flattened summary fields (path; distributions give one row per distinct value with its count)',
    'field_value': 'geometry text, distinct value or field value (copied)', 'count': 'occurrences (blocks for geometry distributions)',
    'note': 'non-numeric distribution entry (e.g. "N more distinct values")', 'count_unit': 'what count counts',
    'check_run_passed': 'check_run technical gate passed', 'format': 'check_run / geometry_summary / summary / generic',
    'run_dir': 'run output directory', 'manifest': 'materials manifest of the run (check_run)',
    'input': 'geometry input file', 'input_sha256': 'its sha256',
}
for _q, _prefix, _family, _meaning in QUANTITIES:
    DESCRIPTIONS.setdefault(_prefix + '_mean', f'{_meaning}: mean (copied)')
    DESCRIPTIONS.setdefault(_prefix + '_ci_low', f'{_meaning}: 95% interval low (copied; NA when se = 0, n < 2 or no interval)')
    DESCRIPTIONS.setdefault(_prefix + '_ci_high', f'{_meaning}: 95% interval high (copied)')
    DESCRIPTIONS.setdefault(_prefix + '_n', f'{_meaning}: n inference units (copied; NA when derived)')


def csv_value(value):
    if value is None:
        return NA
    if isinstance(value, (dict, list)):
        return json.dumps(value, sort_keys=True, ensure_ascii=False)
    return str(value)


def current_umask():
    mask = os.umask(0o022)
    os.umask(mask)
    return mask


def atomic_write(path, text):
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), prefix='.' + path.name + '.', suffix='.tmp')
    try:
        with os.fdopen(fd, 'w', newline='', encoding='utf-8') as f:
            f.write(text)
        os.chmod(tmp, 0o666 & ~current_umask())  # mkstemp creates 0600; keep the usual mode for packing
        os.replace(tmp, path)
    except BaseException:
        if os.path.exists(tmp):
            os.unlink(tmp)
        raise


def write_table(out, name, rows, columns, description, meta):
    buffer = io.StringIO(newline='')
    writer = csv.DictWriter(buffer, fieldnames=list(columns), extrasaction='raise', lineterminator='\n')
    writer.writeheader()
    for r in rows:
        writer.writerow({k: csv_value(r.get(k)) for k in columns})
    atomic_write(out / f'{name}.csv', buffer.getvalue())
    doc = dict(schema=SCHEMA, table=name, description=description, columns=list(columns),
               column_descriptions={k: DESCRIPTIONS.get(k, '') for k in columns}, **meta, rows=rows)
    atomic_write(out / f'{name}.json', json.dumps(doc, indent=1, ensure_ascii=False, allow_nan=False) + '\n')


TABLE_DESCRIPTIONS = {
    'main_table': 'Per model x material x task x readout x kind x severity label (plus a reference row): CC, DD, CD, DC, actual native '
                  'loss CC-DD, recovery DC-DD, recovery interaction, conflict T/C/R and dR(L) under the primary measures, with copied '
                  'intervals and n; in_shared_grid marks the shared physical grid.',
    'sample_sizes': 'Per main-table row and quantity: inference unit and n, planned/used denominators, exclusions by reason '
                    '(analysis.py excluded field), invalid/truncated counts of the invalid-as-incorrect measures (valid-only '
                    'counterpart) and, with --benchmark-scores, scorer per-label/per-cell counts.',
    'sensitivities': 'Benchmark rows: every metric x policy of the scorer beside the primary (GQA strict exact; DocVQA organiser ANLS; '
                     'invalid_as_zero), on CC, DD, CD, DC, CC-DD (paired), DC-DD and the interaction.',
    'capability_calibrated': 'Every calibration.json capability-calibrated entry (model x task, label, r) with its evaluation rows '
                             '(main-table columns) or one "not evaluated" row. ' + CALIBRATED_NOTE + '.',
    'noop_images': 'Lowres no-op source images per label from each import_metadata.json (severity.noop_images), per material set; '
                   'grid or evaluated labels without a record are listed with NA.',
    'grid': 'calibration.json candidates per task with the shared grid (position, target, median r, per-model r) and task warnings.',
    'geometry': 'Processor geometry summaries per model x material x task (and per source label), copied/flattened from --geometry.',
}


def markdown_index(out, tables, inputs, warnings, conventions):
    cell = lambda text: str(text).replace('|', '\\|')
    quantity_cols = {p + s for _, p, _, _ in QUANTITIES for s in ('_mean', '_ci_low', '_ci_high', '_n')}
    lines = ['# v15 final tables', '', f'Written by `scripts/final_tables_v15.py` (schema `{SCHEMA}`). Every value, interval and '
             'denominator is copied from the inputs below; nothing is re-estimated. CSV `NA` = JSON `null` (missing row or no value). '
             'Each JSON holds the same rows plus columns, column_descriptions, inputs, conventions and warnings.', '',
             '## Inputs', '']
    lines += [f"- {x['role']}: `{x['path']}` sha256 `{x['sha256']}`" for x in inputs]
    lines += ['', '## Conventions', ''] + [f'- {c}' for c in conventions]
    lines += ['', '## Quantity columns', '', 'main_table, capability_calibrated and sensitivities hold, per quantity prefix Q, '
              '`Q_mean`, `Q_ci_low`, `Q_ci_high` (copied 95% interval; NA when se = 0, n < 2 or derived) and `Q_n` (inference units).', '',
              '| prefix | quantity (curves) | meaning |', '|---|---|---|']
    lines += [f'| `{p}` | `{q}` | {cell(meaning)} |' for q, p, _, meaning in QUANTITIES]
    for name, (rows, columns) in tables.items():
        lines += ['', f'## {name} ({len(rows)} rows)', '', f'`{name}.csv`, `{name}.json`. {cell(TABLE_DESCRIPTIONS[name])}', '']
        shown = [c for c in columns if c not in quantity_cols]
        if name == 'capability_calibrated':
            shown = [c for c in shown if c not in MAIN_COLUMNS]
            lines += ['Then every main_table column, for the calibrated label (NA on "not evaluated" rows).', '']
        prefixes = [] if name == 'capability_calibrated' else [p for _, p, _, _ in QUANTITIES if p + '_mean' in columns]
        if prefixes:
            lines += ['Quantity columns: ' + ', '.join(f'`{p}_*`' for p in prefixes) + '.', '']
        lines += ['| column | meaning |', '|---|---|'] + [f"| `{c}` | {cell(DESCRIPTIONS.get(c, ''))} |" for c in shown]
    lines += ['', f'## Warnings ({len(warnings)})', ''] + ([f'- {w}' for w in warnings] or ['- none'])
    atomic_write(out / 'tables_index.md', '\n'.join(lines) + '\n')


CONVENTIONS = [
    'Primary measures: two-choice recovery cells and contrasts `accuracy_invalid_as_incorrect`; conflict T/C/R/dR '
    '`b_minus_a_log_odds` (score) or `answer_b_indicator` (generation); GQA strict exact (`gqa_exact|invalid_as_zero`); DocVQA '
    'organiser ANLS (`docvqa_anls_organizer_framework|invalid_as_zero`). Other measures, metrics and policies stay in '
    'severity_curves.csv; benchmark metric/policy sensitivities are in sensitivities.',
    'Actual native loss CC-DD: two-choice (synthetic) = difference of the copied CC and DD means without an interval (curves.py, '
    'value_source derived); benchmark = the scorer\'s paired estimate with its interval (value_source copied).',
    'Two-choice CC is the shared label-free clean cell; benchmark CC is each label\'s own clean cell over that label\'s complete clusters.',
    'Intervals are descriptive 95% intervals clustered by template/image/document (ci95_t: Student-t; scorer: normal), without '
    'multiplicity correction; an interaction is never read as "one significant, one not".',
    'Shared physical grid (in_shared_grid) and capability-calibrated labels are different estimands; the latter is a model-specific severity.',
    'Benchmarks are selected official-validation subsets, not leaderboard scores; calibration-v14reuse benchmark units are named reuse, not replication.',
    'Same physical image is not the same processed input: see processor_geometry / geometry.',
    'Reference rows (row_type reference) hold the clean recovery cell and the conflict effects at the normal main; dR is defined on label rows only.',
]


# ----------------------------------------------------------------------------------------------- main
def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--curves', type=Path, required=True, help='curves.py output directory')
    p.add_argument('--calibration', type=Path, required=True, help='calibrate.py output (calibration.json)')
    p.add_argument('--import-metadata', type=Path, nargs='+', required=True, help='benchmark import_metadata.json files (or their directories)')
    p.add_argument('--geometry', type=Path, nargs='*', default=[], help='processor geometry summaries (check_run.py outputs or similar JSON)')
    p.add_argument('--benchmark-scores', type=Path, nargs='*', default=[], help='scorer JSONs for benchmark per-label/per-cell counts (optional)')
    p.add_argument('--output', type=Path, required=True, help='output directory (not the curves directory)')
    a = p.parse_args(argv)
    if not a.curves.is_dir():
        p.error(f'--curves {a.curves}: not a directory')
    out = a.output
    if out.resolve() == a.curves.resolve():
        p.error('--output must differ from --curves (curves.py capability_calibrated.csv/.json would be replaced)')
    existing = out / 'capability_calibrated.json'
    if existing.exists():
        try:
            if json.loads(existing.read_bytes()).get('schema') == CURVES_SCHEMA:
                p.error(f'{existing} is a curves.py output; choose another --output')
        except (ValueError, AttributeError):
            pass
    warn = Warnings()
    rows, header, curves_prov = read_curves_file(a.curves, 'severity_curves', warn, required=True)
    cap_rows, cap_header, cap_prov = read_curves_file(a.curves, 'capability_calibrated', warn, required=False)
    calibration, calibration_sha = read_calibration(a.calibration)
    inputs = [dict(role='curves', **curves_prov)]
    if cap_prov:
        inputs.append(dict(role='curves capability_calibrated', **cap_prov))
    inputs.append(dict(role='calibration', path=str(a.calibration.resolve()), sha256=calibration_sha))
    used = [x for x in header.get('inputs') or [] if x.get('kind') == 'calibration']
    if header and not used:
        warn('severity_curves.json was written without --calibration: in_shared_grid is NA throughout')
    for x in used:
        if x.get('sha256') != calibration_sha:
            warn(f"curves were made with calibration {x.get('path')} (sha256 {str(x.get('sha256'))[:16]}), not the given "
                 f"{a.calibration} ({calibration_sha[:16]}): in_shared_grid follows the curves' calibration")
    if header.get('shared_grid') not in (None, calibration.get('severity_labels')):
        warn('shared grid recorded in severity_curves.json differs from calibration.json severity_labels')
    for x in header.get('inputs') or []:
        if x.get('kind') in ('analysis', 'benchmark'):
            inputs.append(dict(role=f"curves input ({x['kind']}, {x.get('records')} records)", path=x.get('path'), sha256=x.get('sha256')))
    if cap_rows is None:
        warn('capability_calibrated.json/.csv absent from --curves: entries matched on severity_curves rows with the same (model, task, label)')
        entries = (calibration.get('capability_calibrated') or {}).get('entries') or []
        picks = {(e.get('model'), e.get('task'), e.get('label')) for e in entries}
        cap_rows = [r for r in rows if (r.get('model'), r.get('task'), r.get('severity_label')) in picks]
    splits = sorted({str(r.get('split')) for r in rows})
    if any(s != 'evaluation' for s in splits):
        warn(f'curves rows include splits {splits}; tables keep every row and show its split')

    bench, bench_prov = read_benchmark_scores(a.benchmark_scores, warn)
    inputs += bench_prov
    if not a.benchmark_scores and any(r.get('source') == 'benchmark' for r in rows):
        warn('benchmark rows without --benchmark-scores: sample_sizes bench_* columns (planned questions, per-cell invalid/truncated) are NA')
    geometry_rows = []
    seen_dirs = {}
    for path in a.geometry:
        doc, digest = read_json(path)
        inputs.append(dict(role='geometry', path=str(path.resolve()), sha256=digest))
        found = geometry_records(doc, path, digest, warn)
        for key in {(r['run_dir'], r['format']) for r in found if r.get('run_dir')}:
            if key in seen_dirs:
                warn(f'{key[1]} geometry of {key[0]} given twice ({seen_dirs[key]}, {path}); the later file is used')
                geometry_rows = [r for r in geometry_rows if (r.get('run_dir'), r.get('format')) != key]
            seen_dirs[key] = str(path)
        if not found:
            warn(f'{path}: no geometry rows recognised')
        geometry_rows += found
    label_tasks = defaultdict(set)  # label -> tasks (curves rows, calibration candidates): task of by_source rows of multi-task runs
    for r in rows:
        if r.get('severity_label') is not None:
            label_tasks[r['severity_label']].add(r.get('task'))
    for m in calibration.get('label_medians') or []:
        label_tasks[m.get('label')].add(m.get('task'))
    for r in geometry_rows:
        if r.get('scope') == 'source' and r.get('task') is None and len(label_tasks.get(r.get('image_source'), ())) == 1:
            r['task'] = next(iter(label_tasks[r['image_source']]))
    geometry_rows.sort(key=lambda r: (order_key(r.get('model'), MODEL_ORDER), str(r.get('material')), str(r.get('job')),
                                      order_key(r.get('scope'), ('task', 'source', 'run', 'record')), order_key(r.get('task'), TASK_ORDER),
                                      str(r.get('image_source')), str(r.get('field')),
                                      -(r['count'] if isinstance(r.get('count'), (int, float)) else -1)))
    imports = read_import_metadata(a.import_metadata, warn)
    inputs += [dict(role='import_metadata', path=str(path.resolve()), sha256=digest) for path, digest, _ in imports]

    cv = CurvesIndex(rows, warn)
    lookup = geometry_lookup(geometry_rows) if a.geometry else None
    main_rows, chosen = build_main(cv, lookup, calibration, warn)
    ordered = sorted(main_rows, key=lambda pair: pair_sort_key(pair, main_rows[pair]))
    main_table = [{k: main_rows[pair].get(k) for k in MAIN_COLUMNS} for pair in ordered]
    for row in main_table:
        if row['missing_quantities']:
            warn(f"main_table {row['model']}/{row['task']}/{row['readout']}/{row['kind']} {label_text(row['severity_label'])}: "
                 f"no curves row for {row['missing_quantities']} (NA)")
    sample_sizes = build_sample_sizes(cv, main_rows, chosen, bench)
    sensitivities = build_sensitivities(cv, main_rows)
    capability = build_capability(calibration, cap_rows, cv, main_rows, warn)
    grid_rows, grid_columns, _ = build_grid(calibration)
    curves_labels = defaultdict(set)
    for r in rows:
        if r.get('severity_label') is not None:
            curves_labels[r.get('material_set')].add(r['severity_label'])
    noop = build_noop(imports, calibration, curves_labels, warn)
    geometry_table = [{k: r.get(k) for k in GEOMETRY_COLUMNS} for r in geometry_rows]

    out.mkdir(parents=True, exist_ok=True)
    meta = dict(generated_by=dict(script='scripts/final_tables_v15.py', script_sha256=sha256_bytes(Path(__file__).read_bytes())),
                inputs=inputs, conventions=CONVENTIONS, warnings=list(warn))
    tables = {'main_table': (main_table, MAIN_COLUMNS), 'sample_sizes': (sample_sizes, SAMPLE_COLUMNS),
              'sensitivities': (sensitivities, SENSITIVITY_COLUMNS), 'capability_calibrated': (capability, CAPABILITY_COLUMNS),
              'noop_images': (noop, NOOP_COLUMNS), 'grid': (grid_rows, grid_columns), 'geometry': (geometry_table, GEOMETRY_COLUMNS)}
    for name, (table_rows, columns) in tables.items():
        write_table(out, name, table_rows, columns, TABLE_DESCRIPTIONS[name], meta)
    markdown_index(out, tables, inputs, list(warn), CONVENTIONS)
    for w in warn:
        print('WARNING: ' + w, file=sys.stderr)
    print(json.dumps(dict(output=str(out), rows={name: len(t[0]) for name, t in tables.items()}, warnings=len(warn))))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
