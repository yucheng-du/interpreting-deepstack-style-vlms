#!/usr/bin/env python3
"""Processor geometry each model actually used, from one v15 runner output directory (stdlib only).

Usage:  geometry_summary_v15.py <run_dir> [--output <json>] [--examples N]

Writes ONE JSON object: to --output (atomic replace; a one-line summary then goes to stdout) or to stdout.
--output may not point inside <run_dir> (run directories stay read-only).
Exit 0 whenever the summary was produced (violations are counted in the JSON, never an exit status);
exit 2 on a usage error or a crash. Read-only on the run directory. System python3 3.10, no third-party
modules or GPU required.

WHERE THE NUMBERS COME FROM
  observations.jsonl  streamed physical line by line (0.5-1 GB is fine: measured 10-14 s per GB, <30 MB RSS); only
      compact per-block state is kept. Every row carries visual_metadata.geometry, the processor geometry
      that cloud_prep/measurement_v15/adapter.py FixedLaterComposer recorded for its block. A block is one
      (unit_id, intervention_id): the runner encodes every source the block's conditions read (clean 'a',
      counterfactual 'b', each severity label) once, and the composer raises unless all of them have the
      same geometry, so one geometry per block is expected; every deviation is counted here. Only the
      observations cover every block, so all distributions are built from them (one count per block).
      A broken last line (job still writing) is skipped and reported.
  audits.jsonl        written once per task and run (audit_per_task), i.e. for a few queries only. Where an
      audit carries geometry the observation does not, it is reported as recorded: Qwen
      qwen_geometry_record (per-source image_grid_thw and image-placeholder counts of the processor
      outputs), DeepStack native_geometry_audit (processed_views tensor shape, visual_tokens_per_entry,
      zero-padded later groups), Granite native_geometry_audit (full anyres pack). Every audit geometry
      (also the 'geometry' of the native/self-patch records) is compared with the observation geometry of
      the same query and run.
  run-*.json          model, the image processor actually used (Qwen patch_size/merge_size/pixel limits),
      config source names (clean source, counterfactual source).

PER MODEL FAMILY (detected from the recorded geometry; sizes are 'WxH' unless a field ends in _hw = 'HxW',
Granite's own convention; "processed" = the highest-resolution view the processor makes of the image)
  qwen       grid_thw (patch units), merged grid, visual tokens = sum(t*h*w) / merge_size^2 (the adapter
             raises unless the prompt's image placeholders equal it; evaluated rows' delivered
             row_visual_tokens are checked against it), processed_wh = (grid_w, grid_h) * patch_size.
  granite    LlavaNext anyres pack as recorded: tile_grid_hw, num_views (base + tiles), packed_tokens
             (base + tile + newline tokens), image_hw, resized_hw (the image inside the anyres canvas),
             canvas_hw; processed = resized_hw; evaluated rows' delivered per-row geometries are checked.
  deepstack  native grid gw x gh (CLIP-336 crops by the official resolution rule), image size, active
             later groups = gw*gh; processed views = 1 global 336x336 squash + gw*gh tiles of the
             (gw*336)x(gh*336) squash, i.e. (1+gw*gh)x3x336x336 (derived; DeepStackCore.prepare raises
             unless the official processor returns exactly that shape; audits record it), processed =
             (gw*336)x(gh*336); the LLM sequence holds 576 visual positions (one expanded <image>), the
             later groups are added to those same positions, so llm_visual_tokens is 576 for every image.

OUTPUT JSON  (distributions are {value text: count}, most common first, values beyond the top 12 summarised
under '...'; stats are {n, min, median, max} or null; counts are blocks unless named otherwise)
{
 tool, run_dir, job (basename of run_dir), host, generated_at, elapsed_s,
 inputs: {
   runs: [{file, run_id, status, model, model_family, model_revision, execution_version, recovery_cells,
           started_at, finished_at, clean_source, counterfactual_source, qwen_pixel_limits,
           image_processor (compact actual_image_processor), qwen_patch_merge ([patch, merge] used, or
           null; defaulted_patch_merge true if the run did not record them and 16/2 were assumed)}],
   observations: {present, lines, rows_used, rows_skipped, malformed_lines, broken_last_line,
                  no_final_newline, examples},
   audits: {present, lines, records: {type: count}, unattributed (records whose query has no observation
            row yet), malformed_lines, broken_last_line, no_final_newline, examples}},
 groups: [ one per model x material_set x task:
   {model, family, model_revision, material_set, split, task, execution_versions: {v: rows},
    run_ids: {run_id: rows},
    counts: {blocks, queries, units, images (distinct clean-source sha256), rows},
    geometry: {                     # one entry per block: its first recorded geometry
       distinct: {geometry text: blocks}, n_distinct,
       llm_visual_tokens: {n, min, median, max, distinct, basis},
       processed_wh: {distinct}, processed_long_side: {stats},
       qwen:      grid_thw, merged_grid_hw                              {distinct}
       granite:   tile_grid_hw, num_views, packed_tokens, base_tokens, tile_visual_tokens, newline_tokens,
                  retained_padding_tokens, image_hw, resized_hw, canvas_hw  {distinct}
       deepstack: native_grid_wh, processed_views, active_later_groups, image_wh  {distinct}},
    image_size: {wh: {distinct}, long_side: {stats}, basis},   # clean-source file width x height
    sources: {<source name>: {      # order: clean, counterfactual, severity labels mild -> severe, other
       role ('clean' | 'counterfactual' | 'severity' | 'other'), severity_label, blocks, rows (rows that
       read the source as main, later base or later donor), geometry: {distinct}, geometry_n_distinct,
       grid: {distinct},
       views: {distinct}, llm_visual_tokens: {stats}, processed_wh: {distinct}, processed_long_side: {stats},
       image_wh: {distinct}, image_long_side: {stats}, blocks_differing_from_clean,
       severity (labelled sources only): {family, value, reference_kind: {distinct}, blocks_with_spec,
                 noop_blocks, realized: {numeric key: {stats}}, effective_long_side: {stats},
                 effective_vs_processed: {blocks, effective_ge_processed_both_axes,
                                          effective_long_over_processed_long: {stats}}},
       audit (Qwen per-source records, audited queries only): {queries, geometry: {distinct}}}},
    audits: {records: {type: count}, queries, recorded: {field: {distinct}}},
    consistency: {
       identical_across_sources   true iff violating_blocks == 0
       blocks_checked, violating_blocks (blocks with any of the four cross-source violations),
       violations: {               counts
         geometry_differs_within_block      blocks whose rows of one run carry >1 recorded geometry
         source_geometry_differs_from_clean (block, source) pairs, inside a run whose rows of the block record
                                            >1 geometry, whose geometry set != the clean source's (same run;
                                            the block's clean set if the clean cells came from another run)
         image_size_differs_across_sources  blocks whose used source files differ in width x height
         audit_per_source_differs           Qwen audit records: a per-source grid/count != the query's
         audit_differs_from_observation     audit geometries != observation geometry (same query and run)
         geometry_image_size_differs_from_source_metadata  recorded processor input size != file size
         delivered_differs                  evaluated rows whose delivered tokens/geometry != the cell's
         geometry_differs_across_runs       blocks whose geometry changed between runs (resume) without a
                                            within-block difference
         geometry_missing_rows              rows without a recognisable geometry},
       checked: {delivered_rows, audit_records_compared},
       other_issues (only when present): {block_in_several_groups, source_size_differs_across_runs,
                                          used_source_without_metadata: count},
       examples: {category: [messages]}}}],
 table: [ flat rows, one per model x material_set x task x source, for CSV/report use:
   {model, family, material_set, split, task, source, role, severity_label, blocks, rows,
    geometry_distinct, geometry_mode, geometry_mode_blocks, grid_mode, views_mode,
    llm_visual_tokens_min, llm_visual_tokens_median, llm_visual_tokens_max, processed_wh_mode,
    processed_long_side_median, image_wh_mode, image_long_side_min, image_long_side_median,
    image_long_side_max, effective_long_side_median, noop_blocks, effective_ge_processed_blocks,
    blocks_differing_from_clean}],
 totals: {groups, blocks, rows, violating_blocks, violations: {category: count},
          identical_across_sources (all groups)},
 problems: [input problems, e.g. unreadable run JSON or a missing observations file],
 notes: [...]
}

Parsing conventions (physical-line JSONL reader, geometry text, distribution tails) are adapted from
the checks performed during measurement.
"""
import argparse
import glob
import json
import os
import socket
import statistics
import sys
import time
import traceback
from collections import Counter, defaultdict
from datetime import datetime, timezone

TOOL_VERSION = 'v15-geometry-summary-1'
TOP = 12                      # distinct values kept per distribution
DEFAULT_EXAMPLES = 5          # example messages kept per issue category
MESSAGE_CHARS = 400
DEEPSTACK_VIEW_PX = 336       # CLIP crop size (DeepStack-L, asserted by the adapter)
DEEPSTACK_SEQUENCE_TOKENS = 576
QWEN_DEFAULT_PATCH, QWEN_DEFAULT_MERGE = 16, 2
CROSS_SOURCE = ('geometry_differs_within_block', 'source_geometry_differs_from_clean',
                'image_size_differs_across_sources', 'audit_per_source_differs')
CATEGORIES = CROSS_SOURCE + ('audit_differs_from_observation', 'geometry_image_size_differs_from_source_metadata',
                             'delivered_differs', 'geometry_differs_across_runs', 'geometry_missing_rows')
MORE_SEVERE = {'desaturate': 'lower', 'blur_rel': 'higher', 'lowres': 'lower', 'contrast': 'lower'}  # severity.FAMILIES
GRANITE_FIELDS = ('tile_grid_hw', 'num_views', 'packed_tokens', 'base_tokens', 'tile_visual_tokens', 'newline_tokens',
                  'retained_padding_tokens', 'image_hw', 'resized_hw', 'canvas_hw')
PROCESSOR_KEYS = ('image_processor_type', 'patch_size', 'temporal_patch_size', 'merge_size', 'min_pixels',
                  'max_pixels', 'size', 'crop_size', 'do_pad', 'do_center_crop', 'resample')
TOKEN_BASIS = {
    'qwen': 'sum(t*h*w)/merge_size^2 of grid_thw = image placeholders in the prompt',
    'granite': 'packed_tokens as recorded (base + tile + newline tokens)',
    'deepstack': '576 expanded positions of the single <image> placeholder (adapter constant); later groups '
                 'are added to the same positions',
}


# ---- small helpers ------------------------------------------------------------------------------------------
def is_number(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def wh(values):
    """'AxB...' text of a list of numbers (JSON text for anything else)."""
    if isinstance(values, (list, tuple)) and values and all(is_number(v) for v in values):
        return 'x'.join(str(v) for v in values)
    return json.dumps(values, sort_keys=True)


def stats(values):
    values = [v for v in values if is_number(v)]
    if not values:
        return None
    median = statistics.median(values)
    if isinstance(median, float) and median.is_integer():
        median = int(median)
    return dict(n=len(values), min=min(values), median=median, max=max(values))


def dist(counter, top=TOP):
    """Distinct values with counts, most common first; the tail beyond ``top`` is summarised."""
    items = sorted(counter.items(), key=lambda kv: (-kv[1], str(kv[0])))
    out = {str(k): v for k, v in items[:top]}
    if len(items) > top:
        rest = items[top:]
        out['...'] = f'{len(rest)} more distinct values ({sum(v for _, v in rest)} occurrences)'
    return out


def mode(counter):
    if not counter:
        return None, 0
    return sorted(counter.items(), key=lambda kv: (-kv[1], str(kv[0])))[0]


class Issues:
    """Counted issue categories with the first few example messages each."""

    def __init__(self, limit):
        self.count, self.examples, self.limit = Counter(), defaultdict(list), limit

    def add(self, category, message, n=1):
        self.count[category] += n
        kept = self.examples[category]
        if len(kept) < self.limit:
            kept.append(message if len(message) <= MESSAGE_CHARS else message[:MESSAGE_CHARS] + ' ...')


class JsonlReader:
    """Physical-line JSONL (split on b'\\n' only, like the runner and check_run.py). Malformed lines are
    collected with their number; a broken last line (a job still writing, or a crash tail) is skipped."""

    def __init__(self, path):
        self.path, self.lines, self.bad, self.final_newline = path, 0, [], True

    def __iter__(self):
        with open(self.path, 'rb') as handle:
            for raw in handle:
                self.lines += 1
                self.final_newline = raw.endswith(b'\n')
                if not raw.strip():
                    self.bad.append((self.lines, 'blank line'))
                    continue
                try:
                    value = json.loads(raw)
                except ValueError as exc:  # JSONDecodeError and UnicodeDecodeError
                    self.bad.append((self.lines, f'{type(exc).__name__}: {str(exc)[:100]}'))
                    continue
                if not isinstance(value, dict):
                    self.bad.append((self.lines, 'not a JSON object'))
                    continue
                yield self.lines, value

    def summary(self, limit):
        tail = any(n == self.lines for n, _ in self.bad)
        return dict(lines=self.lines, malformed_lines=len(self.bad) - int(tail), broken_last_line=tail,
                    no_final_newline=bool(self.lines) and not self.final_newline,
                    malformed_examples=[f'line {n}: {why}' for n, why in self.bad[:limit]])


# ---- run metadata -----------------------------------------------------------------------------------------
def compact_processor(processor):
    out = {k: processor[k] for k in PROCESSOR_KEYS if k in processor}
    pinpoints = processor.get('image_grid_pinpoints')
    if isinstance(pinpoints, list) and pinpoints:
        sizes = [p for p in pinpoints if isinstance(p, (list, tuple)) and len(p) == 2 and all(is_number(v) for v in p)]
        out['image_grid_pinpoints'] = dict(count=len(pinpoints),
                                           max_area_hw=wh(max(sizes, key=lambda p: p[0] * p[1])) if sizes else None)
    return out


def load_runs(directory, problems):
    runs = {}
    for path in sorted(glob.glob(os.path.join(directory, 'run-*.json'))):
        name = os.path.basename(path)
        try:
            with open(path) as handle:
                raw = json.load(handle)
            if not isinstance(raw, dict):
                raise ValueError('not a JSON object')
        except (OSError, ValueError) as exc:
            problems.append(f'run metadata unreadable: {name}: {exc}')
            continue
        run_id = raw.get('run_id') or name[len('run-'):-len('.json')]
        model = raw.get('model') if isinstance(raw.get('model'), dict) else {}
        config = raw.get('config') if isinstance(raw.get('config'), dict) else {}
        processor = raw.get('actual_image_processor') if isinstance(raw.get('actual_image_processor'), dict) else {}
        recovery = config.get('recovery_sources') if isinstance(config.get('recovery_sources'), dict) else {}
        families = config.get('families') if isinstance(config.get('families'), list) else []
        patch, merge = processor.get('patch_size'), processor.get('merge_size')
        qwen = model.get('family') == 'qwen' or 'merge_size' in processor
        defaulted = qwen and not (is_number(patch) and is_number(merge))
        runs[run_id] = dict(
            file=name, run_id=run_id, status=raw.get('status'), model=model.get('key'),
            model_family=model.get('family'), model_revision=model.get('revision'),
            execution_version=config.get('execution_version'), recovery_cells=config.get('recovery_cells'),
            started_at=raw.get('started_at'), finished_at=raw.get('finished_at'),
            clean_source=recovery.get('clean') or config.get('later_base') or 'a',
            counterfactual_source=config.get('later_donor') if 'conflict' in families else None,
            qwen_pixel_limits=config.get('qwen_pixel_limits'), image_processor=compact_processor(processor),
            qwen_patch_merge=[int(patch) if is_number(patch) else QWEN_DEFAULT_PATCH,
                              int(merge) if is_number(merge) else QWEN_DEFAULT_MERGE] if qwen else None,
            defaulted_patch_merge=defaulted)
    return runs


def settings_for(runs, run_id):
    """(patch, merge) used to turn a Qwen grid into tokens and pixels for rows of this run."""
    run = runs.get(run_id) or {}
    pm = run.get('qwen_patch_merge')
    return (pm[0], pm[1]) if pm else (QWEN_DEFAULT_PATCH, QWEN_DEFAULT_MERGE)


def clean_source_for(runs, run_id):
    return (runs.get(run_id) or {}).get('clean_source') or 'a'


def counterfactual_for(runs, run_id):
    return (runs.get(run_id) or {}).get('counterfactual_source')


# ---- geometry ----------------------------------------------------------------------------------------------
def family_of(geometry):
    if isinstance(geometry, dict):
        if isinstance(geometry.get('grids'), list):
            return 'qwen'
        if isinstance(geometry.get('packed_geometry'), list):
            return 'granite'
        if isinstance(geometry.get('image_size'), list) and isinstance(geometry.get('grid'), list):
            return 'deepstack'
    return None


def features(geometry, settings):
    """Family-specific summary of one recorded geometry, or None when it is not recognised."""
    family = family_of(geometry)
    try:
        if family == 'qwen':
            patch, merge = settings
            grids = [[int(t), int(h), int(w)] for t, h, w in geometry['grids']]
            if not grids:
                return None
            tokens = sum(t * h * w for t, h, w in grids) // (merge * merge)
            processed = [[w * patch, h * patch] for _, h, w in grids]
            grid_text = ';'.join(wh(g) for g in grids)
            processed_text = ';'.join(wh(p) for p in processed)
            return dict(family=family, text=f'grid_thw={grid_text} tokens={tokens} processed_wh={processed_text}',
                        tokens=tokens, processed=processed[0] if len(grids) == 1 else None,
                        processed_text=processed_text, processed_long=max(max(p) for p in processed),
                        image_wh=None, grid=grid_text, views=None,
                        fields=dict(grid_thw=grid_text,
                                    merged_grid_hw=';'.join(f'{h // merge}x{w // merge}' for _, h, w in grids)))
        if family == 'granite':
            packs = geometry['packed_geometry']
            if not packs or not all(isinstance(p, dict) for p in packs):
                return None

            def join(key):
                return ';'.join(wh(p.get(key)) if isinstance(p.get(key), list) else str(p.get(key)) for p in packs)
            tokens = sum(int(p['packed_tokens']) for p in packs)
            processed = [[int(p['resized_hw'][1]), int(p['resized_hw'][0])] for p in packs]
            image_wh = [int(packs[0]['image_hw'][1]), int(packs[0]['image_hw'][0])] if len(packs) == 1 else None
            text = (f"tile_grid_hw={join('tile_grid_hw')} views={join('num_views')} tokens={tokens} "
                    f"image_hw={join('image_hw')} resized_hw={join('resized_hw')} canvas_hw={join('canvas_hw')}")
            return dict(family=family, text=text, tokens=tokens, processed=processed[0] if len(packs) == 1 else None,
                        processed_text=';'.join(wh(p) for p in processed), processed_long=max(max(p) for p in processed),
                        image_wh=image_wh, grid=join('tile_grid_hw'), views=join('num_views'),
                        fields={k: join(k) for k in GRANITE_FIELDS})
        if family == 'deepstack':
            gw, gh = (int(v) for v in geometry['grid'])
            w, h = (int(v) for v in geometry['image_size'])
            views = f'{1 + gw * gh}x3x{DEEPSTACK_VIEW_PX}x{DEEPSTACK_VIEW_PX}'
            processed = [gw * DEEPSTACK_VIEW_PX, gh * DEEPSTACK_VIEW_PX]
            return dict(family=family, text=f'grid_wh={gw}x{gh} views={views} image_wh={w}x{h}',
                        tokens=DEEPSTACK_SEQUENCE_TOKENS, processed=processed, processed_text=wh(processed),
                        processed_long=max(processed), image_wh=[w, h], grid=f'{gw}x{gh}', views=views,
                        fields=dict(native_grid_wh=f'{gw}x{gh}', processed_views=views,
                                    active_later_groups=str(geometry.get('active_later_groups')), image_wh=f'{w}x{h}'))
    except (TypeError, ValueError, KeyError, IndexError, ZeroDivisionError):
        return None
    return None


def compact_spec(spec):
    if not isinstance(spec, dict):
        return None
    realized = spec.get('realized') if isinstance(spec.get('realized'), dict) else {}
    effective = realized.get('effective_size')
    if not (isinstance(effective, list) and len(effective) == 2 and all(is_number(v) for v in effective)):
        effective = None
    return dict(family=spec.get('family'), value=spec.get('value'), reference_kind=spec.get('reference_kind'),
                noop=realized.get('noop'), effective=effective,
                realized={k: v for k, v in realized.items() if is_number(v)})


def compact_source_metadata(meta):
    """source name -> (width, height, sha256, role, severity_label)."""
    out = {}
    if isinstance(meta, dict):
        for name, item in meta.items():
            if isinstance(item, dict):
                out[name] = (item.get('width'), item.get('height'), item.get('sha256'), item.get('role'),
                             item.get('severity_label'))
    return out


# ---- the summary ---------------------------------------------------------------------------------------------
class Summary:
    def __init__(self, directory, examples):
        self.d = os.path.abspath(directory)
        self.examples = examples
        self.problems = []
        self.blocks = {}              # (unit_id, intervention_id) -> block state
        self.groups = {}              # (model, material_set, task) -> group state
        self.query_obs = {}           # (run_id, query id) -> (block key, geometry index)
        self.query_any = {}           # query id -> (block key, geometry index), any run
        self.fcache = {}              # (block key, geometry index, settings) -> features

    def run(self):
        began = time.monotonic()
        self.runs = load_runs(self.d, self.problems)
        if not self.runs:
            self.problems.append('no readable run-*.json (Qwen patch/merge default to 16/2; clean source to "a")')
        self.obs_input = self.scan_observations()
        self.audit_input = self.scan_audits()
        groups = [self.summarise_group(key) for key in sorted(self.groups, key=lambda k: tuple(map(str, k)))]
        return self.output(groups, time.monotonic() - began)

    # ---- observations ----------------------------------------------------------------------------------------
    def group(self, key):
        group = self.groups.get(key)
        if group is None:
            group = self.groups[key] = dict(
                blocks=[], rows=0, versions=Counter(), run_ids=Counter(), splits=Counter(), revisions=Counter(),
                issues=Issues(self.examples), violating=set(), delivered_rows=0, audit_compared=0,
                audit_records=Counter(), audit_queries=set(), audit_recorded=defaultdict(Counter),
                audit_source=defaultdict(Counter), audit_source_queries=defaultdict(set))
        return group

    def feature(self, bkey, gi, run_id):
        settings = settings_for(self.runs, run_id)
        key = (bkey, gi, settings)
        value = self.fcache.get(key, False)
        if value is False:
            value = self.fcache[key] = features(self.blocks[bkey]['geoms'][gi], settings)
        return value

    def scan_observations(self):
        path = os.path.join(self.d, 'observations.jsonl')
        info = dict(present=os.path.exists(path), rows_used=0, rows_skipped=0, examples=[], lines=0, malformed_lines=0,
                    broken_last_line=False, no_final_newline=False, malformed_examples=[])
        if not info['present']:
            self.problems.append('observations.jsonl not found: no per-block geometry (audits alone cover only a '
                                 'few queries per task)')
            return info
        reader = JsonlReader(path)
        blocks, query_obs, query_any = self.blocks, self.query_obs, self.query_any
        for number, o in reader:
            cond, vm = o.get('condition'), o.get('visual_metadata')
            qid, unit, iid, run_id = o.get('id'), o.get('unit_id'), o.get('intervention_id'), o.get('run_id')
            if not isinstance(cond, dict) or not isinstance(vm, dict) or not isinstance(qid, str) \
                    or unit is None or iid is None:
                info['rows_skipped'] += 1
                if len(info['examples']) < self.examples:
                    info['examples'].append(f'line {number}: id/unit_id/intervention_id/condition/visual_metadata '
                                            'missing or invalid')
                continue
            info['rows_used'] += 1
            run_id = str(run_id)
            model = o.get('model_key') or (self.runs.get(run_id) or {}).get('model')
            gkey = (str(model), str(o.get('material_set')), str(o.get('task')))
            group = self.group(gkey)
            group['rows'] += 1
            group['versions'][str(o.get('execution_version'))] += 1
            group['run_ids'][run_id] += 1
            group['splits'][str(o.get('split'))] += 1
            group['revisions'][str(o.get('model_revision'))] += 1
            bkey = (str(unit), str(iid))
            block = blocks.get(bkey)
            if block is None:
                block = blocks[bkey] = dict(group=gkey, geoms=[], last=0, by_run=defaultdict(set),
                                            src=defaultdict(set), src_run=defaultdict(set), src_rows=Counter(),
                                            rows=0, qids=set(),
                                            meta=compact_source_metadata(o.get('source_metadata')), meta_runs={run_id},
                                            labels={}, specs={}, line=number, first_run=run_id)
                group['blocks'].append(bkey)
            else:
                if block['group'] != gkey:
                    group['issues'].add('block_in_several_groups', f'line {number}: block {bkey[1]} also in {block["group"]}')
                listed = o.get('source_metadata')
                if run_id not in block['meta_runs'] or (isinstance(listed, dict) and not block['meta'].keys() >= listed.keys()):
                    # a continuation run (e.g. added severity levels from an extended manifest) may list more sources
                    block['meta_runs'].add(run_id)
                    for name, item in compact_source_metadata(listed).items():
                        known = block['meta'].setdefault(name, item)
                        if known[:2] != item[:2]:
                            group['issues'].add('source_size_differs_across_runs',
                                                f'line {number}: block {bkey[1]} source {name} is {item[0]}x{item[1]} in run '
                                                f'{run_id} but {known[0]}x{known[1]} in an earlier run')
            geometry = vm.get('geometry')
            geoms = block['geoms']
            if geoms and geoms[block['last']] == geometry:
                gi = block['last']
            else:
                for gi, known in enumerate(geoms):
                    if known == geometry:
                        break
                else:
                    geoms.append(geometry)
                    gi = len(geoms) - 1
                block['last'] = gi
            block['by_run'][run_id].add(gi)
            block['rows'] += 1
            block['qids'].add(qid)
            for source in {cond.get('main_source'), cond.get('later_base'), cond.get('later_donor')}:
                if source is not None:
                    block['src'][source].add(gi)
                    block['src_run'][(run_id, source)].add(gi)
                    block['src_rows'][source] += 1
            query_obs.setdefault((run_id, qid), (bkey, gi))
            query_any.setdefault(qid, (bkey, gi))
            label = cond.get('severity_label')
            if label is not None:
                if cond.get('main_quality') == label:
                    block['labels'].setdefault(cond.get('main_source'), label)
                if cond.get('later_quality') == label:
                    block['labels'].setdefault(cond.get('later_base'), label)
                if label not in block['specs']:
                    block['specs'][label] = compact_spec(o.get('severity_spec'))
            feat = self.feature(bkey, gi, run_id)
            where = f'line {number} {qid} {cond.get("name")} {o.get("kind")}'
            if feat is None:
                group['issues'].add('geometry_missing_rows', f'{where}: geometry {json.dumps(geometry)[:160]} not recognised')
            elif o.get('evaluation_reused') is False:
                self.check_delivery(o, geometry, feat, group, where)
        info.update(reader.summary(self.examples))
        if info['broken_last_line']:
            self.problems.append(f'observations.jsonl last line {reader.lines} is incomplete (job still writing?); '
                                 'skipped')
        return info

    @staticmethod
    def check_delivery(o, geometry, feat, group, where):
        """Physically evaluated rows: the visual the batch actually delivered has the cell's geometry."""
        result = o.get('result') if isinstance(o.get('result'), dict) else {}
        delivery = result.get('delivery') if isinstance(result.get('delivery'), dict) else {}
        if 'reused_from_observation_id' in delivery:
            return
        checked = False
        tokens = delivery.get('row_visual_tokens')                      # Qwen: placeholders per batch row
        if tokens is not None and feat['family'] == 'qwen':
            checked = True
            if not isinstance(tokens, list) or not tokens or any(t != feat['tokens'] for t in tokens):
                group['issues'].add('delivered_differs', f'{where}: row_visual_tokens {str(tokens)[:80]} != '
                                    f'{feat["tokens"]} from grid_thw {feat["grid"]}')
        row = delivery.get('row')                                       # Granite: per-row delivered geometry
        if isinstance(row, dict) and 'geometries' in row and feat['family'] == 'granite':
            checked = True
            if row['geometries'] != geometry.get('packed_geometry'):
                group['issues'].add('delivered_differs', f'{where}: delivered geometries differ from the cell geometry')
        group['delivered_rows'] += checked

    # ---- audits -------------------------------------------------------------------------------------------------
    def scan_audits(self):
        path = os.path.join(self.d, 'audits.jsonl')
        info = dict(present=os.path.exists(path), records=Counter(), unattributed=0, examples=[], lines=0,
                    malformed_lines=0, broken_last_line=False, no_final_newline=False, malformed_examples=[])
        if not info['present']:
            return info
        reader = JsonlReader(path)
        for number, rec in reader:
            run_id, qid = str(rec.get('run_id')), rec.get('id')
            kinds = [k for k in ('qwen_geometry_record', 'native_geometry_audit') if k in rec]
            if 'geometry' in rec and ('native_self_differences' in rec or 'native_self_generation' in rec):
                kinds.append('selfpatch_geometry')
            for kind in kinds:
                info['records'][kind] += 1
            if not kinds:
                continue
            hit, other_run = self.query_obs.get((run_id, qid)), False
            if hit is None:
                hit = self.query_any.get(qid)
                other_run = hit is not None
            if hit is None:
                info['unattributed'] += 1
                if len(info['examples']) < self.examples:
                    info['examples'].append(f'audits line {number}: {qid} (run {run_id}) has no observation row yet')
                continue
            bkey, gi = hit
            block = self.blocks[bkey]
            group = self.groups[block['group']]
            observed = block['geoms'][gi]
            feat = self.feature(bkey, gi, run_id)
            where = f'audits line {number} {qid} run {run_id}' + (' (rows written by another run)' if other_run else '')
            group['audit_queries'].add(qid)
            for kind in kinds:
                group['audit_records'][kind] += 1
            if 'qwen_geometry_record' in rec:
                self.audit_qwen(rec['qwen_geometry_record'], feat, observed, bkey, qid, group, where)
            if 'native_geometry_audit' in rec:
                self.audit_native(rec['native_geometry_audit'], feat, observed, group, where)
            if 'selfpatch_geometry' in kinds:
                group['audit_compared'] += 1
                if rec['geometry'] != observed:
                    group['issues'].add('audit_differs_from_observation',
                                        f'{where}: self-patch audit geometry {json.dumps(rec["geometry"])[:160]} != '
                                        f'observation {feat["text"] if feat else json.dumps(observed)[:160]}')
        info.update(reader.summary(self.examples))
        return info

    def audit_qwen(self, record, feat, observed, bkey, qid, group, where):
        issues = group['issues']
        if not isinstance(record, dict):
            issues.add('audit_differs_from_observation', f'{where}: qwen_geometry_record is not a mapping')
            return
        grid, tokens = record.get('grid_thw'), record.get('visual_tokens')
        grid_text = ';'.join(wh(g) for g in grid) if isinstance(grid, list) else wh(grid)
        recorded = group['audit_recorded']
        recorded['grid_thw'][grid_text] += 1
        recorded['visual_tokens'][str(tokens)] += 1
        recorded['merged_grid_hw'][wh(record.get('merged_grid_hw'))] += 1
        if record.get('canvas_wh') is not None:
            recorded['canvas_wh'][wh(record.get('canvas_wh'))] += 1
        differing = []
        if record.get('image_token_count') != tokens:
            differing.append(f'image_token_count {record.get("image_token_count")} != visual_tokens {tokens}')
        per_source = record.get('per_source') if isinstance(record.get('per_source'), dict) else {}
        if not per_source:
            differing.append('no per_source entries')
        for source, item in sorted(per_source.items()):
            item = item if isinstance(item, dict) else {}
            source_grid = item.get('image_grid_thw')
            text = (f"grid_thw={';'.join(wh(g) for g in source_grid) if isinstance(source_grid, list) else wh(source_grid)}"
                    f" tokens={item.get('image_tokens')}")
            group['audit_source'][source][text] += 1
            group['audit_source_queries'][source].add(qid)
            if source_grid != grid or item.get('image_tokens') != tokens:
                differing.append(f'{source}: {text}')
        if differing:
            group['issues'].add('audit_per_source_differs', f'{where}: query grid {grid_text}/{tokens} tokens; ' + '; '.join(differing))
            group['violating'].add(bkey)
        group['audit_compared'] += 1
        if family_of(observed) != 'qwen' or grid != observed.get('grids') or feat is None or tokens != feat['tokens']:
            issues.add('audit_differs_from_observation',
                       f'{where}: qwen_geometry_record grid {grid_text} / {tokens} tokens != observation '
                       f'{feat["text"] if feat else json.dumps(observed)[:160]}')

    def audit_native(self, record, feat, observed, group, where):
        issues, recorded = group['issues'], group['audit_recorded']
        if not isinstance(record, dict):
            issues.add('audit_differs_from_observation', f'{where}: native_geometry_audit is not a mapping')
            return
        group['audit_compared'] += 1
        if 'native_grid' in record and 'image_size' in record:                 # DeepStack
            views = record.get('processed_views')
            views_text = ';'.join(wh(v) for v in views) if isinstance(views, list) else wh(views)
            recorded['processed_views'][views_text] += 1
            recorded['native_grid_wh'][wh(record.get('native_grid'))] += 1
            recorded['image_wh'][wh(record.get('image_size'))] += 1
            recorded['visual_tokens_per_entry'][str(record.get('visual_tokens_per_entry'))] += 1
            recorded['active_later_groups'][str(record.get('active_later_groups'))] += 1
            recorded['zero_padded_later_groups'][json.dumps(record.get('zero_padded_later_groups'))] += 1
            same = (family_of(observed) == 'deepstack' and feat is not None
                    and record.get('native_grid') == observed.get('grid')
                    and record.get('image_size') == observed.get('image_size')
                    and record.get('active_later_groups') in (None, observed.get('active_later_groups'))
                    and views_text == feat['views']
                    and record.get('visual_tokens_per_entry') in (None, feat['tokens']))
            if not same:
                issues.add('audit_differs_from_observation',
                           f'{where}: DeepStack audit grid {wh(record.get("native_grid"))} image {wh(record.get("image_size"))} '
                           f'views {views_text} tokens/entry {record.get("visual_tokens_per_entry")} != observation '
                           f'{feat["text"] if feat else json.dumps(observed)[:160]}')
        elif isinstance(record.get('geometry'), dict):                         # Granite
            pack = features({'packed_geometry': [record['geometry']]}, (QWEN_DEFAULT_PATCH, QWEN_DEFAULT_MERGE))
            recorded['granite_pack'][pack['text'] if pack else json.dumps(record['geometry'])[:160]] += 1
            if family_of(observed) != 'granite' or observed.get('packed_geometry') != [record['geometry']]:
                issues.add('audit_differs_from_observation',
                           f'{where}: Granite audit pack {pack["text"] if pack else "?"} != observation '
                           f'{feat["text"] if feat else json.dumps(observed)[:160]}')
        else:
            issues.add('audit_differs_from_observation', f'{where}: native_geometry_audit without a recognised geometry')

    # ---- per-group summary ----------------------------------------------------------------------------------------
    def summarise_group(self, key):
        model, material_set, task = key
        group = self.groups[key]
        issues = group['issues']
        families = Counter()
        distinct, fields = Counter(), defaultdict(Counter)
        tokens, processed, processed_long = [], Counter(), []
        image_wh, image_long = Counter(), []
        images, units, queries = set(), set(), set()
        sources = {}
        for bkey in group['blocks']:
            block = self.blocks[bkey]
            run_id = block['first_run']
            clean = clean_source_for(self.runs, run_id)
            feats = [self.feature(bkey, gi, run_id) for gi in range(len(block['geoms']))]
            primary = feats[0]
            units.add(bkey[0])
            queries.update(block['qids'])
            meta = block['meta']
            clean_meta = meta.get(clean)
            if clean_meta and clean_meta[2]:
                images.add(clean_meta[2])
            if clean_meta and is_number(clean_meta[0]) and is_number(clean_meta[1]):
                image_wh[f'{clean_meta[0]}x{clean_meta[1]}'] += 1
                image_long.append(max(clean_meta[0], clean_meta[1]))
            # distributions (one entry per block: its first recorded geometry)
            if primary is not None:
                families[primary['family']] += 1
                distinct[primary['text']] += 1
                for field, value in primary['fields'].items():
                    fields[field][value] += 1
                tokens.append(primary['tokens'])
                processed[primary['processed_text']] += 1
                processed_long.append(primary['processed_long'])
            else:
                distinct['unrecognised'] += 1
            # consistency
            where = f'block {bkey[1]} (unit {bkey[0]}, first line {block["line"]})'
            violating = False
            if any(len(found) > 1 for found in block['by_run'].values()):
                texts = sorted({f['text'] if f else 'unrecognised' for f in feats})
                issues.add('geometry_differs_within_block', f'{where}: rows of one run record {len(texts)} geometries {texts[:3]}')
                violating = True
            elif len(block['geoms']) > 1:
                issues.add('geometry_differs_across_runs', f'{where}: runs {sorted(block["by_run"])} record different geometries')
            # Sources are attributed only inside a run that itself records >1 geometry for the block (the runner
            # shares one composer geometry among all sources of a block within a run): compared with the clean
            # source's set in that run, or with the block's clean set when the clean cells were written by
            # another run (continuation runs that only add severity levels). A change between runs is reported
            # as geometry_differs_across_runs, not as a source difference.
            differing = set()
            for run, found_run in sorted(block['by_run'].items()):
                if len(found_run) < 2:
                    continue
                reference = block['src_run'].get((run, clean)) or block['src'].get(clean)
                for (other, source), found in sorted(block['src_run'].items()):
                    if other == run and reference is not None and found != reference and source not in differing:
                        differing.add(source)
                        issues.add('source_geometry_differs_from_clean',
                                   f'{where}: run {run} source {source} geometries {sorted(found)} != clean {clean} '
                                   f'{sorted(reference)}')
                        violating = True
            missing = sorted(s for s in block['src'] if s not in meta)
            if missing:
                issues.add('used_source_without_metadata', f'{where}: sources {missing} have no source_metadata')
            sizes = {(meta[s][0], meta[s][1]) for s in block['src'] if s in meta}
            if len(sizes) > 1:
                issues.add('image_size_differs_across_sources',
                           f'{where}: used sources have sizes ' + ', '.join(sorted(f'{s}={meta[s][0]}x{meta[s][1]}' for s in block['src'] if s in meta)))
                violating = True
            if primary is not None and primary['image_wh'] is not None and clean_meta \
                    and [clean_meta[0], clean_meta[1]] != primary['image_wh']:
                issues.add('geometry_image_size_differs_from_source_metadata',
                           f'{where}: processor input {wh(primary["image_wh"])} != clean file {clean_meta[0]}x{clean_meta[1]}')
            if violating:
                group['violating'].add(bkey)
            # per source
            for source, found in block['src'].items():
                acc = sources.get(source)
                if acc is None:
                    acc = sources[source] = dict(blocks=0, rows=0, geometry=Counter(), grid=Counter(), views=Counter(),
                                                 tokens=[], processed=Counter(), processed_long=[], image=Counter(),
                                                 image_long=[], differs=0, roles=Counter(), labels=Counter(),
                                                 spec_family=Counter(), spec_value=Counter(), spec_reference=Counter(),
                                                 spec_blocks=0, noop=0, realized=defaultdict(list), effective_long=[],
                                                 compared=0, effective_ge=0, ratio=[])
                acc['blocks'] += 1
                acc['rows'] += block['src_rows'][source]
                for gi in sorted(found):
                    f = feats[gi]
                    acc['geometry'][f['text'] if f else 'unrecognised'] += 1
                    if f:
                        acc['grid'][f['grid']] += 1
                        if f['views'] is not None:
                            acc['views'][f['views']] += 1
                first = feats[min(found)]
                if first:
                    acc['tokens'].append(first['tokens'])
                    acc['processed'][first['processed_text']] += 1
                    acc['processed_long'].append(first['processed_long'])
                if source in differing:
                    acc['differs'] += 1
                m = meta.get(source)
                if m and is_number(m[0]) and is_number(m[1]):
                    acc['image'][f'{m[0]}x{m[1]}'] += 1
                    acc['image_long'].append(max(m[0], m[1]))
                label = block['labels'].get(source) or (m[4] if m else None)
                if label is not None:
                    role = 'severity'
                elif source == clean or (m and m[3] == 'clean'):
                    role = 'clean'
                elif source == counterfactual_for(self.runs, run_id):
                    role = 'counterfactual'
                elif m and m[3] == 'degraded':
                    role = 'severity'
                else:
                    role = 'other'
                acc['roles'][role] += 1
                if label is not None:
                    acc['labels'][label] += 1
                    spec = block['specs'].get(label)
                    if spec:
                        acc['spec_blocks'] += 1
                        acc['spec_family'][str(spec['family'])] += 1
                        value = spec['value']
                        acc['spec_value'][value if value is None or isinstance(value, (str, int, float)) else json.dumps(value)] += 1
                        acc['spec_reference'][str(spec['reference_kind'])] += 1
                        acc['noop'] += spec['noop'] is True
                        for k, v in spec['realized'].items():
                            acc['realized'][k].append(v)
                        effective = spec['effective']
                        if effective:
                            acc['effective_long'].append(max(effective))
                            if first and first['processed'] and max(first['processed']) > 0:
                                acc['compared'] += 1
                                acc['effective_ge'] += effective[0] >= first['processed'][0] and effective[1] >= first['processed'][1]
                                acc['ratio'].append(round(max(effective) / max(first['processed']), 4))
        family = mode(families)[0]
        geometry = dict(distinct=dist(distinct), n_distinct=len(distinct),
                        llm_visual_tokens=dict(**(stats(tokens) or {}), distinct=dist(Counter(tokens)),
                                               basis=TOKEN_BASIS.get(family)),
                        processed_wh=dist(processed), processed_long_side=stats(processed_long))
        for field in sorted(fields):
            geometry[field] = dist(fields[field])
        source_out = {}
        order = {'clean': 0, 'counterfactual': 1, 'severity': 2, 'other': 3}

        def source_order(name):
            """clean, counterfactual, then severity labels from mild to severe (family direction), then others."""
            acc = sources[name]
            role, value, family = mode(acc['roles'])[0], mode(acc['spec_value'])[0], mode(acc['spec_family'])[0]
            if role == 'severity' and is_number(value) and family in MORE_SEVERE:
                return (order[role], 0, value if MORE_SEVERE[family] == 'higher' else -value, name)
            return (order.get(role, 9), 1, 0, name)
        for source in sorted(sources, key=source_order):
            acc = sources[source]
            role = mode(acc['roles'])[0]
            item = dict(role=role if len(acc['roles']) == 1 else dict(acc['roles']),
                        severity_label=mode(acc['labels'])[0] if len(acc['labels']) <= 1 else dict(acc['labels']),
                        blocks=acc['blocks'], rows=acc['rows'], geometry=dist(acc['geometry']),
                        geometry_n_distinct=len(acc['geometry']), grid=dist(acc['grid']),
                        views=dist(acc['views']), llm_visual_tokens=stats(acc['tokens']),
                        processed_wh=dist(acc['processed']), processed_long_side=stats(acc['processed_long']),
                        image_wh=dist(acc['image']), image_long_side=stats(acc['image_long']),
                        blocks_differing_from_clean=acc['differs'])
            if acc['labels']:
                item['severity'] = dict(
                    family=mode(acc['spec_family'])[0] if len(acc['spec_family']) <= 1 else dict(acc['spec_family']),
                    value=mode(acc['spec_value'])[0] if len(acc['spec_value']) <= 1 else dict(acc['spec_value']),
                    reference_kind=dist(acc['spec_reference']), blocks_with_spec=acc['spec_blocks'],
                    noop_blocks=acc['noop'], realized={k: stats(v) for k, v in sorted(acc['realized'].items())},
                    effective_long_side=stats(acc['effective_long']),
                    effective_vs_processed=dict(blocks=acc['compared'], effective_ge_processed_both_axes=acc['effective_ge'],
                                                effective_long_over_processed_long=stats(acc['ratio'])))
            if source in group['audit_source']:
                item['audit'] = dict(queries=len(group['audit_source_queries'][source]),
                                     geometry=dist(group['audit_source'][source]))
            source_out[source] = item
        violations = {c: issues.count.get(c, 0) for c in CATEGORIES}
        consistency = dict(identical_across_sources=not group['violating'], blocks_checked=len(group['blocks']),
                           violating_blocks=len(group['violating']), violations=violations,
                           checked=dict(delivered_rows=group['delivered_rows'], audit_records_compared=group['audit_compared']),
                           examples={c: list(v) for c, v in issues.examples.items() if v})
        other = {c: n for c, n in issues.count.items() if c not in CATEGORIES}
        if other:
            consistency['other_issues'] = other
        return dict(model=model, family=family, model_revision=mode(group['revisions'])[0] if len(group['revisions']) == 1 else dict(group['revisions']),
                    material_set=material_set, split=mode(group['splits'])[0] if len(group['splits']) == 1 else dict(group['splits']),
                    task=task, execution_versions=dict(group['versions']), run_ids=dict(group['run_ids']),
                    counts=dict(blocks=len(group['blocks']), queries=len(queries), units=len(units), images=len(images),
                                rows=group['rows']),
                    geometry=geometry,
                    image_size=dict(wh=dist(image_wh), long_side=stats(image_long),
                                    basis='clean-source file width x height from source_metadata; every used source is '
                                          'compared (image_size_differs_across_sources)'),
                    sources=source_out,
                    audits=dict(records=dict(group['audit_records']), queries=len(group['audit_queries']),
                                recorded={k: dist(v) for k, v in sorted(group['audit_recorded'].items())}),
                    consistency=consistency)

    # ---- output ----------------------------------------------------------------------------------------------------
    @staticmethod
    def table_rows(group):
        def first(distribution):
            """Most common value of a dist() mapping (its first key), with its count."""
            for key, count in distribution.items():
                return key, count
            return None, 0
        rows = []
        for source, item in group['sources'].items():
            geometry_mode, geometry_mode_blocks = first(item['geometry'])
            tokens = item['llm_visual_tokens'] or {}
            image_long = item['image_long_side'] or {}
            severity = item.get('severity') or {}
            rows.append(dict(
                model=group['model'], family=group['family'], material_set=group['material_set'], split=group['split'],
                task=group['task'], source=source, role=item['role'], severity_label=item['severity_label'],
                blocks=item['blocks'], rows=item['rows'], geometry_distinct=item['geometry_n_distinct'],
                geometry_mode=geometry_mode, geometry_mode_blocks=geometry_mode_blocks,
                grid_mode=first(item['grid'])[0], views_mode=first(item['views'])[0],
                llm_visual_tokens_min=tokens.get('min'), llm_visual_tokens_median=tokens.get('median'),
                llm_visual_tokens_max=tokens.get('max'), processed_wh_mode=first(item['processed_wh'])[0],
                processed_long_side_median=(item['processed_long_side'] or {}).get('median'),
                image_wh_mode=first(item['image_wh'])[0], image_long_side_min=image_long.get('min'),
                image_long_side_median=image_long.get('median'), image_long_side_max=image_long.get('max'),
                effective_long_side_median=(severity.get('effective_long_side') or {}).get('median'),
                noop_blocks=severity.get('noop_blocks'),
                effective_ge_processed_blocks=(severity.get('effective_vs_processed') or {}).get('effective_ge_processed_both_axes'),
                blocks_differing_from_clean=item['blocks_differing_from_clean']))
        return rows

    def output(self, groups, elapsed):
        totals = dict(groups=len(groups), blocks=sum(g['counts']['blocks'] for g in groups),
                      rows=sum(g['counts']['rows'] for g in groups),
                      violating_blocks=sum(g['consistency']['violating_blocks'] for g in groups),
                      violations={c: sum(g['consistency']['violations'][c] for g in groups) for c in CATEGORIES},
                      identical_across_sources=all(g['consistency']['identical_across_sources'] for g in groups) if groups else None)
        audits = dict(self.audit_input)
        audits['records'] = dict(audits.get('records') or {})
        runs = [{k: v for k, v in run.items()} for run in sorted(self.runs.values(), key=lambda r: (str(r.get('started_at')), r['file']))]
        notes = [
            'Distributions count blocks: one (unit_id, intervention_id) = one encoded source set; queries of a block '
            'share its geometry. counts.images = distinct clean-source files.',
            'The runner records one geometry per block (FixedLaterComposer raises unless every source the block reads '
            'has the same processed geometry); per-source entries attribute it to every source a row reads (main, '
            'later base, later donor). Qwen audit records add per-source processor outputs for audited queries.',
            'Derived, not recorded per row: Qwen tokens and processed_wh (from grid_thw, patch_size, merge_size), '
            'DeepStack processed views (1+gw*gh)x3x336x336 and processed (gw*336)x(gh*336), DeepStack 576 LLM '
            'visual positions. Granite values are recorded.',
            'severity.effective_vs_processed compares the lowres effective size (severity_spec.realized.effective_size, '
            'the downsampled size before the bicubic upsample back to the canvas) with the processed size per axis: '
            'a processor-geometry expectation only, not a measured effect; the realized damage is CC-DD.',
            'Same physical image is not the same processed input: each model resizes, tiles or pads it differently.',
        ]
        return dict(tool=TOOL_VERSION, run_dir=self.d, job=os.path.basename(self.d.rstrip('/')),
                    host=socket.gethostname(), generated_at=datetime.now(timezone.utc).isoformat(timespec='seconds'),
                    elapsed_s=round(elapsed, 1),
                    inputs=dict(runs=runs, observations=self.obs_input, audits=audits),
                    groups=groups, table=[row for g in groups for row in self.table_rows(g)], totals=totals,
                    problems=self.problems, notes=notes)


def write_json(path, value):
    path = os.path.abspath(path)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    temporary = f'{path}.tmp-{os.getpid()}'
    with open(temporary, 'w') as handle:
        json.dump(value, handle, indent=1, ensure_ascii=False, sort_keys=False)
        handle.write('\n')
    os.replace(temporary, path)


def main(argv=None):
    parser = argparse.ArgumentParser(description='Processor geometry each model actually used in one v15 run directory.')
    parser.add_argument('run_dir')
    parser.add_argument('--output', help='write the JSON here (atomic); default: stdout')
    parser.add_argument('--examples', type=int, default=DEFAULT_EXAMPLES, help='example messages kept per category')
    args = parser.parse_args(argv)
    if not os.path.isdir(args.run_dir):
        sys.stderr.write(f'usage error: run directory not found: {args.run_dir}\n')
        return 2
    if args.output:
        run_dir, output = os.path.realpath(args.run_dir), os.path.realpath(args.output)
        if os.path.commonpath([run_dir, output]) == run_dir:
            sys.stderr.write(f'usage error: --output {args.output} is inside the run directory (kept read-only)\n')
            return 2
    try:
        result = Summary(args.run_dir, max(0, args.examples)).run()
    except Exception:  # crash: traceback on stderr, exit 2
        traceback.print_exc(file=sys.stderr)
        return 2
    if args.output:
        write_json(args.output, result)
        t = result['totals']
        print(json.dumps(dict(output=os.path.abspath(args.output), job=result['job'], groups=t['groups'],
                              blocks=t['blocks'], rows=t['rows'], violating_blocks=t['violating_blocks'],
                              identical_across_sources=t['identical_across_sources'],
                              violations={c: n for c, n in t['violations'].items() if n},
                              problems=len(result['problems']), elapsed_s=result['elapsed_s'])))
    else:
        sys.stdout.write(json.dumps(result, indent=1, ensure_ascii=False) + '\n')
    return 0


if __name__ == '__main__':
    sys.exit(main())
