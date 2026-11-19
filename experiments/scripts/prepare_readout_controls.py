#!/usr/bin/env python3
"""Prepare a fixed, answer-blind BLINK readout-control sample (stdlib only)."""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import os
import random
import re
import sys

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from deepstack.data import BLINK_TASKS, load_manifest, validate_record

DATA_ROOT = Path(os.environ.get("DS_DATA_ROOT", "work")).expanduser().resolve()
ROOT = DATA_ROOT / 'runs/readout-controls-v1'
ROOT_V2 = DATA_ROOT / 'runs/readout-controls-v2'
FULL = DATA_ROOT / 'data/blink/val_seed20260926_all.jsonl'
PRIOR = DATA_ROOT / 'data/blink/val_seed20260926_n8.jsonl'
FOCUS = ('Functional_Correspondence', 'Object_Localization')
SEED = 20260928
SCHEMA = 'qwen-readout-controls-v1'
SCHEMA_V2 = 'qwen-readout-controls-v2'
VARIANTS = ('identity', 'cyclic')
PIXELS = {'low': 262144, 'high': 1048576}
PARSER_VERSION = 'explicit_unambiguous_answer_v1'


def campaign_schema(root):
    resolved = Path(root).resolve()
    if resolved == ROOT:
        return SCHEMA
    if resolved == ROOT_V2:
        return SCHEMA_V2
    raise ValueError('Use an explicitly versioned readout-controls-v1 or readout-controls-v2 campaign')


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def canonical_digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                     separators=(',', ':')).encode()).hexdigest()


def option_suffix(record):
    """Find a contiguous final option block; preserve all preceding bytes.

    Choice prose may differ in case/whitespace from the source choices (IQ_Test).
    We retain the actual prompt's choice prose, never reconstruct its question.
    Unsupported suffixes fail visibly rather than excluding an inconvenient item.
    """
    text, labels = record['prompt'], record['option_labels']
    matches = list(re.finditer(r'(?m)^(\([A-E]\))([ \t]+)([^\r\n]*)(\r?\n|$)', text))
    if len(matches) < len(labels):
        raise ValueError(f'{record["id"]}: cannot locate final option block')
    matches = matches[-len(labels):]
    normalize = lambda x: ' '.join(x.casefold().split())
    if ([m.group(1) for m in matches] != labels
            or any(a.end() != b.start() for a, b in zip(matches, matches[1:]))
            or matches[-1].end() != len(text)
            or any(normalize(m.group(3)) != normalize(choice)
                   for m, choice in zip(matches, record['choices']))):
        raise ValueError(f'{record["id"]}: option suffix is noncontiguous or differs from choices')
    return text[:matches[0].start()], matches


def variant(record, name):
    if name not in VARIANTS:
        raise ValueError(f'Unknown prompt variant: {name}')
    prefix, matches = option_suffix(record)
    labels = record['option_labels']
    order = list(range(len(labels)))
    if name == 'cyclic':
        order = order[1:] + order[:1]
    # A fixed left rotation: displayed A receives original B's choice, etc.
    suffix = ''.join(m.group(1) + m.group(2) + matches[source].group(3) + m.group(4)
                     for m, source in zip(matches, order))
    prompt = prefix + suffix
    if name == 'identity' and prompt != record['prompt']:
        raise ValueError('Identity prompt must preserve every byte')
    mapping = {labels[i]: labels[source] for i, source in enumerate(order)}
    inverse = {original: current for current, original in mapping.items()}
    return {'name': name, 'prompt': prompt, 'prompt_sha256': hashlib.sha256(prompt.encode()).hexdigest(),
            'prefix_sha256': hashlib.sha256(prefix.encode()).hexdigest(), 'option_suffix_start': len(prefix),
            'source_choice_indices': order, 'display_choices': [matches[i].group(3) for i in order],
            'label_to_original': mapping, 'original_to_label': inverse,
            'answer_label': inverse[record['answer_label']], 'original_answer_label': record['answer_label']}


def control_record(record, *, prior_ids=()):
    # Keep the authoritative source row intact, including image order and markers.
    result = dict(record)
    result['readout_control'] = {'schema': SCHEMA, 'selection_seed': SEED,
                                 'in_exploratory112': record['id'] in prior_ids,
                                 'variants': {name: variant(record, name) for name in VARIANTS}}
    return result


def validate_control(record, *, check_files=True):
    validate_record(record, check_files=check_files)
    control = record['readout_control']
    if control['schema'] != SCHEMA or control['selection_seed'] != SEED:
        raise ValueError('Unexpected readout-control identity')
    if set(control['variants']) != set(VARIANTS):
        raise ValueError('Both identity and cyclic conditions are required in the prepared row')
    for name in VARIANTS:
        if control['variants'][name] != variant(record, name):
            raise ValueError(f'{record["id"]}: altered prompt or answer mapping')


def read_controls(path, *, check_files=True):
    rows = [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]
    if not rows or len({r['id'] for r in rows}) != len(rows):
        raise ValueError('Empty or duplicate-ID control manifest')
    for row in rows:
        validate_control(row, check_files=check_files)
    return rows


def select_records(full, prior_ids, *, seed=SEED):
    if seed != SEED:
        raise ValueError('This named exploration uses seed20260928')
    pools = {}
    for task in BLINK_TASKS:
        candidates = sorted((r for r in full if r['task'] == task),
                            key=lambda r: (r['source']['row_index'], r['id']))
        unseen = [r for r in candidates if r['id'] not in prior_ids]
        previous = [r for r in candidates if r['id'] in prior_ids]
        rng = random.Random(f'{seed}:{task}')
        rng.shuffle(unseen); rng.shuffle(previous)
        pools[task] = unseen + previous
        needed = 8 if task in FOCUS else 4
        if len(pools[task]) < needed:
            raise ValueError(f'Not enough items for {task}')
    chosen = [pools[task][rank] for rank in range(4) for task in BLINK_TASKS]
    chosen += [pools[task][rank] for rank in range(4, 8) for task in FOCUS]
    return [control_record(row, prior_ids=prior_ids) for row in chosen]


def parse_generated_label(text, labels):
    """Conservative explicit-label parsing; Point/Box/image marker letters are not answers.

    Accept a complete bare label, an initial parenthesized label, or an explicit
    'answer is/: LABEL' declaration. Competing label mentions are ambiguous.
    Never infer a label from free-form choice prose or an isolated internal letter.
    """
    clean = text.strip().replace('**', '').replace('__', '')
    legal = {label[1:-1]: label for label in labels}
    exact = re.fullmatch(r'(?:\(([A-E])\)|([A-E]))[.!]?', clean)
    candidates = []
    if exact:
        candidates.append(exact.group(1) or exact.group(2))
    initial = re.match(r'^\(([A-E])\)(?=\s|[.!,:;]|$)', clean)
    if initial:
        candidates.append(initial.group(1))
    pattern = r'\b(?:final\s+answer|(?:the\s+)?(?:correct\s+)?answer)\s*(?:is\s*:?|:)\s*(?:\(([A-E])\)|([A-E])\b)'
    for match in re.finditer(pattern, clean, flags=re.IGNORECASE):
        candidates.append((match.group(1) or match.group(2)).upper())
    mentioned = set(re.findall(r'\(([A-E])\)', clean)) | set(candidates)
    if len(mentioned) > 1:
        return {'status': 'ambiguous', 'label': None, 'candidates': sorted(mentioned)}
    if not candidates:
        return {'status': 'missing', 'label': None, 'candidates': sorted(mentioned)}
    letter = candidates[0]
    if letter not in legal:
        return {'status': 'invalid_label', 'label': None, 'candidates': [letter]}
    return {'status': 'parsed', 'label': legal[letter], 'candidates': [letter],
            'exact_label_only': exact is not None}


def generation_stop(token_ids, eos_ids, maximum):
    eos = set(eos_ids)
    ended = bool(token_ids) and token_ids[-1] in eos
    truncated = not ended and len(token_ids) >= maximum
    return {'eos_reached': ended, 'truncated': truncated,
            'stop_reason': 'eos' if ended else ('max_new_tokens' if truncated else 'other_stop')}


def write_once(path, text):
    if path.exists():
        if path.read_text() != text:
            raise ValueError(f'Existing preparation differs; preserve it: {path}')
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x') as handle:
        handle.write(text)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, default=ROOT)
    parser.add_argument('--batch-size', type=int, default=8)
    args = parser.parse_args()
    try:
        schema = campaign_schema(args.output_dir)
    except ValueError as error:
        parser.error(str(error))
    if not 1 <= args.batch_size <= 16:
        parser.error('Use 1–16 items per batch')
    full, prior = load_manifest(FULL), load_manifest(PRIOR)
    prior_ids = {r['id'] for r in prior}
    parent_plan = None
    if schema == SCHEMA_V2:
        parent_plan = json.loads((ROOT/'plan.json').read_text())
        source_path = Path(parent_plan['manifests']['all64']['path'])
        if digest(source_path) != parent_plan['manifests']['all64']['sha256']:
            raise ValueError('Original v1 all64 manifest changed')
        selected = read_controls(source_path)
        if len(selected) != 64 or [r['id'] for r in selected] != parent_plan['manifests']['all64']['ids']:
            raise ValueError('v2 must retain the exact v1 64-item selection/order')
    else:
        selected = select_records(full, prior_ids)
    for row in selected:
        validate_control(row)
    smoke = [next(r for r in selected if r['task'] == task) for task in FOCUS]
    smoke_ids = {r['id'] for r in smoke}
    remaining = [r for r in selected if r['id'] not in smoke_ids]
    manifests = {'all64': selected, 'smoke2': smoke, 'remaining62': remaining}
    for index, start in enumerate(range(0, len(remaining), args.batch_size), 1):
        manifests[f'batches/batch-{index:03d}'] = remaining[start:start + args.batch_size]
    paths = {}
    for label, rows in manifests.items():
        path = args.output_dir / 'manifests' / f'{label}.jsonl'
        write_once(path, ''.join(json.dumps(r, ensure_ascii=False, sort_keys=True) + '\n' for r in rows))
        paths[label] = {'path': str(path), 'sha256': digest(path), 'items': len(rows),
                        'ids': [r['id'] for r in rows]}
    plan = {'schema': schema, 'seed': SEED, 'selection': 'Per-task source-row order shuffled by task-local seed; prefer outside original112; no outcome selection',
            'full_manifest': str(FULL), 'full_manifest_sha256': digest(FULL),
            'prior_manifest': str(PRIOR), 'prior_manifest_sha256': digest(PRIOR),
            'task_counts': dict(Counter(r['task'] for r in selected)), 'items': len(selected),
            'reused_exploratory112_ids': [r['id'] for r in selected if r['id'] in prior_ids],
            'budgets': PIXELS, 'max_sequence': 8192, 'masks': ['111', '000'],
            'generation_masks': ['111'], 'generation_max_new_tokens': 2048 if schema == SCHEMA_V2 else 128,
            'variants': list(VARIANTS), 'default_expected_score_rows': 512,
            'default_expected_generation_rows': 256, 'batch_size': args.batch_size,
            'smoke_reuse': 'Run smoke2 once, then remaining62; all64 is the analysis identity',
            'manifests': paths, 'prior_result_reuse': False,
            'protocol_note': 'New readout protocol: 8192 total tokens, low/high image budgets, remapped answer lists and separate native greedy generation; not merged into original full1901 scores.'}
    if schema == SCHEMA_V2:
        snapshot = ROOT/'protocol_source_snapshot/snapshot.json'
        if not snapshot.is_file():
            raise ValueError('Preserve v1 execution source snapshot before preparing v2')
        plan.update(parent_campaign=str(ROOT), parent_plan_sha256=digest(ROOT/'plan.json'),
                    parent_source_snapshot=str(snapshot), parent_source_snapshot_sha256=digest(snapshot),
                    source_manifest_schema=SCHEMA,
                    runner_extra_arguments=['--campaign-root', str(args.output_dir.resolve())],
                    generation_revision='v2 extends max_new_tokens from 128 to 2048 and binds effective greedy kwargs/use_model_defaults=False, fixing v1 checkpoint sampling fallback; EOS early stopping; no prompt or scoring change',
                    v1_generation_annotation=str(ROOT/'generation_validation_note.json'),
                    v1_result_reuse=False)
        if paths['all64']['sha256'] != parent_plan['manifests']['all64']['sha256'] or paths['smoke2']['ids'] != parent_plan['manifests']['smoke2']['ids']:
            raise ValueError('v2 changed original control rows or smoke IDs')
    write_once(args.output_dir/'plan.json', json.dumps(plan, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({'plan': str(args.output_dir/'plan.json'), 'items': 64,
                      'task_counts': plan['task_counts'], 'prior_overlap': len(plan['reused_exploratory112_ids']),
                      'smoke_ids': list(smoke_ids), 'remaining_batches': len(paths)-3}, indent=2))


if __name__ == '__main__':
    main()
