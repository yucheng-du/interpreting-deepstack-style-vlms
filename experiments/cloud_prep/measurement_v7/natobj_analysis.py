"""Natural-object M x D x P analysis: the v7 spatial analyzer with source-cluster units.

Records come from qwen_spatial on the natural_gqa family (natobj_family). Estimands are spatial_analysis
unchanged: E per query, mapping, format, main strength, layout M, donor pose D (relative to the original layout)
and permutation P, with a/b fixed by the target object's identity; label mappings averaged within unit with
per-mapping rows kept, direct separate; missing cells stay missing; invalid/truncated generations stay in the
planned denominator. Only the unit changes: unit_id, the union-find source cluster, because both assignment
scenes of a quad reuse the same four crops. Torch-free.
"""
from __future__ import annotations
import argparse
from collections import defaultdict
import json
from pathlib import Path

from cloud_prep.portable_data import _guard_path
from cloud_prep.measurement_v7 import spatial_analysis as sa
from cloud_prep.measurement_v7.natobj_family import FAMILY, UNIT_KEY

OUTPUTS = ('query_effects', 'query_vectors', 'unit_vectors', 'unit_layout_cells', 'behavior_pairs')
NEEDED = (UNIT_KEY, 'quad_id', 'stratum', 'construction')


def natural_coverage(records, generations=()):
    """Family and unit guards (one unit per quad) plus natural-subset counts."""
    quads, strata, constructions, scenes = {}, defaultdict(set), set(), set()
    for r in [*records, *generations]:
        fields = r.get('record_fields') or {}
        if r.get('family') != FAMILY or any(k not in fields for k in NEEDED):
            raise ValueError(f'Expected {FAMILY} records whose record_fields carry {list(NEEDED)}')
        if quads.setdefault(fields['quad_id'], fields[UNIT_KEY]) != fields[UNIT_KEY]:
            raise ValueError(f'{fields["quad_id"]}: one quad maps to several units')
        strata[fields['stratum']].add(fields[UNIT_KEY])
        constructions.add(fields['construction'])
        scenes.add(r['scene_id'])
    return {'unit_key': UNIT_KEY, 'units': len(set(quads.values())), 'quads': len(quads), 'scenes': len(scenes),
            'strata': len(strata), 'units_per_stratum': {s: len(u) for s, u in sorted(strata.items())},
            'construction': sorted(constructions)}


def analyze(records, generations=()):
    natural = natural_coverage(records, generations)
    result = sa.analyze(records, generations, unit_key=UNIT_KEY)
    result['natural'] = natural
    result['interpretation'] += (' Natural bridge: real GQA/VG object crops recomposed into fixed boxes '
        '(controlled_recomposition, not whole natural scenes). Units are source clusters (unit_id); assignment, target, '
        'mapping, format, strength, M, D and P are within-unit repeats.')
    return result


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument('--runs', type=Path, nargs='+', required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--include-partial', action='store_true')
    a = p.parse_args(argv)
    records, generations, provenance = sa.load_runs(a.runs, a.include_partial)
    result = analyze(records, generations)
    result.update(provenance, partial_allowed=a.include_partial)
    output = _guard_path(a.output, write=True); output.mkdir(parents=True, exist_ok=False)
    for name in OUTPUTS:
        (output / (name + '.jsonl')).write_text(''.join(json.dumps(row) + '\n' for row in result.pop(name)))
    (output / 'summary.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({'output': str(output), 'score_rows': len(records), 'generation_rows': len(generations),
                      'unit_key': UNIT_KEY, 'units': result['natural']['units'], 'quads': result['natural']['quads'],
                      **{k: v for k, v in result['coverage'].items() if k != 'units'}}))


if __name__ == '__main__':
    main()
