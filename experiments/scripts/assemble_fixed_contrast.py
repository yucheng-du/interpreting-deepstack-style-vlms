"""Build the fixed-contrast tables' JSON inputs from the per-job analysis and scoring files."""
import argparse
import json
from pathlib import Path
import os

if __package__:
    from . import analyze_v14_readout_interactions as analysis
else:
    import analyze_v14_readout_interactions as analysis

def assemble(analyses, scores, output):
    groups = []
    seen = set()
    for path in analyses:
        for index, result in enumerate(json.loads(path.read_text())['results']):
            material = 'natural' if result['task'] == 'natural_scene_color' else 'synthetic'
            panel = 'granite-marginal' if 'marginal' in result['execution_version'] else 'line1'
            job = f"{result['model']}-{material}-{panel}-batched"
            group = {k: result[k] for k in ('model', 'task', 'kind', 'response_format', 'mapping', 'execution_version')}
            readout = analysis.readout_key(group)
            group_id = '/'.join([job, result['task'], result['kind'], readout])
            if group_id in seen:
                raise ValueError(f'Duplicate analysis group {group_id}')
            seen.add(group_id)
            group.update(material=material, job=job, group_id=group_id, readout=readout, analysis=result,
                         source={'file': path.name, 'pointer': f'/results/{index}'})
            groups.append(group)
    groups.sort(key=lambda g: (analysis.MODEL_ORDER.index(g['model']), g['material'], g['task'], g['kind'],
                               analysis.READOUT_ORDER.index(g['readout']), g['job']))
    if len(groups) != 144:
        raise ValueError(f'Expected 144 fixed-contrast groups, received {len(groups)}')
    contrasts = analysis.build_pair_contrasts(groups)
    panels = analysis.build_panels(groups, contrasts)
    reports = []
    for path in scores:
        for report in json.loads(path.read_text())['reports']:
            report['job'] = path.stem
            reports.append(report)
    if len({(r['model'], r['task']) for r in reports}) != 12 or len(reports) != 12:
        raise ValueError('Expected one benchmark score report for each of four models and three tasks')
    output.mkdir(parents=True, exist_ok=True)
    for name, doc in [('summary', {'groups': [analysis.strip_units(g) for g in groups],
                                   'paired_readout_contrasts': contrasts}),
                      ('paper_numbers', panels), ('benchmark_numbers', {'reports': reports})]:
        (output / (name + '.json')).write_text(json.dumps(doc) + '\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--analyses', type=Path, nargs='+', required=True)
    parser.add_argument('--scores', type=Path, nargs='+', required=True)
    parser.add_argument('--output', type=Path, default=Path(os.environ.get('DS_DATA_ROOT', 'work')) /
                        'analysis/v14-integration-20261029')
    args = parser.parse_args()
    assemble(args.analyses, args.scores, args.output)
