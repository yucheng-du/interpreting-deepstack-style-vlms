"""Export single-entry, coding and paired-comparison CSVs from rerun observations."""
import csv
import hashlib
from analysis.paths import ROOT
from analysis.coding_csv import family
from analysis.coding_estimates import CELL_INFO, compute
from paper.tables_19_to_26 import cfg_tex, SINGLE, DIFF_ROWS, diff_status, sort_cfg

def write_csv(path, fields, rows, exports):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=fields, lineterminator='\n')
        w.writeheader()
        for r in rows:
            w.writerow({k: ('' if r.get(k) is None else f'{r[k]:.12g}' if isinstance(r.get(k), float) else r.get(k))
                        for k in fields})
    exports.append(dict(file=path.name, rows=len(rows), sha256=hashlib.sha256(path.read_bytes()).hexdigest()))


def pct(x):
    return None if x is None else 100*x


def paper_cfg(c):
    return cfg_tex(c).strip('$').replace('\\,|\\,', '|').replace('\\mathrm{n}', 'n').replace('\\mathrm{', '').replace(
        '{+}', '+').replace('{-}', '-').replace('}', '').replace('{', '')


STATUS_CFG = {'main_only': 'F', 'single_entry': 'L'}


def config_status(cell, c):
    if cell in ('g531',):
        return 'F'
    if cell == 'g999':
        return 'F'
    return STATUS_CFG.get(family(c), 'L')


def export(tree, E, exports):
    d = tree/'supplementary_data'
    # single-entry: scores, differences, endpoints
    rows = []
    for cell in ('q8', 'q32a', 'g999', 'g531', 'g468', 'ds', 'q32d'):
        info = CELL_INFO[cell]
        for c in SINGLE:
            e = E[f'score/{cell}/{c}']
            rows.append(dict(model=info['model'], task=info['task'], image_set=info['image_set'], estimate='score',
                             config_a=c, config_b='', value=pct(e['est']), ci_low=pct(e['lo']), ci_high=pct(e['hi']),
                             n_units=e['n_units'], n_questions=e['n_questions'], status=config_status(cell, c)))
        for a, b, _ in DIFF_ROWS + [('M|C', 'C|M', ''), ('Z|C', 'M|M', '')]:
            k = f'diff/{cell}/{a}/{b}'
            if cell == 'q32d' and (a, b) != ('C|Z', 'C|C'):
                continue
            if (a, b) == ('Z|C', 'M|M') and cell != 'g999':
                continue
            if (a, b) == ('M|C', 'C|M') and not cell.startswith('g'):
                continue
            e = E[k]
            rows.append(dict(model=info['model'], task=info['task'], image_set=info['image_set'], estimate='difference',
                             config_a=a, config_b=b, value=pct(e['est']), ci_low=pct(e['lo']), ci_high=pct(e['hi']),
                             n_units=e['n_units'], n_questions=e['n_questions'], status=diff_status(cell, a, b)))
    for k in [k for k in E if k.startswith('end/')]:
        e = E[k]
        info = CELL_INFO[e['cell']]
        rows.append(dict(model=info['model'], task=info['task'], image_set=info['image_set'], estimate='endpoint_margin',
                         config_a=e['config'], config_b='', value=pct(e['margin']), ci_low=pct(e['lo']), ci_high=pct(e['hi']),
                         n_units=e['n_units'], n_questions=e['n_questions'], tolerated_fraction=e['f'],
                         loss_over_gain=e['loss_over_JZ'], outcome={'refuted': 'fails'}.get(e['label'], e['label']),
                         status='F' if e['cell'] in ('g531', 'g999') else 'L'))
    write_csv(d/'coding_single_entry.csv', ['model', 'task', 'image_set', 'estimate', 'config_a', 'config_b', 'value',
                                            'ci_low', 'ci_high', 'n_units', 'n_questions', 'tolerated_fraction',
                                            'loss_over_gain', 'outcome', 'status'], rows, exports)
    # configurations
    rows = []
    for cell in ('q8', 'q32a', 'q32d', 'ds', 'g468', 'g531', 'g999'):
        info = CELL_INFO[cell]
        for k in sorted([k for k in E if k.startswith(f'score/{cell}/')], key=lambda k: (family(k.split('/', 2)[2]), sort_cfg(k.split('/', 2)[2]) if family(k.split('/', 2)[2]) != 'single_entry' else (0, 0), k)):
            e = E[k]
            c = e['config']
            rows.append(dict(model=info['model'], task=info['task'], image_set=info['image_set'], config=c,
                             config_paper=paper_cfg(c), family=family(c), status=config_status(cell, c),
                             n_units=e['n_units'], n_questions=e['n_questions'], bytes_total=e['bytes'],
                             bytes_main=e['bytes_main'], bytes_later=e['bytes_later'], score=pct(e['est']),
                             ci_low=pct(e['lo']), ci_high=pct(e['hi'])))
    for k in [k for k in E if k.startswith('tscore/')]:
        e = E[k]
        rows.append(dict(model=e['model'], task='gqa_attribute', image_set='development192', config=e['config'],
                         config_paper=paper_cfg(e['config']), family='training_' + e['arm'], status='E',
                         arm=e['arm'], checkpoint=e['checkpoint'], n_units=e['n_units'], n_questions=e['n_questions'],
                         bytes_total=e['bytes'], score=pct(e['est']), ci_low=pct(e['lo']), ci_high=pct(e['hi'])))
    write_csv(d/'coding_configurations.csv', ['model', 'task', 'image_set', 'config', 'config_paper', 'family', 'arm',
                                              'checkpoint', 'status', 'n_units', 'n_questions', 'bytes_total', 'bytes_main',
                                              'bytes_later', 'score', 'ci_low', 'ci_high'], rows, exports)
    # comparisons
    rows = []
    groups = [('deepstack_pairs', 'ds', [('q-4|Z', 'q-4|q-4', 'F'), ('q-2|Z', 'q-2|q-2', 'F'), ('q-1|Z', 'q-1|q-1', 'F'),
                                         ('q-4|Z', 'q-2|q-2', 'F'), ('q-2|k+2', 'q-2|k0', 'L'), ('q-2|k+2', 'q-2|q-2', 'L'),
                                         ('q-2|Z', 'q-2|k+2', 'F'), ('q-4|Z', 'q-2|k0', 'P')]),
              ('granite_pair', 'g531', [('q0|k+4', 'q-4|q-4', 'F')]),
              ('granite_pair', 'g468', [('q0|k+4', 'q-4|q-4', 'P'), ('q4|k0', 'q-4|q-4', 'P'), ('q0|k+4', 'q4|k0', 'P')]),
              ('qwen_main_only', 'q8', [('q0|Z', 'q0|q0', 'F'), ('q2|Z', 'q2|q2', 'F'), ('q6|Z', 'q6|q6', 'F'),
                                        ('q0|Z', 'q2|q2', 'F'), ('q2|k+4', 'q2|q2', 'L')]),
              ('qwen_main_only', 'q32a', [('q6|Z', 'q6|q6', 'F'), ('q12|Z', 'q12|q12', 'F')]),
              ('qwen_main_only', 'q32d', [('q6|Z', 'q6|q6', 'F'), ('q10|Z', 'q10|q10', 'F')]),
              ('role_direction', 'q8', [('q2|k+4', 'q10|k-4', 'L'), ('q6|k+4', 'q10|k-2', 'L')]),
              ('role_direction', 'q32d', [('q8|k+4', 'q12|k-2', 'L'), ('q12|k+2', 'q16|k-4', 'L')]),
              ('role_direction', 'g468', [('q8|k-4', 'q0|k+2', 'L'), ('q12|k-4', 'q8|k+2', 'L')])]
    for grp, cell, pairs in groups:
        info = CELL_INFO[cell]
        for a, b, st in pairs:
            e = E[f'diff/{cell}/{a}/{b}']
            rows.append(dict(group=grp, model=info['model'], task=info['task'], image_set=info['image_set'], config_a=a,
                             config_b=b, bytes_a=e['a_score']['bytes'], bytes_b=e['b_score']['bytes'],
                             bytes_change_pct=e['bytes_change_pct'], score_a=pct(e['a_score']['est']),
                             score_b=pct(e['b_score']['est']), difference=pct(e['est']), ci_low=pct(e['lo']),
                             ci_high=pct(e['hi']), n_units=e['n_units'], n_questions=e['n_questions'], status=st))
    for k in sorted(k for k in E if k.startswith('match/') and E[k]['inside']):
        e = E[k]
        info = CELL_INFO[e['cell']]
        if e['cell'] == 'ds':
            continue
        rows.append(dict(group='matched_bytes_shared_curve', model=info['model'], task=info['task'],
                         image_set=info['image_set'], config_a=e['config'], config_b='shared curve: ' + ' / '.join(e['bracket']),
                         bytes_a=e['bytes'], score_a=pct(e['score']), score_b=pct(e['curve_score']), difference=pct(e['est']),
                         ci_low=pct(e['lo']), ci_high=pct(e['hi']), n_units=e['n_units'], status='P'))
    for k in sorted(k for k in E if k.startswith('tdiff/')):
        e = E[k]
        rows.append(dict(group='training', model=e['model'], task='gqa_attribute', image_set='development192',
                         config_a=e['config'], config_b=e['config'], arm_a=e['a'], arm_b=e['b'], checkpoint=e['checkpoint'],
                         difference=pct(e['est']), ci_low=pct(e['lo']), ci_high=pct(e['hi']), n_units=e['n_units'],
                         n_questions=e['n_questions'], status='E, P' if 'sft' in (e['a'], e['b']) else 'E'))
    write_csv(d/'coding_comparisons.csv', ['group', 'model', 'task', 'image_set', 'config_a', 'config_b', 'arm_a', 'arm_b',
                                           'checkpoint', 'bytes_a', 'bytes_b', 'bytes_change_pct', 'score_a', 'score_b',
                                           'difference', 'ci_low', 'ci_high', 'n_units', 'n_questions', 'status'],
              rows, exports)



if __name__ == '__main__':
    exports = []
    export(ROOT, compute(), exports)
    for result in exports:
        print(result['file'], result['rows'], result['sha256'])
