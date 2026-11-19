"""Read the three CSV files containing the coding results."""
import csv
from decimal import Decimal
from analysis.paths import CSV

CELLS = {
    ('qwen8b', 'confirm1000'): 'q8', ('deepstack7b', 'confirm1000'): 'ds',
    ('qwen32b', 'confirm1000'): 'q32a', ('qwen32b', 'confirm474'): 'q32d',
    ('ibm4b', 'granite468'): 'g468', ('ibm4b', 'granite531'): 'g531',
    ('ibm4b', 'granite999'): 'g999',
}


def records(stem):
    with (CSV / f'coding_{stem}.csv').open(newline='') as handle:
        yield from csv.DictReader(handle)


def fraction(percent):
    return float(Decimal(percent) / 100)


def estimate(row, value):
    return dict(est=fraction(row[value]), lo=fraction(row['ci_low']),
                hi=fraction(row['ci_high']), n_units=int(row['n_units']))


def read_estimates():
    data = {}
    for row in records('configurations'):
        if row['image_set'] == 'development192':
            continue
        cell = CELLS[row['model'], row['image_set']]
        cfg = row['config']
        data[f'score/{cell}/{cfg}'] = dict(estimate(row, 'score'), cell=cell, config=cfg,
            bytes=float(row['bytes_total']), bytes_main=float(row['bytes_main']),
            bytes_later=float(row['bytes_later']), n_questions=int(row['n_questions']))
        if cell.startswith('q') and row['family'] in ('main_only', 'scale_normalized'):
            data[f'match/{cell}/{cfg}'] = dict(inside=False)
    for row in records('single_entry'):
        cell = CELLS[row['model'], row['image_set']]
        a, b = row['config_a'], row['config_b']
        if row['estimate'] == 'difference':
            data[f'diff/{cell}/{a}/{b}'] = estimate(row, 'value')
        elif row['estimate'] == 'endpoint_margin':
            e = estimate(row, 'value')
            data[f'end/{cell}/{a}'] = dict(e, margin=e['est'], f=float(row['tolerated_fraction']),
                loss_over_JZ=float(row['loss_over_gain']), label={'fails': 'refuted'}.get(row['outcome'], row['outcome']))
    for row in records('comparisons'):
        if row['group'] == 'training':
            continue
        cell = CELLS[row['model'], row['image_set']]
        a, b = row['config_a'], row['config_b']
        e = estimate(row, 'difference')
        if row['group'] == 'matched_bytes_shared_curve':
            data[f'match/{cell}/{a}'] = dict(e, cell=cell, config=a, inside=True,
                bytes=float(row['bytes_a']), score=fraction(row['score_a']),
                curve_score=fraction(row['score_b']), bracket=b.split(': ')[1].split(' / '))
        else:
            data[f'diff/{cell}/{a}/{b}'] = dict(e, a_score=data[f'score/{cell}/{a}'],
                b_score=data[f'score/{cell}/{b}'], bytes_change_pct=float(row['bytes_change_pct']))
    return data


def family(c):
    m, l = c.split('|')
    if m in 'CZM' and l in 'CZM':
        return 'single_entry'
    if l == m:
        return 'shared'
    if l == 'k0':
        return 'scale_normalized'
    if l.startswith('k+'):
        return 'later_coarser'
    if l.startswith('k-'):
        return 'later_finer'
    if l == 'Z':
        return 'main_only'
    return 'other'


def qmain(c):
    return int(c.split('|')[0][1:])


