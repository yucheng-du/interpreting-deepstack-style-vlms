"""v19 confirmation analysis (CPU only). Per cell (model x task) and budget tier, each family (S, N, R, A, or the
two offset sides Kpos/Kneg in interaction and no-judgment cells) takes one config chosen on calibration only; that
config is scored on test. Reported: paired family differences with unit-bootstrap intervals and auxiliary labels,
bytes to a common task target, endpoint tests, BD-style summaries, and the post-hoc test envelope (reference only).

CLI: python -m cloud_prep.measurement_v19.confirm.report <cmd> ...
  lock        [--protocol P] [--out LOCK] [--allow-pending]   hash protocol + confirmation unit lists
  hash-units  [--protocol P]                                  sha256 / size of every unit list in the protocol
  select      [--protocol P] --obs PATH... --out DIR [--mode dev|confirm]
              calibration only: chosen configs and the configs each cell needs on test (test_configs.json)
  report      [--protocol P] --obs PATH... --out DIR [--mode dev|confirm] [--lock LOCK] [--no-plots]
  configs     [--protocol P] --model M --task T [--no-corners]
              exact config list of one cell, one per line: every family candidate (union, protocol order) plus the
              seven corners C|C C|Z Z|C Z|Z C|M M|C M|M and the cell's endpoint configs. Use it for the confirm sweep
              from the configs command; offset_axis configs prints a wider development grid, not the protocol's candidates.
--obs: observations.jsonl files or directories (searched recursively), optionally MODEL=PATH. Confirm-mode report
refuses to run (exit 3) without a lock matching the current protocol and confirmation unit lists.
Exit 4: some cell failed a check (outputs are still written; the cell carries 'error').
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

from cloud_prep.measurement_v19.confirm import core
from cloud_prep.measurement_v19.confirm.core import CC, ZZ, EPS

LABEL_TEXT = {'holds': 'holds', 'refuted': 'refuted', 'undetermined': 'undetermined', None: ''}


def items(d):
    return [(k, v) for k, v in (d or {}).items() if not k.startswith('_')]


# ---- cells -----------------------------------------------------------------------------------

def load_cells(p, obs, mode, models=None, tasks=None):
    """([cell], notes, skipped). cell: model, task, cal View, test View, info."""
    groups, skipped = core.load_groups(p, obs)
    cells, notes = [], []
    for (model, _, task), g in sorted(groups.items(), key=lambda kv: (kv[0][0], kv[0][2])):
        if (models and model not in models) or (tasks and task not in tasks):
            continue
        if task not in p['cells'].get(model, {}):
            notes.append(f'{model}/{task}: not a protocol cell, skipped')
            continue
        ts = p['tasks'][task]
        rows = list(g['rows'].values())
        cal, test, info = core.split_rows(p, task, rows, mode)
        mk = lambda rr: core.View(rr, ts.get('unit_field', 'unit_id'), ts.get('score_field', 'correct'),
                                  ts.get('secondary_field'))
        run_ids = sorted(r['run_id'] for r in test if r.get('run_id'))
        info.update(files=g['files'], duplicates=g['duplicates'], conflicts=g['conflicts'],
                    metric_mismatch=sum(1 for r in rows if ts.get('metric') and r.get('metric', ts['metric']) != ts['metric']),
                    test_run_ids=[run_ids[0], run_ids[-1]] if run_ids else None)
        cells.append(dict(model=model, task=task, cal=mk(cal), test=mk(test), info=info))
    return cells, notes, skipped


def family_check(cal, cands, mode):
    """Per family: candidates in the spec / on calibration, calibration units and evaluations actually read.
    The calibration unit set is the union over all candidates observed on calibration; a candidate missing or not
    covering every one of those units is not usable. Unequal counts, missing or incomplete candidates, unequal
    question or evaluation counts are listed; confirm mode raises on any."""
    seen = [cal.units_of(c) for cs in cands.values() for c in cs if c in cal]
    units = set().union(*seen) if seen else set(cal.units)
    fam, problems = {}, []
    if seen and CC in cal and cal.units_of(CC) != units:
        problems.append(f'{CC} covers {len(cal.units_of(CC))} calibration units, the candidates {len(units)}')
    for f, cs in cands.items():
        found = [c for c in cs if c in cal]
        nob = [c for c in found if not core.has_bytes(cal, c)]
        usable = [c for c in found if cal.units_of(c) == units and c not in nob]
        fam[f] = dict(n_spec=len(cs), n_found=len(found), n_usable=len(usable),
                      missing=[c for c in cs if c not in cal], incomplete=[c for c in found if c not in usable and c not in nob],
                      no_bytes=nob, cal_evals=sum(cal.n(c) for c in usable),
                      cal_units=len(set().union(*[cal.units_of(c) for c in usable])) if usable else 0, usable=usable)
    counts = {f: v['n_spec'] for f, v in fam.items()}
    if len(set(counts.values())) > 1:
        problems.append(f'candidate counts differ across families: {counts}')
    for f, v in fam.items():
        if v['missing']:
            problems.append(f'{f}: {len(v["missing"])}/{v["n_spec"]} candidates not observed on calibration')
        if v['incomplete']:
            problems.append(f'{f}: {len(v["incomplete"])} candidates miss calibration units: {v["incomplete"]}')
        if v['no_bytes']:
            problems.append(f'{f}: {len(v["no_bytes"])} candidates lack real code-length bytes on calibration '
                            f'(sweep without --bytes): {v["no_bytes"]}')
    nq = sorted({cal.n(c) for v in fam.values() for c in v['usable']})
    if len(nq) > 1:
        problems.append(f'usable candidates differ in calibration question count: {nq}')
    evals = {f: v['cal_evals'] for f, v in fam.items()}
    if len(set(evals.values())) > 1:
        problems.append(f'calibration evaluations read differ across families: {evals}')
    if problems and mode == 'confirm':
        raise ValueError('; '.join(problems))
    return fam, problems


def select_cell(p, model, task, cal, mode):
    """Everything chosen from calibration only (the test split is never passed in)."""
    cs = core.cell_spec(p, model, task)
    fam, problems = family_check(cal, cs['candidates'], mode)
    usable = {f: v.pop('usable') for f, v in fam.items()}
    ref = p['budgets']['reference_family']
    if ref not in usable:
        raise ValueError(f'{model}/{task}: reference family {ref} is not a family of this cell')
    tiers = {}
    for name, spec in items(p['budgets']['tiers']):
        try:
            tb = core.tier_budget(cal, spec, usable[ref])
        except KeyError as e:
            tb = dict(budget=None, reason=f'{e.args[0]} on calibration')
        B = tb['budget']
        picks = {}
        for f, cands in usable.items():
            c = core.pick_budget(cal, cands, B)
            picks[f] = dict(config=c, n_feasible=sum(cal.nbytes(x) <= B * (1 + EPS) for x in cands) if B is not None else 0,
                            cal_score=cal.score(c) if c else None, cal_bytes=cal.nbytes(c) if c else None)
        tiers[name] = dict(spec=spec, **tb, picks=picks)
    targets = {}
    for name, spec in items(p.get('targets')):
        try:
            tc = core.score_target(cal, spec)
        except KeyError as e:
            targets[name] = dict(spec=spec, reason=f'{e.args[0]} on calibration')
            continue
        picks = {}
        for f, cands in usable.items():
            c = core.pick_target(cal, cands, tc)
            picks[f] = dict(config=c, cal_score=cal.score(c) if c else None, cal_bytes=cal.nbytes(c) if c else None)
        targets[name] = dict(spec=spec, cal_target=tc, picks=picks)
    return dict(model=model, task=task, type=cs['type'], role=cs.get('role'), describe_only=bool(cs.get('describe_only')),
                spec=cs, families=fam, problems=problems, usable=usable, tiers=tiers, targets=targets,
                pareto={f: core.pareto(cal, c) for f, c in usable.items()},
                cal_units=len(cal.units), cal_cc=cal.score(CC), cal_zz=cal.score(ZZ))


def test_configs(sel):
    need = {CC, ZZ} | {e['config'] for e in sel['spec'].get('endpoints', [])}
    need |= {pk['config'] for t in sel['tiers'].values() for pk in t['picks'].values() if pk['config']}
    need |= {pk['config'] for t in sel['targets'].values() for pk in t.get('picks', {}).values() if pk['config']}
    allc = sorted({c for cs in sel['spec']['candidates'].values() for c in cs} | need)
    return dict(selected=sorted(need), pareto=sorted({c for v in sel['pareto'].values() for c in v}), all_candidates=allc)


# ---- test side -------------------------------------------------------------------------------

def eval_config(test, boot, c):
    if c is None:
        return {}
    if c not in test:
        return dict(test_missing=True)
    d = boot.est(test.unit_mean(c))
    return dict(test_score=d['est'], test_lo=d['lo'], test_hi=d['hi'], test_units=d['n_units'], test_n=test.n(c),
                test_bytes=test.nbytes(c), test_bytes_main=test.nbytes(c, 3), test_bytes_later=test.nbytes(c, 4),
                test_secondary=test.score(c, 5), test_rows_without_bytes=test.missing_bytes.get(c, 0))


def compare(test, boot, ca, cb, cmp, tol, describe):
    pred = cmp.get('predicted', 'a>b') if cmp.get('label') and not describe else None
    r = dict(a=cmp['a'], b=cmp['b'], config_a=ca, config_b=cb, predicted=pred)
    if not ca or not cb or ca not in test or cb not in test:
        r['reason'] = 'no selection or not observed on test'
        return r
    qv = core.qvals([test.by[ca], test.by[cb]], (1, -1))
    d = boot.est(core.unit_means(qv))
    r.update(diff=d['est'], lo=d['lo'], hi=d['hi'], n_units=d['n_units'], n_questions=len(qv),
             bytes_a=test.nbytes(ca), bytes_b=test.nbytes(cb), label=core.label(d, pred) if pred else None,
             **core.tol_flags(d, tol))
    return r


def endpoint_rows(cal, test, boot, specs):
    """Single-stream endpoint E keeps most performance: loss L = C|C - E <= f * J(Z), J(Z) = C|C - Z|Z. Tested as
    the per-question margin E - (1-f) C|C - f Z|Z (= f J - L) with the unit bootstrap; holds if its interval > 0."""
    out = []
    for e in specs:
        c, f = e['config'], float(e['max_loss_frac_of_JZ'])
        r = dict(config=c, max_loss_frac_of_JZ=f, source=e.get('source'))
        for name, v in (('cal', cal), ('test', test)):
            if all(x in v for x in (c, CC, ZZ)):
                cc, zz, ee = v.score(CC), v.score(ZZ), v.score(c)
                r.update({f'{name}_loss': cc - ee, f'{name}_JZ': cc - zz,
                          f'{name}_loss_over_JZ': (cc - ee) / (cc - zz) if abs(cc - zz) > EPS else None})
        if all(x in test for x in (c, CC, ZZ)):
            qv = core.qvals([test.by[c], test.by[CC], test.by[ZZ]], (1, -(1 - f), -f))
            d = boot.est(core.unit_means(qv))
            r.update(margin=d['est'], margin_lo=d['lo'], margin_hi=d['hi'], n_units=d['n_units'], n_questions=len(qv),
                     label=core.label(d))
        else:
            r['reason'] = f'{c}, {CC} or {ZZ} not observed on test'
        out.append(r)
    return out


def curve_rows(model, task, cal, test, cs):
    fam_of = {}
    for f, cl in cs['candidates'].items():
        for c in cl:
            fam_of.setdefault(c, []).append(f)
    rows = []
    for c in sorted(set(cal.configs()) | set(test.configs())):
        k = core.config_kind(c, cs.get('role'))
        r = dict(model=model, task=task, config=c, kind=k['family'], q=k['q'], k=k['k'], side=k['side'],
                 families=','.join(fam_of.get(c, [])))
        for name, v in (('cal', cal), ('test', test)):
            if c in v:
                r.update({f'{name}_score': v.score(c), f'{name}_bytes': v.nbytes(c), f'{name}_bytes_main': v.nbytes(c, 3),
                          f'{name}_bytes_later': v.nbytes(c, 4), f'{name}_n': v.n(c), f'{name}_units': len(v.units_of(c))})
        rows.append(r)
    return rows


def analyze_cell(p, cell, mode):
    model, task, cal, test = cell['model'], cell['task'], cell['cal'], cell['test']
    sel = select_cell(p, model, task, cal, mode)
    cs, describe, tol = sel['spec'], sel['describe_only'], p['tasks'][task]['tolerance']
    bp = p['bootstrap']
    boot = core.Boot(test.units, bp['n_boot'], bp['seed'], bp.get('ci', 0.95))
    head = dict(model=model, task=task)
    tier_rows, cmp_rows, tgt_rows, tgt_cmp, bd_rows = [], [], [], [], []
    problems = list(sel['problems'])
    for name, t in sel['tiers'].items():
        B = t['budget']
        for f, pk in t['picks'].items():
            r = dict(head, tier=name, budget=B, budget_rule=json.dumps(t['spec']), reference_config=t.get('reference_config'),
                     family=f, **pk, **eval_config(test, boot, pk['config']))
            if r.get('test_missing'):
                problems.append(f'{name}/{f}: selected {pk["config"]} not observed on test')
            if r.get('test_rows_without_bytes'):
                problems.append(f'{name}/{f}: {r["test_rows_without_bytes"]} test rows of {pk["config"]} lack real code-length bytes')
            if r.get('test_units') is not None and r['test_units'] < len(test.units):
                problems.append(f'{name}/{f}: selected {pk["config"]} covers {r["test_units"]} of {len(test.units)} test units')
            env = core.pick_budget(test, [x for x in cs['candidates'][f] if x in test], B)
            r.update(env_config=env, env_score=test.score(env) if env else None, env_bytes=test.nbytes(env) if env else None)
            if env and r.get('test_score') is not None:
                r.update(sel_minus_env=r['test_score'] - r['env_score'],
                         within_tol_of_env=bool(r['test_score'] - r['env_score'] >= -tol))
            tier_rows.append(r)
        allc = [x for cl in cs['candidates'].values() for x in cl if x in test]
        env = core.pick_budget(test, allc, B)
        tier_rows.append(dict(head, tier=name, budget=B, family='ALL (test envelope, reference)', env_config=env,
                              env_score=test.score(env) if env else None, env_bytes=test.nbytes(env) if env else None))
        for cmp in cs['comparisons']:
            ca, cb = t['picks'].get(cmp['a'], {}).get('config'), t['picks'].get(cmp['b'], {}).get('config')
            cmp_rows.append(dict(head, tier=name, budget=B, **compare(test, boot, ca, cb, cmp, tol, describe)))
    for name, t in sel['targets'].items():
        if 'reason' in t:
            tgt_rows.append(dict(head, target=name, reason=t['reason']))
            continue
        spec = t['spec']
        try:
            tt = t['cal_target'] if spec['type'] == 'abs' or spec.get('basis', 'split') == 'calibration' else core.score_target(test, spec)
        except KeyError as e:
            tt = None
            problems.append(f'target {name}: {e.args[0]} on test')
        res = {}
        for f, pk in t['picks'].items():
            c = pk['config']
            r = dict(head, target=name, rule=json.dumps(spec), cal_target=t['cal_target'], test_target=tt, family=f,
                     config=c, reached_calibration=c is not None, cal_bytes=pk['cal_bytes'], cal_score=pk['cal_score'])
            if c and c in test:
                r.update(test_bytes=test.nbytes(c), test_score=test.score(c),
                         reached_test=None if tt is None else bool(test.score(c) >= tt - EPS))
            elif c:
                r['test_missing'] = True
            env = core.pick_target(test, [x for x in cs['candidates'][f] if x in test], tt)
            r.update(env_config=env, env_bytes=test.nbytes(env) if env else None)
            res[f] = r
            tgt_rows.append(r)
        for cmp in cs['comparisons']:
            a, b = res.get(cmp['a'], {}), res.get(cmp['b'], {})
            ba, bb = a.get('test_bytes'), b.get('test_bytes')
            tgt_cmp.append(dict(head, target=name, a=cmp['a'], b=cmp['b'], bytes_a=ba, bytes_b=bb,
                                ratio_a_over_b=ba / bb if ba and bb else None,
                                reached_a=a.get('reached_test'), reached_b=b.get('reached_test')))
    bdp = p.get('bd', {})
    for cmp in cs['comparisons']:
        cur = {f: [(test.nbytes(c), test.score(c)) for c in sel['pareto'].get(f, []) if c in test] for f in (cmp['a'], cmp['b'])}
        bd_rows.append(dict(head, a=cmp['a'], b=cmp['b'], points_a=len(cur[cmp['a']]), points_b=len(cur[cmp['b']]),
                            **core.bd(cur[cmp['a']], cur[cmp['b']], bdp.get('min_points', 3), bdp.get('min_points_in_overlap', 2),
                                      bdp.get('min_overlap_score', 0.0), bdp.get('min_overlap_log10', 0.0))))
    ends = [dict(head, **r) for r in endpoint_rows(cal, test, boot, cs.get('endpoints', []))]
    ti = cell['info'].get('test', {})
    if mode == 'confirm' and ti.get('n_listed') is not None and ti.get('n_observed') != ti['n_listed']:
        problems.append(f'test observed {ti.get("n_observed")} of the {ti["n_listed"]} locked confirmation units')
    if cell['info'].get('conflicts'):
        problems.append(f'{cell["info"]["conflicts"]} duplicate question x config rows disagree in score (later file/line kept)')
    if cell['info'].get('metric_mismatch'):
        problems.append(f'{cell["info"]["metric_mismatch"]} rows carry a metric other than {p["tasks"][task].get("metric")}')
    summary = dict(head, type=sel['type'], role=sel['role'], describe_only=describe, cal_units=sel['cal_units'],
                   test_units=len(test.units), cal_cc=sel['cal_cc'], cal_zz=sel['cal_zz'], test_cc=test.score(CC),
                   test_zz=test.score(ZZ), families=sel['families'], usable=sel['usable'],
                   tiers={n: {k: v for k, v in t.items() if k != 'picks'} for n, t in sel['tiers'].items()},
                   problems=problems, info=cell['info'])
    return dict(summary=summary, tiers=tier_rows, comparisons=cmp_rows, targets=tgt_rows, target_comparisons=tgt_cmp,
                endpoints=ends, bd=bd_rows, curves=curve_rows(model, task, cal, test, cs))


# ---- output ----------------------------------------------------------------------------------

def fmt(x, nd=3):
    if x is None:
        return ''
    if isinstance(x, bool):
        return 'yes' if x else 'no'
    if isinstance(x, float):
        return f'{x:.{nd}f}' if abs(x) < 1000 else f'{x:,.0f}'      # scores / differences vs bytes
    return str(x)


def ci(r, a='lo', b='hi'):
    return f'[{fmt(r.get(a))}, {fmt(r.get(b))}]' if r.get(a) is not None else ''


def table(header, rows):
    return ['| ' + ' | '.join(header) + ' |', '|' + '---|' * len(header)] + ['| ' + ' | '.join(r) + ' |' for r in rows]


def markdown(res, head):
    L = [f'# v19 confirmation analysis ({head["mode"]})', '',
         f'Protocol {head["protocol"]["name"]} {head["protocol"]["version"]}, sha256 {head["protocol"]["sha256"]}.']
    if head.get('lock'):
        L.append(f'Lock {head["lock"]["lock_path"]} at {head["lock"]["locked_at_utc"]} (lock sha256 {head["lock"]["lock_sha256"][:16]}).')
    if head['protocol']['pending']:
        L.append('Pending (placeholder numbers): ' + '; '.join(head['protocol']['pending']) + '.')
    L += ['', 'Scores: unit means (GQA image, DocVQA document); intervals: unit bootstrap, percentile. Bytes: mean input code '
          'length per evaluation unit (within-unit mean over questions of the bytes of the image each question uses, then '
          'an equal-weight mean over units; DocVQA units are documents, not sums over pages). Labels: holds = interval wholly on the predicted side, refuted = wholly on the other side, undetermined = otherwise. '
          'Test-envelope columns are post-hoc references, not deployment results.', '']
    for c in res:
        s = c['summary']
        L += [f'## {s["model"]} / {s["task"]} ({s["type"]}{", describe only" if s["describe_only"] else ""})', '']
        if s.get('error'):
            L += [f'**Error:** {s["error"]}', '']
            continue
        L.append(f'Calibration units {s["cal_units"]}, test units {s["test_units"]}. C|C cal {fmt(s["cal_cc"])} / test '
                 f'{fmt(s["test_cc"])}; Z|Z cal {fmt(s["cal_zz"])} / test {fmt(s["test_zz"])}.')
        L.append('Families (spec / usable on calibration / calibration evaluations read): ' + '; '.join(
            f'{f} {v["n_spec"]}/{v["n_usable"]}/{v["cal_evals"]}' for f, v in s['families'].items()) + '.')
        for pr in s['problems']:
            L.append(f'- flag: {pr}')
        L.append('')
        for name, t in s['tiers'].items():
            L.append(f'### Tier {name}: budget {fmt(t.get("budget"))} bytes/image'
                     + (f' (reference {t.get("reference_config")}, target {fmt(t.get("target"))})' if t.get('target') is not None else '')
                     + (f' — {t["reason"]}' if t.get('reason') else ''))
            L.append('')
            rows = [r for r in c['tiers'] if r['tier'] == name]
            if not any(r.get('config') or r.get('env_config') for r in rows):
                L += ['No family has a usable candidate within this budget.', '']
                continue
            L += table(['family', 'config', 'cal bytes', 'cal score', 'test bytes', 'test score [CI]', 'envelope (ref)', 'env score'],
                       [[r['family'], fmt(r.get('config')), fmt(r.get('cal_bytes')), fmt(r.get('cal_score')),
                         fmt(r.get('test_bytes')), (fmt(r.get('test_score')) + ' ' + ci(r, 'test_lo', 'test_hi')) if r.get('test_score') is not None
                         else ('missing on test' if r.get('test_missing') else ''), fmt(r.get('env_config')), fmt(r.get('env_score'))] for r in rows])
            cm = [r for r in c['comparisons'] if r['tier'] == name]
            if cm:
                L += [''] + table(['a − b', 'configs', 'diff [CI]', 'label', 'non-inferior', 'equivalent'],
                                  [[f'{r["a"]} − {r["b"]}', f'{fmt(r.get("config_a"))} vs {fmt(r.get("config_b"))}',
                                    (fmt(r.get('diff')) + ' ' + ci(r)) if r.get('diff') is not None else r.get('reason', ''),
                                    LABEL_TEXT.get(r.get('label'), ''), fmt(r.get('noninferior')), fmt(r.get('equivalent'))] for r in cm])
            L.append('')
        if c['targets']:
            L.append('### Bytes to the common target (chosen on calibration; envelope = reference)')
            L.append('')
            L += table(['target', 'family', 'config', 'cal target', 'test target', 'test bytes', 'test score', 'reached on test', 'envelope (ref)', 'env bytes'],
                       [[r['target'], r.get('family', ''), fmt(r.get('config')) if r.get('reached_calibration') else ('not reached' if 'family' in r else r.get('reason', '')),
                         fmt(r.get('cal_target')), fmt(r.get('test_target')), fmt(r.get('test_bytes')), fmt(r.get('test_score')),
                         fmt(r.get('reached_test')), fmt(r.get('env_config')), fmt(r.get('env_bytes'))] for r in c['targets']])
            L.append('')
        if c['endpoints']:
            L.append('### Endpoint tests (loss ≤ f · J(Z))')
            L.append('')
            L += table(['config', 'f', 'source', 'cal loss/J', 'test loss/J', 'margin f·J − L [CI]', 'label'],
                       [[r['config'], fmt(r['max_loss_frac_of_JZ'], 2), fmt(r.get('source')), fmt(r.get('cal_loss_over_JZ')),
                         fmt(r.get('test_loss_over_JZ')), (fmt(r.get('margin')) + ' ' + ci(r, 'margin_lo', 'margin_hi')) if r.get('margin') is not None
                         else r.get('reason', ''), LABEL_TEXT.get(r.get('label'), '')] for r in c['endpoints']])
            L.append('')
        if c['bd']:
            L.append('### BD-style summaries (calibration-Pareto configs on test)')
            L.append('')
            L += table(['a vs b', 'points', 'BD-score (a − b)', 'BD-rate % (a vs b)', 'note'],
                       [[f'{r["a"]} vs {r["b"]}', f'{r["points_a"]}/{r["points_b"]}', fmt(r.get('bd_score')), fmt(r.get('bd_rate_pct'), 1),
                         '; '.join(x for x in (r.get('reason'), r.get('bd_score_reason'), r.get('bd_rate_reason')) if x)] for r in c['bd']])
            L.append('')
    return '\n'.join(L) + '\n'


def plot(rows, path, title):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    ink, ink2, blue, red, grid = '#0b0b0b', '#52514e', '#2a78d6', '#e34948', '#e4e3df'
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)
    for ax, side in zip(axes, ('cal', 'test')):
        pts = [r for r in rows if r.get(f'{side}_score') is not None]
        ax.set_facecolor('#fcfcfb')
        for c, ls in ((CC, '--'), (ZZ, ':')):
            r = next((x for x in pts if x['config'] == c), None)
            if r:
                ax.axhline(r[f'{side}_score'], color=ink2, lw=1, ls=ls, label=f'{c} (no payload)' if c == ZZ else c)
        s = sorted((x for x in pts if x['kind'] == 'S' and x[f'{side}_bytes'] > 0), key=lambda x: x[f'{side}_bytes'])
        if s:
            ax.plot([x[f'{side}_bytes'] for x in s], [x[f'{side}_score'] for x in s], color=ink, lw=2, ls='--', marker='o', ms=4,
                    label='S shared QP (reference)')
        ax_rows = [x for x in pts if x['kind'] == 'axis' and x[f'{side}_bytes'] > 0]
        ks = sorted({x['k'] for x in ax_rows})
        mk = ['o', 's', '^', 'D', 'v', 'P']
        for k in ks:
            g = sorted((x for x in ax_rows if x['k'] == k), key=lambda x: x[f'{side}_bytes'])
            col = ink2 if k == 0 else blue if k > 0 else red
            ax.plot([x[f'{side}_bytes'] for x in g], [x[f'{side}_score'] for x in g], color=col, lw=2,
                    marker=mk[min(abs(k) // 2, len(mk) - 1)], ms=8, mec='#fcfcfb', mew=1.5,
                    label='N (k0)' if k == 0 else f'k{k:+d} ({"later coarser" if k > 0 else "later finer"})')
        for kind, m, col, lab in (('single_main', 'x', blue, 'q|Z (main only)'), ('single_later', '+', red, 'Z|q (later only)')):
            g = sorted((x for x in pts if x['kind'] == kind and x[f'{side}_bytes'] > 0), key=lambda x: x[f'{side}_bytes'])
            if g:
                ax.plot([x[f'{side}_bytes'] for x in g], [x[f'{side}_score'] for x in g], color=col, lw=1, ls='-',
                        marker=m, ms=8, mew=1.5, label=lab)
        ax.set_xscale('log')
        ax.grid(True, color=grid, lw=0.8)
        ax.set_axisbelow(True)
        for sp in ('top', 'right'):
            ax.spines[sp].set_visible(False)
        ax.set_xlabel('mean bytes per evaluation unit (log)', color=ink2)
        ax.set_title('calibration' if side == 'cal' else 'test', color=ink, fontsize=10)
    axes[0].set_ylabel('task score', color=ink2)
    h, lab = axes[1].get_legend_handles_labels()
    h0, lab0 = axes[0].get_legend_handles_labels()
    if len(lab0) > len(lab):
        h, lab = h0, lab0
    if h:
        fig.legend(h, lab, loc='center left', bbox_to_anchor=(1.0, 0.5), frameon=False, fontsize=8)
    fig.suptitle(title, color=ink, fontsize=11)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=130, bbox_inches='tight', facecolor='white')
    plt.close(fig)


def header(p, mode, obs, lock=None, notes=(), skipped=()):
    return dict(mode=mode, generated_at_utc=core.now_utc(), lock=lock, notes=list(notes), skipped=list(skipped), obs=[str(o) for o in obs],
                protocol=dict(path=p['_path'], name=p.get('name'), version=p.get('version'), sha256=core.protocol_sha(p),
                              pending=list(p.get('pending') or [])))


def run_report(p, obs, out, mode='dev', lock_path=None, plots=True, models=None, tasks=None):
    lock = core.check_lock(p, lock_path) if mode == 'confirm' else None
    cells, notes, skipped = load_cells(p, obs, mode, models, tasks)
    head = header(p, mode, obs, lock, notes, skipped)
    res, errors = [], []
    for cell in cells:
        try:
            res.append(analyze_cell(p, cell, mode))
        except (ValueError, KeyError) as e:
            errors.append(f'{cell["model"]}/{cell["task"]}: {e}')
            ct = p['cells'][cell['model']][cell['task']]
            res.append(dict(summary=dict(model=cell['model'], task=cell['task'], type=ct if isinstance(ct, str) else ct.get('type'),
                                         describe_only=False, error=str(e), info=cell['info']),
                            tiers=[], comparisons=[], targets=[], target_comparisons=[], endpoints=[], bd=[], curves=[]))
    out = Path(out)
    core.write_json(out / 'report.json', dict(header=head, errors=errors, cells=res))
    for k in ('tiers', 'comparisons', 'targets', 'target_comparisons', 'endpoints', 'bd', 'curves'):
        core.write_csv(out / f'{k}.csv', [r for c in res for r in c[k]])
    (out / 'report.md').write_text(markdown(res, head))
    if plots:
        for c in res:
            if c['curves']:
                s = c['summary']
                plot(c['curves'], out / 'plots' / f'{s["model"]}_{s["task"]}.png', f'{s["model"]} / {s["task"]} ({head["mode"]})')
    return head, res, errors


def run_select(p, obs, out, mode='dev', models=None, tasks=None):
    cells, notes, skipped = load_cells(p, obs, mode, models, tasks)
    head = header(p, mode, obs, None, notes, skipped)
    res, rows, need, errors = [], [], {}, []
    for cell in cells:
        try:
            sel = select_cell(p, cell['model'], cell['task'], cell['cal'], mode)
        except (ValueError, KeyError) as e:
            errors.append(f'{cell["model"]}/{cell["task"]}: {e}')
            continue
        need[f'{cell["model"]}/{cell["task"]}'] = test_configs(sel)
        sel = {k: v for k, v in sel.items() if k != 'spec'}
        res.append(sel)
        for name, t in sel['tiers'].items():
            rows += [dict(model=cell['model'], task=cell['task'], tier=name, budget=t['budget'], family=f, **pk)
                     for f, pk in t['picks'].items()]
    out = Path(out)
    core.write_json(out / 'select.json', dict(header=head, errors=errors, cells=res))
    core.write_json(out / 'test_configs.json', need)
    core.write_csv(out / 'selection_cal.csv', rows)
    return head, res, errors


CORNERS = ['C|C', 'C|Z', 'Z|C', 'Z|Z', 'C|M', 'M|C', 'M|M']


def cell_configs(p, model, task, corners=True):
    """Exact config names a cell needs: family candidates (union, first occurrence order), corners, endpoints."""
    t = core.cell_spec(p, model, task)
    out = []
    for f in t['families']:
        out += t['candidates'][f]
    out += (CORNERS if corners else [CC, ZZ]) + [e['config'] for e in t.get('endpoints') or []]
    seen = set()
    return [c for c in out if not (c in seen or seen.add(c))]


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest='cmd', required=True)
    s = sub.add_parser('configs')
    s.add_argument('--protocol', type=Path, default=core.DEFAULT_PROTOCOL)
    s.add_argument('--model', required=True)
    s.add_argument('--task', required=True)
    s.add_argument('--no-corners', action='store_true', help='only C|C and Z|Z (the gap budget rule needs both)')
    s = sub.add_parser('lock')
    s.add_argument('--protocol', type=Path, default=core.DEFAULT_PROTOCOL)
    s.add_argument('--out', type=Path, help='default: <protocol stem>.lock.json next to the protocol')
    s.add_argument('--allow-pending', action='store_true')
    s = sub.add_parser('hash-units')
    s.add_argument('--protocol', type=Path, default=core.DEFAULT_PROTOCOL)
    for name in ('select', 'report'):
        s = sub.add_parser(name)
        s.add_argument('--protocol', type=Path, default=core.DEFAULT_PROTOCOL)
        s.add_argument('--obs', nargs='+', required=True)
        s.add_argument('--out', type=Path, required=True)
        s.add_argument('--mode', choices=['dev', 'confirm'], default='dev')
        s.add_argument('--models', nargs='*')
        s.add_argument('--tasks', nargs='*')
        if name == 'report':
            s.add_argument('--lock', type=Path)
            s.add_argument('--no-plots', action='store_true')
    a = ap.parse_args(argv)
    p = core.load_protocol(a.protocol)
    if a.cmd == 'configs':
        print('\n'.join(cell_configs(p, a.model, a.task, not a.no_corners)))
        return 0
    try:
        if a.cmd == 'lock':
            rec, path = core.write_lock(p, a.out, a.allow_pending)
            print(json.dumps(dict(lock=str(path), **rec), indent=1))
            return 0
        if a.cmd == 'hash-units':
            field = p['units'].get('match_field', 'unit_id')
            out = {}
            for which in ('calibration', 'dev_test', 'confirm'):
                for task, spec in items(p['units'].get(which)):
                    path = core.resolve(p, (spec or {}).get('path'))
                    if path is not None and path.is_file():
                        out[f'{which}.{task}'] = dict(path=str(path), sha256=core.file_sha(path),
                                                      n_units=len(core.read_units(path, field)), protocol_sha256=spec.get('sha256'))
            print(json.dumps(out, indent=1))
            return 0
        if a.cmd == 'select':
            head, res, errors = run_select(p, a.obs, a.out, a.mode, a.models, a.tasks)
        else:
            head, res, errors = run_report(p, a.obs, a.out, a.mode, a.lock, not a.no_plots, a.models, a.tasks)
    except core.LockError as e:
        print(e, file=sys.stderr)
        return 3
    print(json.dumps(dict(out=str(a.out), cells=len(res), errors=errors, protocol_sha256=head['protocol']['sha256'],
                          notes=head['notes'], skipped=head['skipped']), indent=1))
    return 4 if errors else 0


if __name__ == '__main__':
    raise SystemExit(main())
