"""Compute the confirmation and development estimates from per-question records."""
import numpy as np
from analysis import coding_observations as cd
from analysis.coding_observations import core
from analysis.coding_csv import family

CELL_INFO = {
    'q8': dict(model='qwen8b', task='gqa_attribute', image_set='confirm1000', unit='image',
               members=['cf_qwen8b_attr', 'fu_qwen8b_attr']),
    'q32a': dict(model='qwen32b', task='gqa_attribute', image_set='confirm1000', unit='image',
                 members=['cf_qwen32b_attr', 'fu_qwen32b_attr']),
    'q32d': dict(model='qwen32b', task='docvqa', image_set='confirm474', unit='document',
                 members=['cf_qwen32b_doc', 'fu_qwen32b_doc']),
    'ds': dict(model='deepstack7b', task='gqa_attribute', image_set='confirm1000', unit='image',
               members=['cf_deepstack7b_attr', 'fu_deepstack7b_attr']),
    'g468': dict(model='ibm4b', task='gqa_attribute', image_set='granite468', unit='image', members=['cf_ibm4b_attr']),
    'g531': dict(model='ibm4b', task='gqa_attribute', image_set='granite531', unit='image',
                 members=['fu_ibm4b_corners', 'fu_ibm4b_pair']),
    'g999': dict(model='ibm4b', task='gqa_attribute', image_set='granite999', unit='image',
                 members=['cf_ibm4b_attr', 'fu_ibm4b_corners']),
}
def build_cells(D):
    U468 = set(D['txt_ibm4b_excluded']) - {cd.BOUNDARY}
    U531 = {r['unit_id'] for r in D['man_ibm4b_remaining']}
    assert len(U468) == 468 and len(U531) == 531 and not U468 & U531 and cd.BOUNDARY not in U531
    views = {
        'q8': cd.View(D['cf_qwen8b_attr'] + D['fu_qwen8b_attr']),
        'q32a': cd.View(D['cf_qwen32b_attr'] + D['fu_qwen32b_attr']),
        'q32d': cd.View(D['cf_qwen32b_doc'] + D['fu_qwen32b_doc'], unit_field='cluster_id'),
        'ds': cd.View(D['cf_deepstack7b_attr'] + D['fu_deepstack7b_attr']),
        'g468': cd.View(D['cf_ibm4b_attr'], units=U468),
        'g531': cd.View(D['fu_ibm4b_corners'] + D['fu_ibm4b_pair'], units=U531),
        'g999': cd.View([r for r in D['cf_ibm4b_attr'] if r[2] in U468 and r[1] in cd.CORNERS]
                        + [r for r in D['fu_ibm4b_corners'] if r[2] in U531]),
    }
    cells = {k: cd.Cell(k, v) for k, v in views.items()}
    checks = {}
    for k, c in cells.items():
        per_cfg = {cfg: (c.v.n_units(cfg), c.v.n_q(cfg)) for cfg in c.v.by}
        checks[k] = dict(units=len(c.v.units), duplicates=c.v.dups, configs=len(c.v.by),
                         incomplete=[cfg for cfg, (nu, _) in per_cfg.items() if nu != len(c.v.units)],
                         questions=sorted({nq for _, nq in per_cfg.values()}))
    return cells, checks, U468, U531


def vs_curve(cell, test_cfgs, curve_cfgs):
    """Post hoc matched-bytes contrast as norm_vs_shared.py vs_curve (protocol weights of the cell's core.Boot):
    the test configuration's score minus the shared curve interpolated linearly in log10 bytes; per resample the
    curve and the test point are recomputed with the same weights; resamples outside the curve range are dropped."""
    v, boot = cell.v, cell.boot
    w = boot.w
    ws = w.sum(1)
    idx = boot.idx

    def arr(c, col):
        m = v.unit_mean(c, col)
        a = np.zeros(len(boot.units))
        for u, x in m.items():
            a[idx[u]] = x
        return a
    S = {c: (arr(c, 2), arr(c, 1)) for c in curve_cfgs}
    pts = sorted((S[c][0].mean(), S[c][1].mean(), c) for c in curve_cfgs)
    cb = np.array([S[c][0] for c in curve_cfgs])
    cs = np.array([S[c][1] for c in curve_cfgs])
    bb = (w @ cb.T)/ws[:, None]
    sb = (w @ cs.T)/ws[:, None]
    order = np.argsort(bb, axis=1)
    lbb = np.log10(np.take_along_axis(bb, order, 1))
    sbs = np.take_along_axis(sb, order, 1)
    out = {}
    for c in test_cfgs:
        tb_u, ts_u = arr(c, 2), arr(c, 1)
        xb, xs = tb_u.mean(), ts_u.mean()
        lx = np.log10([p[0] for p in pts])
        if not lx[0] <= np.log10(xb) <= lx[-1]:
            out[c] = dict(config=c, inside=False, bytes=xb, score=xs)
            continue
        ref = float(np.interp(np.log10(xb), lx, [p[1] for p in pts]))
        tb, ts = np.log10((w @ tb_u)/ws), (w @ ts_u)/ws
        dd = np.full(len(w), np.nan)
        for i in range(len(w)):
            if lbb[i, 0] <= tb[i] <= lbb[i, -1]:
                dd[i] = ts[i]-np.interp(tb[i], lbb[i], sbs[i])
        ok = np.isfinite(dd)
        lo, hi = np.quantile(dd[ok], [0.025, 0.975])
        k = int(np.searchsorted(lx, np.log10(xb)))
        out[c] = dict(config=c, inside=True, bytes=xb, score=xs, curve_score=ref, est=xs-ref, lo=float(lo), hi=float(hi),
                      dropped=int(len(w)-ok.sum()), bracket=[pts[max(k-1, 0)][2], pts[min(k, len(pts)-1)][2]],
                      n_units=len(boot.units))
    return out


def training(D):
    """Development-panel training evaluations; one core.Boot per model over the union of evaluated units
    (confirm/training.py analyze_model)."""
    res, views = {}, {}
    for model in ('qwen8b', 'ibm4b'):
        ev = {}
        for arm in ('base', 'sft-long', 'generic-long', 'role-long', 'anti-long',
                    'sft-short', 'generic-short', 'role-short', 'anti-short'):
            ev[arm] = cd.View(D[f'tr_{model}_{arm}'])
        units = sorted({u for v in ev.values() for u in v.units})
        boot = core.Boot(units, n_boot=cd.NB, seed=cd.SEED, ci=cd.CI)
        views[model] = (ev, boot)
        for arm, v in ev.items():
            a, ck = (arm, 'none') if arm == 'base' else arm.split('-')
            for c in v.by:
                d = boot.est(v.unit_mean(c))
                res[f'tscore/{model}/{ck}/{a}/{c}'] = dict(est=d['est'], lo=d['lo'], hi=d['hi'], n_units=d['n_units'],
                                                           n_questions=v.n_q(c), bytes=v.mean(c, 2), model=model,
                                                           checkpoint=ck, arm=a, config=c)
        for ck in ('long', 'short'):
            for a, b in (('generic', 'sft'), ('role', 'generic'), ('role', 'anti'), ('generic', 'anti'),
                         ('role', 'sft'), ('anti', 'sft')):
                va, vb = ev[f'{a}-{ck}'], ev[f'{b}-{ck}']
                for c in va.by:
                    qv = core.qvals([va.by[c], vb.by[c]], (1, -1))
                    d = boot.est(core.unit_means(qv))
                    res[f'tdiff/{model}/{ck}/{a}/{b}/{c}'] = dict(est=d['est'], lo=d['lo'], hi=d['hi'],
                                                                  n_units=d['n_units'], n_questions=len(qv), model=model,
                                                                  checkpoint=ck, a=a, b=b, config=c)
    return res


def compute():
    D = cd.load_observations()
    cells, checks, U468, U531 = build_cells(D)
    E = {}
    for k, c in cells.items():
        for cfg in c.v.by:
            E[f'score/{k}/{cfg}'] = dict(c.score(cfg), cell=k)
    diffs = []
    for k in ('q8', 'q32a', 'ds', 'g468', 'g531', 'g999', 'q32d'):
        for a in ('C|Z', 'C|M', 'Z|C', 'M|C'):
            diffs.append((k, a, 'C|C'))
        diffs += [(k, 'C|M', 'M|C'), (k, 'C|Z', 'Z|C'), (k, 'M|C', 'C|M'), (k, 'Z|C', 'M|M')]
    for q in ('q-2', 'q-1', 'q-4'):
        diffs.append(('ds', f'{q}|Z', f'{q}|{q}'))
    diffs += [('ds', 'q-4|Z', 'q-2|q-2'), ('ds', 'q-2|k+2', 'q-2|k0'), ('ds', 'q-2|k+2', 'q-2|q-2'),
              ('ds', 'q-2|Z', 'q-2|k+2'), ('ds', 'q-4|Z', 'q-2|k0'), ('ds', 'q-4|Z', 'q-2|k+2')]
    for q in ('q2', 'q6', 'q0'):
        diffs.append(('q8', f'{q}|Z', f'{q}|{q}'))
    diffs += [('q8', 'q0|Z', 'q2|q2'), ('q8', 'q2|k+4', 'q2|q2'), ('q8', 'q2|k+4', 'q10|k-4'), ('q8', 'q6|k+4', 'q10|k-2')]
    for q in ('q6', 'q12'):
        diffs.append(('q32a', f'{q}|Z', f'{q}|{q}'))
    for q in ('q6', 'q10'):
        diffs.append(('q32d', f'{q}|Z', f'{q}|{q}'))
    diffs += [('q32d', 'q8|k+4', 'q12|k-2'), ('q32d', 'q12|k+2', 'q16|k-4')]
    diffs += [('g531', 'q0|k+4', 'q-4|q-4'), ('g468', 'q0|k+4', 'q-4|q-4'), ('g468', 'q4|k0', 'q-4|q-4'),
              ('g468', 'q0|k+4', 'q4|k0'), ('g468', 'q8|k-4', 'q0|k+2'), ('g468', 'q12|k-4', 'q8|k+2')]
    for k, a, b in diffs:
        E[f'diff/{k}/{a}/{b}'] = dict(cells[k].diff(a, b), cell=k)
    for k, cfg, f in (('q8', 'C|Z', .25), ('ds', 'C|Z', .5), ('ds', 'Z|C', .5), ('q32a', 'C|Z', .5), ('q32a', 'Z|C', .5),
                      ('g468', 'Z|C', .25), ('g531', 'Z|C', .25), ('g999', 'Z|C', .25)):
        E[f'end/{k}/{cfg}'] = dict(cells[k].endpoint(cfg, f), cell=k)
    for k in ('q8', 'q32a', 'q32d', 'ds'):
        shared = [c for c in cells[k].v.by if family(c) == 'shared']
        tests = [c for c in cells[k].v.by if family(c) in ('main_only', 'scale_normalized')]
        for c, r in vs_curve(cells[k], tests, shared).items():
            E[f'match/{k}/{c}'] = dict(r, cell=k)
    E.update(training(D))
    return E
