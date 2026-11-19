"""v19 training-pilot cross-evaluation report (CPU only).

Inputs: evaluation directories written by the runner (observations.jsonl + run-sweep-*.json): the untrained model
(meta lora null) and the four arms (plain SFT, generic compression-aware, role, anti-role), each at its short and
long checkpoint, all evaluated on the same test codestream configs. Arm and checkpoint come from
meta['lora']['meta'][training.meta_fields.*] mapped through training.arm_aliases, or from --evals-json.

Reported per model: full-input score of every arm; scores at the fixed-budget tiers; bytes needed to reach the
common target (training.reference: by default the plain-SFT full-input score at the same checkpoint); paired
differences role-generic, role-anti, generic-anti with unit-bootstrap intervals; the offset-axis role effect
(R - A over matched q, |k|) of every arm and its change from the untrained model; and the mechanical reading.

CLI: python -m cloud_prep.measurement_v19.confirm.training [--protocol P] --eval DIR... [--evals-json F]
       --out DIR [--mode dev|confirm] [--lock LOCK]
--evals-json: [{"dir": ..., "arm": ..., "checkpoint": ...}] (arm / checkpoint override the run meta).
Confirm mode needs the same lock as the confirmation report (exit 3 otherwise).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

from cloud_prep.measurement_v19 import levels
from cloud_prep.measurement_v19.confirm import core
from cloud_prep.measurement_v19.confirm.core import CC, EPS

READING_ZH = {
    'anti_highest': 'Anti-role scores highest. The prediction is refuted.',
    'role_above_generic': 'Role scores above generic. Role information improves on generic adaptation.',
    'role_eq_generic_above_anti': 'Role and generic are tied and both score above anti-role.',
    'no_difference': 'No difference was resolved among the three arms.',
    'undetermined': 'Undetermined',
}


def arm_of(tp, lora, override=None):
    alias = {str(k).lower(): v for k, v in tp.get('arm_aliases', {}).items()}
    if override and override.get('arm'):
        a = alias.get(str(override['arm']).lower(), override['arm'])
        return a, str(override.get('checkpoint') or 'none')
    if lora is None:
        return 'base', 'none'
    meta, f = lora.get('meta') or {}, tp['meta_fields']
    arm = meta.get(f['arm'])
    if arm is None:
        raise ValueError(f'LoRA meta lacks {f["arm"]!r} (keys {sorted(meta)}); set training.meta_fields or use --evals-json')
    a = alias.get(str(arm).lower())
    if a is None:
        raise ValueError(f'unknown arm {arm!r}; add it to training.arm_aliases')
    ck = meta.get(f['checkpoint'])
    if isinstance(ck, dict):            # train/lora_utils lora_meta: checkpoint = {label, step, samples_seen, total_steps}
        ck = ck.get('label')
    return a, str(ck) if ck is not None else 'none'


def load_evals(p, specs, overrides, mode):
    """{model: {(arm, ckpt): dict(cal, test, dir, lora, info)}} for the models in training.models."""
    tp = p['training']
    ev = {}
    for d in specs:
        groups, _ = core.load_groups(p, [d], lora=True)
        for (model, _, task), g in groups.items():
            ms = tp['models'].get(model)
            if not ms or ms['task'] != task:
                continue
            arm, ck = arm_of(tp, g['lora'], overrides.get(str(Path(d).resolve())))
            if (arm, ck) in ev.setdefault(model, {}):
                raise ValueError(f'{model}: two evaluations for arm {arm} checkpoint {ck}: {ev[model][arm, ck]["dir"]}, {d}')
            ts = p['tasks'][task]
            cal, test, info = core.split_rows(p, task, list(g['rows'].values()), mode)
            mk = lambda rr: core.View(rr, ts.get('unit_field', 'unit_id'), ts.get('score_field', 'correct'))
            ev[model][arm, ck] = dict(cal=mk(cal), test=mk(test), dir=str(d), lora=g['lora'], info=info)
    return ev


def tier_choice(e, spec, cands_all):
    """(config, basis) for one evaluation and tier: a fixed config, or the best select-split candidate within the
    byte budget (falls back to the test envelope, labelled, when the evaluation has no select-split rows)."""
    if 'config' in spec:
        return spec['config'], 'fixed config'
    cands = spec.get('candidates') or [c for c in cands_all if c != CC]
    if any(c in e['cal'] for c in cands):
        return core.pick_budget(e['cal'], cands, float(spec['bytes'])), 'chosen on calibration'
    return core.pick_budget(e['test'], cands, float(spec['bytes'])), 'test envelope (reference only)'


def reading(d, tol):
    """Mechanical case from the role-generic, role-anti, generic-anti intervals (tolerance for 'level')."""
    rg, ra, ga = d.get('role-generic'), d.get('role-anti'), d.get('generic-anti')
    if not all(x and x.get('lo') is not None for x in (rg, ra, ga)):
        return None
    pos = lambda x: x['lo'] > 0
    level = lambda x: -tol < x['lo'] and x['hi'] < tol
    if ra['hi'] < 0 and ga['hi'] < 0:
        return 'anti_highest'
    if pos(rg):
        return 'role_above_generic'
    if level(rg) and pos(ra) and pos(ga):
        return 'role_eq_generic_above_anti'
    if level(rg) and level(ra) and level(ga):
        return 'no_difference'
    return 'undetermined'


def analyze_model(p, model, evals):
    tp = p['training']
    ms = tp['models'][model]
    role, tol, bp = int(ms['role']), float(tp['tolerance']), p['bootstrap']
    units = sorted({u for e in evals.values() for u in e['test'].units})
    boot = core.Boot(units, bp['n_boot'], bp['seed'], bp.get('ci', 0.95))
    head = dict(model=model, task=ms['task'])
    keys = sorted(evals, key=lambda k: (k[0] != 'base', k[1], k[0]))
    ckpts = sorted({ck for a, ck in evals if a != 'base'})
    cfgs = sorted(set.intersection(*[set(e['test'].configs()) for e in evals.values()]))
    est = lambda e, c: boot.est(e['test'].unit_mean(c)) if c in e['test'] else dict(est=None, lo=None, hi=None, n_units=0)
    full, per_config, tiers, targets, cmps, effect, reads = [], [], [], [], [], [], []
    for k in keys:
        e = evals[k]
        d = est(e, CC)
        full.append(dict(head, arm=k[0], checkpoint=k[1], score=d['est'], lo=d['lo'], hi=d['hi'], n_units=d['n_units'],
                         n_questions=e['test'].n(CC), dir=e['dir']))
        for c in e['test'].configs():
            per_config.append(dict(head, arm=k[0], checkpoint=k[1], config=c, score=e['test'].score(c),
                                   bytes=e['test'].nbytes(c), n_questions=e['test'].n(c)))
    tspec = {**tp['tiers'], **tp.get('model_tiers', {}).get(model, {})}
    chosen = {}
    for k in keys:
        for name, spec in tspec.items():
            c, basis = tier_choice(evals[k], spec, cfgs)
            chosen[k, name] = c
            d = est(evals[k], c)
            tiers.append(dict(head, arm=k[0], checkpoint=k[1], tier=name, config=c, basis=basis,
                              bytes=evals[k]['test'].nbytes(c) if c in evals[k]['test'] else None,
                              score=d['est'], lo=d['lo'], hi=d['hi'], n_units=d['n_units']))
    ref = tp['reference']
    tcands = cfgs if tp.get('target_candidates', 'all') == 'all' else tp['target_candidates']
    for ref_ck in (ckpts if ref.get('checkpoint', 'same') == 'same' else [ref['checkpoint']]):
        r = evals.get((ref['arm'], ref_ck))
        if r is None:
            targets.append(dict(head, reference=f'{ref["arm"]}/{ref_ck}', reason='reference evaluation missing'))
            continue
        try:
            tt = core.score_target(r['test'], ref['target']) if ref['target']['type'] != 'abs' else float(ref['target']['value'])
        except KeyError as ex:
            targets.append(dict(head, reference=f'{ref["arm"]}/{ref_ck}', reason=f'{ex.args[0]} on test'))
            continue
        try:
            tc = core.score_target(r['cal'], ref['target']) if ref['target']['type'] != 'abs' else tt
        except KeyError:
            tc = None
        for k in keys:
            if k[0] != 'base' and k[1] != ref_ck:
                continue
            e = evals[k]
            c = core.pick_target(e['cal'], tcands, tc) if tc is not None else None
            basis = 'chosen on calibration'
            if c is None and (tc is None or not any(x in e['cal'] for x in tcands)):
                c, basis = core.pick_target(e['test'], tcands, tt), 'test envelope (reference only)'
            row = dict(head, reference=f'{ref["arm"]}/{ref_ck}', cal_target=tc, test_target=tt, arm=k[0], checkpoint=k[1],
                       config=c, basis=basis, reached=c is not None)
            if c and c in e['test']:
                row.update(test_bytes=e['test'].nbytes(c), test_score=e['test'].score(c),
                           reached_test=bool(e['test'].score(c) >= tt - EPS))
            targets.append(row)
    for ck in ckpts:
        for name in tspec:
            dd = {}
            for a, b in tp['comparisons']:
                ea, eb = evals.get((a, ck)), evals.get((b, ck))
                ca, cb = chosen.get(((a, ck), name)), chosen.get(((b, ck), name))
                row = dict(head, checkpoint=ck, tier=name, a=a, b=b, config_a=ca, config_b=cb)
                if not ea or not eb or ca not in ea['test'] or cb not in eb['test']:
                    row['reason'] = 'arm evaluation or config missing'
                else:
                    qv = core.qvals([ea['test'].by[ca], eb['test'].by[cb]], (1, -1))
                    d = boot.est(core.unit_means(qv))
                    row.update(diff=d['est'], lo=d['lo'], hi=d['hi'], n_units=d['n_units'], n_questions=len(qv), **core.tol_flags(d, tol))
                    dd[f'{a}-{b}'] = d
                cmps.append(row)
            if name in tp.get('reading_tiers', []):
                case = reading(dd, tol)
                reads.append(dict(head, checkpoint=ck, tier=name, case=case, case_zh=READING_ZH.get(case, ''),
                                  tolerance=tol, **{f'{n}_{x}': dd.get(n, {}).get(x) for n in ('role-generic', 'role-anti', 'generic-anti')
                                                    for x in ('est', 'lo', 'hi')}))
    pairs = sorted({(q, abs(k)) for c in cfgs for form, q, k in [levels.parse_config(c)] if form == 'offset' and k}
                   if cfgs else set())
    pairs = [(q, n) for q, n in pairs if levels.offset_name(q, role * n) in cfgs and levels.offset_name(q, -role * n) in cfgs]
    qeff = {}
    for k in keys:
        by = evals[k]['test'].by
        qeff[k] = core.avg_qvals([core.qvals([by[levels.offset_name(q, role * n)], by[levels.offset_name(q, -role * n)]], (1, -1))
                                  for q, n in pairs]) if pairs else {}
        d = boot.est(core.unit_means(qeff[k]))
        row = dict(head, arm=k[0], checkpoint=k[1], role=role, pairs=' '.join(f'q{q}/|k|{n}' for q, n in pairs),
                   effect=d['est'], lo=d['lo'], hi=d['hi'], n_units=d['n_units'])
        base = ('base', 'none')
        if k != base and base in qeff and qeff[base] and qeff[k]:
            common = set(qeff[k]) & set(qeff[base])
            dv = boot.est(core.unit_means({q: (qeff[k][q][0], qeff[k][q][1] - qeff[base][q][1]) for q in common}))
            row.update(change_vs_base=dv['est'], change_lo=dv['lo'], change_hi=dv['hi'])
        effect.append(row)
    return dict(summary=dict(head, role=role, evaluations=[f'{a}/{c}' for a, c in keys], common_configs=len(cfgs),
                             test_units=len(units), offset_pairs=len(pairs)),
                full_input=full, per_config=per_config, tiers=tiers, targets=targets, comparisons=cmps,
                role_effect=effect, reading=reads)


def fmt(x, nd=3):
    from cloud_prep.measurement_v19.confirm.report import fmt as f
    return f(x, nd)


def markdown(res, head):
    from cloud_prep.measurement_v19.confirm.report import table
    ci = lambda r, a='lo', b='hi': f'[{fmt(r.get(a))}, {fmt(r.get(b))}]' if r.get(a) is not None else ''
    L = [f'# v19 training pilot cross-evaluation ({head["mode"]})', '',
         f'Protocol {head["protocol"]["name"]} {head["protocol"]["version"]}, sha256 {head["protocol"]["sha256"]}.']
    if head.get('lock'):
        L.append(f'Lock at {head["lock"]["locked_at_utc"]}.')
    L.append('')
    for m in res:
        s = m['summary']
        L += [f'## {s["model"]} / {s["task"]} (role {s["role"]:+d})', '',
              f'Evaluations: {", ".join(s["evaluations"])}. Common configs {s["common_configs"]}, test units {s["test_units"]}, '
              f'offset pairs for the role effect {s["offset_pairs"]}.', '', '### Full input (C|C)', '']
        L += table(['arm', 'checkpoint', 'score [CI]', 'units'],
                   [[r['arm'], r['checkpoint'], f'{fmt(r["score"])} {ci(r)}', fmt(r['n_units'])] for r in m['full_input']])
        L += ['', '### Fixed-budget tiers', '']
        L += table(['tier', 'arm', 'checkpoint', 'config', 'basis', 'bytes', 'score [CI]'],
                   [[r['tier'], r['arm'], r['checkpoint'], fmt(r['config']), r['basis'], fmt(r['bytes']), f'{fmt(r["score"])} {ci(r)}']
                    for r in m['tiers']])
        L += ['', '### Bytes to the common target', '']
        L += table(['reference', 'arm', 'checkpoint', 'test target', 'config', 'basis', 'test bytes', 'test score', 'reached on test'],
                   [[r['reference'], r.get('arm', ''), r.get('checkpoint', ''), fmt(r.get('test_target')),
                     fmt(r.get('config')) if r.get('reached') else ('not reached' if 'arm' in r else r.get('reason', '')),
                     r.get('basis', ''), fmt(r.get('test_bytes')), fmt(r.get('test_score')), fmt(r.get('reached_test'))] for r in m['targets']])
        L += ['', '### Paired arm differences', '']
        L += table(['checkpoint', 'tier', 'a − b', 'configs', 'diff [CI]', 'level within tol'],
                   [[r['checkpoint'], r['tier'], f'{r["a"]} − {r["b"]}', f'{fmt(r["config_a"])} vs {fmt(r["config_b"])}',
                     f'{fmt(r.get("diff"))} {ci(r)}' if r.get('diff') is not None else r.get('reason', ''), fmt(r.get('equivalent'))]
                    for r in m['comparisons']])
        L += ['', '### Offset-axis role effect (R − A), and its change from the untrained model', '']
        L += table(['arm', 'checkpoint', 'R − A [CI]', 'change vs base [CI]'],
                   [[r['arm'], r['checkpoint'], f'{fmt(r["effect"])} {ci(r)}',
                     f'{fmt(r.get("change_vs_base"))} {ci(r, "change_lo", "change_hi")}' if r.get('change_vs_base') is not None else '']
                    for r in m['role_effect']])
        L += ['', '### Reading', '']
        L += table(['checkpoint', 'tier', 'case', 'role − generic', 'role − anti', 'generic − anti'],
                   [[r['checkpoint'], r['tier'], r['case_zh'] or fmt(r['case'])] +
                    [f'{fmt(r[f"{n}_est"])} [{fmt(r[f"{n}_lo"])}, {fmt(r[f"{n}_hi"])}]' if r.get(f'{n}_est') is not None else ''
                     for n in ('role-generic', 'role-anti', 'generic-anti')] for r in m['reading']])
        L.append('')
    return '\n'.join(L) + '\n'


def run(p, specs, out, mode='dev', lock_path=None, overrides=None):
    from cloud_prep.measurement_v19.confirm.report import header
    lock = core.check_lock(p, lock_path) if mode == 'confirm' else None
    ev = load_evals(p, specs, overrides or {}, mode)
    head = header(p, mode, specs, lock)
    res = [analyze_model(p, m, ev[m]) for m in sorted(ev)]
    out = Path(out)
    core.write_json(out / 'training.json', dict(header=head, models=res))
    for k in ('full_input', 'per_config', 'tiers', 'targets', 'comparisons', 'role_effect', 'reading'):
        core.write_csv(out / f'{k}.csv', [r for m in res for r in m[k]])
    (out / 'training.md').write_text(markdown(res, head))
    return head, res


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--protocol', type=Path, default=core.DEFAULT_PROTOCOL)
    ap.add_argument('--eval', nargs='*', default=[], help='evaluation directories (observations.jsonl + run-sweep-*.json)')
    ap.add_argument('--evals-json', type=Path, help='[{"dir", "arm", "checkpoint"}]; adds the dirs and overrides their meta')
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--mode', choices=['dev', 'confirm'], default='dev')
    ap.add_argument('--lock', type=Path)
    a = ap.parse_args(argv)
    p = core.load_protocol(a.protocol)
    specs, overrides = list(a.eval), {}
    if a.evals_json:
        for e in json.loads(a.evals_json.read_text()):
            overrides[str(Path(e['dir']).resolve())] = e
            if e['dir'] not in specs:
                specs.append(e['dir'])
    try:
        head, res = run(p, specs, a.out, a.mode, a.lock, overrides)
    except core.LockError as e:
        print(e, file=sys.stderr)
        return 3
    print(json.dumps(dict(out=str(a.out), models=[m['summary'] for m in res], protocol_sha256=head['protocol']['sha256']), indent=1))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
