"""Read local per-question observations and estimate paired unit-level outcomes."""
import json
import numpy as np
from cloud_prep.measurement_v19.confirm import core
from analysis.paths import INPUT

NB, SEED, CI = 4000, 20261112, 0.95
BOUNDARY = 'gqa-image-2410156'
CORNERS = ['C|C', 'C|Z', 'C|M', 'Z|C', 'M|C', 'Z|Z', 'M|M']
MEMBERS = {
    'confirmation_a': {
        'runs/confirm/qwen8b/gqa-attribute/sweep/observations.jsonl': 'cf_qwen8b_attr',
        'runs/confirm/deepstack7b/gqa-attribute/sweep/observations.jsonl': 'cf_deepstack7b_attr',
        'runs/confirm/ibm4b/gqa-attribute/sweep/observations.jsonl': 'cf_ibm4b_attr',
    },
    'confirmation_b': {
        'runs/confirm/qwen32b/gqa-attribute/sweep/observations.jsonl': 'cf_qwen32b_attr',
        'runs/confirm/qwen32b/docvqa/sweep/observations.jsonl': 'cf_qwen32b_doc',
    },
    'followup_a': {
        'followup/deepstack7b-attr-delater/sweep/observations.jsonl': 'fu_deepstack7b_attr',
        'followup/qwen8b-attr-delater/sweep/observations.jsonl': 'fu_qwen8b_attr',
        'followup/ibm4b-attr-corners/sweep/observations.jsonl': 'fu_ibm4b_corners',
        'followup/ibm4b-attr-posthoc-pair/sweep/observations.jsonl': 'fu_ibm4b_pair',
        'followup/inputs/ibm4b-attr-excluded-units.txt': 'txt_ibm4b_excluded',
        'followup/inputs/ibm4b-attr-remaining-manifest.jsonl': 'man_ibm4b_remaining',
    },
    'followup_b': {
        'followup/qwen32b-attr-delater/sweep/observations.jsonl': 'fu_qwen32b_attr',
        'followup/qwen32b-docvqa-delater/sweep/observations.jsonl': 'fu_qwen32b_doc',
    },
}


def _compact(fobj):
    """Rows with the fields the analysis needs, file order kept (later rows win, as core.load_groups)."""
    out = []
    for line in fobj:
        if not line.strip():
            continue
        r = json.loads(line)
        b = r.get('bytes') or {}
        out.append([r.get('key'), r['config'], r.get('unit_id'), r.get('cluster_id'), r['id'],
                    float(r['correct']), b.get('main'), b.get('later'), r.get('task')])
    return out


class View:
    """by[config][qid] = (unit, score, bytes, main, later); duplicate (qid, config) rows: the later row wins."""

    def __init__(self, rows, unit_field='unit_id', units=None, configs=None):
        self.by, self.dups = {}, 0
        uf = 3 if unit_field == 'cluster_id' else 2
        for r in rows:
            cfg, u = r[1], r[uf]
            if configs is not None and cfg not in configs:
                continue
            if units is not None and u not in units:
                continue
            bm, bl = r[6], r[7]
            bm = float('nan') if bm is None else float(bm)
            bl = float('nan') if bl is None else float(bl)
            d = self.by.setdefault(cfg, {})
            if r[4] in d:
                self.dups += 1
            d[r[4]] = (u, r[5], bm + bl, bm, bl)
        self.units = sorted({v[0] for d in self.by.values() for v in d.values()})

    def __contains__(self, c):
        return c in self.by

    def unit_mean(self, c, col=1):
        acc = {}
        for v in self.by[c].values():
            acc.setdefault(v[0], []).append(v[col])
        return {u: float(np.mean(x)) for u, x in acc.items()}

    def mean(self, c, col=1):
        m = self.unit_mean(c, col)
        return float(np.mean(list(m.values())))

    def n_units(self, c):
        return len({v[0] for v in self.by[c].values()})

    def n_q(self, c):
        return len(self.by[c])


class Cell:
    """One fixed unit set with one shared core.Boot (weights drawn once over the sorted units)."""

    def __init__(self, name, view, status_note=''):
        self.name, self.v = name, view
        self.boot = core.Boot(view.units, n_boot=NB, seed=SEED, ci=CI)
        self.note = status_note

    def score(self, c):
        d = self.boot.est(self.v.unit_mean(c))
        return dict(config=c, est=d['est'], lo=d['lo'], hi=d['hi'], n_units=d['n_units'], n_questions=self.v.n_q(c),
                    bytes=self.v.mean(c, 2), bytes_main=self.v.mean(c, 3), bytes_later=self.v.mean(c, 4))

    def diff(self, a, b):
        qv = core.qvals([self.v.by[a], self.v.by[b]], (1, -1))
        d = self.boot.est(core.unit_means(qv))
        sa, sb = self.score(a), self.score(b)
        return dict(a=a, b=b, est=d['est'], lo=d['lo'], hi=d['hi'], n_units=d['n_units'], n_questions=len(qv),
                    a_score=sa, b_score=sb, bytes_change_pct=(sa['bytes']/sb['bytes']-1)*100 if sb['bytes'] else None)

    def endpoint(self, c, f):
        """report.endpoint_rows: margin E - (1-f) C|C - f Z|Z; loss and J(Z) on the cell."""
        qv = core.qvals([self.v.by[c], self.v.by['C|C'], self.v.by['Z|Z']], (1, -(1-f), -f))
        d = self.boot.est(core.unit_means(qv))
        cc, zz, ee = self.v.mean('C|C'), self.v.mean('Z|Z'), self.v.mean(c)
        lab = core.label(d)
        return dict(config=c, f=f, margin=d['est'], lo=d['lo'], hi=d['hi'], n_units=d['n_units'], n_questions=len(qv),
                    loss=cc-ee, JZ=cc-zz, loss_over_JZ=(cc-ee)/(cc-zz), label=lab)



def load_observations():
    """Read local runner outputs using their experiment-relative names under coding/."""
    root = INPUT / 'coding'
    data = {}
    for group in MEMBERS.values():
        for name, tag in group.items():
            with (root/name).open() as handle:
                if tag.startswith('txt_'):
                    data[tag] = [line.strip() for line in handle if line.strip()]
                elif tag.startswith('man_'):
                    data[tag] = [json.loads(line) for line in handle if line.strip()]
                else:
                    data[tag] = _compact(handle)
    for model in ('qwen8b','ibm4b'):
        for path in sorted((root/'runs/train-eval-panel'/model/'pilot1').glob('*/observations.jsonl')):
            with path.open() as handle:
                data[f'tr_{model}_{path.parent.name}'] = _compact(handle)
    return data
