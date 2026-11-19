"""CPU tests of v15 severity rows: pairing, identities, reuse and readout mappings."""
import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from PIL import Image, ImageDraw

from cloud_prep.measurement_v15 import design, severity
from cloud_prep.measurement_v15.analysis import analyze
from cloud_prep.measurement_v15.runner import check_resume_execution, make_variant, outcome_fields

CONFIGS = Path(__file__).resolve().parents[1] / 'configs'
LABELS = ['blurrel-period0.18', 'blurrel-period0.25', 'blurrel-period0.32']
DIRECT = dict(response_format='direct', mapping='none')


def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_row(root, labels=LABELS, size=(64, 48), period=8):
    """One stripe query with clean a, counterfactual b and one degraded file per label."""
    root = Path(root); (root/'images').mkdir(exist_ok=True)
    a = Image.new('RGB', size, (230, 230, 230)); d = ImageDraw.Draw(a)
    for x in range(0, size[0], period): d.line((x, 0, x, size[1]), fill=(40, 40, 40), width=3)
    b = a.rotate(90, expand=False)
    images = {'a': a, 'b': b}; specs = {}
    for label in labels:
        images[label], specs[label] = severity.apply(a, label, period)
    paths = {}; meta = {}
    for name, image in images.items():
        path = root/'images'/f'{name}.png'; image.save(path); paths[name] = f'images/{name}.png'
        meta[name] = dict(width=image.width, height=image.height, sha256=sha(path))
    row = dict(id='u1-edit0-q0', unit_id='u1', intervention_id='u1-edit0', task='stripe_orientation', query_role='changed',
        query='What pattern?', choices=['vertical stripes', 'horizontal stripes'], option_labels=['(A)', '(B)'],
        answer_index=0, answer_label='(A)', reference_answers=['vertical stripes'], direction_a_index=0, direction_b_index=1,
        donor_roi_bbox_xyxy=[0, 0, size[0], size[1]], source_images=paths, source_image_metadata=meta,
        severity_sources={x: x for x in labels}, degradations=specs, split='evaluation', material_set='test')
    manifest = root/'materials.jsonl'; manifest.write_text(json.dumps(row)+'\n')
    return manifest, row


def config(name='line1_severity_batched.json', **changes):
    value = design.settings(CONFIGS/name); value.update(changes); return value


def keys(row, cfg, model={'key': 'm', 'revision': 'r'}):
    return {(c['name'], json.dumps(ro, sort_keys=True), kind): design.observation_key(row, c, ro, kind, model, cfg)
            for c in design.conditions(row, cfg) for ro in design.readouts(row, cfg) for kind in ('score', 'generation')}


class SeverityConditionTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(); self.manifest, _ = write_row(self.tmp.name)
        self.cfg = config(severity_labels=None)
        self.row = design.load_rows(self.manifest, self.cfg)[0]

    def tearDown(self): self.tmp.cleanup()

    def test_four_cell_and_native_only_condition_sets(self):
        names = [c['name'] for c in design.conditions(self.row, self.cfg)]
        self.assertEqual(len(names), 18); self.assertEqual(len(set(names)), 18)
        self.assertEqual(sum(n.startswith('conflict__') for n in names), 2+2*len(LABELS))
        native = config('calibration_native.json')
        cells = {(c['main_quality'], c['later_quality']) for c in design.conditions(self.row, native)}
        self.assertEqual(cells, {('clean', 'clean')} | {(x, x) for x in LABELS})

    def test_exact_four_cell_pairing_shares_one_degraded_file(self):
        cc = {(c['family'], c['main_quality'], c.get('later_quality', c.get('later_endpoint'))): c
              for c in design.conditions(self.row, self.cfg)}
        clean = cc[('recovery', 'clean', 'clean')]
        self.assertEqual((clean['main_source'], clean['later_base'], clean['severity_label']), ('a', 'a', None))
        for x in LABELS:
            cd, dc, dd = cc[('recovery', 'clean', x)], cc[('recovery', x, 'clean')], cc[('recovery', x, x)]
            self.assertEqual((cd['main_source'], cd['later_base'], cd['later_donor']), ('a', x, x))
            self.assertEqual((dc['main_source'], dc['later_base'], dc['later_donor']), (x, 'a', 'a'))
            self.assertEqual((dd['main_source'], dd['later_base'], dd['later_donor']), (x, x, x))
            self.assertEqual({cd['severity_label'], dc['severity_label'], dd['severity_label']}, {x})
            for endpoint in 'ab':
                conflict = cc[('conflict', x, endpoint)]
                self.assertEqual((conflict['main_source'], conflict['later_base'], conflict['later_donor']), (x, 'a', 'b'))
            # One file realizes the label in every cell that names it.
            self.assertEqual(len({self.row['source_images'][c['main_source']] for c in (dc, dd)} |
                                 {self.row['source_images'][cd['later_base']]}), 1)

    def test_label_selection_forms_and_typos(self):
        listed = config(severity_labels=LABELS[:2])
        self.assertEqual(design.severity_labels(self.row, listed), LABELS[:2])
        mapped = config(severity_labels={'stripe_orientation': [LABELS[2]]})
        self.assertEqual(design.severity_labels(self.row, mapped), [LABELS[2]])
        with self.assertRaisesRegex(ValueError, 'no entry for task'):
            design.severity_labels(self.row, config(severity_labels={'color': ['desat-c0.2']}))
        with self.assertRaisesRegex(ValueError, 'lacks selected'):
            design.severity_labels(self.row, config(severity_labels={'stripe_orientation': ['blurrel-period0.4']}))
        with self.assertRaisesRegex(ValueError, 'No selected severity'):
            design.conditions(self.row, config(severity_labels=['blurrel-period0.4']))
        with self.assertRaisesRegex(ValueError, 'absent from every row'):
            design.execution_plan([self.row], {'key': 'm'}, config(severity_labels=LABELS+['desat-c0.2']))
        with tempfile.TemporaryDirectory() as t:
            path = Path(t)/'c.json'; path.write_text(json.dumps({'execution_version': 'x', 'severity_sources': 'row', 'severity_labels': ['blur-0.2']}))
            with self.assertRaises(ValueError): design.settings(path)

    def test_plan_counts_logical_rows(self):
        plan = design.execution_plan([self.row], {'key': 'm'}, self.cfg)
        self.assertEqual(plan['expected_logical_rows'], {'score': 18*3, 'generation': 18*3})
        self.assertEqual(plan['severity_labels_by_task'], {'stripe_orientation': LABELS})


class IdentityTests(unittest.TestCase):
    def test_unaffected_observations_keep_keys(self):
        with tempfile.TemporaryDirectory() as t1, tempfile.TemporaryDirectory() as t2:
            m3, _ = write_row(t1); m4, _ = write_row(t2, LABELS+['blurrel-period0.4'])
            cfg = config(severity_labels=None)
            three = keys(design.load_rows(m3, cfg)[0], cfg); four = keys(design.load_rows(m4, cfg)[0], cfg)
            self.assertTrue(set(three.items()) <= set(four.items()))  # adding a severity (and moving paths) reuses all
            self.assertEqual(len(four)-len(three), 5*6)  # conflict L x {a,b} + recovery (clean,L),(L,clean),(L,L)
            dropped = copy.deepcopy(design.load_rows(m3, cfg)[0])
            del dropped['severity_sources'][LABELS[2]], dropped['degradations'][LABELS[2]]
            del dropped['source_images'][LABELS[2]], dropped['source_image_metadata'][LABELS[2]]
            kept = keys(dropped, cfg)
            self.assertTrue(set(kept.items()) <= set(three.items())); self.assertEqual(len(three)-len(kept), 5*6)
            # Selecting a subset of labels in the config leaves every selected key unchanged.
            subset = keys(design.load_rows(m3, cfg)[0], config(severity_labels=LABELS[:1]))
            self.assertTrue(set(subset.items()) <= set(three.items()))
            # max_batch is execution scheduling, recorded per batch; not identity (as accepted in v14).
            self.assertEqual(keys(design.load_rows(m3, cfg)[0], {**cfg, 'max_batch': 3}), three)

    def test_value_changing_changes_get_new_keys(self):
        with tempfile.TemporaryDirectory() as t:
            manifest, _ = write_row(t); cfg = config(severity_labels=None); row = design.load_rows(manifest, cfg)[0]
            base = keys(row, cfg)
            changed = copy.deepcopy(row); label = LABELS[0]
            changed['source_image_metadata'][label]['sha256'] = '0'*64
            moved = keys(changed, cfg)
            for k, v in base.items():
                uses = label in (k[0].split('__')[1], k[0].split('__')[2])
                self.assertEqual(moved[k] != v, uses, k)
            other_b = copy.deepcopy(row); other_b['source_image_metadata']['b']['sha256'] = '1'*64
            for k, v in keys(other_b, cfg).items():
                self.assertEqual(v != base[k], k[0].startswith('conflict__'), k)
            spec = copy.deepcopy(row); spec['degradations'][label]['value'] = 0.2
            self.assertNotEqual(keys(spec, cfg)[('recovery__'+label+'__'+label, json.dumps(DIRECT, sort_keys=True), 'score')],
                                base[('recovery__'+label+'__'+label, json.dumps(DIRECT, sort_keys=True), 'score')])
            version = keys(row, {**cfg, 'execution_version': 'other'})
            self.assertFalse(set(version.values()) & set(base.values()))
            tokens = keys(row, {**cfg, 'generate_tokens': 16})
            for k, v in base.items(): self.assertEqual(tokens[k] != v, k[2] == 'generation')
            self.assertFalse(set(keys(row, cfg, {'key': 'm', 'revision': 'r2'}).values()) & set(base.values()))
            query = copy.deepcopy(row); query['query'] = 'Which pattern?'
            self.assertFalse(set(keys(query, cfg).values()) & set(base.values()))
            self.assertEqual(len(set(base.values())), len(base))

    def test_canvas_identity_order_and_spec_checks(self):
        cases = []
        with tempfile.TemporaryDirectory() as t:
            manifest, row = write_row(t); cfg = config(severity_labels=None)
            def broken(edit):
                r = copy.deepcopy(row); edit(r); manifest.write_text(json.dumps(r)+'\n')
                with self.assertRaises(ValueError): design.load_rows(manifest, cfg)
                cases.append(edit)
            def resize(r):
                path = Path(t)/r['source_images'][LABELS[1]]; Image.new('RGB', (65, 48)).save(path)
                r['source_image_metadata'][LABELS[1]].update(width=65, sha256=sha(path))
            broken(lambda r: r['source_image_metadata'][LABELS[0]].update(sha256='f'*64))
            broken(lambda r: r['degradations'][LABELS[0]].update(value=0.5))
            broken(lambda r: r.update(severity_sources=dict(reversed(list(r['severity_sources'].items())))))
            broken(lambda r: r['severity_sources'].update({LABELS[0]: 'a'}))
            broken(lambda r: r['degradations'].pop(LABELS[2]))
            broken(resize)
        self.assertEqual(len(cases), 6)


class ExecutionChangeTests(unittest.TestCase):
    LIBS = {'torch': '2.11.0', 'transformers': '5.8.0', 'Pillow': '12.3.0'}

    def output_with_run(self, root, **config):
        cfg = {**design.settings(CONFIGS/'line1_severity_batched.json'), **config}
        (Path(root)/'run-r1.json').write_text(json.dumps(dict(run_id='r1', config=cfg, libraries=self.LIBS, cuda_device='NVIDIA A100 80GB PCIe')))
        return {'k1': {'observation_id': 'k1', 'run_id': 'r1'}}, cfg

    def test_batch_or_library_change_is_refused_unless_recorded(self):
        with tempfile.TemporaryDirectory() as t:
            done, cfg = self.output_with_run(t)
            same = design.execution_signature(cfg, self.LIBS, 'NVIDIA A100 80GB PCIe')
            self.assertIsNone(check_resume_execution(t, done, same))
            for change in ({'max_batch': 4}, {'cross_query_cache': False}, {'batch_conditions': False}):
                other = design.execution_signature({**cfg, **change}, self.LIBS, 'NVIDIA A100 80GB PCIe')
                with self.assertRaisesRegex(RuntimeError, 'numerical execution settings'): check_resume_execution(t, done, other)
                self.assertEqual(set(check_resume_execution(t, done, other, allow=True)['differing_runs']), {'r1'})
            newer = design.execution_signature(cfg, {**self.LIBS, 'torch': '2.12.0'}, 'NVIDIA A100 80GB PCIe')
            with self.assertRaises(RuntimeError): check_resume_execution(t, done, newer)
            with self.assertRaises(RuntimeError): check_resume_execution(t, {'k2': {'observation_id': 'k2', 'run_id': 'gone'}}, same)
            self.assertIsNone(check_resume_execution(t, {}, newer))  # empty output: nothing to mix

    def test_batch_keys_irrelevant_without_batching(self):
        plain = design.settings()
        self.assertEqual(design.execution_signature({**plain, 'max_batch': 3}), design.execution_signature(plain))

    def test_analysis_reports_pooled_execution_settings(self):
        base = dict(task='t', kind='score', readout=DIRECT, model_key='m', model_revision='r', execution_version='v', unit_id='u',
                    intervention_id='e', id='q', query_role='changed', result={'b_minus_a_log_odds': 0.},
                    condition=dict(name='n', family='conflict', main_quality='normal', later_endpoint='a'))
        records = [dict(base, observation_id='1', run_id='r1'), dict(base, observation_id='2', run_id='r2', id='q2')]
        cfg = design.settings(CONFIGS/'line1_severity_batched.json')
        metas = [dict(run_id='r1', model={'key': 'm'}, config=cfg, libraries=self.LIBS, cuda_device='A100'),
                 dict(run_id='r2', model={'key': 'm'}, config={**cfg, 'max_batch': 4}, libraries=self.LIBS, cuda_device='A100')]
        out = analyze(records, run_metadata=metas)
        self.assertEqual(len(out['execution_settings'][0]['signatures']), 2)
        self.assertTrue(any('numerical execution settings' in w for w in out['warnings']))


class ReadoutAndAnalysisTests(unittest.TestCase):
    def test_retained_readouts_map_answers_both_directions(self):
        with tempfile.TemporaryDirectory() as t:
            manifest, _ = write_row(t); cfg = config(severity_labels=None); row = design.load_rows(manifest, cfg)[0]
        self.assertEqual(design.readouts(row, cfg), [dict(response_format='direct', mapping='none'),
            dict(response_format='label', mapping='identity'), dict(response_format='label', mapping='cyclic')])
        seen = {}
        for answer in (0, 1):
            r = {**row, 'answer_index': answer, 'answer_label': row['option_labels'][answer]}
            for ro in design.readouts(r, cfg):
                v = make_variant(r, ro)
                self.assertEqual(v['label_to_original'][v['answer_label']], r['answer_label'])
                self.assertEqual(outcome_fields(r, v, v['answer_label'])['clean_reference_correct'], 1)
                wrong = next(x for x in v['labels'] if x != v['answer_label'])
                self.assertEqual(outcome_fields(r, v, wrong)['clean_reference_correct'], 0)
                seen[(answer, ro['mapping'])] = v['answer_label']
        self.assertNotEqual(seen[(0, 'identity')], seen[(0, 'cyclic')])
        benchmark = {k: v for k, v in row.items() if k not in ('choices', 'option_labels', 'answer_index', 'answer_label')}
        self.assertEqual(design.readouts(benchmark, design.settings(CONFIGS/'benchmark_recovery_severity_batched.json')),
                         [dict(response_format='freeform', mapping='none')])

    def test_restored_analysis_reports_each_severity_against_shared_clean_cell(self):
        with tempfile.TemporaryDirectory() as t:
            manifest, _ = write_row(t); cfg = config(families=['recovery'], severity_labels=None, readouts=[DIRECT])
            row = design.load_rows(manifest, cfg, check_files=False)[0]
        truth = {('clean', 'clean'): 1}
        for i, x in enumerate(LABELS):
            truth.update({('clean', x): 1, (x, 'clean'): int(i < 2), (x, x): int(i == 0)})
        records = []
        for c in design.conditions(row, cfg):
            records.append(dict(observation_id=str(len(records)), run_id='run', id=row['id'], unit_id='u1', intervention_id=row['intervention_id'],
                task=row['task'], kind='generation', readout=DIRECT, model_key='m', model_revision='r', execution_version=cfg['execution_version'],
                query_role='changed', condition=c, result=dict(clean_reference_correct=truth[(c['main_quality'], c['later_quality'])], scorable=True, truncated=False)))
        meta = dict(run_id='run', model=dict(key='m', revision='r'), config=cfg, conditions_per_query={row['id']: 10})
        result = analyze(records, materials=[row], run_metadata=[meta])['results'][0]
        benefit = {x['main_quality']: x['mean'] for x in result['recovery_contrasts']['accuracy_valid_only']['clean_later_benefit_with_degraded_main']}
        self.assertEqual(benefit, {LABELS[0]: 0, LABELS[1]: 1, LABELS[2]: 0})
        cells = {(x['main_quality'], x['later_quality']) for x in result['recovery_clean_reference_accuracy']}
        self.assertEqual(cells, set(truth))
        self.assertEqual(result['reference_labels']['recovery_clean_quality'], 'clean')


if __name__ == '__main__': unittest.main()
