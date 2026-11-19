"""Natural-selectivity planning and scoped reuse of the accepted CUDA loop."""
import argparse
from contextlib import contextmanager
import json
import os
from pathlib import Path
import os as _path_os
DATA_ROOT = Path(_path_os.environ.get("DS_DATA_ROOT", "work")).expanduser().resolve()
import sys
from cloud_prep.measurement_v8.model_registry import MODELS, get_model
from cloud_prep.measurement_v8.runtime import plan_runtime, prepare_runtime, sha256, write_json
from cloud_prep.measurement_v9.runtime import validate_shared_interventions
from cloud_prep.measurement_v10.geometry import preflight_geometry
from cloud_prep.measurement_v4.content_intervention import cloud_gpu_lock
from cloud_prep.measurement_v12 import p1_design as design, p1_prepare as data
from cloud_prep.measurement_v12.v11_v2_vendor import verify_vendor

ROOT = Path(__file__).resolve().parents[2]


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model', required=True, choices=tuple(MODELS))
    p.add_argument('--manifest', type=Path, default=(DATA_ROOT / 'datasets/natural-selectivity-v12/manifest.jsonl'))
    p.add_argument('--panel', choices=('main','smoke'), default='main')
    p.add_argument('--photo-ids', nargs='+')
    p.add_argument('--campaign-root', type=Path, required=True)
    p.add_argument('--cache-root', type=Path, required=True)
    p.add_argument('--output', type=Path)
    p.add_argument('--plan-output', type=Path)
    p.add_argument('--generate-tokens', type=int, default=32)
    p.add_argument('--max-sequence', type=int, default=4096)
    p.add_argument('--audit-tolerance', type=float, default=.001)
    p.add_argument('--gpu', default=os.environ.get('CUDA_VISIBLE_DEVICES','0'))
    p.add_argument('--seed', type=int, default=2026102201)
    p.add_argument('--local-files-only', action='store_true')
    p.add_argument('--dry-run', action='store_true')
    a = p.parse_args(argv)
    if not a.dry_run and a.output is None: p.error('--output required')
    if a.generate_tokens < 0 or a.max_sequence < 1 or not 0 <= a.audit_tolerance <= .01: p.error('Invalid runtime bounds')
    if not a.gpu or any(c not in '0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ-_' for c in a.gpu): p.error('Expose one GPU')
    a.arm='dose'; a.entry_groups=['all']; a.mappings=['identity','cyclic']; a.design_config=None
    plan_runtime(a)
    return a


def prepare(a):
    rows = data.load_manifest(a.manifest)
    rows = [{**r, 'query_type': 'identity', 'query_target': r['query_role']} for r in rows]
    ids = sorted({r['scene_id'] for r in rows})
    selected = set(a.photo_ids) if a.photo_ids else set(ids[:2] if a.panel=='smoke' else ids)
    if not selected or selected-set(ids): raise ValueError('Unknown/empty photo shard')
    rows = sorted((r for r in rows if r['scene_id'] in selected), key=lambda r:(r['intervention_id'],r['query_role']))
    plan = design.execution_plan(rows, a.panel, a.mappings, a.generate_tokens)
    plan.update(validate_shared_interventions(rows))
    needed = design.required_sources(plan['conditions'])
    geometry = []
    for row in rows:
        if needed-set(row['source_images']): raise ValueError('Missing P1 sources')
        g=preflight_geometry(row,plan['conditions'])
        if any(v<=0 for counts in g['nominal_roi_token_counts'].values() for v in counts.values()): raise ValueError('Empty nominal ROI support')
        geometry.append({'id':row['id'],**g})
    plan.update(model=get_model(a.model), runtime=plan_runtime(a), manifest_sha256=sha256(a.manifest),
        execution_schema='natural-selectivity-cuda-v12-v1', ids=[r['id'] for r in rows],
        shard_units=[{'id':r['id'],'unit':r['scene_id']} for r in rows], geometry_preflight=geometry,
        accepted_v11_v2_dependency=verify_vendor(),
        p1_source_sha256={str(p.relative_to(ROOT)):sha256(p) for p in sorted((ROOT/'cloud_prep/measurement_v12').glob('p1_*.py'))},
        backend_scope='CPU preparation only; actual processor/ROI/self-patch/dose smoke required on fixed BF16 CUDA')
    plan['physical_work_accounting']=physical_work_counts(rows,plan,needed)
    return rows,plan


def physical_work_counts(rows,plan,needed):
    audits={};audit_rows=0
    for row in rows:
        key=(row['attribute'],str(row.get('geometry')),row.get('panel_family'))
        audits.setdefault(key,row['intervention_id'])
        audit_rows+=int(plan['panel']=='smoke' or audits[key]==row['intervention_id'])
    reads=len(plan['readouts'])
    refs={c['main'] for c in plan['conditions'] if c['role']=='self_patch'}
    counts=dict(model=plan['model']['key'],arm='p1_natural',photos=plan['independent_source_photos'],
        logical_score_rows=plan['expected_score_rows'],logical_generation_rows=plan['expected_generation_rows'],
        planned_unique_score_calls=plan['expected_unique_score_evaluations'],
        planned_unique_generation_calls=plan['expected_unique_generation_evaluations'],
        native_self_audit_score_calls=2*len(needed)*audit_rows*reads,
        non_audited_composition_self_reference_score_calls=(len(rows)-audit_rows)*reads*len(refs),
        baseline_reuse_audit_score_calls=plan['expected_dedup_score_audit_evaluations'],
        baseline_reuse_audit_generation_calls=plan['expected_dedup_generation_audit_evaluations'],
        calibration_anchor_forwards=plan['expected_anchor_forwards'],
        source_encode_calls=len(needed)*len({r['intervention_id'] for r in rows}))
    counts['total_score_options_calls']=sum(counts[k] for k in ('planned_unique_score_calls',
        'native_self_audit_score_calls','non_audited_composition_self_reference_score_calls','baseline_reuse_audit_score_calls'))
    counts['total_generation_calls']=counts['planned_unique_generation_calls']+counts['baseline_reuse_audit_generation_calls']
    counts['boundary']='score_options calls are not candidate forward calls; each shard restarts technical audits; no cross-query scalar reuse'
    return counts


@contextmanager
def execution_bridge(executor, adapter_module):
    from cloud_prep.measurement_v12.p1_adapter import NaturalDoseAdapter
    saved = executor.record_fields, executor.make_adapter, executor.EXECUTION_SCHEMA, executor.REUSED_SOURCES, adapter_module.design
    def make_adapter(spec,snapshot,arm):
        base,meta=saved[1](spec,snapshot,'natural')
        family='ibm' if spec['model'].startswith('ibm-granite/') else 'qwen'
        return NaturalDoseAdapter(base,family),meta
    executor.record_fields=lambda row,variant,condition,prediction,arm: design.record_fields(row,variant,condition,prediction)
    executor.make_adapter=make_adapter
    executor.EXECUTION_SCHEMA='natural-selectivity-cuda-v12-v1'
    executor.REUSED_SOURCES=tuple(dict.fromkeys((*saved[3],*(str(p.relative_to(ROOT)) for p in sorted((ROOT/'cloud_prep/measurement_v12').glob('p1_*.py'))))))
    adapter_module.design=design
    try: yield
    finally:
        executor.record_fields,executor.make_adapter,executor.EXECUTION_SCHEMA,executor.REUSED_SOURCES,adapter_module.design=saved


def main(argv=None):
    a=parse_args(argv);rows,plan=prepare(a)
    if a.plan_output: write_json(data._guard_path(a.plan_output,write=True),plan)
    if a.dry_run:
        if 'torch' in sys.modules: raise RuntimeError('Planning imported torch')
        print(json.dumps({k:v for k,v in plan.items() if k not in ('conditions','queries','geometry_preflight','ids','shard_units')}
                         | {'dry_run':True,'torch_imported':False},indent=2));return 0
    prepare_runtime(a)
    from cloud_prep.measurement_v12 import v11_v2_runner as executor,v11_v2_dose_adapter as adapter_module
    try:
        with cloud_gpu_lock(a.campaign_root,a.gpu),execution_bridge(executor,adapter_module):
            return executor.execute(a,rows,plan,design)
    except BlockingIOError:
        print(json.dumps({'event':'gpu_busy','gpu':a.gpu}),flush=True);return 75


if __name__=='__main__': raise SystemExit(main())
