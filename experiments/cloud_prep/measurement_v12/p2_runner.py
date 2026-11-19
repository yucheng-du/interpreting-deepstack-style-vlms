"""P2 tensor-free CLI planning and explicit reuse of validated CUDA executors."""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import json
import os
from pathlib import Path
import sys

from cloud_prep.measurement_v8.model_registry import MODELS, get_model
from cloud_prep.measurement_v8.runtime import plan_runtime, prepare_runtime, sha256, write_json
from cloud_prep.measurement_v10.geometry import preflight_geometry
from cloud_prep.measurement_v10.runtime import required_sources, validate_shared_interventions
from cloud_prep.measurement_v4.content_intervention import cloud_gpu_lock
from cloud_prep.measurement_v12 import p2_data, p2_design
from cloud_prep.measurement_v12.v11_v2_vendor import verify_vendor

ROOT=Path(__file__).resolve().parents[2]


def parse_args(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model',required=True,choices=tuple(MODELS))
    p.add_argument('--arm',required=True,choices=p2_design.ARMS)
    p.add_argument('--manifest',type=Path,default=p2_data.DEFAULT_ROOT/'manifest.jsonl')
    p.add_argument('--panel',choices=('main','smoke'),default='main')
    p.add_argument('--template-ids',nargs='+')
    p.add_argument('--campaign-root',type=Path,required=True)
    p.add_argument('--cache-root',type=Path,required=True)
    p.add_argument('--output',type=Path)
    p.add_argument('--plan-output',type=Path)
    p.add_argument('--generate-tokens',type=int,default=32)
    p.add_argument('--max-sequence',type=int,default=4096)
    p.add_argument('--audit-tolerance',type=float,default=.001)
    p.add_argument('--seed',type=int,default=2026102202)
    p.add_argument('--gpu',default=os.environ.get('CUDA_VISIBLE_DEVICES','0'))
    p.add_argument('--local-files-only',action='store_true')
    p.add_argument('--dry-run',action='store_true')
    a=p.parse_args(argv)
    if not a.dry_run and a.output is None: p.error('--output required for execution')
    if a.generate_tokens<0 or a.max_sequence<1 or not 0<=a.audit_tolerance<=.01:
        p.error('Invalid generation/context/audit bound')
    if not a.gpu or any(c not in '0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ-_' for c in a.gpu):
        p.error('Expose one GPU index/UUID per process')
    a.entry_groups=['all']; a.mappings=['identity','cyclic']; a.design_config=None
    plan_runtime(a)
    return a


def prepare(a):
    rows=p2_data.select_rows(p2_data.load_manifest(a.manifest),a.panel,a.arm=='dose',a.template_ids)
    plan=p2_design.execution_plan(rows,a.arm,a.panel,a.mappings,a.generate_tokens)
    plan.update(validate_shared_interventions(rows))
    needed=p2_design.dose.required_sources(plan['conditions']) if a.arm=='dose' else required_sources(plan['conditions'])
    checked={}
    for row in rows:
        if needed-set(row['source_images']): raise ValueError('Missing source')
        geometry=preflight_geometry(row,plan['conditions'])
        key=str(geometry['image_hw'])+str(geometry['nominal_roi_token_counts'])
        checked.setdefault(key,geometry)
    source_files=sorted((ROOT/'cloud_prep/measurement_v12').glob('p2_*.py'))
    plan.update(execution_schema='paired-attribute-cuda-v12-v1',arm=a.arm,model=get_model(a.model),
        manifest_sha256=sha256(a.manifest),runtime=plan_runtime(a),geometry_preflight=list(checked.values()),
        ids=[r['id'] for r in rows],shard_units=[dict(id=r['id'],unit=r['template_id']) for r in rows],
        p2_source_sha256={str(p.relative_to(ROOT)):sha256(p) for p in source_files},
        accepted_v11_v2_dependency=verify_vendor(),
        backend_scope='CPU prep passed; formal runtime remains fixed BF16 CUDA and requires actual processor/self-patch smoke')
    plan['physical_work_accounting']=physical_work_counts(rows,plan,needed)
    return rows,plan


def physical_work_counts(rows,plan,needed):
    """Count actual executor calls, separating scoring calls from model forwards."""
    seen={}; audit_rows=0
    for row in rows:
        group=(row['attribute'],str(row.get('geometry')),row.get('panel_family'))
        seen.setdefault(group,row['intervention_id'])
        audit_rows+=int(plan['panel']=='smoke' or seen[group]==row['intervention_id'])
    readouts=len(plan['readouts'])
    audit_native_self=2*len(needed)*audit_rows*readouts
    self_sources={c['main'] for c in plan['conditions'] if c['role']=='self_patch'}
    nonaudit_self=(len(rows)-audit_rows)*readouts*len(self_sources)
    result=dict(model=plan['model']['key'],arm=plan['arm'],templates=plan['scene_templates'],
        query_rows=len(rows),conditions=len(plan['conditions']),
        logical_score_rows=plan['expected_score_rows'],logical_generation_rows=plan['expected_generation_rows'],
        planned_unique_score_calls=plan['expected_unique_score_evaluations'],
        planned_unique_generation_calls=plan['expected_unique_generation_evaluations'],
        native_self_audit_score_calls=audit_native_self,
        non_audited_composition_self_reference_score_calls=nonaudit_self,
        baseline_reuse_audit_score_calls=plan.get('expected_dedup_score_audit_evaluations',0),
        baseline_reuse_audit_generation_calls=plan.get('expected_dedup_generation_audit_evaluations',0),
        calibration_anchor_forwards=plan.get('expected_anchor_forwards',0),
        source_encode_calls=len(needed)*len({r['intervention_id'] for r in rows}),
        boundary='score calls are score_options invocations, not candidate forward calls. Matching succeeds bound; unmatched rows retained. No cross-arm cache reuse. Each shard restarts audits; sum the actual shard plans for scheduling.')
    result['total_score_options_calls']=result['planned_unique_score_calls']+audit_native_self+nonaudit_self+result['baseline_reuse_audit_score_calls']
    result['total_generation_calls']=result['planned_unique_generation_calls']+result['baseline_reuse_audit_generation_calls']
    return result


@contextmanager
def execution_bridge(module, arm):
    """Scoped callback replacement, no edits to old modules or global files."""
    original_fields=module.record_fields
    original_schema=module.EXECUTION_SCHEMA
    original_sources=module.REUSED_SOURCES
    def record_fields(row,variant,condition,prediction,unused_arm):
        return p2_design.record_fields(row,variant,condition,prediction,arm)
    module.record_fields=record_fields
    module.EXECUTION_SCHEMA='paired-attribute-cuda-v12-v1'
    module.REUSED_SOURCES=tuple(dict.fromkeys((*original_sources,
        *(str(p.relative_to(ROOT)) for p in sorted((ROOT/'cloud_prep/measurement_v12').glob('p2_*.py'))))))
    try:
        yield
    finally:
        module.record_fields=original_fields
        module.EXECUTION_SCHEMA=original_schema
        module.REUSED_SOURCES=original_sources


def main(argv=None):
    a=parse_args(argv); rows,plan=prepare(a)
    if a.plan_output:
        target=p2_data._guard_path(a.plan_output,write=True)
        write_json(target,plan)
    if a.dry_run:
        if 'torch' in sys.modules: raise RuntimeError('CPU planning imported torch')
        fields=('schema','arm','panel','model','items','scene_templates','attribute_scenes',
                'unique_visual_interventions','expected_score_rows','expected_generation_rows',
                'expected_unique_score_evaluations','expected_unique_generation_evaluations',
                'expected_anchor_forwards','expected_dedup_score_audit_evaluations',
                'expected_dedup_generation_audit_evaluations','manifest_sha256','independence_unit')
        print(json.dumps({k:plan[k] for k in fields if k in plan}
                         | dict(torch_imported=False,dry_run=True,plan_output=str(a.plan_output) if a.plan_output else None),indent=2))
        return 0
    # Both executors enforce Linux/CUDA before model load. No Mac tensor work.
    prepare_runtime(a)
    if a.arm=='dose':
        from cloud_prep.measurement_v12 import v11_v2_runner as executor
    else:
        from cloud_prep.measurement_v10 import runner as executor
    try:
        with cloud_gpu_lock(a.campaign_root,a.gpu), execution_bridge(executor,a.arm):
            return executor.execute(a,rows,plan,p2_design.ExecutionDesign)
    except BlockingIOError:
        print(json.dumps(dict(event='gpu_busy',gpu=a.gpu)),flush=True)
        return 75


if __name__=='__main__': raise SystemExit(main())
