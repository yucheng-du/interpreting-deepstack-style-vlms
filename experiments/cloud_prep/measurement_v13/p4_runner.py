"""P4 plans on CPU; actual model work runs on Linux with CUDA."""
from __future__ import annotations
import argparse
from contextlib import nullcontext
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import importlib.metadata
from itertools import groupby
import json
import os
from pathlib import Path
import platform
import sys
import time

from cloud_prep.measurement_v8.runtime import plan_runtime,prepare_runtime,resolve_snapshot,sha256,write_json
from cloud_prep.measurement_v4.content_intervention import cloud_gpu_lock
from cloud_prep.measurement_v10.runtime import direction_readout,validate_shared_interventions
from cloud_prep.measurement_v12 import p2_data
from cloud_prep.measurement_v13 import p4_design as design

ROOT=Path(__file__).resolve().parents[2]


def utc_now(): return datetime.now(timezone.utc).isoformat()


LIBRARIES=('torch','transformers','Pillow','huggingface_hub','safetensors','tokenizers','accelerate','peft','numpy')


def library_versions(names=LIBRARIES):
    out={}
    for name in names:
        try: out[name]=importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError: out[name]=None
    return out


def parse_args(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model',choices=design.MODELS,required=True)
    p.add_argument('--manifest',type=Path,default=p2_data.DEFAULT_ROOT/'manifest.jsonl')
    p.add_argument('--config',type=Path)
    p.add_argument('--panel',choices=('main','smoke'),default='main')
    p.add_argument('--template-ids',nargs='+')
    p.add_argument('--campaign-root',type=Path,required=True)
    p.add_argument('--cache-root',type=Path,required=True)
    p.add_argument('--output',type=Path)
    p.add_argument('--plan-output',type=Path)
    p.add_argument('--generate-tokens',type=int,default=32)
    p.add_argument('--max-sequence',type=int,default=4096)
    p.add_argument('--audit-tolerance',type=float,default=.001)
    p.add_argument('--seed',type=int,default=20261024)
    p.add_argument('--gpu',default=os.environ.get('CUDA_VISIBLE_DEVICES','0'))
    p.add_argument('--local-files-only',action='store_true')
    p.add_argument('--dry-run',action='store_true')
    a=p.parse_args(argv)
    if not a.dry_run and a.output is None: p.error('--output required for execution')
    if a.generate_tokens<0 or a.max_sequence<1 or not 0<=a.audit_tolerance<=.01: p.error('Invalid runtime bounds')
    if not a.gpu or any(c not in '0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ-_' for c in a.gpu): p.error('Expose one GPU per process')
    plan_runtime(a)
    return a


def prepare(a):
    d=design.settings(a.config)
    rows=design.select_rows(p2_data.load_manifest(a.manifest),d,a.panel,a.template_ids)
    plan=design.execution_plan(rows,a.model,d,a.panel,a.generate_tokens)
    plan.update(validate_shared_interventions(rows))
    plan.update(manifest_sha256=sha256(a.manifest),runtime=plan_runtime(a),
        shard_units=[dict(id=r['id'],unit=r['template_id']) for r in rows],
        p4_source_sha256={p.name:sha256(p) for p in sorted(Path(__file__).parent.glob('p4_*.py'))})
    return rows,plan


def record_fields(row,readout,condition,variant,prediction):
    original=variant['label_to_original'].get(prediction) if prediction is not None else None
    return {**{k:row[k] for k in ('id','template_id','intervention_id','attribute','changed_object','changed_index','queried_object','queried_index','query_role','direction_a_index','direction_b_index')},
        **readout,'condition':condition['name'],'condition_spec':condition,
        'original_prediction':original,**direction_readout(row,variant)}


def reuse_key(c):
    if len(c['entries'])==1 and c['mode'] in ('common_low_rss','common_low_l1'):
        return (c['context']['name'],tuple(c['entries']),'common_low_singleton')
    return None


def execute(a,rows,plan):
    runtime=prepare_runtime(a)
    a.output.mkdir(parents=True,exist_ok=False)
    meta={**plan,'runtime':runtime,'started_at':utc_now(),'platform':platform.platform(),'python':sys.version,
        'arguments':{k:str(v) if isinstance(v,Path) else v for k,v in vars(a).items()},
        'completed_score_rows':0,'completed_generation_rows':0,'completed_anchor_forwards':0,
        'executed_score_evaluations':0,'executed_generation_evaluations':0,
        'source_sha256':{str(p.relative_to(ROOT)):sha256(p) for p in sorted(Path(__file__).parent.glob('p4_*.py'))}}
    write_json(a.output/'metadata.json',meta)
    active=None
    try:
        import torch
        from transformers import StoppingCriteria,StoppingCriteriaList
        from deepstack.scoring import score_options
        from deepstack.readout_v3 import tokenize_variant,fp32_output_head,PARSER_VERSION
        from cloud_prep.measurement_v6.content_readout import make_variant,parse_response
        from cloud_prep.measurement_v8.generation import native_greedy,resolve_greedy_config
        from cloud_prep.measurement_v3.runtime import validate_cuda,memory,check_swap,swap_used_mb
        from cloud_prep.measurement_v12.v11_v2_runner import make_adapter,tensor_digest
        from scripts.prepare_readout_controls import generation_stop
        from cloud_prep.measurement_v13.p4_adapter import P4Adapter
        meta.update(cuda=validate_cuda(torch),initial_swap_mb=swap_used_mb(),parser=PARSER_VERSION)
        torch.set_num_threads(4); torch.manual_seed(a.seed)
        snapshot=resolve_snapshot(plan['model'],a.cache_root,a.local_files_only)
        base,adapter_meta=make_adapter(plan['model'],snapshot,'p4')
        adapter=P4Adapter(base,'ibm' if a.model=='ibm4b' else 'qwen')
        model,processor=adapter.model,adapter.processor
        meta.update(model=snapshot,adapter=adapter_meta,libraries=library_versions(),
            executable=sys.executable,torch_cuda=torch.version.cuda,
            image_processor=processor.image_processor.to_dict())
        config=None; eos_ids=[]
        if a.generate_tokens:
            config=deepcopy(model.generation_config)
            config.max_new_tokens=a.generate_tokens; config.do_sample=False; config.num_beams=1
            config.num_return_sequences=1; config.use_cache=True
            config=resolve_greedy_config(model,config)
            eos_ids=config.eos_token_id
            eos_ids=[eos_ids] if isinstance(eos_ids,int) else list(eos_ids or [])
            if not eos_ids: raise ValueError('Explicit generation EOS required')
            meta['generation_config']=config.to_dict()

        class SwapGuard(StoppingCriteria):
            def __init__(self): self.checked=0.; self.error=None
            def __call__(self,input_ids,scores,**kwargs):
                if time.monotonic()-self.checked>10:
                    self.checked=time.monotonic()
                    try: check_swap(meta['initial_swap_mb'])
                    except RuntimeError as exc: self.error=str(exc)
                return bool(self.error)

        needed=sorted({'d_a','d_b',*(c['source'] for c in design.contexts(plan['config']))})
        audited_attributes=set()
        with (a.output/'results.jsonl').open('w',buffering=1) as out, \
             (a.output/'generations.jsonl').open('w',buffering=1) as gout, \
             (a.output/'calibrations.jsonl').open('w',buffering=1) as cout, \
             (a.output/'audits.jsonl').open('w',buffering=1) as audits, \
             torch.inference_mode(),fp32_output_head(model):
            for iid,group in groupby(rows,key=lambda r:r['intervention_id']):
                block=list(group); anchor=next(r for r in block if r['query_role']=='changed')
                active=anchor['id']; avariant=make_variant(anchor,'direct','none')
                sources={}; pixel_hashes={}; anchors={}; contexts={}; calibrations={}
                audit=a.panel=='smoke' or anchor['attribute'] not in audited_attributes
                for name in needed:
                    inp,prompt=adapter.prepare(anchor,avariant,name)
                    if inp['input_ids'].shape[1]>a.max_sequence: raise ValueError('Anchor context too long')
                    sources[name]=adapter.encode(inp)
                    pixel_hashes[name]=tensor_digest({k:v for k,v in inp.items() if k in ('pixel_values','image_sizes','image_grid_thw')},torch)
                    anchors[name]=(inp,prompt)
                source_before=tensor_digest(sources,torch) if audit else None
                for ctx in design.contexts(plan['config']):
                    inp,prompt=anchors[ctx['source']]
                    B,D,masks=adapter.common_visuals(sources,ctx,anchor,inp)
                    contexts[ctx['name']]=(B,D,masks)
                    def forward_once():
                        result=model(**inp,use_cache=False)
                        del result
                    cal=adapter.calibrate(B,D,masks,ctx,anchor,inp,forward_once,hashlib.sha256(prompt.encode()).hexdigest(),snapshot['revision'])
                    calibrations[ctx['name']]=cal; cout.write(json.dumps(cal,allow_nan=False)+'\n')
                    meta['completed_anchor_forwards']+=1
                shared_visual={}
                for row in block:
                    active=row['id']
                    for readout in plan['readouts']:
                        variant=make_variant(row,**readout); prepared={}; options=None; text_ref=None
                        for name in needed:
                            inp,prompt=adapter.prepare(row,variant,name)
                            if text_ref is not None and (prompt!=text_ref or not torch.equal(inp['input_ids'],next(iter(prepared.values()))['input_ids'])):
                                raise ValueError('Source changed query/token geometry')
                            text_ref=prompt; prepared[name]=inp
                            current=tensor_digest({k:v for k,v in inp.items() if k in ('pixel_values','image_sizes','image_grid_thw')},torch)
                            if current!=pixel_hashes[name]: raise ValueError('Query/readout changed processed image')
                        options=tokenize_variant(processor.tokenizer,variant,text_ref)
                        seq_len=next(iter(prepared.values()))['input_ids'].shape[1]
                        if seq_len+max(max(map(len,options.token_ids)),a.generate_tokens)>a.max_sequence: raise ValueError('Continuation exceeds context')
                        if audit:
                            # Native/self patch checks all actual source images; do not filter by accuracy.
                            diffs={}
                            for name,inp in prepared.items():
                                native=score_options(model,inp,options)
                                with adapter.use(sources[name],inp) as delivery: patched=score_options(model,inp,options)
                                diff=max(abs(native.scores[k]-patched.scores[k]) for k in native.scores)
                                if delivery['calls']!=patched.forward_count or diff>a.audit_tolerance: raise ValueError('Native/self patch failed: '+name)
                                diffs[name]=diff
                            audits.write(json.dumps(dict(id=row['id'],**readout,native_self_max_differences=diffs,adapter=adapter.audit_sources(prepared,sources,row)))+'\n')
                        cache={}
                        for c in plan['conditions']:
                            began=time.monotonic(); cal=None; visual=None; dose=None
                            source=c['source'] if c['role']=='native' else c['context']['source']; inp=prepared[source]
                            if c['role']!='native':
                                cal=calibrations[c['context']['name']]
                                visual,scale=adapter.compose(*contexts[c['context']['name']],c,cal,
                                    plan['config']['target_multiplier'],plan['config']['match_tolerance'])
                                dose=visual.metadata if visual else dict(matching=scale,effective_k=None)
                            status='unmatched_dose' if dose and dose['matching']['status']=='unmatched' else 'executed'
                            provenance=dict(schema=design.SCHEMA,execution_version=plan['execution_version'],
                                scientific_identity_sha256=plan['scientific_identity_sha256'],model_key=a.model,
                                model_revision=snapshot['revision'],protocol_status=status,source_pixel_sha256=pixel_hashes,
                                prompt_sha256=options.prompt_sha256,dose_metadata=dose,
                                calibration_id=cal['calibration_id'] if cal else None,
                                anchor_query_id=cal['anchor_query_id'] if cal else None,
                                anchor_prompt_sha256=cal['anchor_prompt_sha256'] if cal else None)
                            if status=='unmatched_dose':
                                record={**record_fields(row,readout,c,variant,None),**provenance,'kind':'score','scores':{},'b_minus_a_log_odds':None,'forward_count':0}
                                generated={**record,'kind':'generation','scorable':False,'truncated':False,'prediction':None,'stop_reason':'not_run_unmatched_dose','generated_token_ids':[]}
                            else:
                                vk=tensor_digest(visual.visual,torch) if visual and audit else None
                                if audit and vk is not None:
                                    old=shared_visual.setdefault(c['name'],vk)
                                    if old!=vk: raise ValueError('Intervention differs across query/readout')
                                rk=reuse_key(c); saved=cache.get(rk) if rk is not None else None
                                if saved is None:
                                    with (nullcontext(None) if c['role']=='native' else adapter.use(visual,inp)) as delivery:
                                        scored=score_options(model,inp,options)
                                    if delivery is not None and delivery['calls']!=scored.forward_count: raise ValueError('Missed score visual delivery')
                                    score_value=scored.to_dict(); meta['executed_score_evaluations']+=1
                                    gen_value=None; gd=None
                                    if a.generate_tokens:
                                        guard=SwapGuard()
                                        with (nullcontext(None) if c['role']=='native' else adapter.use(visual,inp)) as gd:
                                            seq=native_greedy(model,inp,config,stopping_criteria=StoppingCriteriaList([guard]))
                                        if gd is not None and gd['calls']!=1: raise ValueError('Generation must use one visual prefill')
                                        if guard.error: raise RuntimeError(guard.error)
                                        ids=seq[0,seq_len:].detach().cpu().tolist(); del seq
                                        text=processor.tokenizer.decode(ids,skip_special_tokens=True); parsed=parse_response(text,variant)
                                        stop=generation_stop(ids,eos_ids,a.generate_tokens)
                                        scorable=parsed['status']=='parsed' and stop['eos_reached']
                                        gen_value=dict(generated_token_ids=ids,generated_text=text,
                                            generated_text_raw=processor.tokenizer.decode(ids,skip_special_tokens=False),
                                            parse=parsed,scorable=scorable,prediction=parsed['label'] if scorable else None,**stop)
                                        meta['executed_generation_evaluations']+=1
                                    if rk is not None: cache[rk]=(deepcopy(score_value),deepcopy(gen_value),c['name'],vk)
                                    reused=False; reuse_source=c['name']
                                else:
                                    score_value,gen_value,reuse_source,old_vk=saved; reused=True; delivery=None; gd=None
                                    if audit and old_vk!=vk: raise ValueError('Singleton RSS/L1 reuse tensors differ')
                                lineage=dict(reused=reused,source_condition=reuse_source,
                                    reason='same singleton k and common B across RSS/L1' if reused else None)
                                record={**record_fields(row,readout,c,variant,score_value['prediction']),**provenance,
                                    'kind':'score',**score_value,**direction_readout(row,variant,score_value['scores']),
                                    'visual_delivery':delivery,'visual_tensor_sha256':vk,'evaluation_reuse':lineage,
                                    'forward_count':0 if reused else score_value['forward_count']}
                                generated={**record_fields(row,readout,c,variant,gen_value['prediction']),**provenance,
                                    'kind':'generation',**gen_value,'visual_delivery':gd,'evaluation_reuse':lineage} if gen_value else None
                                if c['role']=='native':
                                    native_index=row['source_answer_indices'][source]
                                    native_original=row['option_labels'][native_index]
                                    record['native_correct']=record['original_prediction']==native_original
                                    if generated is not None: generated['native_correct']=generated['original_prediction']==native_original if generated['scorable'] else None
                            record['elapsed_seconds']=time.monotonic()-began
                            out.write(json.dumps(record,allow_nan=False)+'\n'); meta['completed_score_rows']+=1
                            if a.generate_tokens:
                                gout.write(json.dumps(generated,allow_nan=False)+'\n'); meta['completed_generation_rows']+=1
                            check_swap(meta['initial_swap_mb'])
                if audit:
                    if source_before!=tensor_digest(sources,torch): raise ValueError('Source cache mutated')
                    audits.write(json.dumps(dict(intervention_id=iid,source_cache_unchanged=True,shared_interventions_verified=True))+'\n')
                    audited_attributes.add(anchor['attribute'])
                del sources,contexts,anchors,calibrations
                torch.cuda.empty_cache(); meta['memory']=memory(torch); write_json(a.output/'metadata.json',meta)
                print(json.dumps(dict(event='intervention_complete',intervention_id=iid,score_rows=meta['completed_score_rows'])),flush=True)
        for kind in ('score','generation'):
            if meta['completed_'+kind+'_rows']!=plan['expected_'+kind+'_rows']: raise ValueError('Incomplete '+kind+' output')
        if meta['completed_anchor_forwards']!=plan['expected_anchor_forwards']: raise ValueError('Incomplete common B anchors')
        meta.update(finished_at=utc_now(),status='completed',memory=memory(torch)); write_json(a.output/'metadata.json',meta)
        return 0
    except BaseException as exc:
        with (a.output/'failures.jsonl').open('a') as f: f.write(json.dumps(dict(id=active,error=repr(exc),at=utc_now()))+'\n')
        meta.update(failed_at=utc_now(),error=repr(exc)); write_json(a.output/'metadata.json',meta)
        raise


def main(argv=None):
    a=parse_args(argv); rows,plan=prepare(a)
    if a.plan_output: write_json(p2_data._guard_path(a.plan_output,write=True),plan)
    if a.dry_run:
        if 'torch' in sys.modules: raise RuntimeError('P4 planning imported torch')
        print(json.dumps({k:plan[k] for k in ('schema','model','panel','template_n','expected_score_rows','expected_generation_rows','expected_unique_score_evaluations','expected_anchor_forwards')},indent=2))
        return 0
    prepare_runtime(a)
    try:
        with cloud_gpu_lock(a.campaign_root,a.gpu): return execute(a,rows,plan)
    except BlockingIOError:
        print(json.dumps(dict(event='gpu_busy',gpu=a.gpu)),flush=True); return 75


if __name__=='__main__': raise SystemExit(main())
