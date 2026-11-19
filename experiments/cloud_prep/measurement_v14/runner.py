"""Plan on CPU; run new follow-up rows on a user-provided Linux CUDA machine."""
from __future__ import annotations
import argparse
from contextlib import contextmanager
from copy import deepcopy
from datetime import datetime,timezone
import hashlib
import importlib.metadata
from itertools import groupby
import json
import os
from pathlib import Path
import platform
import sys
import time

from cloud_prep.measurement_v14 import design
from cloud_prep.measurement_v8.runtime import prepare_runtime,resolve_snapshot,write_json,sha256


def utc_now(): return datetime.now(timezone.utc).isoformat()


def model_spec(key):
    if key=='deepstack7b':
        from cloud_prep.measurement_v14.accepted_deepstack_core import SPEC
        return {'key':key,**SPEC}
    from cloud_prep.measurement_v8.model_registry import get_model
    return get_model(key)


def parse_args(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model',required=True)
    p.add_argument('--manifest',type=Path,required=True)
    p.add_argument('--config',type=Path)
    p.add_argument('--unit-ids',nargs='+')
    p.add_argument('--limit-units',type=int)
    p.add_argument('--campaign-root',type=Path)
    p.add_argument('--cache-root',type=Path)
    p.add_argument('--output',type=Path)
    p.add_argument('--plan-output',type=Path)
    p.add_argument('--generation-only',action='store_true',help='Use row.query and answer_instruction without multiple-choice prompting')
    p.add_argument('--local-files-only',action='store_true')
    p.add_argument('--gpu',default=os.environ.get('CUDA_VISIBLE_DEVICES','0'))
    p.add_argument('--gpu-slot',type=int,default=int(os.environ.get('DS_GPU_SLOT','0')),
                   help='Concurrent lane on the same GPU (own lock file; 0 keeps gpu-N.lock). Dispatch only; no value change.')
    p.add_argument('--dry-run',action='store_true')
    args=p.parse_args(argv)
    if args.limit_units is not None and args.limit_units<1:p.error('Positive limit-units required')
    if not args.dry_run and any(getattr(args,k) is None for k in ('output','cache_root','campaign_root')):
        p.error('Execution needs --output --cache-root --campaign-root')
    if not 0<=args.gpu_slot<=7: p.error('--gpu-slot must be 0..7')
    if not args.gpu or any(c not in '0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ-_' for c in args.gpu):
        p.error('Expose one GPU per process')
    return args


def prepare(args):
    config=design.settings(args.config)
    if args.generation_only:
        config['readouts']=[dict(response_format='freeform',mapping='none')]
    rows=design.load_rows(args.manifest,config,unit_ids=args.unit_ids,limit_units=args.limit_units)
    if args.generation_only:
        for row in rows: row['readouts']=config['readouts']
    # Catch prompt/label mapping errors on CPU, before reserving a model host.
    for row in rows:
        for readout in design.readouts(row,config): make_variant(row,readout)
    spec=model_spec(args.model)
    plan=design.execution_plan(rows,spec,config)
    plan.update(manifest_sha256=sha256(args.manifest),manifest=str(args.manifest.resolve()),
        code_sha256={p.name:sha256(p) for p in sorted(Path(__file__).parent.glob('*.py'))})
    return rows,plan


def completed_rows(path):
    """Skip complete observations only; malformed/duplicate records are visible errors."""
    found={}
    if not Path(path).exists(): return found
    with Path(path).open() as f:
        for number,line in enumerate(f,1):
            try: row=json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f'Incomplete JSONL line {number}; preserve file, repair tail or use a new output directory') from exc
            key=row['observation_id']
            if key in found: raise ValueError('Duplicate completed observation: '+key)
            found[key]=row
    return found


@contextmanager
def output_lock(directory):
    import fcntl
    directory.mkdir(parents=True,exist_ok=True)
    with (directory/'.writer.lock').open('a') as handle:
        try: fcntl.flock(handle,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError as exc: raise RuntimeError('Another writer owns this output directory') from exc
        try: yield
        finally: fcntl.flock(handle,fcntl.LOCK_UN)


def make_variant(row,readout):
    if readout['response_format']=='freeform':
        return dict(prompt=row['query']+('\n'+row['answer_instruction'] if row.get('answer_instruction') else ''),
                    assistant_prefix='',response_format='freeform',mapping='none')
    from cloud_prep.measurement_v6.content_readout import make_variant as content_variant
    return content_variant(row,**readout)


def load_adapter(spec,snapshot,cache_root,local_files_only):
    if spec['key']=='deepstack7b':
        from huggingface_hub import snapshot_download
        from cloud_prep.measurement_v14.accepted_deepstack_adapter import load
        from cloud_prep.measurement_v14.accepted_deepstack_core import DeepStackCoreAdapter
        clip=snapshot_download(repo_id=spec['clip_repository'],revision=spec['clip_revision'],
            cache_dir=str(cache_root/'hub'),local_files_only=local_files_only,allow_patterns=['config.json','preprocessor_config.json'])
        adapter=DeepStackCoreAdapter(load(snapshot['path'],clip,device='cuda',attention='sdpa'))
        return adapter,'deepstack',adapter.metadata
    from cloud_prep.measurement_v12.v11_v2_runner import make_adapter
    adapter,meta=make_adapter(spec,snapshot,'v14')
    if spec['key']!='ibm4b':
        from cloud_prep.measurement_v14.adapter import flexible_qwen
        adapter=flexible_qwen(adapter)
        meta={**meta,'v14_prepare':'native processor resize; cross-source actual grid equality checked; old fixed panel dimensions removed'}
    return adapter,('ibm' if spec['key']=='ibm4b' else 'qwen'),meta


def tokenize(tokenizer,variant,prompt,family):
    if family!='deepstack':
        from deepstack.readout_v3 import tokenize_variant
        return tokenize_variant(tokenizer,variant,prompt)
    from deepstack.scoring import TokenizedOptions,tokenize_options
    values=tokenize_options(tokenizer,[' '+c for c in variant['continuations']],prompt_text=prompt)
    return TokenizedOptions(tuple(variant['labels']),values.token_ids,values.prompt_sha256)


def generate(model,inputs,config,family,guard):
    if family!='deepstack':
        from cloud_prep.measurement_v8.generation import native_greedy
        sequence=native_greedy(model,inputs,config,stopping_criteria=guard)
        return sequence[0,inputs['input_ids'].shape[1]:]
    from cloud_prep.measurement_v8.generation import greedy_kwargs
    if any(inputs.get(k) is not None for k in ('past_key_values','cache_position','position_ids','inputs_embeds','labels')):
        raise ValueError('DeepStack generation requires fresh processor inputs')
    try:
        sequence=model.generate(inputs=inputs['input_ids'],**{k:v for k,v in inputs.items() if k!='input_ids'},
            generation_config=config,stopping_criteria=guard,**greedy_kwargs(model))
    finally: model.kwargs_for_generation=None
    if sequence.ndim!=2 or sequence.shape[0]!=1 or sequence.shape[1]>config.max_new_tokens:
        raise ValueError('DeepStack must return new tokens only')
    return sequence[0]


def outcome_fields(row,variant,prediction,scores=None):
    original=variant.get('label_to_original',{}).get(prediction)
    values={'original_prediction':original}
    if 'direction_a_index' in row and 'direction_b_index' in row:
        labels=row['option_labels']; a=labels[row['direction_a_index']]; b=labels[row['direction_b_index']]
        values.update(direction_a_original=a,direction_b_original=b,
            answer_b_indicator=(int(original==b) if original in (a,b) else None))
        if scores is not None:
            inv=variant['original_to_label']
            values['b_minus_a_log_odds']=scores[inv[b]]-scores[inv[a]]
    if row.get('has_reference_answer'):
        values['clean_reference_correct']=int(original==row['answer_label']) if original is not None else None
    return values


def code_provenance(root):
    """Executing checkout commit and local changes; run metadata only, never a gate."""
    import subprocess
    def git(*argv):
        return subprocess.run(['git','--no-optional-locks','-C',str(root),*argv],capture_output=True,text=True,check=True,timeout=30).stdout
    try:
        changes=git('status','--porcelain','--untracked-files=normal').splitlines()
        return {'root':str(root),'commit':git('rev-parse','HEAD').strip(),'local_changes':changes[:200],
                'local_change_count':len(changes)}
    except (OSError,subprocess.SubprocessError) as exc:
        return {'root':str(root),'commit':None,'error':repr(exc)}


def execute(args,rows,plan):
    runtime=prepare_runtime(args)
    os.environ['DS_GPU_SLOT']=str(args.gpu_slot)
    from cloud_prep.measurement_v4.content_intervention import cloud_gpu_lock
    with output_lock(args.output),cloud_gpu_lock(args.campaign_root,args.gpu):
        return execute_locked(args,rows,plan,runtime)


def execute_locked(args,rows,plan,runtime):
    import torch
    from transformers import StoppingCriteria,StoppingCriteriaList
    from deepstack.scoring import score_options
    from deepstack.readout_v3 import fp32_output_head,PARSER_VERSION
    from cloud_prep.measurement_v6.content_readout import parse_response
    from cloud_prep.measurement_v8.generation import resolve_greedy_config
    from cloud_prep.measurement_v3.runtime import memory,check_swap,swap_used_mb
    from cloud_prep.measurement_v12.v11_v2_runner import tensor_digest
    from cloud_prep.measurement_v14.adapter import FixedLaterComposer,tensors
    from cloud_prep.measurement_v14.audits import native_geometry_audit,qwen_geometry_record
    from scripts.prepare_readout_controls import generation_stop
    config=plan['config']; output=args.output/'observations.jsonl'; done=completed_rows(output)
    # batch_conditions false (all pre-batched configs) keeps the batch-one path below verbatim.
    batched=config['batch_conditions']
    if batched:
        runtime={**runtime,'batch_size':None,'max_batch_size':config['max_batch'],
                 'batch_size_policy':'physically distinct conditions per query/readout; actual sizes recorded per batch'}
    if not torch.cuda.is_available() or torch.cuda.device_count()!=1 or torch.cuda.get_device_capability(0)[0]<8:
        raise RuntimeError('One native BF16 CUDA GPU is required')
    torch.backends.cuda.matmul.allow_tf32=False; torch.backends.cudnn.allow_tf32=False
    torch.set_num_threads(4); torch.manual_seed(config['seed']); torch.cuda.reset_peak_memory_stats()
    initial_swap=swap_used_mb()
    started=utc_now(); run_id=started.replace(':','').replace('+','_')
    meta={**plan,'run_id':run_id,'runtime':runtime,'started_at':started,'status':'running',
        'platform':platform.platform(),'libraries':{name:importlib.metadata.version(name) for name in ('torch','transformers','Pillow')},
        'cuda_device':torch.cuda.get_device_name(0),'completed_before_resume':len(done),'executed_score':0,'executed_generation':0,
        'parser':PARSER_VERSION,'actual_code_hashes_are_provenance_not_launch_gate':True,'gpu_slot':args.gpu_slot}
    if batched:meta['code_provenance']=code_provenance(Path(__file__).resolve().parents[2])
    metadata_path=args.output/('run-'+run_id+'.json'); write_json(metadata_path,meta)
    snapshot=resolve_snapshot(plan['model'],args.cache_root,args.local_files_only)
    adapter,family,adapter_meta=load_adapter(plan['model'],snapshot,args.cache_root,args.local_files_only)
    model,processor=adapter.model,adapter.processor
    if family=='qwen' and config['qwen_pixel_limits'] is not None:
        limits=config['qwen_pixel_limits'];low,high=int(limits['min_pixels']),int(limits['max_pixels'])
        if low<1 or high<low:raise ValueError('Ordered positive Qwen pixel limits required')
        processor.image_processor.min_pixels=low;processor.image_processor.max_pixels=high
        processor.image_processor.size={'shortest_edge':low,'longest_edge':high}
    meta.update(snapshot=snapshot,adapter=adapter_meta)
    meta['actual_image_processor']=processor.image_processor.to_dict()
    genconfig=deepcopy(model.generation_config);genconfig.max_new_tokens=config['generate_tokens']
    genconfig.do_sample=False;genconfig.num_beams=1;genconfig.num_return_sequences=1;genconfig.use_cache=True
    genconfig=resolve_greedy_config(model,genconfig)
    eos=genconfig.eos_token_id; eos=[eos] if isinstance(eos,int) else list(eos or [])
    if config['generate_tokens'] and not eos: raise ValueError('Generation EOS must be explicit')
    meta['generation_config']=genconfig.to_dict()
    class SwapGuard(StoppingCriteria):
        def __init__(self):self.checked=0.;self.error=None
        def __call__(self,input_ids,scores,**kwargs):
            if time.monotonic()-self.checked>10:
                self.checked=time.monotonic()
                try:check_swap(initial_swap)
                except RuntimeError as exc:self.error=str(exc)
            return bool(self.error)
    audited=set()
    try:
        with output.open('a',buffering=1) as out,(args.output/'audits.jsonl').open('a',buffering=1) as auditout,torch.inference_mode(),fp32_output_head(model):
            batcher=None
            if batched:
                from cloud_prep.measurement_v14.batched import ConditionBatcher
                def write_observation(record):
                    out.write(json.dumps(record,ensure_ascii=False,allow_nan=False)+'\n');done[record['observation_id']]=record
                def make_guard():
                    guard=SwapGuard();return guard,StoppingCriteriaList([guard])
                batcher=ConditionBatcher(adapter=adapter,family=family,config=config,plan=plan,model_key=args.model,
                    run_id=run_id,meta=meta,done=done,genconfig=genconfig,eos=eos,tokenizer=processor.tokenizer,
                    write_observation=write_observation,write_audit=lambda record:auditout.write(json.dumps(record)+'\n'),
                    make_guard=make_guard,parse_response=parse_response,generation_stop=generation_stop,
                    outcome_fields=outcome_fields)
            for iid,group in groupby(rows,key=lambda r:(r['unit_id'],r['intervention_id'])):
                block=list(group); anchor=block[0]; condition_map={}
                if batcher is not None:batcher.start_unit(anchor['unit_id'])
                for r in block:
                    for c in design.conditions(r,config):
                        if condition_map.setdefault(c['name'],c)!=c:raise ValueError('Same condition name changes within an intervention')
                conds=list(condition_map.values())
                pending=[(r,c,ro,kind) for r in block for c in design.conditions(r,config) for ro in design.readouts(r,config)
                    for kind in (['generation'] if ro['response_format']=='freeform' else ['score','generation'])
                    if (kind!='generation' or config['generate_tokens']) and design.observation_key(r,c,ro,kind,plan['model'],config) not in done]
                if not pending: continue
                names=sorted({c[k] for c in conds for k in ('main_source','later_base','later_donor')})
                sources={}; pixels={}; anchor_inputs={}; avariant=make_variant(anchor,design.readouts(anchor,config)[0])
                for name in names:
                    inp,prompt=adapter.prepare(anchor,avariant,name)
                    sources[name]=adapter.encode(inp);anchor_inputs[name]=inp
                    pixel_values=adapter.pixel_digest_inputs(inp) if family=='deepstack' else {k:v for k,v in inp.items() if k in ('pixel_values','image_sizes','image_grid_thw')}
                    pixels[name]=tensor_digest(pixel_values,torch)
                before=tensor_digest(sources,torch)
                composer=FixedLaterComposer(sources,anchor,family)
                visuals={};fixed_later={}
                for c in conds:
                    visual,delivery_meta=composer.compose(c)
                    tv=tensors(visual,family);main_key=composer.main_key
                    later_hash=tensor_digest({k:v for k,v in tv.items() if k!=main_key},torch)
                    fixed_key=design.digest({k:c[k] for k in ('later_base','later_donor','t','k','entries','region')})
                    if fixed_later.setdefault(fixed_key,later_hash)!=later_hash:raise ValueError('Later visual tensor changed with main source')
                    full_hash=tensor_digest(tv,torch)
                    visuals[c['name']]=(visual,{**delivery_meta,'visual_sha256':full_hash,'later_visual_sha256':later_hash})
                audit=config['audit_per_task'] and anchor['task'] not in audited
                if audit and family!='qwen':
                    auditout.write(json.dumps({'run_id':run_id,'id':anchor['id'],
                        'native_geometry_audit':native_geometry_audit(adapter,family,anchor_inputs,sources,anchor)})+'\n')
                elif audit and batcher is not None:
                    auditout.write(json.dumps({'run_id':run_id,'id':anchor['id'],
                        'qwen_geometry_record':qwen_geometry_record(adapter,anchor_inputs,sources,anchor)})+'\n')
                for row in block:
                    for readout in design.readouts(row,config):
                        variant=make_variant(row,readout); prepared={};prompt_ref=None; ids_ref=None
                        for name in names:
                            inp,prompt=adapter.prepare(row,variant,name)
                            if prompt_ref is not None and (prompt!=prompt_ref or not torch.equal(inp['input_ids'],ids_ref)):
                                raise ValueError('Main/source quality changed textual prompt or token geometry')
                            prompt_ref=prompt;ids_ref=inp['input_ids']
                            pv=adapter.pixel_digest_inputs(inp) if family=='deepstack' else {k:v for k,v in inp.items() if k in ('pixel_values','image_sizes','image_grid_thw')}
                            if tensor_digest(pv,torch)!=pixels[name]: raise ValueError('Query/readout changed source pixels')
                            prepared[name]=inp
                        options=None if readout['response_format']=='freeform' else tokenize(processor.tokenizer,variant,prompt_ref,family)
                        seq_len=next(iter(prepared.values()))['input_ids'].shape[1]+(575 if family=='deepstack' else 0)
                        bound=min(config['max_sequence'],getattr(adapter,'max_context',config['max_sequence']))
                        if seq_len+max(config['generate_tokens'],max(map(len,options.token_ids)) if options else 0)>bound:raise ValueError('Expanded context exceeds budget')
                        # Qwen batched vision can differ from its batch-one source cache in BF16.
                        # Audit the actual cache against batch-one native processing as well as
                        # the separate batched delivery audit performed by ConditionBatcher.
                        cache_audit=audit and (batcher is None or family=='qwen')
                        if cache_audit and options is not None:
                            differences={}
                            for name,inp in prepared.items():
                                native=score_options(model,inp,options)
                                with adapter.use(sources[name],inp) as d: patched=score_options(model,inp,options)
                                difference=max(abs(native.scores[k]-patched.scores[k]) for k in native.scores)
                                if d['calls']!=patched.forward_count or difference>config['self_patch_tolerance']:raise ValueError(f'Native/self-patch failed {name}: {difference}')
                                differences[name]=difference
                            auditout.write(json.dumps({'run_id':run_id,'id':row['id'],**readout,'native_self_differences':differences,
                                **({'audit_mode':'source_cache_batch_one'} if batcher is not None else {}),
                                'geometry':composer.geometry,'fixed_later_hashes':fixed_later})+'\n')
                        elif cache_audit and config['generate_tokens']:
                            checks={}
                            for name,inp in prepared.items():
                                native_ids=generate(model,inp,genconfig,family,None).detach().cpu().tolist()
                                with adapter.use(sources[name],inp) as d:
                                    patched_ids=generate(model,inp,genconfig,family,None).detach().cpu().tolist()
                                if native_ids!=patched_ids or d['calls']!=1:raise ValueError('Freeform native/self-patch generation differs: '+name)
                                checks[name]=dict(equal=True,new_tokens=len(native_ids))
                            auditout.write(json.dumps({'run_id':run_id,'id':row['id'],**readout,'native_self_generation':checks,
                                **({'audit_mode':'source_cache_batch_one'} if batcher is not None else {}),
                                'geometry':composer.geometry,'fixed_later_hashes':fixed_later})+'\n')
                        if batcher is not None:
                            batcher.readout(row=row,readout=readout,variant=variant,prepared=prepared,prompt=prompt_ref,
                                options=options,visuals=visuals,pixels=pixels,sources=sources,audit=audit,
                                geometry=composer.geometry,fixed_later=fixed_later)
                            continue
                        evaluation_cache={}
                        for c in design.conditions(row,config):
                            visual,vm=visuals[c['name']];inp=prepared[c['main_source']]
                            for kind in (['generation'] if options is None else ['score','generation']):
                                if kind=='generation' and not config['generate_tokens']:continue
                                key=design.observation_key(row,c,readout,kind,plan['model'],config)
                                if key in done:continue
                                began=time.monotonic();reuse_key=(kind,vm['visual_sha256']);reused=reuse_key in evaluation_cache
                                if reused: result=deepcopy(evaluation_cache[reuse_key])
                                elif kind=='score':
                                    with adapter.use(visual,inp) as d: scored=score_options(model,inp,options)
                                    if d['calls']!=scored.forward_count:raise ValueError('Score missed native visual delivery')
                                    result={**scored.to_dict(),**outcome_fields(row,variant,scored.prediction,scored.scores),
                                        'delivery':d,'score_event':'specified string continuation; no EOS',
                                        'score_separator':' ' if family=='deepstack' else ''}
                                    meta['executed_score']+=1
                                else:
                                    guard=SwapGuard()
                                    with adapter.use(visual,inp) as d: tokens=generate(model,inp,genconfig,family,StoppingCriteriaList([guard]))
                                    if guard.error:raise RuntimeError(guard.error)
                                    if d['calls']!=1:raise ValueError('Generation needs exactly one fresh visual prefill')
                                    ids=tokens.detach().cpu().tolist();text=processor.tokenizer.decode(ids,skip_special_tokens=True)
                                    stop=generation_stop(ids,eos,config['generate_tokens'])
                                    parsed=parse_response(text,variant) if options is not None else {'status':'freeform_retained','label':None}
                                    scorable=parsed['status']=='parsed' and stop['eos_reached'];prediction=parsed['label'] if scorable else None
                                    result=dict(generated_text=text,generated_text_raw=processor.tokenizer.decode(ids,skip_special_tokens=False),
                                        generated_token_ids=ids,parse=parsed,scorable=scorable if options is not None else None,
                                        prediction=prediction,**stop,**outcome_fields(row,variant,prediction),delivery=d,
                                        freeform_metric_status='official metric pending' if options is None else None)
                                    meta['executed_generation']+=1
                                if not reused:evaluation_cache[reuse_key]=deepcopy(result)
                                record=dict(schema=design.SCHEMA,observation_id=key,run_id=run_id,execution_version=config['execution_version'],
                                    id=row['id'],unit_id=row['unit_id'],intervention_id=row['intervention_id'],task=row['task'],
                                    query_role=row['query_role'],kind=kind,readout=readout,condition=c,model_key=args.model,
                                    model_revision=plan['model']['revision'],query=row['query'],reference_answers=row.get('reference_answers'),
                                    material_provenance=row.get('source',{}),benchmark=row.get('benchmark'),
                                    source_metadata=row['source_image_metadata'],source_pixel_sha256=pixels,
                                    prompt_sha256=hashlib.sha256(prompt_ref.encode()).hexdigest(),visual_metadata=vm,
                                    result=result,evaluation_reused=reused,elapsed_seconds=time.monotonic()-began)
                                out.write(json.dumps(record,ensure_ascii=False,allow_nan=False)+'\n');done[key]=record
                        check_swap(initial_swap)
                if tensor_digest(sources,torch)!=before:raise ValueError('Cached visual source mutated')
                audited.add(anchor['task']);meta['completed_total_rows']=len(done);meta['memory']=memory(torch)
                write_json(metadata_path,meta)
                print(json.dumps({'event':'intervention_complete','unit_id':anchor['unit_id'],'intervention_id':anchor['intervention_id'],'completed_rows':len(done)}),flush=True)
        meta['status']='complete'
    except BaseException as exc:
        meta.update(status='interrupted_or_failed',error=repr(exc));raise
    finally:
        meta.update(finished_at=utc_now(),completed_total_rows=len(done));write_json(metadata_path,meta)
    return meta


def main(argv=None):
    args=parse_args(argv);rows,plan=prepare(args)
    if args.plan_output:write_json(args.plan_output,plan)
    if args.dry_run:
        print(json.dumps(plan,ensure_ascii=False,indent=2));return 0
    execute(args,rows,plan);return 0


if __name__=='__main__':raise SystemExit(main())
