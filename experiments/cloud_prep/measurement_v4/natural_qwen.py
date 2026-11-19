"""Natural Visual_Similarity single/joint DS candidate-content swaps on CUDA."""
from __future__ import annotations
import argparse
from copy import deepcopy
from datetime import datetime,timezone
import itertools
import json
import inspect
import importlib.metadata
import os
import sys
from pathlib import Path
import time
import traceback

from cloud_prep.common import sha256,write_json
from cloud_prep.portable_data import load_records
from cloud_prep.measurement_v3 import native_runtime as rt,qwen_native as backend
from cloud_prep.measurement_v3.native import emit,score_check,image_variant
from cloud_prep.measurement_v4.natural_panel import rank_id
from deepstack.readout_v3 import format_variant,PARSER_VERSION

SCHEMA='natural-qwen-ds-content-v4-1'


def conditions():
    return [{'name':f'm{main}_ds{bits}','main':'swapped' if main else 'original',
             'ds_sources':['swapped' if b=='1' else 'original' for b in bits],
             'main_swapped':bool(main),'ds_swap_bits':bits}
            for main in (0,1) for bits in ('000','100','010','001','110','101','011','111')]


def parse_args(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('manifest','snapshot','campaign-root','cache-root','output'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--image-root',type=Path);p.add_argument('--panel',choices=('smoke','main'),default='main')
    p.add_argument('--offset',type=int,default=0);p.add_argument('--limit',type=int,default=135)
    p.add_argument('--generate-tokens',type=int,default=64);p.add_argument('--max-sequence',type=int,default=8192)
    p.add_argument('--format',default='strict_letters',choices=('base','strict_letters','strict_digits','prefix_digits'))
    p.add_argument('--mappings',nargs='+',choices=('identity','cyclic'),default=['identity','cyclic'])
    p.add_argument('--gpu',default='0');p.add_argument('--local-files-only',action='store_true');p.add_argument('--dry-run',action='store_true')
    a=p.parse_args(argv)
    if a.offset<0 or not 1<=a.limit<=135 or not 0<=a.generate_tokens<=256 or not 1<=a.max_sequence<=8192:p.error('Invalid bounded run size')
    if len(a.mappings)!=len(set(a.mappings)):p.error('Duplicate mapping')
    rt.guard_paths(a.campaign_root,a.cache_root,a.output)
    return a


def prepare(a):
    spec=rt.read_spec(a.snapshot)
    if spec['family']!='qwen3_vl':raise ValueError('This natural DS arm requires Qwen3-VL')
    rows=load_records(a.manifest,a.image_root)
    if a.panel=='smoke':rows=sorted(rows,key=lambda r:rank_id(r['id']))[:4]
    rows=rows[a.offset:a.offset+a.limit]
    if not rows:raise ValueError('Empty selected panel')
    for row in rows:
        if row['task']!='Visual_Similarity' or row.get('natural_v4',{}).get('slot_order')!=[0,2,1]:
            raise ValueError('Use versioned complete natural VS panel')
    cfg=conditions()
    plan={'schema':SCHEMA,'phase':'exploratory','panel':a.panel,
        'checkpoint':{k:spec[k] for k in ('model','revision','total_parameters')},
        'ids':[r['id'] for r in rows],'manifest_sha256':sha256(a.manifest),'conditions':cfg,
        'format':a.format,'mappings':a.mappings,'expected_score_rows':len(rows)*len(a.mappings)*len(cfg),
        'expected_generation_rows':sum(r['natural_v4']['generation_selected'] for r in rows)*len(a.mappings)*len(cfg) if a.generate_tokens else 0,
        'generate_tokens':a.generate_tokens,'max_sequence':a.max_sequence,'parser':PARSER_VERSION,
        'content_endpoint':'swapped-candidate label logp minus original-candidate label logp; within fixed main/mapping',
        'sample_status':'complete previously explored natural VS validation set; no confirmation claim',
        'gold_policy':'original task gold is retained for conflicted feature arms; swapped image gold retained separately',
        'generation_selection':'fixed output-independent SHA256 lowest32 item IDs; every condition retained',
        'interaction_policy':'evaluate each DS subset within each main and mapping; do not require same direction',
        'cuda_verified':False}
    plan['protocol_sha256']=rt.canonical_sha(plan)
    return spec,rows,plan


def run(a,spec,rows,plan):
    rt.prepare_runtime(a.campaign_root,a.cache_root,a.output);a.output.mkdir(parents=True,exist_ok=False)
    meta={**plan,'started_at':datetime.now(timezone.utc).isoformat(),'completed_score_rows':0,'completed_generation_rows':0,
          'source_sha256':{name:sha256(Path(__file__).resolve().parents[2]/name) for name in ['cloud_prep/measurement_v4/natural_qwen.py', 'cloud_prep/measurement_v4/natural_panel.py', 'deepstack/scoring.py', 'deepstack/readout_v3.py', 'cloud_prep/measurement_v3/native_runtime.py', 'cloud_prep/measurement_v3/native.py', 'cloud_prep/generation.py', 'cloud_prep/measurement_v4/content_intervention.py', 'deepstack/intervention.py', 'deepstack/slot_swap.py', 'cloud_prep/measurement_v3/qwen_native.py']}}
    write_json(a.output/'metadata.json',meta)
    try:
        import torch
        from cloud_prep.measurement_v3.runtime import validate_cuda,memory,swap_used_mb,check_swap
        from cloud_prep.measurement_v4.content_intervention import compose_visual_output
        from deepstack.slot_swap import encode_visual,use_visual_output,reorder_tensor,compare_outputs
        from deepstack.intervention import DeepStackIntervention
        from deepstack.scoring import score_options
        from deepstack.readout_v3 import tokenize_variant,fp32_output_head,parse_answer
        from cloud_prep.generation import prepare_native_generation
        from scripts.prepare_readout_controls import generation_stop
        meta['cuda']=validate_cuda(torch);meta['initial_swap_mb']=swap_used_mb()
        snapshot,provenance=rt.resolve_snapshot(spec,a.cache_root,local_files_only=a.local_files_only)
        torch.manual_seed(20261004);torch.set_num_threads(4)
        model,processor,backend_meta=backend.load_backend(spec,snapshot)
        meta.update(checkpoint_files=provenance,backend_metadata=backend_meta,
            libraries={name:importlib.metadata.version(name) for name in ('torch','transformers','Pillow')},
            model_source_sha256=sha256(inspect.getfile(type(model))))
        settings=deepcopy(spec['generation']);settings.update(max_new_tokens=max(1,a.generate_tokens),min_new_tokens=0,
            do_sample=False,num_beams=1,num_return_sequences=1,use_cache=True,stop_strings=None)
        generator=prepare_native_generation(model,settings);eos=generator.resolved_config.eos_token_id
        eos_ids=[eos] if isinstance(eos,int) else list(eos or [])
        with (a.output/'scores.jsonl').open('x') as sf,(a.output/'generations.jsonl').open('x') as gf, \
             (a.output/'semantic_checks.jsonl').open('x') as vf,torch.inference_mode(),fp32_output_head(model):
            for index,row in enumerate(rows):
                source=None;grids=None
                for mapping in a.mappings:
                    variant=format_variant(row,a.format,mapping)
                    inputs,prompt,input_meta=backend.make_inputs(processor,row,variant,spec)
                    current_grids=input_meta['image_grid_thw']
                    if len(current_grids)!=3 or current_grids[1]!=current_grids[2]:raise ValueError('Candidate grids differ')
                    options=tokenize_variant(processor.tokenizer,variant,prompt)
                    if inputs['input_ids'].shape[1]+max(a.generate_tokens,max(map(len,options.token_ids)))>a.max_sequence:
                        raise ValueError('Sequence budget exceeded; no silent truncation')
                    if source is None:
                        grids=current_grids;original=encode_visual(model,inputs)
                        swapped=(reorder_tensor(original[0],grids,[0,2,1]),
                                 [reorder_tensor(x,grids,[0,2,1]) for x in original[1]])
                        source={'original':original,'swapped':swapped}
                    elif grids!=current_grids:raise ValueError('Mapping changed visual grid')
                    if index==0:
                        native=score_options(model,inputs,options)
                        with use_visual_output(model,source['original'],grids),DeepStackIntervention(model,'111',audit=True) as adapter:
                            checked=score_options(model,inputs,options)
                        emit(vf,{'id':row['id'],'mapping':mapping,'control':'native_feature_identity',
                            'max_score_delta':score_check(native.scores,checked.scores),'records':adapter.records})
                        swapped_row=image_variant(row,'candidate_swap');swap_variant=format_variant(swapped_row,a.format,mapping)
                        swapped_inputs,_,swap_meta=backend.make_inputs(processor,swapped_row,swap_variant,spec)
                        if swap_meta['image_grid_thw']!=grids:raise ValueError('Real swap grid changed')
                        actual_swapped=encode_visual(model,swapped_inputs)
                        comparisons=compare_outputs(source['swapped'],actual_swapped)
                        if any(x['max_abs_difference']>.001 for x in comparisons):raise RuntimeError('Real-image/reordered-feature bridge failed')
                        real_score=score_options(model,swapped_inputs,options)
                        with use_visual_output(model,source['swapped'],grids),DeepStackIntervention(model,'111'):
                            cached_swap_score=score_options(model,inputs,options)
                        emit(vf,{'id':row['id'],'mapping':mapping,'control':'real_candidate_swap_bridge',
                            'feature_comparison':comparisons,'max_score_delta':score_check(real_score.scores,cached_swap_score.scores),
                            'scope':'first selected item per mapping only'})
                        del actual_swapped,swapped_inputs
                    original_label=variant['answer_label']
                    swapped_label=variant['original_to_label'][row['natural_v4']['swapped_answer_label']]
                    for cfg in plan['conditions']:
                        output=compose_visual_output(source,cfg,grids)
                        shared={'id':row['id'],'task':row['task'],'format':a.format,'mapping':mapping,'condition':cfg['name'],
                            'main':cfg['main'],'ds_swap_bits':cfg['ds_swap_bits'],'ds_sources':cfg['ds_sources'],
                            'image_condition':'feature_blocks','protocol_sha256':plan['protocol_sha256'],
                            'original_answer_label':row['answer_label'],'answer_label':original_label,'swapped_answer_label':swapped_label,
                            'label_to_original':variant['label_to_original'],'original_to_label':variant['original_to_label'],
                            'image_sha256':[m['sha256'] for m in row['image_metadata']],'input_metadata':input_meta,
                            'prompt_text':prompt,'input_ids':inputs['input_ids'][0].tolist(),'option_tokens':options.to_dict()}
                        started=time.monotonic()
                        with use_visual_output(model,output,grids),DeepStackIntervention(model,'111',audit=index==0) as adapter:
                            scored=score_options(model,inputs,options,correct_label=original_label)
                        if len(adapter.records)!=scored.forward_count or not all(r['completed'] for r in adapter.records):
                            raise RuntimeError('DS patch did not cover every scoring forward')
                        emit(sf,{**shared,**scored.to_dict(),'correct':scored.prediction==original_label,
                            'follows_swapped_gold':scored.prediction==swapped_label,
                            'directed_content_score':scored.scores[swapped_label]-scored.scores[original_label],
                            'intervention_records':adapter.records,'elapsed_seconds':time.monotonic()-started,'memory':memory(torch)})
                        sf.flush();meta['completed_score_rows']+=1
                        if a.generate_tokens and row['natural_v4']['generation_selected']:
                            with use_visual_output(model,output,grids),DeepStackIntervention(model,'111'):
                                generated=generator.generate(model,inputs)
                            tokens=generated[0,inputs['input_ids'].shape[1]:].tolist()
                            text=processor.tokenizer.decode(tokens,skip_special_tokens=True);stop=generation_stop(tokens,eos_ids,a.generate_tokens)
                            parsed=parse_answer(text,variant['labels'],variant['assistant_prefix'])
                            scorable=parsed['status']=='parsed' and stop['eos_reached'] and not stop['truncated']
                            emit(gf,{**shared,'generated_token_ids':tokens,'text':text,
                                'text_with_special_tokens':processor.tokenizer.decode(tokens,skip_special_tokens=False),
                                'parse':parsed,**stop,'scorable':scorable,'prediction':parsed['label'] if scorable else None,
                                'correct':bool(scorable and parsed['label']==original_label),
                                'follows_swapped_gold':bool(scorable and parsed['label']==swapped_label),'invalid_retained':True})
                            gf.flush();meta['completed_generation_rows']+=1
                        check_swap(meta['initial_swap_mb']);del output
                    del inputs
                del source,original,swapped
                write_json(a.output/'metadata.json',meta)
                print(json.dumps({'event':'item_complete','id':row['id'],'index':index+1,'total':len(rows)}),flush=True)
        if (meta['completed_score_rows'],meta['completed_generation_rows'])!=(plan['expected_score_rows'],plan['expected_generation_rows']):raise RuntimeError('Incomplete declared matrix')
        meta.update(finished_at=datetime.now(timezone.utc).isoformat(),memory_final=memory(torch),cuda_verified=True)
    except BaseException as error:
        meta.update(failed_at=datetime.now(timezone.utc).isoformat(),failure=repr(error),traceback=traceback.format_exc())
        write_json(a.output/'metadata.json',meta);raise
    write_json(a.output/'metadata.json',meta);return 0

def main(argv=None):
    a=parse_args(argv);spec,rows,plan=prepare(a)
    if a.dry_run:
        print(json.dumps(plan,indent=2));return 0
    if sys.platform!='linux':
        raise RuntimeError('Linux CUDA only; no local model/tensor execution')
    from cloud_prep.measurement_v4.content_intervention import cloud_gpu_lock
    os.environ['CUDA_VISIBLE_DEVICES']=a.gpu
    try:
        with cloud_gpu_lock(a.campaign_root,a.gpu):return run(a,spec,rows,plan)
    except BlockingIOError:
        print(json.dumps({'status':'busy','gpu':a.gpu,'exit_code':75}));return 75

if __name__=='__main__':raise SystemExit(main())
