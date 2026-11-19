"""Run the paired BLINK pilot with batch size one."""
from __future__ import annotations
import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import inspect
import json
import os
from pathlib import Path
import os as _path_os
DATA_ROOT = Path(_path_os.environ.get("DS_DATA_ROOT", "work")).expanduser().resolve()
import platform
import subprocess
import re
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def json_dump(path, obj):
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False) + '\n')


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


@contextmanager
def cached_vision(model):
    """Cache one item's visual output only; every LM forward recomputes its states."""
    visual = model.model.visual
    original = visual.forward
    cache = []
    def forward(*args, **kwargs):
        if not cache:
            cache.append(original(*args, **kwargs))
        return cache[0]
    visual.forward = forward
    try:
        yield cache
    finally:
        visual.forward = original


def swap_used_mb():
    if sys.platform != 'darwin':
        return None
    text = subprocess.check_output(['/usr/sbin/sysctl', '-n', 'vm.swapusage'], text=True)
    found = re.search(r'used = ([0-9.]+)M', text)
    return float(found.group(1)) if found else None


def memory(torch, device):
    if device == 'mps':
        torch.mps.synchronize()
        return {'mps_allocated_bytes': torch.mps.current_allocated_memory(),
                'mps_driver_bytes': torch.mps.driver_allocated_memory(),
                'mps_recommended_bytes': torch.mps.recommended_max_memory()}
    return {}


def make_processor(snapshot_path, min_pixels, max_pixels):
    from transformers import AutoProcessor
    processor = AutoProcessor.from_pretrained(snapshot_path, local_files_only=True, use_fast=False)
    # In transformers 4.57.6 the slow image processor can retain a default size
    # dictionary when min/max kwargs are applied through AutoProcessor. Bind
    # all three documented attributes so processing and provenance agree.
    image_processor = processor.image_processor
    image_processor.min_pixels = min_pixels
    image_processor.max_pixels = max_pixels
    image_processor.size = {'shortest_edge':min_pixels,'longest_edge':max_pixels}
    return processor


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest', required=True, type=Path)
    p.add_argument('--output', required=True, type=Path)
    p.add_argument('--snapshot', type=Path, default=(Path(__file__).resolve().parents[1] / 'cloud_prep/measurement_v3/snapshot_qwen4b.json'))
    p.add_argument('--limit', type=int, default=1)
    p.add_argument('--offset', type=int, default=0)
    p.add_argument('--device', choices=['mps', 'cpu', 'cuda'], default='mps')
    p.add_argument('--dtype', choices=['bfloat16','float16','float32'], default='bfloat16')
    p.add_argument('--max-pixels', type=int, default=262144)
    p.add_argument('--min-pixels', type=int, default=4096)
    p.add_argument('--max-sequence', type=int, default=2048)
    p.add_argument('--masks', nargs='+', default=['111','110','101','011','100','010','001','000'])
    p.add_argument('--generate-tokens', type=int, default=0)
    args = p.parse_args()
    legal_masks = {'111','110','101','011','100','010','001','000'}
    if len(set(args.masks)) != len(args.masks) or not set(args.masks) <= legal_masks or '111' not in args.masks:
        p.error('Masks must be unique three-bit conditions and include 111 for validation')
    if args.limit < 1 or args.offset < 0:
        p.error('limit must be positive and offset nonnegative')
    if not args.output.resolve().is_relative_to((DATA_ROOT / 'runs')):
        p.error('Raw outputs must be under the external project runs directory')
    args.output.mkdir(parents=True, exist_ok=False)
    import torch
    from PIL import Image
    from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
    from deepstack.data import load_manifest
    from deepstack.intervention import DeepStackIntervention
    from deepstack.scoring import tokenize_options, score_options
    from transformers.models.qwen3_vl import modeling_qwen3_vl
    torch.set_num_threads(4)
    torch.manual_seed(20260926)
    if args.device == 'mps' and not torch.backends.mps.is_available():
        raise RuntimeError('MPS is unavailable')
    spec = json.loads(args.snapshot.read_text())
    rows = load_manifest(args.manifest)[args.offset:args.offset+args.limit]
    meta = {'started_at': datetime.now(timezone.utc).isoformat(), 'model': spec,
            'manifest': str(args.manifest), 'manifest_sha256': sha256(args.manifest),
            'ids': [r['id'] for r in rows], 'arguments': {k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()},
            'platform': platform.platform(), 'python':sys.version,
            'libraries': {n:importlib.metadata.version(n) for n in ['torch','transformers','Pillow','huggingface_hub']},
            'transformers_source_sha256':sha256(inspect.getfile(modeling_qwen3_vl)),
            'source_sha256': {str(x.relative_to(ROOT)):sha256(x) for x in [Path(__file__), *sorted((ROOT/'deepstack').glob('*.py'))]},
            'score_protocol': 'Sum of log probabilities of each complete parenthesized option label after the same assistant generation prefix; no length normalization; no generation accuracy implied',
            'phase':'exploratory reduced-resolution local pilot', 'cache_validation':'First item compares a fully uncached original scorer against cached visual features with 111 intervention'}
    json_dump(args.output/'metadata.json', meta)
    print(json.dumps({'event':'loading', 'model':spec, 'device':args.device,'dtype':args.dtype}), flush=True)
    initial_swap = swap_used_mb()
    meta['initial_swap_mb'] = initial_swap
    started = time.monotonic()
    model = Qwen3VLForConditionalGeneration.from_pretrained(spec.get('path') or str(Path(os.environ.get('HF_HOME', DATA_ROOT / 'huggingface')) / 'hub' / ('models--' + spec['model'].replace('/', '--')) / 'snapshots' / spec['revision']), local_files_only=True,
            torch_dtype=getattr(torch,args.dtype), attn_implementation='sdpa').eval().to(args.device)
    processor = make_processor(spec.get('path') or str(Path(os.environ.get('HF_HOME', DATA_ROOT / 'huggingface')) / 'hub' / ('models--' + spec['model'].replace('/', '--')) / 'snapshots' / spec['revision']), args.min_pixels, args.max_pixels)
    if processor.image_processor.patch_size != model.config.vision_config.patch_size:
        raise RuntimeError('Processor patch size disagrees with checkpoint')
    meta.update({'load_seconds':time.monotonic()-started,'parameter_count':sum(p.numel() for p in model.parameters()),
                'vision_indexes':model.config.vision_config.deepstack_visual_indexes,
                'processor':processor.to_dict(), 'image_processor':processor.image_processor.to_dict(), 'memory_after_load':memory(torch,args.device)})
    json_dump(args.output/'metadata.json', meta)
    print(json.dumps({'event':'loaded','seconds':meta['load_seconds'],'parameters':meta['parameter_count'],**meta['memory_after_load']}),flush=True)
    output = (args.output/'results.jsonl').open('a', buffering=1)
    failures = (args.output/'failures.jsonl').open('a', buffering=1)
    validation_done = False
    with torch.inference_mode():
        for row in rows:
            try:
                images = [Image.open(x).convert('RGB') for x in row['images']]
                # Match official BLINK text-first then image_1..image_4 order.
                messages=[{'role':'user','content':[{'type':'text','text':row['prompt']}, *[{'type':'image'} for _ in images]]}]
                text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                inputs = processor(text=[text], images=images, return_tensors='pt').to(args.device)
                for im in images: im.close()
                sequence_length = inputs.input_ids.shape[1]
                grid_pixels = inputs.image_grid_thw[:,1:].prod(dim=1) * processor.image_processor.patch_size ** 2
                if bool((grid_pixels > args.max_pixels).any().item()):
                    raise RuntimeError('Actual image grid exceeds the declared pixel budget')
                if sequence_length > args.max_sequence:
                    raise RuntimeError(f'Sequence length {sequence_length} exceeds pilot cap {args.max_sequence}')
                options = tokenize_options(processor.tokenizer, row['option_labels'], prompt_text=text)
                item_meta={'sequence_length':sequence_length,'image_grid_thw':inputs.image_grid_thw.cpu().tolist(),
                           'visual_token_count':int((inputs.input_ids==model.config.image_token_id).sum().item()),
                           'prompt_text':text, 'option_tokens':options.to_dict(),
                           'image_metadata':row.get('image_metadata'), 'source':row.get('source'),
                           'raw_answer':row.get('raw_answer'), 'github_answer_label':row.get('github_answer_label'),
                           'github_label_agrees':row.get('github_label_agrees')}
                # Each item has its own visual cache; KV is never shared between masks.
                baseline = None
                if not validation_done:
                    baseline = score_options(model,inputs,options).scores
                with cached_vision(model):
                    for mask in args.masks:
                        t=time.monotonic()
                        with DeepStackIntervention(model,mask=mask,audit=not validation_done) as intervention:
                            score_result=score_options(model,inputs,options,correct_label=row['answer_label'])
                            scores=score_result.scores
                        current_swap = swap_used_mb()
                        if initial_swap is not None and current_swap is not None and current_swap > initial_swap + 128:
                            raise RuntimeError(f'Swap grew from {initial_swap}MB to {current_swap}MB; stopping own pilot')
                        pred=max(scores,key=scores.get)
                        correct_score=scores[row['answer_label']]
                        margin=correct_score-max(v for k,v in scores.items() if k!=row['answer_label'])
                        record={'id':row['id'],'task':row['task'],'mask':mask,'scores':scores,
                                'prediction':pred,'answer_label':row['answer_label'],'correct':pred==row['answer_label'],
                                'margin':margin,'scoring_method':score_result.method,'forward_count':score_result.forward_count,'elapsed_seconds':time.monotonic()-t, **item_meta,
                                'memory':memory(torch,args.device), 'system_swap_used_mb':current_swap}
                        if baseline is not None and mask=='111':
                            diff=max(abs(scores[k]-baseline[k]) for k in scores)
                            record['original_vs_111_max_score_difference']=diff
                            if diff>0.001:
                                raise RuntimeError(f'111 differs from original: {diff}')
                        if not validation_done:
                            record['intervention_audit']=intervention.records
                        output.write(json.dumps(record,ensure_ascii=False)+'\n')
                        print(json.dumps({k:record[k] for k in ['id','task','mask','prediction','correct','margin','elapsed_seconds']}),flush=True)
                    validation_done=True
                    if args.generate_tokens:
                        generated=model.generate(**inputs,max_new_tokens=args.generate_tokens,do_sample=False)
                        text_out=processor.tokenizer.decode(generated[0,sequence_length:],skip_special_tokens=True)
                        with (args.output/'generations.jsonl').open('a') as g:
                            g.write(json.dumps({'id':row['id'],'task':row['task'],'mask':'111','text':text_out,'max_new_tokens':args.generate_tokens})+'\n')
                del inputs
                if args.device=='mps': torch.mps.empty_cache()
            except Exception as exc:
                failures.write(json.dumps({'id':row['id'],'task':row['task'],'error':repr(exc)})+'\n')
                raise
    output.close()
    failures.close()
    meta['finished_at']=datetime.now(timezone.utc).isoformat()
    meta['memory_at_end']=memory(torch,args.device)
    json_dump(args.output/'metadata.json',meta)
    print(json.dumps({'event':'complete','output':str(args.output)}),flush=True)

if __name__=='__main__':
    main()
