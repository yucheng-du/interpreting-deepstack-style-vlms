"""Load the pinned adapters on CUDA or Apple silicon."""
from cloud_prep.measurement_v8.runtime import snapshot_weight_files

def load_adapter(a,spec,config):
    """Keep the accepted CUDA loaders; add an explicitly labelled local Qwen pilot."""
    if a.snapshot.name != spec['revision']:
        raise ValueError('Use the fixed HF revision snapshot directory named in the model registry')
    snapshot={**spec,'path':str(a.snapshot.resolve()),'weight_files':snapshot_weight_files(a.snapshot)}
    if a.device=='cuda':
        if a.dtype!='bfloat16':raise ValueError('Formal CUDA path uses the accepted BF16 precision')
        from cloud_prep.measurement_v15.runner import load_adapter as original_load
        return (*original_load(spec,snapshot,a.cache_root,a.local_files_only),snapshot)
    if spec.get('backend')=='granite' and spec['total_parameters']<14_000_000_000:
        from cloud_prep.measurement_v17.local_loaders import load_granite_mps
        return (*load_granite_mps(a.snapshot,dtype=a.dtype),snapshot)
    if spec.get('family')=='deepstack_original' and spec['total_parameters']<14_000_000_000:
        from cloud_prep.measurement_v17.deepstack_local import load_deepstack_mps
        return (*load_deepstack_mps(a.snapshot,dtype=a.dtype),snapshot)
    if spec.get('family')!='qwen' or spec.get('backend')!='qwen_dense' or spec['total_parameters']>=14_000_000_000:
        raise ValueError('The local pilot supports dense Qwen, Granite and original DeepStack checkpoints under 14B')
    import torch
    import transformers
    from cloud_prep.measurement_v8.qwen_adapter import _validate_config
    from cloud_prep.measurement_v10.adapter import QwenAdapter
    from cloud_prep.measurement_v15.adapter import flexible_qwen
    if not torch.backends.mps.is_available(): raise RuntimeError('MPS unavailable')
    model,loading=transformers.Qwen3VLForConditionalGeneration.from_pretrained(
        str(a.snapshot),local_files_only=True,dtype=getattr(torch,a.dtype),
        device_map={'':a.device},attn_implementation='sdpa',output_loading_info=True)
    bad={k:v for k,v in loading.items() if k in ('missing_keys','unexpected_keys','mismatched_keys','error_msgs') and v}
    if bad:raise ValueError('Incomplete local checkpoint load: '+str(bad))
    _validate_config(model.config,spec)
    count=sum(p.numel() for p in model.parameters())
    if count!=spec['total_parameters']:raise ValueError('Loaded parameter count differs from fixed checkpoint')
    model.eval().requires_grad_(False)
    processor=transformers.AutoProcessor.from_pretrained(str(a.snapshot),local_files_only=True,use_fast=False)
    # These are the accepted loader limits; do not inherit mutable Hub defaults.
    limits=config['qwen_pixel_limits'] or spec['image_processor']
    lo,hi=int(limits['min_pixels']),int(limits['max_pixels'])
    processor.image_processor.min_pixels=lo;processor.image_processor.max_pixels=hi
    processor.image_processor.size={'shortest_edge':lo,'longest_edge':hi}
    adapter=flexible_qwen(QwenAdapter(model,processor,device=a.device))
    return adapter,'qwen',dict(parameter_count=count,device=a.device,dtype=a.dtype,
        role='local feasibility and timing pilot; not pooled with formal CUDA results',
        loading_info={k:sorted(v,key=str) if isinstance(v,(set,frozenset)) else v for k,v in loading.items()}),snapshot
