"""Fixed visual sources with fresh, main-dependent language-model state."""
from __future__ import annotations
from dataclasses import asdict
from cloud_prep.measurement_v12.v11_v2_dose_adapter import TorchOps


def flexible_qwen(base):
    """Extend only preparation beyond the old fixed-size material panel.

    Native processor resize is retained. Cross-quality grid equality is checked
    after encoding. ROI coordinates remain normalized original-image support.
    """
    from cloud_prep.measurement_v10.adapter import QwenAdapter
    class NativeSizeQwenAdapter(QwenAdapter):
        def prepare(self,row,variant,source):
            from PIL import Image
            from cloud_prep.measurement_v8.qwen_adapter import _grids,token_geometry
            path=row['source_images'][source]
            with Image.open(path) as raw:image=raw.convert('RGB')
            try:
                messages=[{'role':'user','content':[{'type':'text','text':variant['prompt']},{'type':'image'}]}]
                prompt=self.processor.apply_chat_template(messages,tokenize=False,add_generation_prompt=True)+variant.get('assistant_prefix','')
                inputs=self.processor(text=[prompt],images=[image],return_tensors='pt').to(self.device)
            finally:image.close()
            grids=_grids(inputs);hh,ww=token_geometry(grids)
            actual=int((inputs['input_ids']==self.model.config.image_token_id).sum().item())
            if actual!=hh*ww:raise ValueError('Native image placeholders differ from processed grid')
            if 'mm_token_type_ids' not in inputs or inputs['mm_token_type_ids'].shape!=inputs['input_ids'].shape:
                raise ValueError('Multimodal type ids do not align with prompt')
            return inputs,prompt
    return NativeSizeQwenAdapter(base.model,base.processor,device=base.device)


def tensors(visual,family):
    if family=='qwen': return {'main':visual.main,**{f'later{i}':v for i,v in enumerate(visual.deepstack)}}
    if family=='deepstack': return {'main':visual.main,**{f'later{i}':v for i,v in enumerate(visual.later)}}
    if any(len(parts)!=1 for _,parts in visual.entries): raise ValueError('Batch-one one-image Granite required')
    return {f'layer{layer}':parts[0] for layer,parts in visual.entries}


def geometry(visual,family):
    if family=='qwen': return {'grids':visual.grids}
    if family=='deepstack': return {'image_size':list(visual.image_size),'grid':list(visual.grid),'active_later_groups':visual.grid[0]*visual.grid[1]}
    return {'packed_geometry':[g.summary() for g in visual.geometries]}


def wrap(reference,values,family,metadata):
    if family=='qwen':
        from cloud_prep.measurement_v8.qwen_adapter import QwenVisual
        return QwenVisual(values['main'],[values[f'later{i}'] for i in range(3)],reference.grids,reference.output_type,metadata)
    if family=='deepstack':
        from cloud_prep.measurement_v15.accepted_deepstack_adapter import DeepStackVisual
        return DeepStackVisual(values['main'],[values[f'later{i}'] for i in range(4)],reference.image_size,reference.grid,metadata)
    from cloud_prep.measurement_v8.ibm_adapter import IBMVisual
    return IBMVisual(tuple((layer,(values[f'layer{layer}'],)) for layer,_ in reference.entries),reference.geometries,metadata)


def support(visual,row,family,region):
    values=tensors(visual,family); main='layer0' if family=='ibm' else 'main'
    if region=='all':
        result={k:list(range(v.shape[0])) for k,v in values.items() if k!=main}
    elif family=='qwen':
        from cloud_prep.measurement_v8.qwen_adapter import roi_token_indices
        meta=next(iter(row['source_image_metadata'].values()))
        ids=roi_token_indices(visual.grids,row['donor_roi_bbox_xyxy'],(meta['width'],meta['height']))
        result={k:ids for k in values if k!=main}
    elif family=='ibm':
        from cloud_prep.measurement_v8.ibm_geometry import roi_token_indices
        ids=roi_token_indices(visual.geometries[0],row['donor_roi_bbox_xyxy'])
        result={k:ids for k in values if k!=main}
    else:
        from cloud_prep.measurement_v15.accepted_deepstack_core import selected_indices
        result={k:selected_indices(visual.image_size,row['donor_roi_bbox_xyxy'],k,visual.grid) for k in values if k!=main}
    if family=='deepstack':
        for i in range(visual.grid[0]*visual.grid[1],4): result[f'later{i}']=[]
    return result


def compose_values(main_values,base_values,donor_values,masks,condition,main_key,ops):
    """The main source never determines the later baseline, delta, or k.

    A tiny operations seam permits CPU list-based tests without importing torch.
    All later slots are copied from base_values, including non-ROI slots.
    """
    keys=set(base_values)-{main_key}
    selected=keys if condition['entries']=='all_later' else set(condition['entries'])
    if selected-keys: raise ValueError('Unknown/non-later entries: '+str(selected-keys))
    out={main_key:ops.clone(main_values[main_key])}; measured=[]
    for name in sorted(keys):
        a,b=base_values[name],donor_values[name]
        if ops.shape(a)!=ops.shape(b) or ops.shape(a)!=ops.shape(main_values[name]): raise ValueError('Source tensor shape differs')
        ids=masks[name]
        k=condition['k'].get(name,1.) if isinstance(condition['k'],dict) else condition['k']
        effective=condition['t']*k if name in selected else 0.
        out[name]=ops.mix(a,b,ids,effective)
        record=dict(entry=name,selected=name in selected,selected_token_indices=ids,
            k=k,t=condition['t'],effective_k=effective,status='active' if ids else 'structurally_inactive_or_empty_support')
        if ids:
            av,bv,ov=(ops.select(x,ids) for x in (a,b,out[name]))
            record.update(source_delta=ops.measure(ops.diff(bv,av)),realized_delta=ops.measure(ops.diff(ov,av)))
        measured.append(record)
    return out,measured


class FixedLaterComposer:
    def __init__(self,sources,row,family,ops=None):
        self.sources,self.row,self.family=sources,row,family; self.ops=ops or TorchOps()
        self.geometry=geometry(next(iter(sources.values())),family)
        if any(geometry(v,family)!=self.geometry for v in sources.values()): raise ValueError('Native processed geometry differs across qualities')
        self.values={k:tensors(v,family) for k,v in sources.items()}
        self.main_key='layer0' if family=='ibm' else 'main'
        self.later_cache={}

    def compose(self,condition):
        main=condition['main_source']; baseline=condition['later_base']; donor=condition['later_donor']
        mask=support(self.sources[baseline],self.row,self.family,condition['region'])
        values,entries=compose_values(self.values[main],self.values[baseline],self.values[donor],mask,condition,self.main_key,self.ops)
        metadata=dict(schema='v14-fixed-later-delivery',main_source=main,later_base=baseline,later_donor=donor,
            entries=entries,geometry=self.geometry,region=condition['region'],
            dose_units='absolute post-projection selected-token RMS; no main-dependent normalization',
            main_dependent_receiver_state='recomputed by a fresh model forward; not cached',
            later_baseline_and_nonROI='fixed source '+baseline)
        return wrap(self.sources[main],values,self.family,metadata),metadata
