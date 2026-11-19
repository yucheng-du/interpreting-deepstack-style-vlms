"""P4 common-B composition and one-forward all-entry calibration."""
from __future__ import annotations
from contextlib import contextmanager
import math

from cloud_prep.measurement_v8.qwen_adapter import QwenVisual
from cloud_prep.measurement_v8.ibm_adapter import IBMVisual
from cloud_prep.measurement_v12.v11_v2_dose_adapter import (
    TorchOps, DoseVisual, SiteProbe, entry_specs, entry_tensors)


PAYLOAD_SCHEMA='p4-applied-payload-v2'


class P4Ops(TorchOps):
    """TorchOps plus the few full-visual-slot operations of the exact payload witness."""
    def slots(self,x,positions): return x[0,positions,:].detach().clone()
    def add_native(self,pre,entry): return pre+entry.to(pre.device,pre.dtype)
    def cast_like(self,entry,ref): return entry.to(ref.device,ref.dtype)
    def max_abs_diff(self,a,b): return float((a.float()-b.float()).abs().max().item())
    def numel(self,x): return int(x.numel())


def exact_payload(ops,post,pre,prepared,baseline=None):
    """actual_post == pre + prepared_entry in native BF16 order (or == prepared for
    decoder-0 visual slots); optionally whether the same assertion rejects B."""
    expected=ops.add_native(pre,prepared) if pre is not None else ops.cast_like(prepared,post)
    exact=bool(ops.equal(post,expected))
    out=dict(schema=PAYLOAD_SCHEMA,exact=exact,elements=ops.numel(post),additive=pre is not None)
    if not exact: out['max_abs_diff']=ops.max_abs_diff(post,expected)
    if baseline is not None:
        expected_b=ops.add_native(pre,baseline) if pre is not None else ops.cast_like(baseline,post)
        out['B_applied_would_fail']=not bool(ops.equal(post,expected_b))
    return out


class P4SiteProbe(SiteProbe):
    """SiteProbe plus an exact applied-payload witness at every native entry.

    The unchanged SiteProbe hooks run first. Over the full visual slots the witness
    asserts actual_post == pre + prepared_entry (additive entries, native BF16
    order) or actual_post == prepared_entry (decoder-0 slots: Qwen main
    replacement, Granite zeroed slots), prepared_entry being this condition's
    composed DoseVisual entry; for selected entries it records whether the same
    assertion would reject a B delivery. Read-only; witness exceptions are recorded.
    """
    def __init__(self,adapter,visual,inputs):
        super().__init__(adapter,visual,inputs)
        self.pre_slots={}; self.witness_error=None
        try:
            self.prepared=entry_tensors(visual.visual,adapter.family)
            self.baseline=getattr(visual,'baseline_entries',None)
            self.selected=set(visual.metadata.get('selected_entries') or [])
        except Exception as exc:
            self.prepared=None; self.witness_error=repr(exc)

    def _pre_entry(self,spec):
        base=SiteProbe._pre_entry(self,spec)
        def hook(module,args,output):
            had=spec['entry'] in self.pending
            base(module,args,output)
            if not had and spec['entry'] in self.pending:
                try: self.pre_slots[spec['entry']]=self.ops.slots(self.ops.hidden(output),self.positions)
                except Exception as exc: self.witness_error=repr(exc)
        return hook

    def _post_entry(self,spec):
        base=SiteProbe._post_entry(self,spec)
        def hook(module,args,kwargs):
            n=len(self.rows)
            base(module,args,kwargs)
            if len(self.rows)==n: return  # decode step without the original visual positions
            entry=spec['entry']; pre=self.pre_slots.pop(entry,None)
            try:
                if self.witness_error: raise RuntimeError(self.witness_error)
                if spec['decoder_after'] is not None and pre is None: raise ValueError('missing pre-entry visual slots')
                raw=kwargs.get('hidden_states',args[0] if args else None)
                post=self.ops.slots(self.ops.hidden(raw),self.positions)
                b=self.baseline[entry] if self.baseline is not None and entry in self.selected else None
                payload=exact_payload(self.ops,post,pre,self.prepared[entry],b)
            except Exception as exc:
                payload=dict(schema=PAYLOAD_SCHEMA,witness_error=repr(exc))
            self.rows[-1]['applied_payload']=payload
        return hook


def aggregate(ratios,metric):
    if len(ratios)==1: return ratios[0]
    return math.sqrt(sum(v*v for v in ratios)) if metric=='rss' else sum(ratios)


def resolve_scale(condition,calibration,multiplier=1.):
    if condition['mode']=='baseline': return dict(status='baseline',k=0.,target=None,metric=None)
    if condition['mode']=='raw': return dict(status='raw',k=1.,target=None,metric=None)
    metric=condition['mode'].rsplit('_',1)[1]
    ratios=calibration['native_ratios']
    if any(x is None or not math.isfinite(x) or x<=0 for x in ratios.values()):
        return dict(status='unmatched',reason='zero_or_unavailable_required_singleton',k=None,target=None,metric=metric)
    target=min(ratios.values())*multiplier
    native=aggregate([ratios[e] for e in condition['entries']],metric)
    return dict(status='requested_match',k=target/native,target=target,metric=metric,
        native_group_ratio=native,target_multiplier=multiplier,amplifies_group=target/native>1.)


class P4Adapter:
    def __init__(self,base,family,ops=None):
        self.base_adapter=base; self.family=family; self.ops=ops or P4Ops()

    def __getattr__(self,name): return getattr(self.base_adapter,name)

    def _wrap(self,reference,tensors,metadata):
        if self.family=='qwen':
            return QwenVisual(tensors['main'],[tensors[f'later{i}'] for i in range(3)],reference.grids,reference.output_type,metadata)
        return IBMVisual(tuple((layer,(tensors[f'layer{layer}'],)) for layer,_ in reference.entries),reference.geometries,metadata)

    def common_visuals(self,sources,context,row,inputs):
        """Install donor a/b ROI at *all* entries before selecting any group."""
        sides=[]; masks={}
        for side in ('a','b'):
            role_visuals={}
            for role in ('main','later'):
                c=dict(main=context['source'],context_source=context['source'],donor_source='d_'+side,
                    changed_entry=role,intervention_type='local_role',entry_group='all',region='roi',token_permutation='identity')
                visual=self.base_adapter.compose(sources,c,row,inputs=inputs)
                role_visuals[role]=visual
                metadata=visual.metadata if self.family=='qwen' else visual.intervention
                selected=metadata['selected_token_indices']
                for spec in entry_specs(self.family,role): masks[spec['entry']]=selected[0]
            a=entry_tensors(role_visuals['main'],self.family)
            b=entry_tensors(role_visuals['later'],self.family)
            main='main' if self.family=='qwen' else 'layer0'
            merged={key:a[key] if key==main else b[key] for key in a}
            sides.append(self._wrap(role_visuals['main'],merged,dict(common_B=True,side=side,context=context)))
        return (*sides,masks)

    def compose(self,B,D,masks,condition,calibration=None,multiplier=1.,tolerance=.03):
        aa,bb=entry_tensors(B,self.family),entry_tensors(D,self.family)
        scale=resolve_scale(condition,calibration,multiplier)
        if scale['status']=='unmatched': return None,scale
        k=scale['k']; output=dict(aa); measured=[]
        selected=set(condition['entries'])
        # Baseline probes all entries in one B forward, not one forward per group.
        specs=entry_specs(self.family,'main')+entry_specs(self.family,'later')
        for spec in specs:
            key=spec['entry']; indices=masks[key]; av,bv=aa[key],bb[key]
            out=self.ops.mix(av,bv,indices,k) if key in selected else av
            output[key]=out
            a,b,o=(self.ops.select(x,indices) for x in (av,bv,out))
            den=(calibration or {}).get('entries_by_name',{}).get(key,{}).get('baseline_state')
            realized=self.ops.measure(self.ops.diff(o,a))
            measured.append({**spec,'selected_token_indices':indices,
                'selected_token_count':len(indices),'is_selected':key in selected,
                'source_delta':self.ops.measure(self.ops.diff(b,a)),
                'realized_delta':realized,'baseline_state':den,
                'realized_to_baseline_state':realized['rms']/den['rms'] if den and den['rms']>0 else None})
            if key not in selected and not self.ops.equal(out,av): raise ValueError('Unselected entry changed')
            if k==0 and not self.ops.equal(out,av): raise ValueError('k0 baseline changed')
        actual=[e['realized_to_baseline_state'] for e in measured if e['is_selected']]
        by_metric={metric:aggregate(actual,metric) if actual and all(v is not None for v in actual) else None for metric in ('rss','l1')}
        if scale['status']=='requested_match':
            value=by_metric[scale['metric']]
            error=abs(value-scale['target'])/scale['target'] if value is not None else None
            scale.update(realized=value,relative_error=error,status='matched' if error is not None and error<=tolerance else 'unmatched',
                reason=None if error is not None and error<=tolerance else 'bf16_realized_dose_outside_tolerance')
        metadata=dict(schema='p4-delivery-v1',condition=condition['name'],entries=measured,
            selected_entries=condition['entries'],effective_k=k,matching=scale,
            realized_by_metric=by_metric,context=condition['context'],
            calibration_id=(calibration or {}).get('calibration_id'),
            baseline='complete common B; every unselected entry unchanged')
        dose_visual=DoseVisual(self._wrap(B,output,metadata),metadata)
        dose_visual.baseline_entries=aa  # B entry tensors for the runtime bypass test; never serialized
        return dose_visual,scale

    @contextmanager
    def use(self,visual,inputs,*,audit=False):
        if not isinstance(visual,DoseVisual):
            with self.base_adapter.use(visual,inputs,audit=audit) as state: yield state
            return
        probe=P4SiteProbe(self,visual,inputs)
        with probe,self.base_adapter.use(visual.visual,inputs,audit=audit) as state:
            state['dose_observations']=probe.rows
            yield state
        if any(sum(r['entry']==e['entry'] for r in probe.rows)!=state['calls'] for e in visual.metadata['entries']):
            raise ValueError('Missing native entry measurement on fresh visual forward')
        state['decode_hook_steps_skipped']=probe.decode_steps_skipped

    def calibrate(self,B,D,masks,context,anchor,inputs,forward_once,prompt_sha,revision):
        c=dict(name=context['name']+'__B',entries=[],context=context,mode='baseline')
        visual,_=self.compose(B,D,masks,c)
        with self.use(visual,inputs) as delivery: forward_once()
        if delivery['calls']!=1: raise ValueError('Calibration needs one complete B forward')
        observations={e['entry']:e for e in delivery['dose_observations']}
        entries={}
        for e in visual.metadata['entries']:
            o=observations[e['entry']]
            entries[e['entry']]={**e,'baseline_state':o['post_entry_state'],'pre_residual':o['pre_residual']}
        ratios={key:e['source_delta']['rms']/e['baseline_state']['rms'] if e['baseline_state']['rms']>0 else None for key,e in entries.items()}
        return dict(calibration_id=anchor['intervention_id']+'__'+context['name'],context=context,
            anchor_query_id=anchor['id'],anchor_prompt_sha256=prompt_sha,model_revision=revision,
            entries_by_name=entries,native_ratios=ratios,fresh_forwards=1,
            shared_across_queries_and_readouts=True,baseline_forward='one full B for every singleton and group')
