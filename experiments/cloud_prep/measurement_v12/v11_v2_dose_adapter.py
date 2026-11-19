# V12 vendor copy of accepted archived v11 opposite-native-v2; upstream algorithm unchanged.
"""Difference-dose composition and scalar-only native-site measurement.

Imports are tensor-free. The CUDA runner wraps the visual delivery adapter
with removable decoder hooks.
"""
from contextlib import contextmanager
from dataclasses import dataclass
import math

from cloud_prep.measurement_v12 import v11_v2_dose_design as design
from cloud_prep.measurement_v8.qwen_adapter import QwenVisual
from cloud_prep.measurement_v8.ibm_adapter import IBMVisual
from cloud_prep.measurement_v10.full_data import bbox
from cloud_prep.measurement_v9.geometry import qwen_region_indices, ibm_region_indices


class UnmatchedDose(ValueError):
    def __init__(self, report):
        self.report = {**report,'status':'unmatched'}
        super().__init__('Requested relative dose is unmatched: ' + report['reason'])


@dataclass
class DoseVisual:
    visual: object
    metadata: dict


class TorchOps:
    """Small operations seam; tests supply plain-list arithmetic, never torch."""
    def shape(self, x): return tuple(x.shape)
    def select(self, x, indices): return x[indices]
    def clone(self, x): return x.clone()
    def diff(self, b, a): return b.detach().float() - a.detach().float()
    def dot(self, a, b): return float((a.detach().float()*b.detach().float()).sum().item())
    def sum_entries(self, values):
        result = values[0].detach().float().clone()
        for value in values[1:]: result.add_(value.detach().float())
        return result
    def mix(self, a, b, indices, k):
        result = a.clone()
        if k == 0: return result
        if k == 1: result[indices] = b[indices]
        else: result[indices] = (a[indices].float() + k*(b[indices].float()-a[indices].float())).to(a.dtype)
        return result
    def equal(self, a, b):
        import torch
        return bool(torch.equal(a,b))
    def measure(self, x):
        f = x.detach().float()
        if not bool(f.isfinite().all().item()): raise ValueError('Nonfinite feature/state: stop dose execution')
        n = f.numel()
        if n < 1: raise ValueError('Empty selected feature/state')
        norm = float(f.norm().item())
        return {'norm':norm, 'rms':norm/math.sqrt(n), 'elements':n}
    def positions(self, inputs, token):
        ids = inputs['input_ids'].detach().cpu().tolist()
        if len(ids)!=1: raise ValueError('Dose measurement requires batch=1')
        return [i for i,t in enumerate(ids[0]) if t==token]
    def hidden(self, value):
        return value[0] if isinstance(value,(tuple,list)) else value
    def select_hidden(self, x, positions, indices): return x[0, [positions[i] for i in indices], :]


def entry_tensors(visual, family):
    if family == 'qwen': return {'main':visual.main, **{f'later{i}':v for i,v in enumerate(visual.deepstack)}}
    return {f'layer{layer}':features[0] for layer,features in visual.entries}


def entry_specs(family, role):
    if family == 'qwen':
        return [{'entry':'main','decoder_before':0,'decoder_after':None,'kind':'input_embedding_replacement','pre_residual_reason':'replacement_has_no_additive_pre_residual'}] if role=='main' else [
            {'entry':f'later{i}','decoder_before':i+1,'decoder_after':i,'kind':'post_decoder_addition','pre_residual_reason':None} for i in range(3)]
    layers = [0] if role=='main' else [3,6,9,12,15,18,21]
    return [{'entry':f'layer{i}','decoder_before':i,'decoder_after':i-1 if i else None,
             'kind':'pre_decoder_addition_to_zeroed_visual_slots' if i==0 else 'pre_decoder_addition',
             'pre_residual_reason':'structural_zero_visual_slots_before_layer0' if i==0 else None} for i in layers]


def describe_delta_geometry(ops, deltas, measures=None):
    """Describe entry vectors in raw shared coordinates, without causal transport.

    RSS is a direct-sum convention, not an independence/orthogonality claim.
    Summing vectors from different decoder times ignores intervening nonlinear
    computation. Even cosine=1/-1 cannot imply causal reinforcement/cancellation.
    """
    names = list(deltas)
    if not names: raise ValueError('At least one entry delta required')
    measured = measures or {name:ops.measure(value) for name,value in deltas.items()}
    if set(measured)!=set(deltas): raise ValueError('Geometry measurements do not match entry identities')
    norms = [measured[name]['norm'] for name in names]
    rms = [measured[name]['rms'] for name in names]
    result = {'schema':'cross-entry-delta-geometry-v1','entry_order':names,
        'scope':'Descriptive alignment by nominal token and hidden-channel coordinates; different decoder times are not transported to a common causal site',
        'rss_definition':'Direct-sum norm of entry RMS values, no cross-entry orthogonality assumption',
        'l1_definition':'Sum of per-entry RMS values; not the elementwise L1 norm of one tensor',
        'per_entry_rms':dict(zip(names,rms)),
        'entry_rms_rss':math.sqrt(sum(x*x for x in rms)), 'entry_rms_l1':sum(rms),
        'raw_coordinate_sum_norm':None,'raw_coordinate_sum_rms':None,'pairwise_cosine':[]}
    shapes = {ops.shape(x) for x in deltas.values()}
    if len(shapes)!=1:
        return {**result,'status':'unavailable_cross_entry_shape_mismatch'}
    summed = ops.measure(ops.sum_entries(list(deltas.values())))
    result.update(status='available',shared_shape=list(next(iter(shapes))),
                  raw_coordinate_sum_norm=summed['norm'],raw_coordinate_sum_rms=summed['rms'])
    for i,left in enumerate(names):
        for j in range(i+1,len(names)):
            right = names[j]
            dot = ops.dot(deltas[left],deltas[right])
            if not math.isfinite(dot): raise ValueError('Nonfinite descriptive delta dot product')
            zero = norms[i]==0 or norms[j]==0
            cosine = None if zero else max(-1.,min(1.,dot/(norms[i]*norms[j])))
            result['pairwise_cosine'].append({'entry_i':left,'entry_j':right,'dot':dot,
                'cosine':cosine,'status':'undefined_zero_delta' if zero else 'available'})
    return result


def native_dose(calibration, denominator='baseline_state'):
    """Normalize each entry before aggregation; missing is a protocol outcome."""
    if calibration is None:
        return {'rss':None, 'l1':None, 'entry_ratios':[], 'reason':'missing_calibration'}
    ratios = []; missing = []
    for e in calibration['entries']:
        den = e.get(denominator)
        delta = e['source_delta']['rms']
        if not math.isfinite(delta) or delta < 0:
            raise ValueError('Nonfinite or negative measured feature delta')
        if den is not None and not math.isfinite(den['rms']):
            raise ValueError('Nonfinite measured residual denominator')
        if den is None or den['rms'] <= 0:
            missing.append(f'{e["entry"]}:missing_or_zero_{denominator}')
        else:
            ratios.append(delta/den['rms'])
    if missing or not ratios:
        return {'rss':None,'l1':None,'entry_ratios':ratios,
            'reason':missing[0] if missing else 'missing_entries'}
    rss, l1 = math.hypot(*ratios), sum(ratios)
    if not math.isfinite(rss) or not math.isfinite(l1):
        raise ValueError('Nonfinite group relative dose')
    return {'rss':rss,'l1':l1,'entry_ratios':ratios,'reason':None if rss>0 else 'zero_group_delta'}


def calibration_binding(calibration, prefix):
    fields = ('calibration_id','anchor_query_id','anchor_query_sha256','anchor_prompt_sha256',
              'visual_signature','context_signature','model_revision')
    return {prefix+'_'+key:calibration.get(key) if calibration else None for key in fields}


def resolve_scale(condition, calibration, opposite_calibration=None):
    """Never clip: each target is resolved from this intervention's own anchors."""
    mode, value = condition['dose_mode'], condition['dose_value']
    policy, metric = condition.get('target_policy','fixed'), condition.get('dose_metric','rss')
    if policy not in ('fixed','opposite_native') or metric not in (('raw','rss','l1') if mode=='raw' else ('rss','l1')):
        raise ValueError('Unknown target policy or dose metric')
    if mode != 'raw' and policy == 'opposite_native' and mode != 'baseline_state':
        raise ValueError('Opposite-native requires baseline-state normalization')
    opposite = opposite_calibration or (calibration or {}).get('opposite_calibration')
    denfield = 'pre_residual' if mode == 'pre_residual' else 'baseline_state'
    own = native_dose(calibration, denfield)
    other = native_dose(opposite, denfield)
    by_metric = {m:{'native_self_dose':own[m],'target_other_native_dose':other[m],
        'suggested_k':other[m]/own[m] if own[m] and other[m] else None} for m in ('rss','l1')}
    common = {'target_policy':policy,'dose_metric':metric,'denominator':denfield,
        **calibration_binding(calibration,'source'), **calibration_binding(opposite,'target'),
        'native_self_rho':own['rss'],'target_other_native_rho':other['rss'],
        'native_self_by_metric':{m:own[m] for m in ('rss','l1')},
        'target_other_native_by_metric':{m:other[m] for m in ('rss','l1')},'by_metric':by_metric,
        'target_provenance':'paired_opposite_role_canonical_anchor' if policy=='opposite_native' and mode!='raw'
            else ('configured_fixed_target' if mode!='raw' else 'configured_raw_k'),
        'max_k':condition['max_calibrated_k'],'resolved_target':None,'requested_k':None,
        'within_configured_k_limit':None,'realized_precision_verified':False}
    if mode == 'raw':
        return {**common,'status':'raw','k':float(value),'requested_k':float(value),
            'target_ratio':None,'reference_group_ratio':None if metric=='raw' else own[metric],
            'reachability':'raw_not_matched'}
    target = other[metric] if policy=='opposite_native' else value
    common.update(resolved_target=target, target_ratio=target, reference_group_ratio=own[metric],
                  entry_reference_ratios=own['entry_ratios'])
    reason = own['reason'] or ('opposite:'+other['reason'] if policy=='opposite_native' and other['reason'] else None)
    if reason:
        return {**common,'status':'unmatched','k':None,'reason':reason,'reachability':'unavailable_native_dose'}
    if not isinstance(target,(int,float)) or not math.isfinite(target) or target <= 0:
        raise ValueError('Invalid resolved relative dose target')
    k = target/own[metric]
    if not math.isfinite(k):
        raise ValueError('Nonfinite requested scale')
    common.update(requested_k=k,within_configured_k_limit=k<=condition['max_calibrated_k'])
    if k > condition['max_calibrated_k']:
        return {**common,'status':'unmatched','k':None,'reason':'requested_k_exceeds_explicit_limit',
            'reachability':'exceeds_explicit_k_limit'}
    return {**common,'status':'requested_match','k':k,'reachability':'within_k_limit_pending_realization'}


class SiteProbe:
    """Native decoder boundaries: no hooks modify inputs or outputs.

    Qwen later i: output of decoder i is pre-add; input to decoder i+1 is
    post-add. Granite layer L: decoder L-1 output is pre-add; decoder L input
    is post-add. Qwen main and Granite layer0 only have a usable post-entry
    baseline. Decode tokens without the original visual positions are skipped.
    """
    def __init__(self, adapter, visual, inputs):
        self.adapter, self.visual, self.inputs = adapter, visual, inputs
        self.ops = adapter.ops
        self.specs = visual.metadata['entries']
        model = adapter.model
        self.text = model.model.language_model
        token = getattr(model.config,'image_token_id',None)
        if token is None: token = model.config.image_token_index
        self.positions = self.ops.positions(inputs,token)
        if not self.positions: raise ValueError('No actual image-token positions for dose probe')
        self.rows, self.handles = [], []
        self.pending = {}
        self.decode_steps_skipped = 0

    def _selected(self, raw, spec):
        x = self.ops.hidden(raw); shape = self.ops.shape(x)
        if len(shape)!=3 or shape[0]!=1: raise ValueError('Unexpected decoder state shape')
        if shape[1] <= max(self.positions):
            self.decode_steps_skipped += 1
            return None
        return self.ops.select_hidden(x,self.positions,spec['selected_token_indices'])

    def _pre_entry(self, spec):
        def hook(module,args,output):
            x = self._selected(output,spec)
            if x is None: return
            key = spec['entry']
            if key in self.pending: raise ValueError('Duplicate pre-entry observation before delivery')
            self.pending[key] = self.ops.measure(x)
        return hook

    def _post_entry(self, spec):
        def hook(module,args,kwargs):
            raw = kwargs.get('hidden_states',args[0] if args else None)
            if raw is None: raise ValueError('No hidden_states at decoder entry')
            x = self._selected(raw,spec)
            if x is None: return
            pre = self.pending.pop(spec['entry'],None)
            if spec['decoder_after'] is not None and pre is None: raise ValueError('Missing native pre-entry state observation')
            post = self.ops.measure(x)
            delta_rms = spec['realized_delta']['rms']
            self.rows.append({'entry':spec['entry'],'decoder_before':spec['decoder_before'],
                'pre_residual':pre,'pre_residual_missing_reason':spec['pre_residual_reason'],
                'post_entry_state':post, 'selected_token_count':len(spec['selected_token_indices']),
                'realized_delta_to_actual_pre_residual':delta_rms/pre['rms'] if pre and pre['rms']>0 else None,
                'realized_delta_to_actual_post_state':delta_rms/post['rms'] if post['rms']>0 else None})
        return hook

    def __enter__(self):
        try:
            for spec in self.specs:
                if spec['decoder_after'] is not None:
                    self.handles.append(self.text.layers[spec['decoder_after']].register_forward_hook(self._pre_entry(spec)))
                self.handles.append(self.text.layers[spec['decoder_before']].register_forward_pre_hook(self._post_entry(spec),with_kwargs=True))
        except BaseException:
            self.__exit__(None,None,None); raise
        return self

    def __exit__(self,*exc):
        for h in self.handles: h.remove()
        self.handles.clear()
        if exc[0] is None and self.pending: raise ValueError('Unpaired pre-entry observations')


class DoseAdapter:
    def __init__(self, base_adapter, family, ops=None):
        if family not in ('qwen','ibm'): raise ValueError('Only reviewed Qwen/Granite families supported')
        self.base_adapter, self.family, self.ops = base_adapter, family, ops or TorchOps()
        self.model, self.processor = base_adapter.model, base_adapter.processor
        self.cache = {}
        self.geometry_cache = {}
        self.geometry_intervention = None

    def __getattr__(self,name): return getattr(self.base_adapter,name)

    def _reference(self,sources,reference,row,inputs):
        return self.base_adapter.compose(sources, {'main':reference['main'],'donor_source':reference['later'],
            'region':'all','entry_group':'all','token_permutation':'identity'},row,inputs=inputs)

    def _selected(self,visual,condition,row,n):
        if condition['region']=='all': return list(range(n))
        box = bbox(row['scene'],row['changed_object'],condition['changing_layout'])
        if self.family=='qwen':
            source = condition['reference_a'][condition['changed_entry']]
            meta = row['source_image_metadata'][source]
            return qwen_region_indices(visual.grids,row['tile_grid'],'roi',box,(meta['width'],meta['height']),'identity')
        return ibm_region_indices(visual.geometries[0],row['tile_grid'],'roi',box,'identity')

    def compose(self,sources,condition,row,inputs=None,calibration=None,opposite_calibration=None):
        if condition.get('role')!='dose_pair': return self.base_adapter.compose(sources,condition,row,inputs=inputs)
        signature = design.visual_signature(row,condition)
        if self.geometry_intervention != row['intervention_id']:
            self.geometry_cache.clear()
            self.geometry_intervention = row['intervention_id']
        if calibration is not None:
            if (calibration['visual_signature'],calibration['calibration_id'],calibration['intervention_id']) != (signature,condition['calibration_id'],row['intervention_id']):
                raise ValueError('Calibration from a different visual intervention')
        opposite = opposite_calibration or (calibration or {}).get('opposite_calibration')
        if condition.get('target_policy') == 'opposite_native' and opposite is not None:
            other_c = design.opposite_condition(condition)
            for cal, expected in ((calibration,condition),(opposite,other_c)):
                if cal is None: continue
                if (cal.get('visual_signature'),cal.get('calibration_id'),cal.get('intervention_id'),
                    cal.get('context_signature'),cal.get('changed_entry')) != (
                        design.visual_signature(row,expected),expected['calibration_id'],row['intervention_id'],
                        design.context_signature(row,expected),expected['changed_entry']):
                    raise ValueError('Opposite calibration from a different visual intervention/context')
            if calibration is not None:
                for field in ('anchor_query_id','anchor_query_sha256','anchor_prompt_sha256','model_revision'):
                    if calibration.get(field) != opposite.get(field):
                        raise ValueError('Paired calibration anchor/revision mismatch: '+field)
        scale = resolve_scale(condition,calibration,opposite)
        if scale['status']=='unmatched': raise UnmatchedDose(scale)
        k = 0. if condition['donor_assignment']=='a' else scale['k']
        a = self._reference(sources,condition['reference_a'],row,inputs)
        b = self._reference(sources,condition['reference_b'],row,inputs)
        aa, bb = entry_tensors(a,self.family),entry_tensors(b,self.family)
        result = dict(aa); metrics = []
        source_geometry_key = (signature,'source')
        realized_geometry_key = (signature,'realized',k)
        need_source = source_geometry_key not in self.geometry_cache
        need_realized = realized_geometry_key not in self.geometry_cache
        source_deltas, realized_deltas = {}, {}
        cals = {e['entry']:e for e in calibration['entries']} if calibration else {}
        for spec in entry_specs(self.family,condition['changed_entry']):
            key = spec['entry']; av,bv = aa[key],bb[key]
            if self.ops.shape(av)!=self.ops.shape(bv): raise ValueError('Different a/b feature geometry')
            selected = self._selected(a,condition,row,self.ops.shape(av)[0])
            if not selected: raise ValueError('No selected dose tokens')
            out = self.ops.mix(av,bv,selected,k)
            result[key] = out
            sa,sb,so = (self.ops.select(t,selected) for t in (av,bv,out))
            source_delta = self.ops.diff(sb,sa)
            realized_delta = self.ops.diff(so,sa)
            if need_source: source_deltas[key] = source_delta
            if need_realized: realized_deltas[key] = realized_delta
            metrics.append({**spec,'selected_token_indices':selected,'selected_token_count':len(selected),
                'hidden_size':self.ops.shape(av)[1], 'requested_k':k,
                'baseline_feature':self.ops.measure(sa), 'donor_feature':self.ops.measure(sb),
                'source_delta':self.ops.measure(source_delta),
                'applied_feature':self.ops.measure(so), 'realized_delta':self.ops.measure(realized_delta),
                'full_baseline_norm':self.ops.measure(av)['norm'],'full_donor_norm':self.ops.measure(bv)['norm'],
                'full_source_delta_norm':self.ops.measure(self.ops.diff(bv,av))['norm'],
                'full_applied_feature_norm':self.ops.measure(out)['norm'],
                'full_realized_delta_norm':self.ops.measure(self.ops.diff(out,av))['norm'],
                'baseline_state':cals.get(key,{}).get('baseline_state'),
                'pre_residual':cals.get(key,{}).get('pre_residual')})
            if k==0 and not self.ops.equal(out,av): raise ValueError('k=0 must be exact a baseline')
            if k==1 and condition['region']=='all' and not self.ops.equal(out,bv): raise ValueError('k=1 must exactly reproduce v10 replacement')
        selected_keys = {s['entry'] for s in metrics}
        if any(not self.ops.equal(result[e],aa[e]) for e in result if e not in selected_keys): raise ValueError('Unselected entry changed')
        if need_source:
            self.geometry_cache[source_geometry_key] = describe_delta_geometry(self.ops,source_deltas,
                {m['entry']:m['source_delta'] for m in metrics})
        if need_realized:
            self.geometry_cache[realized_geometry_key] = describe_delta_geometry(self.ops,realized_deltas,
                {m['entry']:m['realized_delta'] for m in metrics})
        matched = dict(scale)
        for mode in ('baseline_state','pre_residual'):
            ratios = []
            for m in metrics:
                den = m.get(mode)
                ratio = m['realized_delta']['rms']/den['rms'] if den and den['rms']>0 else None
                m['realized_to_'+mode] = ratio
                ratios.append(ratio)
            matched['realized_'+mode+'_group_ratio'] = math.sqrt(sum(r*r for r in ratios)) if all(r is not None for r in ratios) else None
            matched['realized_'+mode+'_entry_ratio_l1'] = sum(ratios) if all(r is not None for r in ratios) else None
            if any(value is not None and not math.isfinite(value) for value in (
                matched['realized_'+mode+'_group_ratio'],matched['realized_'+mode+'_entry_ratio_l1'])):
                raise ValueError('Nonfinite realized relative dose')
        if scale['status']=='requested_match':
            metric_suffix = '_group_ratio' if scale['dose_metric']=='rss' else '_entry_ratio_l1'
            actual = matched['realized_'+condition['dose_mode']+metric_suffix]
            target = 0. if condition['donor_assignment']=='a' else scale['resolved_target']
            if actual is None or not math.isfinite(actual): raise ValueError('Nonfinite realized relative dose')
            rel_error = abs(actual-target)/target if target else abs(actual)
            matched.update(actual_target=target,relative_error=rel_error,
                realized_precision_verified=rel_error <= condition['match_tolerance'],
                reachability='matched' if rel_error <= condition['match_tolerance'] else 'unmatched_realized_precision',
                status='matched' if rel_error <= condition['match_tolerance'] else 'unmatched_realized_precision')
            if matched['status']!='matched': raise UnmatchedDose({**matched,'reason':'realized_dose_outside_tolerance'})
        metadata = {'schema':design.DESIGN_SCHEMA,'family':self.family,'condition':condition['name'],
            'calibration_id':condition['calibration_id'],'visual_signature':signature,
            'changed_entry':condition['changed_entry'],'donor_assignment':condition['donor_assignment'],
            'dose_mode':condition['dose_mode'],'dose_value':condition['dose_value'],'effective_k':k,
            'target_policy':condition.get('target_policy','fixed'),'dose_metric':condition.get('dose_metric','rss'),
            'context_signature':design.context_signature(row,condition),
            'model_revision':calibration.get('model_revision') if calibration else getattr(self,'model_revision',None),
            'anchor_query_sha256':calibration.get('anchor_query_sha256') if calibration else None,
            **{field:matched.get(field) for field in ('source_calibration_id','target_calibration_id',
                'source_anchor_query_id','target_anchor_query_id','source_anchor_query_sha256','target_anchor_query_sha256',
                'source_anchor_prompt_sha256','target_anchor_prompt_sha256','source_visual_signature','target_visual_signature',
                'source_context_signature','target_context_signature','source_model_revision','target_model_revision',
                'native_self_by_metric','target_other_native_by_metric','target_provenance','resolved_target',
                'requested_k','reachability')},
            'realized_baseline_state_by_metric':{
                'rss':matched['realized_baseline_state_group_ratio'],
                'l1':matched['realized_baseline_state_entry_ratio_l1']},
            'entry_count':len(metrics),'entries':metrics,'matching':matched,
            'delta_geometry':{'source':self.geometry_cache[source_geometry_key],
                              'realized':self.geometry_cache[realized_geometry_key]},
            'anchor_query_id':calibration.get('anchor_query_id') if calibration else None,
            'anchor_prompt_sha256':calibration.get('anchor_prompt_sha256') if calibration else None,
            'source_cache_immutable':True,'k0_exact_a':k==0,'k1_exact_v10_full':k==1 and condition['region']=='all'}
        if self.family=='qwen':
            visual = QwenVisual(result['main'],[result[f'later{i}'] for i in range(3)],a.grids,a.output_type,metadata)
        else:
            visual = IBMVisual(tuple((layer,(result[f'layer{layer}'],)) for layer,_ in a.entries),a.geometries,metadata)
        return DoseVisual(visual,metadata)

    @contextmanager
    def use(self,visual,inputs,*,audit=False):
        if not isinstance(visual,DoseVisual):
            with self.base_adapter.use(visual,inputs,audit=audit) as state: yield state
            return
        probe = SiteProbe(self,visual,inputs)
        with probe, self.base_adapter.use(visual.visual,inputs,audit=audit) as state:
            state['dose_observations'] = probe.rows
            state['dose_metadata'] = visual.metadata
            yield state
        counts = {e['entry']:sum(r['entry']==e['entry'] for r in probe.rows) for e in visual.metadata['entries']}
        if any(n != state['calls'] for n in counts.values()):
            raise ValueError('Each fresh visual delivery must have one scalar observation per selected entry')
        state['dose_decode_hook_steps_skipped'] = probe.decode_steps_skipped

    def calibrate(self,sources,condition,anchor_row,anchor_inputs,forward_once,*,anchor_prompt_sha256):
        if anchor_row['query_type']!='identity' or anchor_row['query_target']!='changed': raise ValueError('Calibration requires canonical changed identity anchor')
        if not isinstance(anchor_prompt_sha256,str) or len(anchor_prompt_sha256)!=64: raise ValueError('Actual canonical direct prompt SHA256 required')
        key = design.calibration_key(anchor_row,condition)
        signature = design.visual_signature(anchor_row,condition)
        if key in self.cache:
            cached = self.cache[key]
            if (cached['visual_signature']!=signature or cached['anchor_prompt_sha256']!=anchor_prompt_sha256
                or cached['anchor_query_id']!=anchor_row['id']
                or cached.get('anchor_query_sha256')!=design.anchor_query_sha256(anchor_row)):
                raise ValueError('Calibration cache key collision / changed anchor prompt')
            return cached
        base_c = {**condition,'dose_mode':'raw','dose_value':0.,'donor_assignment':'a'}
        base = self.compose(sources,base_c,anchor_row,inputs=anchor_inputs)
        with self.use(base,anchor_inputs,audit=False) as delivery:
            forward_once()
        if delivery['calls']!=1: raise ValueError('Calibration must be exactly one fresh anchor forward')
        observed = {r['entry']:r for r in delivery['dose_observations']}
        if len(observed)!=len(base.metadata['entries']): raise ValueError('Incomplete calibration entry observations')
        entries = []
        for e in base.metadata['entries']:
            o = observed[e['entry']]
            if not math.isfinite(o['post_entry_state']['rms']):
                raise ValueError('Invalid baseline post-entry state denominator')
            entries.append({**e,'baseline_state':o['post_entry_state'],'pre_residual':o['pre_residual'],
                'pre_residual_missing_reason':o['pre_residual_missing_reason']})
        result = {'schema':'dose-calibration-v11','intervention_id':anchor_row['intervention_id'],
            'calibration_id':condition['calibration_id'],'visual_signature':signature,'family':self.family,
            'changed_entry':condition['changed_entry'],'context_signature':design.context_signature(anchor_row,condition),
            'anchor_query_sha256':design.anchor_query_sha256(anchor_row),
            'model_revision':getattr(self,'model_revision',None),
            'anchor_query_id':anchor_row['id'],'anchor_response_format':'direct','anchor_mapping':'none',
            'anchor_prompt_sha256':anchor_prompt_sha256,'entries':entries,'fresh_forwards':1,
            'source_delta_geometry':base.metadata['delta_geometry']['source'],
            'baseline_state_definition':'a-side post-entry residual state on exactly selected visual slots',
            'pre_residual_definition':'before native additive injection; Qwen main absent, Granite layer0 structural zero',
            'status':'measured','shared_across_queries_and_readouts':True}
        result['native_baseline_state_by_metric'] = {
            m:v for m,v in native_dose(result).items() if m in ('rss','l1')}
        self.cache[key] = result
        return result
