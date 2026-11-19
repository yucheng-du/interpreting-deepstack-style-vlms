"""Optional runtime bridge audit for reciprocal natural later-role conditions.

Model loading and scientific composition stay in the validated v10 adapter.
This helper is called by the v11 runner only after a real adapter has loaded.
"""
from cloud_prep.measurement_v10 import natural_design as old_design


def audit_later_bridge(adapter, sources, row, condition, *, inputs=None,
                       visual=None, design=None, equal=None):
    """Require bit-exact same-role tensors; do not compare old/new model scores.

    `equal` is injectable for pure-Python CPU sentinels. Runtime defaults to
    torch.equal, which is imported only when the audit is actually executed.
    """
    if condition.get('changed_entry') != 'later' or condition.get('role') not in ('natural_roi_pair','self_patch'):
        raise ValueError('Bridge audit requires a natural later-role condition')
    if equal is None:
        import torch
        equal = torch.equal
    name=condition.get('v10_bridge_condition')
    candidates=[c for c in old_design.conditions(design) if c['name']==name]
    if len(candidates)!=1:
        raise ValueError('Missing frozen v10 bridge condition')
    prior=candidates[0]
    old=adapter.compose(sources,prior,row,inputs=inputs)
    current=visual if visual is not None else adapter.compose(sources,condition,row,inputs=inputs)
    if hasattr(current,'main'):
        matched=[equal(current.main,old.main)] + [equal(a,b) for a,b in zip(current.deepstack,old.deepstack)]
        if len(current.deepstack)!=len(old.deepstack):
            raise ValueError('Natural later bridge entry count mismatch')
    else:
        if [layer for layer,_ in current.entries] != [layer for layer,_ in old.entries]:
            raise ValueError('Natural later bridge layer set mismatch')
        matched=[]
        for (_,new_parts),(_,old_parts) in zip(current.entries,old.entries):
            if len(new_parts)!=len(old_parts):
                raise ValueError('Natural later bridge packed part count mismatch')
            matched.extend(equal(a,b) for a,b in zip(new_parts,old_parts))
    if not matched or not all(matched):
        raise ValueError('Natural reciprocal later tensors differ from frozen v10 composition')
    return {'kind':'natural_later_v10_bridge','v10_condition':name,
        'v11_condition':condition['name'],'compared_tensors':len(matched),
        'exact_tensor_equality':True,
        'scope':'Same current backend/source tensors; archived model outputs are not pooled'}
