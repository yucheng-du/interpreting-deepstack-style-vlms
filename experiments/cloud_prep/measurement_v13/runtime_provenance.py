"""Record actual implementation identity without freezing cloud engineering.

The accepted v12 entry points retain their historical strict vendor check.
Only the v13 launch context replaces that byte gate with explicit provenance.
Scientific input, composition, pairing and actual CUDA audits remain in force.
"""
from contextlib import contextmanager
import hashlib
import json
from pathlib import Path

from cloud_prep.measurement_v12 import v11_v2_vendor as vendor


def vendor_provenance(root=None):
    root = Path(root) if root is not None else Path(vendor.__file__).parent
    files = {}
    for name, (upstream, expected) in vendor.FILES.items():
        path = root / ('v11_v2_' + name + '.py')
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
        files[name] = {
            'upstream_member': vendor.UPSTREAM_PREFIX + name + '.py',
            'upstream_sha256': upstream, 'accepted_work_copy_sha256': expected,
            'work_copy_sha256': actual, 'changed_since_accepted_copy': actual != expected,
        }
    changed = [name for name, item in files.items() if item['changed_since_accepted_copy']]
    return {
        'archive_sha256': vendor.UPSTREAM_ARCHIVE_SHA256, 'files': files,
        'changed_files': changed,
        'algorithm': 'accepted opposite-native-v2 baseline; actual implementation recorded',
        'policy': 'Implementation edits are permitted. Validate affected behavior; record a new execution version when observations can change. Never silently mix incompatible outputs.',
        'byte_identity_is_launch_gate': False,
    }


@contextmanager
def repairable_runtime():
    """Scope provenance-only handling to one non-concurrent runner invocation."""
    from cloud_prep.measurement_v12 import p1_runner, p2_runner
    modules = (p1_runner, p2_runner)
    saved = [module.verify_vendor for module in modules]
    for module in modules:
        module.verify_vendor = vendor_provenance
    try:
        yield
    finally:
        for module, original in zip(modules, saved):
            module.verify_vendor = original


def scientific_identity(plan):
    """A small semantic comparison aid, not a launch lock or preregistration."""
    fields = ('schema', 'design_schema', 'arm', 'panel', 'manifest_sha256',
              'conditions', 'queries', 'readouts', 'ids', 'shard_units')
    identity = {key: plan[key] for key in fields if key in plan}
    model = plan.get('model', {})
    identity['model'] = {key: model[key] for key in ('key', 'model', 'revision') if key in model}
    encoded = json.dumps(identity, ensure_ascii=False, sort_keys=True, separators=(',', ':')).encode()
    return {'sha256': hashlib.sha256(encoded).hexdigest(), 'fields': list(identity),
            'purpose': 'Compare input/model/intervention/readout identity independently of deployment paths and source provenance; not a hard gate.'}
