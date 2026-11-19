"""Identity and completion checks for the shared accepted-v11-v2 work copy.

No model imports. P1/P2 wrappers supply their own data, design and record callbacks
to ``v11_v2_runner.execute(args, rows, plan, design_facade)``. The legacy runner's
standalone preparation is not the v12 manifest validator.
"""
import hashlib
from pathlib import Path

UPSTREAM_ARCHIVE_SHA256 = '13f17b48f131ab8df70a433aa0d5934acc7d50ab48d347c9d53653eea31a7d6b'
UPSTREAM_PREFIX = 'remote/DeepStack/v11-work/candidates/dose-opposite-native-v2/code/cloud_prep/measurement_v11/'
FILES = {
    'dose_design': ('9a3a4e8c353db7358ef36263aa1a3d2423f1e498c15a5494e021cfb71053f837', '6c4f8aa4244a14faa0642d056213a83a0008222e84419723d29c18b50913fda3'),
    'dose_adapter': ('78231f2c9ff6a8ec973702196074d02bdddc116d6eb5fcc66b19b87869fe1895', '1519e5ab7244f285646c587a5f5355aae2c6f87f01d5ff917ec5b93be84b00ab'),
    'runtime': ('51a345ff2fc68588ea4c06d245515b1383b4cb26fee1812c3083062b40a543c8', '428123964224dc1cfd91477e94ac476c3c526cdddd4e70659794d4fd906b05bb'),
    'runner': ('44875fcf95d9614b80464ec6310d712d58efc59530bbe0f2fad918051be1222c', '23972ab37e9fbd487ecbb73e2276552bd2b8769ed1fb80da81c2d13c77034ad8'),
    'dose_analysis': ('d391de5265bcde44aa0209c2620afe5408324ad52fe9531e4676a3128494170b', 'a6297a05e5cf551017cd9880f79d2f11bb57dd73dec0ebad3c40f14a09431088'),
    'reference_contrasts': ('cab190d03d2df467e5159b3f05a97fc3a0e4dfb1d90129e149885c58b83feac0', 'ccb7603bd23ab6c5963de4c89fa3fda8548d320da2e7f992aeb0e7ae1c260df5'),
}


def verify_vendor():
    paths = {}
    for name,(upstream,expected) in FILES.items():
        path=Path(__file__).with_name('v11_v2_'+name+'.py')
        actual=hashlib.sha256(path.read_bytes()).hexdigest()
        if actual!=expected:
            raise ValueError('Vendored file changed without versioned provenance: '+str(path))
        paths[name]=dict(upstream_member=UPSTREAM_PREFIX+name+'.py',upstream_sha256=upstream,
                         work_copy_sha256=actual)
    return dict(archive_sha256=UPSTREAM_ARCHIVE_SHA256, files=paths,
        modifications='provenance header, import redirection and self-provenance metadata only',
        algorithm='accepted opposite-native-v2 unchanged')


def validate_completed_metadata(meta):
    """Minimal technical completeness, not a claim/outcome-significance gate."""
    if not meta.get('finished_at') or meta.get('error') or meta.get('failed_at'):
        raise ValueError('Run not successfully finished')
    for kind in ('score','generation'):
        if meta.get('completed_'+kind+'_rows')!=meta['expected_'+kind+'_rows']:
            raise ValueError('Incomplete logical matrix: '+kind)
    if meta.get('completed_anchor_forwards',0)!=meta.get('expected_anchor_forwards',0):
        raise ValueError('Incomplete canonical-anchor matrix')
    if not meta.get('shared_visual_contracts_validated'):
        raise ValueError('Shared visual input contract not recorded')
    return dict(status='complete',unmatched_dose_score_rows=meta.get('unmatched_dose_score_rows',0),
                unmatched_retained=True, model_results=True,
                scope='metadata completeness only; raw parsing, technical audits and scientific estimation remain separate')
