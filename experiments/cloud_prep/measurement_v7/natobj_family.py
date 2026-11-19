"""Natural-object rows (GQA/VG controlled recomposition) as a family of the shared v7 Qwen spatial runner.

  qwen_spatial --family-module cloud_prep.measurement_v7.natobj_family --manifest <natobj manifest.jsonl> ...

natobj_data.load_manifest validates the line-B design, answers and provenance; spatial_data.normalize_rows
then renames m_sNNN_M1 -> m1_sNNN, recomputes the object / original-slot / current-slot answer maps and
cross-checks them against line B's maps. The independent unit is the source cluster: union-find over the
quads' source photos and crops, recomputed here and required to equal the manifest unit_id partition (the two
assignment scenes of a quad always share their four crops). record_fields carry the unit, quad, stratum and
source ids into every runner record for natobj_analysis. Torch-free.
"""
from __future__ import annotations

from cloud_prep.measurement_v7 import natobj_data as nd
from cloud_prep.measurement_v7.spatial_data import register_family, normalize_rows, default_smoke

FAMILY = 'natural_gqa'
MODULE = 'cloud_prep.measurement_v7.natobj_family'  # stable even when imported via --family-module
UNIT_KEY = 'unit_id'
SEP = '|'
RECORD_FIELDS = ('unit_id', 'quad_id', 'stratum', 'category_pair', 'color_pair', 'scene_layout', 'assignment',
                 'replicate', 'construction', 'source_photos', 'source_crops')


def source_clusters(groups):
    """Union-find over {key: source ids}: keys sharing any id share a cluster labelled by its smallest key."""
    parent, owner = {k: k for k in groups}, {}

    def find(k):
        while parent[k] != k:
            parent[k] = parent[parent[k]]
            k = parent[k]
        return k
    for k in sorted(groups):
        for i in sorted(groups[k]):
            if i in owner:
                a, b = find(k), find(owner[i])
                parent[max(a, b)] = min(a, b)
            else:
                owner[i] = k
    return {k: find(k) for k in groups}


def quad_sources(rows):
    """{quad_id: {'photo:<id>', 'crop:<id>'}} from crop provenance; both scenes of a quad must agree."""
    groups = {}
    for r in rows:
        crops = r['natural']['crops'].values()
        ids = {'photo:' + c['image_id'] for c in crops} | {'crop:' + c['candidate_id'] for c in crops}
        if groups.setdefault(r['quad_id'], ids) != ids:
            raise ValueError(f'{r["quad_id"]}: scenes of one quad must share their four crops')
    return groups


def check_units(rows, unit_key=UNIT_KEY):
    """The unit field must be exactly the source-cluster partition: no cluster split, no extra merge."""
    clusters, unit = source_clusters(quad_sources(rows)), {}
    for r in rows:
        if unit.setdefault(r['quad_id'], r[unit_key]) != r[unit_key]:
            raise ValueError(f'{r["quad_id"]}: quad split across units')
    pairs = {(clusters[q], unit[q]) for q in clusters}
    if len(pairs) != len(set(clusters.values())) or len(pairs) != len(set(unit.values())):
        raise ValueError(f'{unit_key} differs from the source-cluster partition (union-find over source photos/crops)')
    return clusters


def moved(spec):
    return spec['pose'] == 'cross' if spec['type'] == 'donor' else spec['layout'] == 'M1'


def load(manifest, image_root=None, check_hashes=True):
    raw = nd.load_manifest(manifest, image_root, check_hashes=check_hashes)
    check_units(raw)
    rows = normalize_rows(raw, FAMILY, generation_group='replicate')
    for row in rows:
        rename, t = row['source_name_map'], row['target_main_slot']
        slots = {rename[n]: v for n, v in row['source_object_slots'].items()}
        if slots != {n: None if s['type'] == 'blank' else t ^ moved(s) for n, s in row['source_specs'].items()}:
            raise ValueError(f'{row["id"]}: line-B target slots disagree with the normalized layouts')
        crops = row['natural']['crops'].values()
        row['source_object_slots'] = slots
        row['record_fields'] = {'unit_id': row['unit_id'], 'quad_id': row['quad_id'], 'stratum': row['stratum'],
            'category_pair': SEP.join(row['category_pair']), 'color_pair': SEP.join(row['color_pair']),
            'scene_layout': row['layout'], 'assignment': row['assignment'], 'replicate': row['replicate'],
            'construction': row['natural']['construction'],
            'source_photos': SEP.join(sorted({c['image_id'] for c in crops}, key=lambda k: (len(k), k))),
            'source_crops': SEP.join(sorted(c['candidate_id'] for c in crops))}
    return rows


def smoke(rows):
    """Fixed before model output: first scene-layout-0 / assignment-0 query on the first object (vehicle) and the
    first scene-layout-1 / assignment-1 query on the second object from another stratum (both layouts, both
    assignments, both target slots and two strata)."""
    first = next((r for r in rows if (r['layout'], r['assignment'], r['target_index']) == (0, 0, 0)), None)
    second = next((r for r in rows if first and (r['layout'], r['assignment'], r['target_index']) == (1, 1, 1)
                   and r['stratum'] != first['stratum']), None)
    return [first, second] if first and second else default_smoke(rows)


register_family(FAMILY, (nd.SCHEMA,), load, smoke=smoke, module=MODULE)
