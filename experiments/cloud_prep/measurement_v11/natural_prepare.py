"""Write CPU-only natural reciprocal plans, photo shards and validation."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import os as _path_os
DATA_ROOT = Path(_path_os.environ.get("DS_DATA_ROOT", "work")).expanduser().resolve()
import sys

from cloud_prep.measurement_v10.geometry import preflight_geometry
from cloud_prep.measurement_v11 import natural_data as data, natural_design as design
from cloud_prep.measurement_v11.natural_analysis import analyze
from cloud_prep.portable_data import _guard_path

DEFAULT_OUTPUT = (DATA_ROOT / 'runs/measurement-v11/preparation/natural')


def photo_shards(rows, photos_per_shard=8):
    """Contiguous query offsets, never splitting a declared paired-photo group."""
    if type(photos_per_shard) is not int or photos_per_shard < 1:
        raise ValueError('Positive photos_per_shard required')
    ordered=[]
    for row in rows:
        if not ordered or ordered[-1] != row['scene_id']:
            if row['scene_id'] in ordered:
                raise ValueError('Manifest must group queries from each photo contiguously')
            ordered.append(row['scene_id'])
    shards=[]; offset=0
    for start in range(0,len(ordered),photos_per_shard):
        scenes=ordered[start:start+photos_per_shard]
        selected=[q for q in rows if q['scene_id'] in scenes]
        data.validate_query_groups(selected)
        shards.append({'offset':offset,'limit':len(selected),'source_photos':scenes,
            'query_ids':[q['id'] for q in selected]})
        offset+=len(selected)
    if offset != len(rows):
        raise AssertionError('Shard coverage mismatch')
    return shards


def prepare(manifest, output, photos_per_shard=8):
    manifest=_guard_path(Path(manifest)); output=_guard_path(Path(output),write=True)
    rows=data.load_manifest(manifest)
    pixels=data.validate(manifest)
    plans={panel:design.execution_plan(data.select_panel(rows,panel),panel) for panel in ('main','smoke')}
    geometry=[{'id':q['id'],**preflight_geometry(q,plans['main']['conditions'])} for q in rows]
    shards=photo_shards(rows,photos_per_shard)
    shard_plans=[]
    for spec in shards:
        selected=rows[spec['offset']:spec['offset']+spec['limit']]
        shard_plans.append(design.execution_plan(selected))
    coverage={k:sum(p[k] for p in shard_plans) for k in ('expected_score_rows','expected_generation_rows')}
    if any(coverage[k]!=plans['main'][k] for k in coverage):
        raise AssertionError('Shard row-count mismatch')
    # Empty observations verify planned denominators without fabricating results.
    empty=analyze([],[],plans['main'])
    report={'schema':'natural-reciprocal-preparation-v11','status':'CPU preparation only; no model execution',
        'manifest':str(manifest),'manifest_sha256':hashlib.sha256(manifest.read_bytes()).hexdigest(),
        'data_reused_without_selection':True,'data_root':str(manifest.parent),
        'pixel_validation':pixels,'geometry_rows':len(geometry),
        'planned_source_photos':32,'planned_queries':len(rows),
        'conditions':len(plans['main']['conditions']),
        'per_model':{panel:{k:p[k] for k in ('expected_score_rows','expected_generation_rows','items','independent_source_photos')} for panel,p in plans.items()},
        'shards':shards,'shard_coverage':coverage,
        'empty_observation_coverage':empty['coverage'],
        'bridge_scope':'v10 later-only dispatch matched at CPU composition; actual BF16 equivalence requires testing on the selected GPU',
        'torch_imported':'torch' in sys.modules,'transformers_imported':'transformers' in sys.modules}
    if report['torch_imported'] or report['transformers_imported']:
        raise RuntimeError('CPU preparation imported a model/tensor library')
    output.mkdir(parents=True,exist_ok=True)
    for name,value in {**{f'{k}-plan':v for k,v in plans.items()},'geometry':geometry,
                       'preparation':report,**{f'main-shard-{i:02d}-plan':v for i,v in enumerate(shard_plans)}}.items():
        (output/f'{name}.json').write_text(json.dumps(value,indent=2)+'\n')
    return report


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest',type=Path,default=data.DEFAULT_ROOT/'manifest.jsonl')
    parser.add_argument('--output',type=Path,default=DEFAULT_OUTPUT)
    parser.add_argument('--photos-per-shard',type=int,default=8)
    args=parser.parse_args(argv)
    print(json.dumps(prepare(args.manifest,args.output,args.photos_per_shard),indent=2))


if __name__=='__main__':
    main()
