"""Select the paired estimates used in Figures 4 through 7 and Figure 9."""
from collections import defaultdict
import json
from analysis.paths import INPUT

MODELS = ('ibm4b', 'qwen8b', 'qwen32b')
SETTINGS = [('raw', 'raw'), ('main_native', 'rss'), ('main_native', 'l1'),
            ('later_native', 'rss'), ('later_native', 'l1')]


def one(rows, **fields):
    hits = [row for row in rows if all(row.get(k) == v for k, v in fields.items())]
    if len(hits) != 1:
        raise ValueError((fields, len(hits)))
    return hits[0]


def photo(rows, setting, entry, measure):
    target, metric = SETTINGS[setting]
    endpoint = {'E_target': 'E', 'E_control': 'E', 'R': 'R'}[measure]
    if entry == 'main_minus_later':
        endpoint = 'R_main_minus_later'
    query = {'E_target': 'changed', 'E_control': 'unchanged', 'R': 'changed_minus_unchanged'}[measure]
    return one(rows, main_layout='M0', main_strength=1, dose_value=1, target=target,
               dose_metric=metric, response_format='direct', mapping='none', endpoint=endpoint,
               query_role=query, changed_entry=entry)


def paired_dose(attribute, setting, entry):
    target, metric = SETTINGS[setting]
    params = dict(response_format='direct', mapping='none', endpoint='R' if entry != 'main_minus_later' else 'R_main_minus_R_later',
                  changed_entry=entry, changing_layout='M0', fixed_layout='M0', changing_strength=1,
                  fixed_strength=1, fixed_other_assignment='a', region='all', dose_metric=metric)
    if setting == 0:
        return one(attribute['generation_selectivity']['summaries'], **params, dose_value=1, dose_mode='raw')
    return one(attribute['generation_matched_target_selectivity']['summaries'], **params, reference_target=target)


def point(row, **labels):
    stats = row.get('statistics', row)
    interval = stats.get('normal_95_interval')
    if interval is None:
        interval = [stats['ci_low'], stats['ci_high']]
    return dict(labels, original_row=row, mean=stats['mean'], ci95=interval,
                n=stats.get('n', stats.get('complete_photos')))


def main():
    result = defaultdict(list)
    for model in MODELS:
        for kind in ('generation','score'):
            path = INPUT / 'photographs' / model / (kind+'.json')
            rows = json.loads(path.read_text())['summaries']
            estimates = result['natural_'+kind]
            for entry in ('main','later'):
                for metric in ('E_target','E_control','R'):
                    estimates.append(point(photo(rows,0,entry,metric),model=model,setting=0,endpoint=entry+'.'+metric))
            for setting in range(5):
                row = photo(rows,setting,'main_minus_later','R')
                estimates.append(point(row,model=model,setting=setting,endpoint='main_minus_later.R'))
                if kind == 'generation':
                    result['dose_reference_45'].append(point(row,model=model,setting=setting,panel='natural'))
        path = INPUT / 'paired' / model
        dose = json.loads((path/'dose.json').read_text())['by_attribute']
        for panel in ('color','stripe_orientation'):
            for setting in range(5):
                row = paired_dose(dose[panel],setting,'main_minus_later')
                result['dose_reference_45'].append(point(row,model=model,setting=setting,panel=panel))
        if model=='qwen32b':
            for setting in (1,2):
                for entry in ('main','later','main_minus_later'):
                    row=paired_dose(dose['stripe_orientation'],setting,entry)
                    result['dose_condition_boundaries'].append(point(row,model=model,setting=setting,entry=entry,fixed='a'))
        del dose
        local = json.loads((path/'local.json').read_text())['by_attribute']['stripe_orientation']['legacy_estimator_output']
        for entry in ('main','later','main_minus_later'):
            prefix = 'role_by_' if entry=='main_minus_later' else ''
            for mapping,readout in [('none','Direct'),('identity','Identity'),('cyclic','Cyclic')]:
                row=one(local['per_mapping_behavior_contrast_summaries'], contrast=prefix+'layout_by_destination_by_query',
                        changed_entry=entry, main_strength=1, entry_group='all', mapping=mapping,
                        response_format='direct' if mapping=='none' else 'label')
                result['stripe_spatial_interaction'].append(point(row,model=model,entry=entry,readout=readout))
        del local
    out=INPUT/'matched'
    out.mkdir(parents=True,exist_ok=True)
    for name,rows in result.items():
        (out/(name+'.json')).write_text(json.dumps(rows)+'\n')


if __name__ == '__main__':
    main()
