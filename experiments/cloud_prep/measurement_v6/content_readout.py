"""Label mappings and strict direct-content readout, with no torch on import."""
import re
from deepstack.readout_v3 import format_variant,parse_answer


def make_variant(row,response_format,mapping):
    if response_format=='label':
        return {**format_variant(row,'strict_digits',mapping),'response_format':'label'}
    if response_format!='direct' or mapping!='none':raise ValueError('Unknown response format/mapping')
    labels=list(row['choices']);original=row['option_labels']
    return {'format':'direct_content','response_format':'direct','mapping':'none',
        'prompt':row['query']+'\nAnswer with exactly one of these phrases: '+ '; '.join(labels)+'. Write only the phrase.',
        'assistant_prefix':'','labels':labels,'continuations':labels,
        'label_to_original':dict(zip(labels,original)),'original_to_label':dict(zip(original,labels)),
        'answer_label':labels[row['answer_index']],'original_answer_label':row['answer_label'],
        'display_choices':labels}


def parse_response(text,variant):
    if variant['response_format']=='label':return parse_answer(text,variant['labels'],variant['assistant_prefix'])
    # Strict direct-content parsing: no inference from explanatory prose.
    clean=re.sub(r'\s+',' ',text.strip()).casefold()
    if clean.endswith('.'):clean=clean[:-1]
    matches=[x for x in variant['labels'] if clean==x.casefold()]
    if len(matches)==1:return {'status':'parsed','label':matches[0],'exact_label_only':text.strip()==matches[0]}
    mentioned=[x for x in variant['labels'] if re.search(r'(?<!\w)'+re.escape(x.casefold())+r'(?!\w)',clean)]
    return {'status':'ambiguous' if len(mentioned)>1 else 'missing','label':None,'exact_label_only':False}
