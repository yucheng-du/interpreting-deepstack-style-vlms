"""Apply the accepted v11-v2 dose provenance checks to P1's ROI schema.

Only dose_pair records are inspected here. The P1 analyzer remains responsible
for all planned native/self and general record identity checks. Missing records
remain missing; this adds no scientific outcome or significance gate.
"""
from contextlib import contextmanager

from cloud_prep.measurement_v12 import p1_design
from cloud_prep.measurement_v12 import v11_v2_dose_analysis as accepted


@contextmanager
def _schema_bridge():
    replacements={
        'DESIGN_SCHEMA':p1_design.DESIGN_SCHEMA,
        'QUERY_FIELDS':p1_design.QUERY_FIELDS,
        'opposite_condition':p1_design.opposite_condition,
        'anchor_query_sha256':lambda query:query['query_sha256'],
    }
    previous={name:getattr(accepted,name) for name in replacements}
    try:
        for name,value in replacements.items(): setattr(accepted,name,value)
        yield
    finally:
        for name,value in previous.items(): setattr(accepted,name,value)


def validate_dose_records(records,expected,kind,plan):
    """Return the accepted-v2 index for the planned P1 dose cells only.

``expected`` maps (query id, response format, mapping, condition name) to
(query, readout, condition), exactly as constructed by p1_analysis. The complete
parent plan is retained so canonical anchor IDs/hashes bind to the same query.
This scoped bridge is for a single-threaded analysis call, like the CLI.
"""
    if kind not in ('score','generation'): raise ValueError('Unknown P1 readout kind')
    if plan.get('schema')!=p1_design.DESIGN_SCHEMA: raise ValueError('P1 execution plan required')
    dose_expected={key:spec for key,spec in expected.items() if spec[2].get('role')=='dose_pair'}
    dose_rows=[]
    for row in records:
        key=accepted.key(row)
        if key in dose_expected:
            dose_rows.append(row)
        elif key not in expected and row.get('condition_spec',{}).get('role')=='dose_pair':
            # Keep an unplanned dose row so the accepted validator rejects it.
            dose_rows.append(row)
    with _schema_bridge():
        return accepted._index(dose_rows,dose_expected,kind,plan)

