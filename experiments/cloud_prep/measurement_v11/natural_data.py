"""Reuse the frozen v10 whole-photo panel without outcome-dependent selection.

The row schema and image provenance remain v10. The v11 scientific execution
schema belongs to natural_design, and old model outputs are never input data.
"""
from cloud_prep.measurement_v10.natural_data import (
    DEFAULT_ROOT, ROW_SCHEMA, load_manifest, select_panel, validate,
    validate_query_groups,
)

__all__ = ['DEFAULT_ROOT', 'ROW_SCHEMA', 'load_manifest', 'select_panel',
           'validate', 'validate_query_groups']
