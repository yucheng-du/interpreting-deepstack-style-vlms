"""Natural ROI composition on the accepted v11 scalar-dose/probe implementation."""
from cloud_prep.measurement_v12.v11_v2_dose_adapter import DoseAdapter, entry_tensors
from cloud_prep.measurement_v9.geometry import qwen_region_indices, ibm_region_indices


class NaturalDoseAdapter(DoseAdapter):
    def _reference(self, sources, reference, row, inputs):
        return self.base_adapter.compose(sources, {
            'main': reference['main'], 'context_source': reference['later'],
            'donor_source': reference['donor_source'], 'changed_entry': reference['changed_entry'],
            'intervention_type': 'local_role', 'region': 'roi', 'entry_group': 'all',
            'token_permutation': 'identity', 'source_destination': 'original'}, row, inputs=inputs)

    def _selected(self, visual, condition, row, n):
        if condition['region'] != 'roi': raise ValueError('Natural dose requires fixed donor ROI')
        box = row['donor_roi_bbox_xyxy']
        if self.family == 'qwen':
            meta = row['source_image_metadata'][condition['reference_a']['donor_source']]
            return qwen_region_indices(visual.grids, row['tile_grid'], 'roi', box,
                                       (meta['width'], meta['height']), 'identity')
        return ibm_region_indices(visual.geometries[0], row['tile_grid'], 'roi', box, 'identity')

    def compose(self, sources, condition, row, inputs=None, calibration=None, opposite_calibration=None):
        result = super().compose(sources, condition, row, inputs=inputs,
                                  calibration=calibration, opposite_calibration=opposite_calibration)
        if condition.get('role') == 'dose_pair':
            k = result.metadata['effective_k']
            if k in (0., 1.):
                reference = self._reference(sources, condition['reference_a' if k == 0 else 'reference_b'], row, inputs)
                current = entry_tensors(result.visual, self.family); expected = entry_tensors(reference, self.family)
                if set(current) != set(expected) or any(not self.ops.equal(current[e], expected[e]) for e in current):
                    raise ValueError('Natural k0/k1 does not reproduce exact local a/b composition')
            result.metadata.update(natural_fixed_roi=list(row['donor_roi_bbox_xyxy']),
                k0_exact_natural_a=k==0., k1_exact_natural_b=k==1.,
                roi_semantics='fixed donor coordinates across both queries; no natural spatial-rebinding claim')
        return result
