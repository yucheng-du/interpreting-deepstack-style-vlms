"""Executable Granite Vision 4.1 adapter; module import/CPU tests need no torch.

The official get_image_features returns [(decoder_layer, [packed_image_tensor])].
Replacing that result preserves the native forward, including zeroed image
embeddings, embedding/logit scaling, cache handling and pre-layer injection.
No decoder hooks or reinterpretation of the eight entries as Qwen paths occur.
"""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, field
import hashlib
import inspect
from pathlib import Path
import sys
from typing import Any, Mapping

from cloud_prep.measurement_v8.ibm_geometry import (
    ENTRY_ORDER, REVISION, ImageGeometry,
    entry_layers, image_geometry, region_token_indices, roi_token_indices,
    spatial_token_indices, validate_config,
)

MODEL_ID = "ibm-granite/granite-vision-4.1-4b"
SOURCE_SHA256 = {
    "modeling.py": "f13023583433adaf771e5a10675a1dc4244b46b5bce5d9165b91669d23c3b144",
    "downsampling.py": "556c955476d5b84d557c1cf6ede7bed322e0420f3df86c8baf8e0083a9d8d058",
    "processing.py": "4512b42a868c469aea9869457b55432e13223f5eb22f293a0ca61f598356aa9d",
    "configuration.py": "ef3e97557d38b92c47f976588a84af854bd4b8318676a0983802398ac72f8bfc",
}
_OWNER = "_deepstack_v8_granite_owner"
_MISSING = object()


def _plain(value):
    return value.detach().cpu().tolist() if hasattr(value, "detach") else value


def _shape(value):
    return tuple(value.shape)


def _hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


@dataclass(frozen=True)
class IBMVisual:
    entries: tuple[tuple[int, tuple[Any, ...]], ...]
    geometries: tuple[ImageGeometry, ...]
    intervention: Mapping[str, Any] = field(default_factory=dict)

    def as_native(self):
        """Fresh Python containers; tensor values are never mutated by this adapter."""
        return [(layer, list(features)) for layer, features in self.entries]


def validate_visual(visual: IBMVisual, *, hidden_size: int = 2560) -> None:
    if tuple(layer for layer, _ in visual.entries) != ENTRY_ORDER:
        raise ValueError("Granite feature entry order/identity changed")
    if len(visual.geometries) != 1:
        raise ValueError("v8 Granite experiment uses exactly one image per batch")
    for layer, features in visual.entries:
        if len(features) != len(visual.geometries):
            raise ValueError(f"Wrong image count at Granite layer {layer}")
        for feature, geometry in zip(features, visual.geometries):
            if _shape(feature) != (len(geometry.tokens), hidden_size):
                raise ValueError(f"Post-pack feature shape changed at Granite layer {layer}: {_shape(feature)}")


def _roi_bbox(row: Mapping[str, Any], condition: Mapping[str, Any] | None = None):
    condition = condition or {}
    box = condition.get("donor_roi_bbox_xyxy", row.get("donor_roi_bbox_xyxy"))
    if isinstance(box, Mapping):
        source = condition.get("donor_source", condition.get("donor"))
        if source is None:
            raise ValueError("Source-specific ROI boxes require donor_source")
        box = box[source]
    return box


def compose_visual(sources: Mapping[str, IBMVisual], condition: Mapping[str, Any],
                   row: Mapping[str, Any], *, hidden_size: int = 2560) -> IBMVisual:
    """Replace selected post-pack entries, retaining main's layer0 and other entries.

    all replaces every packed token, including newline/padding; complement is
    exactly all minus ROI. P fixes special *positions* but preserves donor
    source semantics there. ROI is selected in donor coordinates before P.
    """
    main = sources[condition["main"]]
    donor_name = condition.get("donor_source", condition.get("donor"))
    if donor_name is None:
        old = condition.get("ds_sources", ())
        if len(set(old)) != 1:
            raise ValueError("IBM needs one donor_source, not three Qwen stream sources")
        donor_name = old[0]
    donor = sources[donor_name]
    validate_visual(main, hidden_size=hidden_size)
    validate_visual(donor, hidden_size=hidden_size)
    if main.geometries != donor.geometries:
        raise ValueError("Main/donor post-pack coordinates differ")
    group = condition.get("entry_group", "all")
    selected_layers = entry_layers(group)
    region = condition.get("region", "all")
    permutation = condition.get("token_permutation", "identity")
    box = _roi_bbox(row, condition)
    masks, permutations, counts = [], [], []
    for geometry in main.geometries:
        ids = spatial_token_indices(geometry, permutation)
        mask = region_token_indices(geometry, region, box, permutation=permutation)
        masks.append(mask)
        permutations.append(ids)
        roi = roi_token_indices(geometry, box) if box is not None else None
        counts.append({"total": len(geometry.tokens), "selected": len(mask),
                       "roi": len(roi) if roi is not None else None,
                       "complement": len(geometry.tokens) - len(roi) if roi is not None else None,
                       "special": len(geometry.special_indices),
                       "selected_special": len(set(mask) & set(geometry.special_indices)),
                       "selected_base": sum(geometry.tokens[i].view == "base" for i in mask),
                       "selected_tiles": sum(geometry.tokens[i].view == "tiles" for i in mask)})
    donor_by_layer = dict(donor.entries)
    output = []
    for layer, main_features in main.entries:
        if layer not in selected_layers:
            output.append((layer, main_features))
            continue
        replaced = []
        for current, other, mask, ids in zip(main_features, donor_by_layer[layer], masks, permutations):
            if getattr(current, "dtype", None) != getattr(other, "dtype", None) or \
                    getattr(current, "device", None) != getattr(other, "device", None):
                raise ValueError("Main/donor tensor dtype or device differs")
            if region == "all":
                result = other if permutation == "identity" else other[ids]
            else:
                result = current.clone()
                result[mask] = other[[ids[i] for i in mask]]
            replaced.append(result)
        output.append((layer, tuple(replaced)))
    result = IBMVisual(tuple(output), main.geometries, {
        "main_source": condition["main"], "donor_source": donor_name,
        "entry_group": group, "replaced_layers": list(selected_layers),
        "preserved_layers": [layer for layer in ENTRY_ORDER if layer not in selected_layers],
        "layer0_source": condition["main"], "token_permutation": permutation,
        "region": region, "donor_roi_bbox_xyxy": list(box) if box is not None else None,
        "region_counts": counts,
        "roi_semantics": "nominal post-projection slot support, not encoder/QFormer receptive field",
    })
    if dict(result.entries)[0] is not dict(main.entries)[0]:
        raise AssertionError("The main layer0 entry was replaced")
    return result


class GraniteAdapter:
    def __init__(self, model, processor, *, device="cuda", metadata=None):
        self.model, self.processor, self.device = model, processor, device
        self.base = model.model
        self.metadata = {**validate_config(model.config), **(metadata or {})}
        self._check_runtime_structure()
        self._check_processor()

    @classmethod
    def load(cls, snapshot_path, *, device="cuda", dtype="bfloat16", local_files_only=True,
             attn_implementation="sdpa"):
        """Cloud-only loading of the fixed, locally downloaded remote-code snapshot.

        Snapshot downloading/cache placement belongs to the common runtime.
        This entry point cannot silently download weights or load a Mac model.
        """
        if sys.platform != "linux" or not str(device).startswith("cuda"):
            raise RuntimeError("v8 Granite loading is restricted to the authorized Linux CUDA runtime")
        if not local_files_only:
            raise ValueError("Download the pinned snapshot through the common cache runtime first")
        snapshot = Path(snapshot_path).resolve()
        if not snapshot.is_dir():
            raise ValueError("A local pinned model snapshot directory is required")
        actual = {name: _hash(snapshot / name) for name in SOURCE_SHA256}
        if actual != SOURCE_SHA256:
            raise ValueError("Granite remote-code files differ from the reviewed fixed revision")
        import torch
        from transformers import AutoModelForImageTextToText, AutoProcessor
        if dtype != "bfloat16":
            raise ValueError("The v8 Granite execution contract is BF16")
        processor = AutoProcessor.from_pretrained(str(snapshot), trust_remote_code=True,
            local_files_only=True, use_fast=False)
        model = AutoModelForImageTextToText.from_pretrained(str(snapshot), trust_remote_code=True,
            local_files_only=True, torch_dtype=torch.bfloat16,
            attn_implementation=attn_implementation).eval().to(device)
        if _hash(inspect.getfile(type(model))) != SOURCE_SHA256["modeling.py"] or \
                _hash(inspect.getfile(type(processor))) != SOURCE_SHA256["processing.py"]:
            raise ValueError("Loader did not select the pinned IBM remote model/processor implementations")
        meta = {"model": MODEL_ID, "revision": REVISION, "snapshot_path": str(snapshot),
                "remote_code_sha256": actual, "attn_implementation": attn_implementation,
                "dtype": dtype, "native_forward_preserved": True,
                "loaded_model_source": inspect.getfile(type(model)),
                "model_smoke_status": "pending_runner_self_patch_and_geometry_audit"}
        return cls(model, processor, device=device, metadata=meta)

    def _check_runtime_structure(self):
        if not callable(getattr(self.base, "get_image_features", None)) or \
                not callable(getattr(self.base, "pack_and_unpad_image_features", None)):
            raise TypeError("Pinned Granite feature/pack methods unavailable")
        if len(self.base.layerwise_projectors) != 4 or len(self.base.spatial_projectors) != 4:
            raise ValueError("Expected four Layer and four Spatial projectors")
        if len(self.base.language_model.layers) != 40 or self.base.image_newline is None:
            raise ValueError("Granite decoder depth/newline differs")
        for index, projector in enumerate((*self.base.layerwise_projectors, *self.base.spatial_projectors)):
            if (projector.image_side, projector.window_side, projector.query_side) != (24, 8, 4):
                raise ValueError("Pinned WindowQFormer geometry differs")
            if index >= 4:
                expected = ((0, 0), (0, 1), (1, 0), (1, 1))[index - 4]
                if (projector.downsampler.offset_h, projector.downsampler.offset_w) != expected:
                    raise ValueError("Spatial sampling offset changed")

    def _check_processor(self):
        p, ip = self.processor, self.processor.image_processor
        if (p.patch_size, p.downsample_rate, p.vision_feature_select_strategy,
                p.num_additional_image_tokens) != (16, "4/8", "full", 0):
            raise ValueError("Processor image-token count contract differs")
        if tuple(map(tuple, ip.image_grid_pinpoints)) != tuple(map(tuple, self.model.config.image_grid_pinpoints)):
            raise ValueError("Processor/model anyres candidates differ")
        for name in ("size", "crop_size"):
            size = getattr(ip, name)
            if isinstance(size, Mapping):
                hw = (size.get("height"), size.get("width"))
            else:
                hw = (getattr(size, "height", None), getattr(size, "width", None))
            if hw != (384, 384):
                raise ValueError(f"Processor {name} must remain 384x384")

    def _geometry(self, inputs):
        sizes = _plain(inputs["image_sizes"])
        if len(sizes) != 1 or _shape(inputs["input_ids"])[0] != 1:
            raise ValueError("v8 Granite adapter requires batch=1 and one image")
        geometry = image_geometry(tuple(map(int, sizes[0])))
        # Planning can tolerate other geometry; this actual use verifies every
        # feature and placeholder count rather than assuming a fixed 248.
        token_id = getattr(self.model.config, "image_token_id", None)
        if token_id is None:
            token_id = self.model.config.image_token_index
        ids = _plain(inputs["input_ids"])
        if sum(token == token_id for row in ids for token in row) != len(geometry.tokens):
            raise ValueError("Actual Granite image-placeholder count differs from post-pack geometry")
        shape = _shape(inputs["pixel_values"])
        if shape != (1, geometry.num_views, 3, 384, 384):
            raise ValueError(f"Actual Granite processor view shape differs: {shape}")
        return (geometry,)

    def prepare(self, row, variant, source):
        from cloud_prep.measurement_v6.content_data import model_row
        from deepstack.readout_v3 import make_inputs
        inputs, prompt = make_inputs(self.processor, model_row(row, source), variant, device=self.device)
        self._geometry(inputs)
        return inputs, prompt

    def encode(self, inputs) -> IBMVisual:
        if getattr(self.base, _OWNER, None) is not None or "get_image_features" in self.base.__dict__:
            raise RuntimeError("Real donor encoding requires an unpatched Granite feature method")
        geometries = self._geometry(inputs)
        values = self.base.get_image_features(inputs["pixel_values"], inputs["image_sizes"],
            vision_feature_select_strategy="full")
        visual = IBMVisual(tuple((int(layer), tuple(parts)) for layer, parts in values), geometries)
        validate_visual(visual)
        return visual

    def compose(self, sources, condition, row, *, inputs=None):
        result = compose_visual(sources, condition, row)
        if inputs is not None and self._geometry(inputs) != result.geometries:
            raise ValueError("Cached feature geometry differs from the actual main input")
        return result

    @contextmanager
    def use(self, visual, inputs, *, audit=False):
        """Yield delivery counts; restore the exact method state on every exit."""
        validate_visual(visual)
        if self._geometry(inputs) != visual.geometries:
            raise ValueError("Cached output and actual main input geometry differ")
        if getattr(self.base, _OWNER, None) is not None or "get_image_features" in self.base.__dict__:
            raise RuntimeError("Nested/foreign Granite feature replacement is not supported")
        original = self.base.__dict__.get("get_image_features", _MISSING)
        expected_sizes, expected_pixels = _plain(inputs["image_sizes"]), _shape(inputs["pixel_values"])
        state = {"calls": 0, "entry_layers": list(ENTRY_ORDER),
                 "audit_requested": bool(audit),
                 "geometries": [g.summary() for g in visual.geometries], **visual.intervention}

        def deliver(pixel_values, image_sizes, vision_feature_layer=None,
                    vision_feature_select_strategy=None, **kwargs):
            if _plain(image_sizes) != expected_sizes or _shape(pixel_values) != expected_pixels:
                raise ValueError("Fresh Granite visual call changed image sizes/view shape")
            if vision_feature_select_strategy not in (None, "full"):
                raise ValueError("Fresh Granite visual call changed feature selection")
            state["calls"] += 1
            return visual.as_native()

        setattr(self.base, _OWNER, state)
        self.base.get_image_features = deliver
        try:
            yield state
        finally:
            if original is _MISSING:
                delattr(self.base, "get_image_features")
            else:
                self.base.get_image_features = original
            delattr(self.base, _OWNER)

    def audit_sources(self, source_inputs, sources, row):
        """Small cloud tensor checks against actual pack/processor implementations.

        These run only on a real model host. They are not claimed by CPU tests.
        Native/self-patch scores, generation delivery and cache hashes are
        checked by the common runner separately.
        """
        import torch
        first = next(iter(sources.values()))
        geometry = first.geometries[0]
        for name, source in sources.items():
            validate_visual(source)
            if source.geometries != first.geometries or self._geometry(source_inputs[name]) != first.geometries:
                raise ValueError("Source geometry mismatch during Granite audit")
        side = geometry.projected_side
        device = source_inputs[next(iter(sources))]["pixel_values"].device
        sentinel = torch.arange(geometry.num_views * side * side, device=device, dtype=torch.float32)
        sentinel = sentinel.reshape(geometry.num_views, side * side, 1)
        packed, lens = self.base.pack_and_unpad_image_features([sentinel],
            source_inputs[next(iter(sources))]["image_sizes"], "full",
            image_newline=torch.tensor([-1.], device=device))
        if packed[0].flatten().tolist() != geometry.sentinel_order() or _plain(lens) != [len(geometry.tokens)]:
            raise RuntimeError("Actual Granite packing/unpadding order differs from coordinate mapping")
        p = self.base.layerwise_projectors[0]
        raster = torch.arange(24 * 24, device=device).reshape(1, 24 * 24, 1)
        if not torch.equal(p._unwin(p._win(raster, 24, 8), 3, 8), raster):
            raise RuntimeError("Actual QFormer window/unwindow raster mapping differs")
        spatial_checks = []
        for projector, (oh, ow), layer in zip(self.base.spatial_projectors,
                ((0, 0), (0, 1), (1, 0), (1, 1)), (12, 15, 18, 21)):
            expected = [(2 * r + oh) * 24 + 2 * c + ow for r in range(12) for c in range(12)]
            actual = projector.downsampler(raster).flatten().tolist()
            if actual != expected:
                raise RuntimeError("Actual SpatialDeepstack sampled offsets differ from slot coordinates")
            spatial_checks.append({"layer": layer, "offset": [oh, ow], "sampled_raster_exact": True})
        area = p.downsampler(raster.float()).flatten().tolist()
        if area != [2 * r * 24 + 2 * c + 12.5 for r in range(12) for c in range(12)]:
            raise RuntimeError("Actual LayerDeepstack area-query downsampling differs")
        ids = spatial_token_indices(geometry)
        feature_checks = []
        for name, source in sources.items():
            for layer, features in source.entries:
                if not torch.equal(features[0][ids][ids], features[0]):
                    raise RuntimeError("Granite post-pack permutation inverse failed")
                feature_checks.append({"source": name, "layer": layer, "inverse_exact": True})
        pairs = []
        for name in sources:
            other = (name.replace("m_s", "m1_s", 1) if name.startswith("m_s") else
                     name.replace("d_same_", "d_cross_", 1) if name.startswith("d_same_") else None)
            if other in sources:
                pairs.append((name, other))
        processor_checks = []
        for left, right in pairs:
            expected = self._pixel_half_swap(source_inputs[left]["pixel_values"], geometry, torch)
            actual = source_inputs[right]["pixel_values"]
            equal = torch.equal(expected, actual)
            if not equal:
                raise RuntimeError(f"Actual Granite processor violates declared pixel half swap: {left} -> {right}")
            processor_checks.append({"source": left, "half_swapped_source": right,
                                     "base_and_tiles_pixel_half_swap_exact": equal})
        box = _roi_bbox(row)
        roi = roi_token_indices(geometry, box) if box is not None else None
        return {"architecture": self.metadata, "geometry": geometry.summary(),
                "actual_pack_sentinel_exact": True, "qformer_window_raster_inverse_exact": True,
                "layer_area_query_raster_exact": True, "spatial_offset_checks": spatial_checks,
                "permutation_involution_exact": True,
                "special_positions_fixed": all(ids[i] == i for i in geometry.special_indices),
                "feature_inverse_checks": feature_checks, "processor_half_swap_checks": processor_checks,
                "roi": {"count": len(roi), "complement_count": len(geometry.tokens) - len(roi),
                        "base_count": sum(geometry.tokens[i].view == "base" for i in roi),
                        "tile_count": sum(geometry.tokens[i].view == "tiles" for i in roi),
                        "bbox_xyxy": list(box)} if roi is not None else None,
                "feature_pixel_equivariance_required": False,
                "native_self_patch_and_generation": "verified_separately_by_common_runner"}

    @staticmethod
    def _pixel_half_swap(pixels, geometry, torch):
        """Swap content halves in the real processor's base and tiled canvases."""
        result = pixels.clone()
        size = geometry.view_size
        result[0, 0] = torch.cat((pixels[0, 0, :, :, size // 2:], pixels[0, 0, :, :, :size // 2]), dim=-1)
        th, tw = geometry.tile_grid_hw
        mosaic = pixels[0, 1:].reshape(th, tw, 3, size, size).permute(2, 0, 3, 1, 4).reshape(3, th * size, tw * size)
        rh, rw = geometry.resized_hw
        py, px = (th * size - rh) // 2, (tw * size - rw) // 2
        if rw % 2:
            raise ValueError("Actual resized image cannot be split into equal pixel halves")
        swapped = mosaic.clone()
        content = mosaic[:, py:py + rh, px:px + rw]
        swapped[:, py:py + rh, px:px + rw] = torch.cat((content[:, :, rw // 2:], content[:, :, :rw // 2]), dim=-1)
        result[0, 1:] = swapped.reshape(3, th, size, tw, size).permute(1, 3, 0, 2, 4).reshape(th * tw, 3, size, size)
        return result
