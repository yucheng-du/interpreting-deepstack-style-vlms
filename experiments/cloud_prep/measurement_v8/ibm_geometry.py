"""Granite Vision 4.1 post-projector/pack coordinates, without tensor imports.

Coordinates describe output *slots*. QFormer cross-attention and the preceding
ViT contextualize their values; a slot's box is not its causal receptive field.
The packing order mirrors IBM revision 37d591f and LlavaNext unpad_image.
"""
from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from math import ceil, isclose, isfinite
from typing import Any, Mapping, Sequence

REVISION = "37d591f06319e8f1638b5adcf58bdf50e0f84f7a"
LAYER_MAP = ((-19, 9), (-13, 6), (-7, 3), (-1, 0))
SPATIAL_LAYERS = (12, 15, 18, 21)
ENTRY_ORDER = (9, 6, 3, 0, *SPATIAL_LAYERS)
ENTRY_GROUPS = {"all": (3, 6, 9, 12, 15, 18, 21), "layer": (3, 6, 9),
                "spatial": SPATIAL_LAYERS}
GRID_PINPOINTS = ((384, 384), (384, 768), (384, 1152), (384, 1536),
                 (384, 1920), (384, 2304), (384, 2688), (384, 3072),
                 (384, 3456), (384, 3840), (768, 384), (768, 768),
                 (768, 1152), (768, 1536), (768, 1920), (1152, 384),
                 (1152, 768), (1152, 1152), (1536, 384), (1536, 768),
                 (1920, 384), (1920, 768), (2304, 384), (2688, 384),
                 (3072, 384), (3456, 384), (3840, 384))


def _get(obj: Any, key: str, default: Any = None) -> Any:
    return obj.get(key, default) if isinstance(obj, Mapping) else getattr(obj, key, default)


def validate_config(config: Any) -> dict[str, Any]:
    """Fail on an architecture that would change the meaning of this adapter."""
    vision, text = _get(config, "vision_config"), _get(config, "text_config")
    expected = {
        "model_type": "granite4_vision", "downsample_rate": "4/8",
        "vision_feature_select_strategy": "full", "spatial_stride": 2,
        "spatial_vision_layer": -1, "use_spatial_sampling": True,
        "use_image_newline_parameter": True,
    }
    for key, value in expected.items():
        if _get(config, key) != value:
            raise ValueError(f"Granite pinned {key} differs: {_get(config, key)!r}")
    if tuple(map(tuple, _get(config, "deepstack_layer_map", ()))) != LAYER_MAP:
        raise ValueError("Granite LayerDeepstack source/target map differs")
    if tuple(_get(config, "spatial_target_layers", ())) != SPATIAL_LAYERS:
        raise ValueError("Granite SpatialDeepstack targets differ")
    if tuple(map(tuple, _get(config, "image_grid_pinpoints", ()))) != GRID_PINPOINTS:
        raise ValueError("Granite anyres resolution candidates differ")
    if (_get(vision, "image_size"), _get(vision, "patch_size")) != (384, 16):
        raise ValueError("Expected 384px SigLIP views with patch16")
    if (_get(text, "model_type"), _get(text, "num_hidden_layers"),
            _get(text, "hidden_size")) != ("granite", 40, 2560):
        raise ValueError("Expected the pinned Granite dense decoder")
    return {
        "adapter": "granite-vision-4.1-postpack-v8", "revision": REVISION,
        "entry_order": list(ENTRY_ORDER), "main_entry": 0,
        "entry_groups": {key: list(value) for key, value in ENTRY_GROUPS.items()},
        "injection": "add_before_decoder_layer_at_image_placeholders",
        "image_placeholder_embeddings_zeroed": True,
        "layer0_preserved_from_main": True,
        "original_patch_grid": [24, 24], "projected_grid": [12, 12],
        "qformer_window_patch_grid": [8, 8], "qformer_query_grid": [4, 4],
        "spatial_offsets": [[0, 0], [0, 1], [1, 0], [1, 1]],
        "roi_semantics": "positive-area intersection with nominal output-slot box; not receptive field",
    }


def select_best_resolution(image_hw: Sequence[int], pinpoints=GRID_PINPOINTS) -> tuple[int, int]:
    """Match HF's effective-resolution then wasted-resolution rule and tie order."""
    h, w = image_hw
    if any(type(v) is not int or v <= 0 for v in (h, w)):
        raise ValueError("Positive integer original (height, width) required")
    best, effective_best, waste_best = None, 0, float("inf")
    for hh, ww in pinpoints:
        scale = min(ww / w, hh / h)
        effective = min(int(w * scale) * int(h * scale), h * w)
        wasted = hh * ww - effective
        if effective > effective_best or (effective == effective_best and wasted < waste_best):
            best, effective_best, waste_best = (hh, ww), effective, wasted
    if best is None:
        raise ValueError("No candidate resolution")
    return best


def resized_content_hw(image_hw: Sequence[int], target_hw: Sequence[int]) -> tuple[int, int]:
    """LlavaNext get_patch_output_size: fit inside canvas with ceil rounding."""
    h, w = image_hw
    th, tw = target_hw
    if tw / w < th / h:
        return min(ceil(h * tw / w), th), tw
    return th, min(ceil(w * th / h), tw)


def unpad_bounds(image_hw: Sequence[int], grid_hw: Sequence[int]) -> tuple[int, int, int, int]:
    """Return row_start,row_stop,col_start,col_stop with HF's symmetric floor crop.

    Odd padding intentionally leaves one extra cell. Do not silently replace
    this with a mathematically ideal aspect-ratio crop.
    """
    h, w = image_hw
    gh, gw = grid_hw
    if w / h > gw / gh:
        nh = int(round(h * (gw / w), 7))
        pad = (gh - nh) // 2
        return pad, gh - pad, 0, gw
    nw = int(round(w * (gh / h), 7))
    pad = (gw - nw) // 2
    return 0, gh, pad, gw - pad


@dataclass(frozen=True)
class TokenSlot:
    index: int
    kind: str                         # visual, padding, newline
    view: str                         # base or tiles (tiles form one raster)
    row: int
    col: int
    source_view: int | None            # processor base=0; row-major tiles=1...
    source_row: int | None
    source_col: int | None
    bbox_xyxy: tuple[float, float, float, float] | None


@dataclass(frozen=True)
class ImageGeometry:
    image_hw: tuple[int, int]
    canvas_hw: tuple[int, int]
    resized_hw: tuple[int, int]
    view_size: int
    projected_side: int
    tile_grid_hw: tuple[int, int]
    unpad_rc: tuple[int, int, int, int]
    tokens: tuple[TokenSlot, ...]

    @property
    def num_views(self) -> int:
        return 1 + self.tile_grid_hw[0] * self.tile_grid_hw[1]

    @property
    def special_indices(self) -> tuple[int, ...]:
        return tuple(t.index for t in self.tokens if t.kind != "visual")

    def summary(self) -> dict[str, Any]:
        return {"image_hw": list(self.image_hw), "canvas_hw": list(self.canvas_hw),
                "resized_hw": list(self.resized_hw), "num_views": self.num_views,
                "projected_side": self.projected_side, "tile_grid_hw": list(self.tile_grid_hw),
                "unpad_rc": list(self.unpad_rc), "packed_tokens": len(self.tokens),
                "base_tokens": sum(t.view == "base" for t in self.tokens),
                "tile_visual_tokens": sum(t.view == "tiles" and t.kind == "visual" for t in self.tokens),
                "newline_tokens": sum(t.kind == "newline" for t in self.tokens),
                "retained_padding_tokens": sum(t.kind == "padding" for t in self.tokens),
                "special_indices": list(self.special_indices),
                "roi_coordinate_system": "original-image pixels, xyxy, positive-area overlap"}

    def sentinel_order(self) -> list[int]:
        """Expected packed values from arange(n_views * side**2), newline=-1."""
        side = self.projected_side
        return [-1 if t.kind == "newline" else
                t.source_view * side * side + t.source_row * side + t.source_col for t in self.tokens]


def image_geometry(image_hw: Sequence[int], *, pinpoints=GRID_PINPOINTS,
                   view_size: int = 384, patch_size: int = 16,
                   downsample_rate: str = "4/8", image_newline: bool = True) -> ImageGeometry:
    """Build the actual base + globally packed/unpadded tile order.

    Base view is resized to a square (it is not a crop). Tile boxes are mapped
    through the fitted-and-padded canvas back to original pixel coordinates.
    Whole retained padding cells have no image support, but remain in all and
    complement interventions. Partly padded cells retain their clipped support.
    """
    h, w = tuple(image_hw)
    canvas = select_best_resolution((h, w), pinpoints)
    if any(v % view_size for v in canvas):
        raise ValueError("Canvas must contain whole views")
    side = int((view_size // patch_size) * Fraction(downsample_rate))
    if side < 1:
        raise ValueError("Empty projected grid")
    th, tw = canvas[0] // view_size, canvas[1] // view_size
    rh, rw = resized_content_hw((h, w), canvas)
    py, px = (canvas[0] - rh) // 2, (canvas[1] - rw) // 2
    top, bottom, left, right = unpad_bounds((h, w), (th * side, tw * side))
    tokens: list[TokenSlot] = []

    def emit(kind, view, r, c, sv, sr, sc, box):
        tokens.append(TokenSlot(len(tokens), kind, view, r, c, sv, sr, sc, box))

    for r in range(side):
        for c in range(side):
            emit("visual", "base", r, c, 0, r, c,
                 (c * w / side, r * h / side, (c + 1) * w / side, (r + 1) * h / side))
    cell = view_size / side
    for r in range(top, bottom):
        for c in range(left, right):
            x0, y0 = max(0., (c * cell - px) * w / rw), max(0., (r * cell - py) * h / rh)
            x1, y1 = min(float(w), ((c + 1) * cell - px) * w / rw), min(float(h), ((r + 1) * cell - py) * h / rh)
            box = (x0, y0, x1, y1) if x1 > x0 and y1 > y0 else None
            emit("visual" if box else "padding", "tiles", r, c,
                 1 + (r // side) * tw + c // side, r % side, c % side, box)
        if image_newline:
            emit("newline", "tiles", r, right, None, None, None, None)
    return ImageGeometry((h, w), canvas, (rh, rw), view_size, side, (th, tw),
                         (top, bottom, left, right), tuple(tokens))


def spatial_token_indices(geometry: ImageGeometry, permutation: str = "swap_halves") -> list[int]:
    """Output-token -> source-token indices; exchange image halves in each view.

    A valid exact permutation must exchange corresponding image-coordinate
    boxes. Unequal/partially padded horizontal grids fail, instead of silently
    permuting padding into image content. Current 384x256 panels are exact.
    """
    ids = list(range(len(geometry.tokens)))
    if permutation == "identity":
        return ids
    if permutation != "swap_halves":
        raise ValueError(f"Unknown permutation {permutation!r}")
    rows: dict[tuple[str, int], list[TokenSlot]] = {}
    for token in geometry.tokens:
        if token.kind == "visual":
            rows.setdefault((token.view, token.row), []).append(token)
    width = geometry.image_hw[1]
    for key, row in rows.items():
        if len(row) % 2:
            raise ValueError(f"Unequal horizontal image half-token counts in {key}")
        half = len(row) // 2
        for col, token in enumerate(row):
            donor = row[(col + half) % len(row)]
            shift = width / 2 if col < half else -width / 2
            a, b = token.bbox_xyxy, donor.bbox_xyxy
            expected = (a[0] + shift, a[1], a[2] + shift, a[3])
            if not all(isclose(x, y, abs_tol=1e-7) for x, y in zip(expected, b)):
                raise ValueError("Post-unpad cells do not form an exact pixel-coordinate half swap")
            ids[token.index] = donor.index
    if sorted(ids) != list(range(len(ids))) or any(ids[ids[i]] != i for i in range(len(ids))):
        raise AssertionError("Half permutation is not an involutive bijection")
    if any(ids[i] != i for i in geometry.special_indices):
        raise AssertionError("Special positions moved")
    return ids


def roi_token_indices(geometry: ImageGeometry, bbox_xyxy: Sequence[float]) -> list[int]:
    """Positive-area overlap of nominal output slots with original-image ROI."""
    if len(bbox_xyxy) != 4 or any(not isfinite(float(v)) for v in bbox_xyxy):
        raise ValueError("Finite ROI pixel xyxy coordinates required")
    x0, y0, x1, y1 = map(float, bbox_xyxy)
    h, w = geometry.image_hw
    if not (0 <= x0 < x1 <= w and 0 <= y0 < y1 <= h):
        raise ValueError("ROI must be a nonempty box inside the original image")
    return [t.index for t in geometry.tokens if t.bbox_xyxy is not None
            and min(x1, t.bbox_xyxy[2]) > max(x0, t.bbox_xyxy[0])
            and min(y1, t.bbox_xyxy[3]) > max(y0, t.bbox_xyxy[1])]


def region_token_indices(geometry: ImageGeometry, region: str,
                         bbox_xyxy: Sequence[float] | None = None, *,
                         permutation: str = "identity") -> list[int]:
    """Select slots after P using donor-coordinate ROI membership before P.

    Thus complement is exactly all minus roi, including special positions.
    Mechanism experiments currently use identity P, but their definition stays
    coherent if a later version crosses P with ROI.
    """
    if region == "all":
        return list(range(len(geometry.tokens)))
    if region not in ("roi", "complement") or bbox_xyxy is None:
        raise ValueError("roi/complement need donor_roi_bbox_xyxy")
    roi = set(roi_token_indices(geometry, bbox_xyxy))
    ids = spatial_token_indices(geometry, permutation)
    return [i for i, source in enumerate(ids) if (source in roi) == (region == "roi")]


def entry_layers(group: str) -> tuple[int, ...]:
    if group not in ENTRY_GROUPS:
        raise ValueError(f"Unknown IBM entry_group {group!r}")
    return ENTRY_GROUPS[group]
