"""Qwen path amplitude/content controls, separate from the original adapter.

Importing this module does not import torch. Constructing an intervention imports torch.
"""
from __future__ import annotations

import hashlib
import math

MASKS = ("111", "110", "101", "011", "100", "010", "001", "000")


def conditions():
    """Thirty exploratory conditions; endpoint names are the original masks."""
    result = [{"name": m, "kind": "scale", "alpha": [int(x) for x in m]} for m in MASKS]
    for value in (.25, .5, .75):
        result.append({"name": f"all_alpha_{value:g}", "kind": "scale", "alpha": [value] * 3})
    for path in range(3):
        alpha = [1., 1., 1.]
        alpha[path] = .5
        result.append({"name": f"path{path}_alpha_0.5", "kind": "scale", "alpha": alpha})
    for pair in ((0, 1), (0, 2), (1, 2)):
        for value in (.25, .5, .75):
            alpha = [1., 1., 1.]
            for path in pair:
                alpha[path] = value
            result.append({"name": f"pair{pair[0]}{pair[1]}_alpha_{value:g}", "kind": "scale", "alpha": alpha})
    for kind in ("token_permute", "channel_orthogonal"):
        for path in range(3):
            result.append({"name": f"path{path}_{kind}", "kind": kind, "path": path, "alpha": [1., 1., 1.]})
    result.append({"name": "selfrestore", "kind": "selfrestore", "alpha": [1., 1., 1.]})
    return result


def validate_condition(condition):
    if condition.get("kind") not in {"scale", "token_permute", "channel_orthogonal", "selfrestore"}:
        raise ValueError("Unknown path-control kind")
    alpha = condition.get("alpha", [])
    if len(alpha) != 3 or any(not math.isfinite(x) or not 0 <= x <= 1 for x in alpha):
        raise ValueError("Need three finite amplitudes in [0,1]")
    if condition["kind"] in {"token_permute", "channel_orthogonal"}:
        if condition.get("path") not in (0, 1, 2) or alpha != [1, 1, 1]:
            raise ValueError("Content controls transform exactly one intact path")


def stable_seed(seed, item_id, path, kind):
    return int.from_bytes(hashlib.sha256(f"{seed}|{item_id}|{path}|{kind}".encode()).digest()[:8], "big") % (2**63 - 1)


def transform_features(features, condition, image_token_counts, *, seed, item_id):
    """Return new transformed tensors without modifying any source tensor.

    Token permutations stay within each image's merged-token segment. Channel
    controls use one signed permutation matrix per path/item, shared across its
    image tokens: an exact orthogonal map up to norm-reduction rounding. No
    resizing, cross-item geometry matching, or Gaussian renormalization occurs.
    """
    import torch
    validate_condition(condition)
    if len(features) != 3 or any(x.ndim != 2 for x in features):
        raise ValueError("Expected exactly three [visual_tokens, hidden] features")
    if not image_token_counts or any(not isinstance(n, int) or n < 1 for n in image_token_counts):
        raise ValueError("Need positive per-image merged-token counts")
    if any(x.shape[0] != sum(image_token_counts) for x in features):
        raise ValueError("Per-image grid counts disagree with actual merged features")
    transformed, details = [], []
    for path, source in enumerate(features):
        detail = {"path": path, "alpha": condition["alpha"][path], "transform": "scale"}
        kind = condition["kind"]
        if kind == "selfrestore":
            target = torch.zeros_like(source)
            target.copy_(source)
            detail["transform"] = "zero_then_restore_original"
        elif kind in {"token_permute", "channel_orthogonal"} and condition["path"] == path:
            generator = torch.Generator(device="cpu")
            local_seed = stable_seed(seed, item_id, path, kind)
            generator.manual_seed(local_seed)
            detail.update(transform=kind, seed=local_seed)
            if kind == "token_permute":
                permutation, offset = [], 0
                for count in image_token_counts:
                    indices = torch.randperm(count, generator=generator)
                    if count > 1 and torch.equal(indices, torch.arange(count)):
                        indices = indices.roll(1)
                    permutation.extend((indices + offset).tolist())
                    offset += count
                indices = torch.tensor(permutation, device=source.device)
                target = source.index_select(0, indices)
                detail.update(permutation=permutation, image_token_counts=list(image_token_counts),
                              changed_indices=sum(i != value for i, value in enumerate(permutation)))
            else:
                width = source.shape[1]
                permutation = torch.randperm(width, generator=generator)
                signs = torch.randint(0, 2, (width,), generator=generator) * 2 - 1
                target = source.index_select(1, permutation.to(source.device)) * signs.to(device=source.device, dtype=source.dtype)
                detail.update(channel_permutation_sha256=hashlib.sha256(bytes(str(permutation.tolist()), "ascii")).hexdigest(),
                              channel_sign_sha256=hashlib.sha256(bytes(str(signs.tolist()), "ascii")).hexdigest(),
                              negative_channels=int((signs < 0).sum().item()),
                              orthogonal_map="signed channel permutation; shared across all image tokens")
        else:
            value = condition["alpha"][path]
            target = source if value == 1 else torch.zeros_like(source) if value == 0 else source * value
        transformed.append(target)
        details.append(detail)
    return transformed, details


def PathControlIntervention(model, condition, image_token_counts, *, seed, item_id, audit=False):
    """Factory keeps preparation/analysis torch-free; subclass reuses layer audit."""
    import torch
    from deepstack.intervention import DeepStackIntervention

    class Control(DeepStackIntervention):
        def __init__(self):
            super().__init__(model, "111", audit=audit)
            validate_condition(condition)
            self._source_features = None
            self._transformed = None

        def _forward(self, *args, **kwargs):
            bound = self._signature.bind_partial(*args, **kwargs)
            sources = bound.arguments.get("deepstack_visual_embeds")
            if sources is None:
                return super()._forward(*args, **kwargs)
            saved = [x.clone() for x in sources] if self.audit else None
            if self._source_features is None or any(a is not b for a, b in zip(self._source_features, sources)):
                self._transformed, self._details = transform_features(sources, condition, image_token_counts, seed=seed, item_id=item_id)
                self._source_features = list(sources)
            updated = dict(kwargs)
            call_args = args
            if "deepstack_visual_embeds" in updated:
                updated["deepstack_visual_embeds"] = self._transformed
            else:
                names = list(self._signature.parameters)
                positional = list(args)
                positional[names.index("deepstack_visual_embeds")] = self._transformed
                call_args = tuple(positional)
            start = len(self.records)
            result = super()._forward(*call_args, **updated)
            for record in self.records[start:]:
                record["condition"] = dict(condition)
                record["image_token_counts"] = list(image_token_counts)
                record.pop("mask", None)
                for path, (source, applied, detail) in enumerate(zip(sources, self._transformed, self._details)):
                    record["paths"][path].update(detail)
                    record["paths"][path]["enabled"] = detail["alpha"] != 0
                    if self.audit:
                        original_norm = float(source.float().norm().item())
                        applied_norm = float(applied.float().norm().item())
                        unchanged = bool(torch.equal(source, saved[path]))
                        record["paths"][path].update(original_norm=original_norm, applied_norm=applied_norm,
                            source_unchanged=unchanged, actually_changed=not bool(torch.equal(source, applied)),
                            relative_norm_change=(applied_norm / original_norm - 1) if original_norm else None)
                        if not unchanged:
                            raise RuntimeError("Original cached visual feature was mutated")
                        if detail["transform"] in {"token_permute", "channel_orthogonal"} and abs(applied_norm - original_norm) > max(1e-5, original_norm * 1e-5):
                            raise RuntimeError("Norm-preserving content control changed global norm")
            return result

    return Control()
