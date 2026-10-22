"""Triplet objectives and fixed affine BCE without a classifier network."""
import math
from dataclasses import dataclass
import torch
from torch.nn import functional as F


@dataclass(frozen=True)
class BCEPolicy:
    offset: float
    scale: float
    weight: float

    def __post_init__(self):
        if not all(math.isfinite(x) for x in (self.offset, self.scale, self.weight)):
            raise ValueError("BCE policy must be finite")
        if self.scale <= 0 or self.weight < 0:
            raise ValueError("BCE requires positive scale and nonnegative weight")


def objective(d_pos, d_neg, mtre_mm, margin="mtre", alpha=0.001, cap_mm=10., bce=None):
    dp, dn = d_pos.reshape(-1), d_neg.reshape(-1)
    error = torch.as_tensor(mtre_mm, device=dp.device, dtype=dp.dtype).reshape(-1)
    if dp.shape != dn.shape or dp.shape != error.shape or not dp.numel():
        raise ValueError("One positive, negative and actual mTRE per triplet")
    if not all(torch.isfinite(x).all() for x in (dp, dn, error)) or (error < 0).any():
        raise ValueError("Nonfinite scores or invalid mTRE")
    if margin not in ("zero", "mtre") or not math.isfinite(alpha) or alpha < 0 or not math.isfinite(cap_mm) or cap_mm <= 0:
        raise ValueError("Invalid margin policy")
    margins = alpha * error.clamp(max=cap_mm) if margin == "mtre" else torch.zeros_like(error)
    triplet = (dp - dn + margins).clamp_min(0).mean()
    auxiliary = dp.new_zeros(())
    if bce is not None:
        scores = torch.cat([dp, dn])
        labels = torch.cat([torch.zeros_like(dp), (error >= 2).to(dp.dtype)])
        auxiliary = F.binary_cross_entropy_with_logits((scores-bce.offset)/bce.scale, labels)
    total = triplet + (bce.weight * auxiliary if bce is not None else 0)
    return {"loss": total, "triplet": triplet, "bce": auxiliary,
            "ordering": (dp < dn).float().mean(), "margin": margins.mean()}
