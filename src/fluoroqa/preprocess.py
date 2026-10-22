"""Explicit image-domain conversions. No pose or ground-truth access."""
import numpy as np
import torch
from scipy.ndimage import gaussian_filter
from torch.nn.functional import interpolate


def fluoroscopy(raw, resolution, border=50):
    """Historical projection pipeline: crop, Gaussian, max-normalize, resize."""
    image = np.asarray(raw, dtype=np.float32)
    if image.ndim != 2 or not np.isfinite(image).all() or image.min() < 0:
        raise ValueError("Expected finite, nonnegative 2D projection")
    if border < 0 or min(image.shape) <= 2*border or resolution < 32:
        raise ValueError("Invalid crop or resolution")
    if border:
        image = image[border:-border, border:-border]
    image = gaussian_filter(image, 1, truncate=2)
    if image.max() <= 0:
        raise ValueError("Empty projection")
    image = torch.from_numpy((image/image.max()).copy())[None, None]
    return interpolate(image, (resolution, resolution), mode="bilinear", align_corners=False)


def absorption_drr(raw):
    """Min/max normalize rendered absorption and invert ONCE into fixed polarity.

    This does not convert arbitrary line integrals to absorption. The renderer
    must supply the agreed exponential/absorption image domain and geometry.
    """
    image = torch.as_tensor(raw, dtype=torch.float32)
    if image.ndim != 2 or not torch.isfinite(image).all():
        raise ValueError("Expected finite 2D rendered absorption")
    if image.max() <= image.min():
        raise ValueError("Constant DRR; check geometry and renderer")
    return (1 - (image-image.min())/(image.max()-image.min()+1e-10))[None, None]
