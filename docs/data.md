# Data and renderer adapter contract

`--provider module:build` imports trusted Python code. `build(config)` returns an
object implementing:

```python
class Provider:
    def identity(self) -> str:
        ...  # stable digest of data version, split, image and geometry contracts
    def train_batches(self, epoch):
        ...  # iterable of independent triplet batch dictionaries
    def val_batches(self):
        ...  # repeatable iterable over a fixed validation bank
    def close(self):
        ...  # optional resource cleanup
```

These ellipses describe the interface, not a bundled medical-data implementation.
The working synthetic provider is in `src/fluoroqa/synthetic.py`.

Each batch has `fixed`, `positive`, `negative`: N×1×R×R finite float tensors in
[0,1], and `mtre_mm`: N actual landmark errors in millimeters. `positive` is the
DRR rendered at GT for that fixed image; `negative` is one perturbed DRR. Generate
new training candidates online, not one cached perturbation for all epochs.
Use deterministic random keys based on seed/epoch/frame/candidate. Do not let
validation consume or alter the training RNG. No test/held-out frames belong in
either iterator. The provider owns split audits, geometry checks and trace logs.

## Image contract

The reference fixed-image pipeline is: crop 50 native pixels per side, Gaussian
sigma 1/truncate 2, divide by the positive maximum, bilinear resize with
`align_corners=False`. `preprocess.fluoroscopy` implements this pipeline. It is
not a universal recipe for arbitrary fluoroscopes: camera geometry and crop must
be updated together. Reject negative or empty projections rather than silently
normalizing them differently.

The reference renderer produces **gray exponential/absorption DRRs**, not binary
masks and not arbitrary raw line-integral images. Render at the target resolution,
min/max normalize, then invert once (`1-absorption`) into fixed-image polarity.
`preprocess.absorption_drr` performs only normalization and inversion; it does
not implement the renderer or convert its intensity physics.

Both tensors are then repeated across three channels and mapped from [0,1] to
[-1,1] inside the model. Upstream LPIPS performs its own scaling. Do not apply
ImageNet normalization a second time. Use identical preprocessing for train,
val, test, each local neighbor, and any later pose optimization. A renderer with
a different intensity domain must be validated, not silently substituted.

## Dataset access and geometry

No dataset is distributed. Obtain data through its official source and follow
its access terms; this repository grants no right to patient data. The internal
experiments used DeepFluoro, but this release does not distribute its HDF5,
patient IDs, private splits, CT masks, camera calibration or pose metadata.

Supply a rigid model-to-camera pose in millimeter coordinates to
`current_neighbors`. The model-space center must come from the model/CT, not a
GT pose. Neighbors are camera-axis translations ±2 mm and rotations ±2 degrees
about that center transformed by the CURRENT pose. Render all neighbors with
the same camera and intensity contract. Rotations are left-multiplied. If your
renderer uses inverse poses, transform conventions explicitly and test them.
