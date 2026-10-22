# Third-party provenance and license boundary

FluoroQA first-party code is under the MIT license in `LICENSE`.
No third-party source tree, pretrained weights or dataset is vendored here.
Python dependencies are installed separately and retain their own licenses.

The companion Projection Lab is an external link, not part of this source
distribution. Its medical examples have separately displayed CC BY-NC 4.0 terms.
The diagrams here are original schematics; the result chart contains aggregates
only. No website medical image or third-party illustration is copied here.

| Dependency | Use | Upstream license source |
|---|---|---|
| LPIPS 0.1.4 | VGG feature distance and learned channel layers | [BSD-2-Clause](https://github.com/richzhang/PerceptualSimilarity/blob/master/LICENSE) |
| PyTorch | Tensors, gradients and checkpoints | [BSD-style license and notices](https://github.com/pytorch/pytorch/blob/main/LICENSE) |
| torchvision | Upstream VGG backbone | [BSD-3-Clause](https://github.com/pytorch/vision/blob/main/LICENSE) |
| NumPy | Array utilities | [BSD-3-Clause and bundled notices](https://github.com/numpy/numpy/blob/main/LICENSE.txt) |
| SciPy | Gaussian filtering, ranks and rotations | [BSD-3-Clause and bundled notices](https://github.com/scipy/scipy/blob/main/LICENSE.txt) |
| TensorBoard | Local training curves | [Apache-2.0](https://github.com/tensorflow/tensorboard/blob/master/LICENSE) |

LPIPS credit: Copyright (c) 2018, Richard Zhang, Phillip Isola, Alexei A. Efros,
Eli Shechtman, Oliver Wang. The VGG-based LPIPS implementation remains theirs;
the FluoroQA wrapper is not a replacement attribution.

The source-code licenses do not grant rights to ImageNet, LPIPS training data,
DeepFluoro or other medical datasets. Pretrained weights and data require a
separate terms review. No FluoroQA weights are redistributed in this release.
External DRR/registration libraries are not included because their source and
redistribution scope have not been cleared for this release. Supply your own
authorized renderer through the documented provider interface.

If redistributing an installed environment, wheel bundle or container, retain
all dependency licenses and bundled third-party notices, not only this summary.
This technical review is not a legal opinion.
