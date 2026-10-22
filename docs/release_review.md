# Initial release review

Review date: 2026-10-22.

## Checks performed

- Ten automated CPU unit tests: triplet margins, BCE boundary and gradient sign,
  invalid inputs, threshold tie behavior, AUROC ties and undefined cases, split
  restrictions, local mean/z scoring, current-pose geometry, image polarity,
  checkpoint round trip, and actual offline VGG freeze/forward/backward checks.
- A one-epoch offline synthetic run through the training entry point, including
  local TensorBoard events and an epoch checkpoint. No pretrained download,
  medical data or GPU was used.
- File allowlist, size and sensitive-string scans of the curated release; manual
  review of the source and documentation. New independent Git history.
- Upstream LPIPS, PyTorch, torchvision, NumPy, SciPy and TensorBoard license
  sources checked. No dependency source, dataset or weight files redistributed.

## Limitations

These are software/plumbing checks, not a medical performance validation or an
exhaustive security/legal audit. They do not certify the private renderer or
prove equivalence of a new data adapter. Upstream LPIPS emits torchvision's
deprecated `pretrained` API warnings; the tested versions still work.
See `requirements-tested.txt` for the checked environment.

The first local test attempt hit Windows sandbox temporary-directory/encoding
restrictions. After fixing UTF-8 logging, the offline CPU checks passed outside
that sandbox. No experiment data or historical results were changed.

Missing components are explicit: medical data adapter, renderer, trained
FluoroQA weights and historical split manifests. Publication readiness of this
research core is not a claim of complete paper reproducibility.
