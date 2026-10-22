# Privacy and publication boundary

Only curated source, public configuration values, documentation, aggregate
figures and synthetic tests are normally intended for this repository.
An explicitly authorized exception is `assets/deepfluoro/`: two PNGs showing
one already-public DeepFluoro acquisition and derived visualization panels.
These have separate attribution and CC BY-NC 4.0 terms. No other medical images,
CT volumes, private patient identifiers, poses, camera matrices, private split manifests,
checkpoints, experiment logs, host addresses, tokens or original repository
history are included in this release. Public subject/view IDs identify the source,
not a license to publish additional records.

Generated outputs may contain sensitive data even if filenames do not. Keep all
`runs/`, `data/`, `weights/`, TensorBoard events, images and renderer outputs
private. Ignore rules reduce accidents; they are not a security boundary.
Inspect the staged file list and contents before every publication. Never upload
medical artifacts as GitHub Issues attachments. Report privacy issues without
posting the affected patient data or credentials.

No telemetry or W&B upload is enabled by this package. TensorBoard logging is
local. Upstream pretrained initialization can fetch weights; use offline smoke
or load a trusted checkpoint with `pretrained=False` to avoid this download.
