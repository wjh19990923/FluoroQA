# Real DeepFluoro example — image license and provenance

Both PNG files in this directory are **CC BY-NC 4.0**, not MIT.
Use is subject to attribution and noncommercial terms:
https://creativecommons.org/licenses/by-nc/4.0/
No endorsement by the data authors is implied. The code license does not
override these image terms.

## Attribution

Robert B. Grupp, Mathias Unberath, Cong Gao, Rachel A. Hegeman, Ryan J. Murphy,
Clayton P. Alexander, Yoshito Otake, Benjamin A. McArthur, Mehran Armand,
Russell H. Taylor. *Automatic annotation of hip anatomy in fluoroscopy for
robust and efficient 2D/3D registration*. IJCARS 15, 759–769 (2020).
[Paper](https://doi.org/10.1007/s11548-020-02162-7) ·
[Original project](https://github.com/rg2/DeepFluoroLabeling-IPCAI2020) ·
[Original archive](https://doi.org/10.7281/T1/IFSXNV).

Website example: **subject02 / view000**.
[Public example source and terms](https://jinhaolab.com/data/2d3dreg/deepfluoro-subject02-000/README.txt) ·
[Public manifest](https://jinhaolab.com/data/2d3dreg/deepfluoro-subject02-000/manifest.json) ·
[Interactive viewer](https://jinhaolab.com/tools/2d3dreg/).
Website conversion source: eigenvivek/xvr-data, revision
`a17273e3eadbd793bd861f3598a80ce4590c1124`.

## Files and changes

- `public-xray.png`: hosted real X-ray, re-encoded without hidden PNG metadata.
  Website processing: crop 50 pixels on each edge, horizontal flip, 718-pixel
  resize, negative-log intensity and 1st–99th percentile display window.
- `registration-and-heatmaps.png`: same acquisition, from archived 256-pixel
  benchmark inputs. Real X-ray, two saved DRRs, cyan/red overlays, and actual
  FluoroQA-BCE learned-distance contribution maps. No CT or pose files included.
  All model panels are flipped vertically together for website-view orientation
  and grayscale is inverted for bone-bright display **after inference**.
  Model inputs retain historical preprocessing: smoothed, max-normalized real
  detector intensity and inverted absorption DRR. Website display pixels are
  not substituted for model input. Thus their contrast is intentionally different.

The two candidates were selected by mTRE proximity to 1 and 10 mm, respectively,
from the existing bank, not by favorable model scores. Actual mTRE is 0.92971 and
7.84824 mm. Success means <2 mm; failure means >=2 mm. These are candidate-pose
labels, not evidence that a registration optimizer succeeded. The specimen was
excluded from training, but this is one selected example, not a generalization test.

## Model and heatmap verification (2026-10-22)

BCE last epoch350 checkpoint SHA-256:
`f4a9dcde94b352f8576a938ae85c65678843dcf66941fc6f5dd8b34781798158`.
CPU eval mode, unchanged weights, no training or new rendering.
The fixed train-selected Raw threshold is 0.47625354. Scores are 0.45519373
(accept) and 0.49075437 (flag). They replay the archived evaluation within
2.99e-8. The threshold was not tuned on these examples.

Each map sums learned channel-weighted squared normalized VGG feature differences
across LPIPS scales. Each scale is bilinearly resized; a constant mean correction
preserves its contribution to the scalar distance. Map spatial means reproduce
the model distance within 1e-6. Both maps use the same color scale. This is a
distance decomposition, **not Grad-CAM, a probability, a segmentation, or a
ground-truth anatomical error map**. Background texture can contribute strongly.

Source-image identity was checked from public subject/view mapping and pixel
correlation (0.99829 after orientation and negative-log display alignment).
The downloaded website PNG matched its published SHA-256:
`dcacc503695890a8ad06e5ea174bb6bb8521636916f84f95dd396339f5323e84`.
The paired images were visually inspected; no embedded patient text was visible.
No metadata, CT, model weights, or additional acquisition is published.
