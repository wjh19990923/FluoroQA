# Protocol and reproduction boundaries

## Three variants

- Zero: frozen VGG, train only the 1,472 LPIPS channel coefficients.
- mTRE: same network; margin 0.001 times actual mTRE, capped at 10 mm.
- mTRE+BCE: start from mTRE last epoch 200, unfreeze conv4_3 and conv5_1/2/3,
  continue to epoch 350. Adam moments for the channel head are retained; the
  newly unfrozen parameters start with empty Adam state. Head LR 1e-4, tail
  LR 1e-5, betas (0.9,0.999), epsilon 1e-8, no decay or scheduler.

VGG runs in eval mode throughout; LPIPS head dropout (0.5) is active during
training and disabled during evaluation. Each sample is one fixed/GT/negative
triplet. A larger batch is not a within-frame multi-negative ranking objective.

The 256-pixel BCE constants in the config come from a training-only probe at the
start of the continuation phase. Do not tune them on test data. The procedure was:

1. Collect 64 training triplets, in 16 batches, without optimizer updates.
2. Compute pair scores with GT pairs labeled success and negative pairs labeled
   failure if actual mTRE is at least 2 mm.
3. Offset = midpoint of the two class means. Scale = population standard
   deviation of all pair distances. Reject a zero/invalid scale or missing class.
4. Replay the same dropout RNG for each gradient measurement. Measure L2 norms
   over all trainable parameters for triplet and unweighted BCE separately.
5. Weight = 0.25 × median(positive triplet-gradient norms) / median(BCE norms).
   Require at least four active triplet batches and finite positive denominators.
6. Restore RNG, confirm weights unchanged, and lock the policy before training.

The portable CLI accepts a locked policy; it does not automatically run this
historical probe. A new dataset/resolution requires a new, recorded train-only
probe, not blind reuse of the reference constants.

## Calibration

Success: actual mTRE <2 mm. Failure: mTRE >=2 mm. Equality is failure.
Raw thresholds maximize balanced accuracy on training scores; ties choose the
largest threshold. Local thresholds use validation scores under a distinct
protocol. Classification predicts failure when score >= threshold.

For local z scores, divide center-minus-neighbor-mean by population standard
deviation; if standard deviation is exactly zero, divide by 1. No additional
epsilon is introduced. Neighbors never use a GT-centered candidate mean.

AUROC does not depend on the selected decision threshold. Validation used for
model selection is not an independent test cohort. Repeated exploration of test
data and one held-out patient do not establish general clinical validity.

## Reproduction status

This is a portable research core, not an archival clone of the internal runs.
The release intentionally excludes external renderer source and medical data.
It does not ship the historical split manifest, trained checkpoints, original
multi-best selection runner or bootstrap analysis. Therefore its synthetic tests
do not reproduce or validate clinical performance numbers. Users must implement
and audit a compatible provider before attempting a numerical reproduction.

The core supports 64,128,256 image sizes. Changing the resolution in zero/mTRE
configs is supported, but a model/threshold trained at one resolution must not
be silently reused at another. The supplied BCE policy is for 256 only.
