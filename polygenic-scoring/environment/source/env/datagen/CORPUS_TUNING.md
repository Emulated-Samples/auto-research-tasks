# SV-PGS corpus design and shipping protocol

The shipped benchmark is an authenticated, fixed 15-category × 3-replicate
study. The old three-category pilot and the biased 13-category raw-SV-PGS grid
are retired. `dense_infinitesimal` and `low_prevalence` are restored: the shipped
reference is the exact public NumPy best of hierarchical empirical Bayes and
ridge logistic, so categories are not removed merely because one reference family
is the wrong inductive bias.

## Fixed shipping contract

- Schema: v8 only.
- Grid: exactly 15 ordered categories and replicates `0..2`, for 45 datasets.
- Requested dimensions: `N=12000`, `P=2500`, with a canonical 15%/85%
  train/test split. A category may apply a declared
  load-bearing shape transformation, such as the `p > n_train` sparse regime.
- Weight: `1.0` for every dataset; aggregation gives every category equal weight.
- Identity: opaque keyed IDs. Neither RNG seeds nor effective seeds are stored.
- Authentication: the manifest, every anchor, every development result, and the
  frozen development report are HMAC-authenticated with an external 32-byte key.
- Replay resistance: development and shipping derive independent `dgp`,
  `materialize`, `bootstrap`, and ID streams from purpose-separated HMAC contexts
  that also bind the generation-pipeline digest, category, and replicate.

The immutable constants live in `grader/contract.py`. Builders and validators do
not accept alternate sizes, dimensions, replicate counts, or unkeyed modes.

## Capability matrix

`datagen/categories.py` is the authoritative ordered matrix:

1. `svld_class` — annotation-adaptive shrinkage under strong LD.
2. `svld_strong` — joint decorrelation under very strong LD.
3. `sparse_heavy_tail` — sparse spikes with `p > n_train`.
4. `dense_infinitesimal` — dense, near-Gaussian small effects.
5. `sparse_dense_mix` — a dense floor plus sparse heavy-tailed spikes.
6. `rare_variant_maf` — MAF-dependent rare-variant signal.
7. `low_prevalence` — calibration under a rare binary outcome.
8. `weak_ld` — a regime where marginal methods remain competitive.
9. `soft_membership` — fractional class membership.
10. `nonlinear_annotation` — nonlinear annotation-to-scale structure.
11. `annotation_interaction` — class-by-length interaction.
12. `decoy_annotations` — declared noise and unlisted nuisance columns.
13. `ld_shift` — train/test LD shift with annotation-identifiable causal variants.
14. `suppressor_ld` — correlated equal-and-opposite effects.
15. `ancestry_shift` — transfer across ancestry-distribution shift.

The breadth is intentional: sparse and dense estimators, marginal and joint
estimators, and simple and flexible annotation models must change rank across
categories. A single hard-coded inductive bias should not dominate the matrix.

## Development gate

Before a corpus can finalize, `validation/model_zoo.py` must produce a current,
passing, authenticated development report over the same exact 15×3 shape under
the separate `development` purpose. The gate requires:

- reliable reference-minus-naive separation;
- training-only model selection before held-out labels are loaded;
- useful between-model variance relative to replicate noise;
- low category-profile redundancy;
- diverse winning model families;
- no contender that reliably beats the strong reference; and
- basic tuned models that do not saturate the benchmark.

The zoo includes the strong reference, SV-PGS, ridge, dense-plus-spike logistic,
elastic net, marginal P+T, class-adaptive ridge, and annotation-adaptive elastic
net. The shipped audit then refits only the frozen strong reference; shipped
outcomes never select categories, hyperparameters, or model rankings.

## Build

Use an explicit interpreter that can import the intended SV-PGS engine and an
absolute external key path:

```bash
datagen/build_corpus.sh \
  --key-file /run/secrets/svpgsbench-corpus.key \
  --python /opt/svpgs-venv/bin/python
```

The script runs one dataset per process, checks each completed dataset, and only
then finalizes the exact corpus. The direct immutable commands are:

```text
build_corpus.py --key-file KEY ONE <category> <replicate>
build_corpus.py --key-file KEY CHECK <category> <replicate>
build_corpus.py --key-file KEY CHECK_FINALIZED <category> <replicate>
build_corpus.py --key-file KEY FINALIZE
build_corpus.py --key-file KEY ALL
```

Finalization rejects missing, duplicate, reordered, stale, unexpected, tampered,
unconverged, or unauthenticated datasets. It also pins the exact passing
development report and every public/truth file digest.
