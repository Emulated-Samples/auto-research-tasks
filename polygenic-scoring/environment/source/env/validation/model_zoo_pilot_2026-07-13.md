# svpgsbench model-zoo pilot — N=4000 (2026-07-13)

> GRADER-SIDE / HIDDEN (validation/). First real model-zoo pilot over the hardened
> 14-category matrix. Raw scores: `validation/model_zoo_report_N4000.json`.
> Purpose: category SELECTION (between-model variance / correlation / gradeability).

## Run
- 14 categories × 5 seeds = **70 datasets**, N=4000 P=800, real SV-PGS reference.
- Resilience fix (retry ladder + skip-and-continue) worked: **70/70 built, 0 skips,
  0 early-stopped references** (every fit converged at the full 25 EM iterations).
  The N=12000 divergence did NOT recur at N=4000.
- 9-model zoo: covariates_only, pt (P+T), ridge, lasso, elastic-net, adaptive_lasso,
  awen (annotation-weighted EN), group_ridge, residualized_en.

## Headline result: separation is COMPRESSED at N=4000
- Best model on any category rarely exceeds 0.25 skill (targets: EN 0.30–0.50,
  annotation-EN 0.45–0.65). Everything clusters near the naive floor.
- Between-model variance is low everywhere (max 0.0087 for low_prevalence; target
  between-agent SD ≥ 0.15 → variance ≥ ~0.0225). The zoo does not spread enough.
- This is largely a SIZE artifact: at N=4000 (n_train 2400, P=800, n/p≈3) there is
  too little data for from-scratch models to pull ahead of naive or differentiate,
  while SV-PGS (full Bayesian machinery) still extracts the modest ref-naive gap.
  The original 1.16 rollout was at N=12000 (n/p≈9), where the zoo separates.

## Per-category (skill of best zoo model; ref_gap = SV-PGS AUC over naive)
| category | ref_gap | best model | best skill | between-model var |
|---|---|---|---|---|
| svld_strong | 0.089 | awen | 0.151 | 0.0030 |
| svld_class | 0.089 | awen | 0.140 | 0.0014 |
| soft_membership | 0.087 | awen | 0.178 | 0.0023 |
| annotation_interaction | 0.083 | awen | 0.111 | 0.0013 |
| signed_ld | 0.062 | awen | 0.246 | 0.0071 |
| nonlinear_annotation | 0.061 | adaptive_lasso | 0.005 | 0.0000 |
| decoy_annotations | 0.061 | awen | 0.212 | 0.0046 |
| sparse_dense_mix | 0.061 | awen | 0.037 | 0.0002 |
| weak_ld | 0.056 | adaptive_lasso | 0.023 | 0.0001 |
| rare_variant_maf | 0.044 | covariates_only | 0.000 | 0.0000 |
| sparse_heavy_tail | 0.038 | en | 0.064 | 0.0001 |
| ancestry_shift | 0.030 | awen | 0.142 | 0.0006 |
| dense_infinitesimal | 0.024 | covariates_only | 0.000 | 0.0000 |
| low_prevalence | 0.0145 | pt | 0.280 | 0.0087 |

## Redundancy: 11 category pairs correlate > 0.8
decoy↔svld_class 0.966; soft↔svld_strong 0.957; signed↔svld_class 0.904;
sparse_dense↔weak_ld 0.902; annotation_interaction↔signed_ld 0.899; … Several new
categories collapse onto svld_class/svld_strong's ordering.

## Read (preliminary — confirm at ship size N=12000)
- CLEAN: early-stop health (no ungradeable-via-truncation anchor). The resilience
  guard is doing its job.
- ZERO model resolution at N=4000 (all zoo models ≈ 0): dense_infinitesimal,
  rare_variant_maf, nonlinear_annotation, weak_ld, sparse_dense_mix — only the
  reference beats naive; no from-scratch model does. Either genuinely hard (keep)
  or too-hard-at-4000 (size artifact). MUST re-measure at 12000 before pruning.
- NOISY: low_prevalence — pt's 0.28 rides a tiny ref_gap (0.0145) → denominator
  noise (the noise-aware SE floor dampens it, but the category is fragile). Prune
  or raise prevalence.
- BEST-SEPARATING distinct mechanisms: signed_ld (awen 0.246), ancestry_shift
  (awen 0.142, distinct top vs others), svld_strong/svld_class (largest gaps).

## Decision
Do NOT finalize the category set from N=4000 — it is too compressed to trust the
difficulty ladder or between-model variance. Re-run the model-zoo at the SHIP size
**N=12000** (resilience fix makes it safe) to get the real ladder, then prune:
drop categories that STILL show zero model resolution at 12000 (false difficulty),
and de-duplicate the >0.8-correlated pairs (keep the distinct-mechanism
representative). Full-size run launched.
