# Release contract

A release is one atomic tuple: prompt, generator, authenticated corpus, reference
anchors, scientific scoring contract, wrapper, development report, final shipping
audit, mastery-calibration artifact, and release lock. A source-only change or a
partially rebuilt corpus is not a release. Production accepts schema 8 only; there
is no legacy-compatible grading path.

## Frozen scientific contract

The schema-v8 shipping grid is exactly 15 ordered categories with three replicates
per category (45 datasets). The authoritative list and dimensions live in
`grader/contract.py`; the authenticated `corpus/manifest.json` must match them
exactly.

Each dataset has two training-only anchors:

- **naive = 0**: a covariates-only model;
- **reference = 1**: the exact packageable public NumPy best of
  `{hierarchical_eb, ridge_logistic}`, selected
  by deterministic inner-validation AUC using training rows only, then refit on
  the full training set. Ridge must beat hierarchical EB by the declared margin.

The family deliberately excludes cheats and weak diagnostic baselines. The build
records the selected method, both candidates' selection AUCs, convergence evidence,
the total selection-plus-refit time, and the winning refit's time. All anchor fits
finish before prediction data are opened, and hidden labels are loaded only after
all predictions are complete.

Per-dataset scientific reward is signed, winsorized AUC skill:

```text
auc_skill = (AUC_submission - AUC_naive) / (AUC_reference - AUC_naive)
calibration_factor = geometric_mean(exp(-proper_score_regret / scale))
reward = auc_skill * calibration_factor   when auc_skill > 0
reward = auc_skill                        when auc_skill <= 0
```

Both the paired reference-minus-naive SE and the naive estimate's SE are mandatory.
The AUC denominator must pass the reliability and resolution gates before the
dataset can ship. Scores are bounded to `[-0.5, 1.5]`; invalid execution receives
`-0.5`, so selective crashing cannot beat a valid bad prediction.

Brier and log loss carry no additive reward weight. Their worse-than-reference
proper-score regrets form a **continuous calibration factor** multiplying positive
AUC skill. Exact reference predictions retain factor 1; prevalence-compressed
rankers lose credit despite unchanged AUC; severe overconfidence tends smoothly
toward zero. Negative AUC skill is never shrunk toward zero.

Runtime is report-only. The runner enforces the disclosed hard caps—3,600 seconds
for `build.sh`, then 170 seconds for fit and 30 seconds for predict independently on
each dataset. Runtime headroom and host-normalized efficiency remain in diagnostics,
but neither multiplies scientific reward and no speed bonus exists. The wrapper
timeout is 14,400 seconds; the platform verifier timeout is 16,200 seconds.

The headline is category-equal and tail-aware:

```text
headline = 0.6 * mean(category scores)
         + 0.4 * mean(lowest ceil(0.20 * 15) category scores)
```

Thus the weakest three categories carry the tail term. The wrapper independently
reconstructs dataset, category, tail, and headline values from trusted detail and
rejects any disagreement.

## Native and platform score scales

The authoritative scientific scale is the signed native scale `[-0.5, 1.5]`, where
naive is 0 and reference is 1. Some platform ingestion paths clamp negative values,
so emitted `TestResult.score` values use the monotone affine transform

```text
platform_score = (native_score + 0.5) / 1.5
```

Consequently native `-0.5, 0, 1, 1.5` map to platform `0, 1/3, 1, 4/3`.
The native headline remains explicitly present as `native_score` in benchmark
output and is the value used for scientific analysis and mastery. Because the
transform is affine and category coefficients sum to one, transformed category
scores reproduce the transformed headline exactly.

## Mastery and pass@k

Continuous score and mastery are different claims. Schema v8 initially ships with
both `MASTERY_THRESHOLD` and `MASTERY_TAIL_THRESHOLD` set to `None`. In that state
the wrapper emits no mastery verdict and `validation/passk.py` refuses to report
pass@k. Best-score@k remains reportable.

Calibration and evaluation each use one complete Opus 4.8 launched run from the
exact released environment. Scores always require complete packages. Either phase
accepts retained-only terminal-failure packages solely to classify a non-completion;
they can never supply a score or threshold. Each accepted bundle is copied to
`validation/calibration_evidence/<bundle-sha256>/<rollout-id>.zip` and is reparsed by
the release validator:

1. Publish the fully locked **uncalibrated** schema-v8 tuple.
2. Run one `claude-opus-4-8` calibration run on that exact environment commit,
   retaining every declared attempt. It needs at least three non-infrastructure
   attempts and at least one gradable attempt.
   `validation/mastery_calibration.py:evidence_from_bundle` binds each row to the
   bundle bytes, benchmark output, run, rollout, model, and environment commit and
   verifies the native/platform mapping.
3. Run `mastery_calibration.py freeze` over every bundle in that run. It rejects
   omitted siblings/retry mixtures, retains failures, and derives both thresholds
   together from the best observed **gradable** row before writing the authenticated
   `threshold_frozen` artifact; callers cannot supply either threshold.
4. Copy the two printed values exactly into `grader.contract`, regenerate the
   release lock, commit, and publish that threshold-frozen tuple.
5. Run one disjoint Opus 4.8 evaluation run on the threshold-frozen
   environment commit, retaining a bundle for **every launched attempt**. The
   capability denominator must contain at least three attempts and at least one must
   be gradable. Then run `mastery_calibration.py finalize` with both bundle sets. It
   refuses calibration/evaluation overlap, a reused commit, a different model, or
   gradable evaluation output that did not carry the frozen thresholds. Every row
   binds `run.json.totalRollouts` and its exact ordered rollout IDs; omitting any
   sibling from an included run fails closed.
6. Commit the resulting `calibrated` artifact, regenerate/check the lock, republish,
   and estimate pass@k from the complete disjoint evaluation attempt set. Explicit
   stopped/fail/complete agent non-completions count at `SKILL_LO`; provider rate-limit deaths are
   infrastructure-censored and excluded. Calibration rollouts
   never enter the reported evaluation denominator.

A calibrated artifact requires one complete launched run per phase, at least three
non-infrastructure attempts and one gradable attempt in each, one exact environment commit per phase,
matching manifest/scoring/wrapper digests, unique
rollout ids and bundle bytes, retained content-addressed evidence, and disjoint
calibration/evaluation ids.

## Required build and evidence gates

No tuple ships until all of these are true:

- The prompt surfaces are generated from `tasks/prompt_spec.py` and are byte-current.
- The development study covers all 45 authenticated cells and every declared
  model-zoo gate passes: reference reliability, model diversity, ranking reversal,
  bounded redundancy, and non-saturation.
- Corpus build produces all 45 cells. Missing, stale, unexpected, or failed cells
  fail the build; finalization never accepts survivors.
- Every reference and runtime anchor is converged, finite, provenance-matched, and
  feasible under its hard cap. The selected reference method agrees with the
  recorded inner-validation winner.
- `validation/shipping_audit.json` is a passing final audit over the exact manifest
  and reproduces the shipped anchors.
- Python validation, wrapper TypeScript compilation, wrapper tests, security
  probes, preflight, and end-to-end checks pass on the declared environment shape.
- Final calibrated publication has at least one successful complete Opus 4.8 bundle
  from the candidate/threshold-frozen harness, manually inspected end to end. Gold
  predictions never substitute for this live harness QA.
- A reference-quality witness proves that native score 1 is achievable without
  hidden information. It is either a complete score-1 hfdev bundle or a public gold
  execution proof containing exact inspectable source and no publisher-supplied
  predictions. Trusted witness creation and validation execute that source over all
  45 datasets in the mandatory sandbox and score the exact emitted bytes; a
  publisher-authored prediction or reward claim is forbidden as evidence.
- Every scored rollout is retained as a full bundle containing run metadata,
  complete grading logs, prompt, transcript, and solution. Retained-only evidence
  is accepted solely for fail-closed terminal-attempt classification.

Exact provider-rate-limit attempts are infra-censored and excluded from capability
denominators. A non-gradable attempt counts as agent failure only when retained
metadata says `workerStatus=stopped`, `problemStatus=fail`, and
`currentPhase=complete`; unknown/error/setup/partial states fail validation rather
than bias capability down. Agent failures count at the score floor in best-score@k
and as non-mastery in pass@k.

## Atomic release gate

Run these commands only after the corpus, development report, and final shipping
audit have been regenerated and synced into the candidate tree. `<key>` is an
absolute path to the owner-controlled 32-byte corpus key.

```bash
python tasks/prompt_spec.py
python validation/release_gate.py prepare-uncalibrated --key-file <key>
```

Commit the generated `validation/mastery_calibration.json`, return to a clean
`main`, then create the authenticated **candidate** lock:

```bash
python validation/release_gate.py write-candidate-lock --key-file <key>
```

Commit `validation/release_lock.json`, return to a clean `main`, run the reproduction
check, and publish this candidate tuple so its identity is fixed before solvability QA:

```bash
python validation/release_gate.py check --key-file <key>
```

Build a public-only **gold execution request** containing the authenticated
`candidate_release_lock.json`, exact inspectable gold source, and
`gold_execution.json`. Do not include predictions, machine-authored reward claims,
or `reward_detail.json`. Trusted creation and every release check snapshot the exact
submission, execute fit/predict over every manifest row in the mandatory networkless
sandbox under the public phase caps, score the exact emitted `pred.csv` bytes, and
retain exit codes, timings, model/output hashes, and source identity. Every dataset,
headline, and tail must be at least `1 - 1e-6`; values above 1 are stronger valid
witnesses. Revalidation reruns the source and requires the model/output hashes to
reproduce; arbitrary adjacent prediction bytes are rejected.
The execution submission must contain exactly `fit`, `predict`, and `pgs_core.py`,
byte-identical to locked `gold/fit`, `gold/predict`, and `gold/pgs_core.py`; those
three authoritative files are part of candidate, production, and compatibility
digests.

Separately, before final publish, require at least one successful complete Opus 4.8
hfdev bundle on the candidate tuple and manually inspect its prompt, transcript,
solution, setup/agent/test logs, and all category results. That is the live
harness/grading QA witness; it need not score 1. A complete hfdev bundle that truly
scores at least `1 - 1e-6` can also serve as the score-1 witness. Every retained
member and the complete gold submission tree are hashed:

```bash
python validation/score1_witness.py create \
  --bundle <public-score1-gold.zip> \
  --key-file <key>
```

Commit `validation/score1_witness.json` and its content-addressed evidence bundle
without changing any candidate-tuple file. On clean `main`, transition to the final
**production** lock and reproduce it:

```bash
python validation/release_gate.py write-production-lock --key-file <key>
# commit validation/release_lock.json, then:
python validation/release_gate.py check --key-file <key>
```

Both write commands and `check` reject a dirty worktree or a branch other than
`main`. Candidate deployment is an explicit staging state which omits only the
not-yet-observable score-1 witness. Initial production uses the witnessed candidate.
Later mastery-only republishes may reuse that witness only because a compatibility
digest covers every locked byte while normalizing the two threshold assignments and
excluding the authenticated calibration artifact. Any grader, runner, prompt,
corpus, wrapper, or other source drift invalidates reuse. A full hfdev score-1
witness records its executed candidate commit. A gold proof instead embeds the
exact authenticated candidate lock and causally executes the locked public gold
fit and predict programs over all 45 datasets in the mandatory sandbox, retaining
exit, timing, model, and output identities. Neither witness is equated with the
later commit that adds the witness and production lock.

The two post-release calibration transitions are explicit commands. Repeat
`--bundle` once per launched attempt: a complete package for a gradable row, or a
retained-only package only for fail-closed terminal classification:

```bash
python validation/mastery_calibration.py freeze \
  --bundle <calibration-0.zip> \
  --bundle <calibration-1.zip> \
  --bundle <calibration-2.zip> \
  --key-file <key>
```

Copy the printed headline/tail thresholds exactly into `grader/contract.py`, write
and check a new candidate→production lock, and publish the threshold-frozen commit.
Every gradable calibration bundle must report authenticated production provenance;
all attempts must share one environment commit/model and all gradable rows one
executable tuple. After the disjoint
evaluation run, finalize from the same calibration packages plus the new packages:

```bash
python validation/mastery_calibration.py finalize \
  --calibration-bundle <calibration-0.zip> \
  --calibration-bundle <calibration-1.zip> \
  --calibration-bundle <calibration-2.zip> \
  --evaluation-bundle <evaluation-0.zip> \
  --evaluation-bundle <evaluation-1.zip> \
  --evaluation-bundle <evaluation-2.zip> \
  --key-file <key>
```

Pass every bundle from each included evaluation run, including terminal failures.
Gradable evaluation rows must share one distinct threshold-frozen executable tuple while
preserving the calibration phase's scientific tuple. Mixing candidate runs,
production tuples within a phase, omitting a declared `run.json` sibling, or a
science-changing overlay fails closed. Provider rate-limit errors are the sole
infra-censored failure signature; only stopped/fail/complete non-gradable attempts
are agent failures, and every unknown terminal state is rejected. The
earlier score-1 witness remains valid only when the score-1 compatibility digest
proves that the delta is confined to the mastery overlay.

Commit the content-addressed bundles and calibrated artifact, regenerate/check the
release lock, and publish the final calibrated tuple. Produce the deterministic
pass@k report only through the validator-backed artifact path:

```bash
python validation/mastery_calibration.py report --key-file <key>
```

This command HMAC-validates the calibrated artifact, reparses every locked bundle,
and constructs the denominator from all classified disjoint evaluation attempts; it does not
accept caller-created `Rollout` objects or a substitute threshold.

## Corpus-key boundary

Dataset identities, streams, anchors, reports, manifest, calibration overlay, and
release lock are authenticated from a private 32-byte key. Build and verification
commands take it via `--key-file <absolute path>`. Losing every owner copy makes the
corpus identities and signatures unrecoverable; exposing it permits regeneration
and forgery. HMAC protects the solver boundary and detects release drift, but it
does not protect against compromise of the private verifier repository or its
deployed key material.
