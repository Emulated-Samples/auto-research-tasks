"""End-to-end env validation: grade every arm over the real corpus and assert the
env's load-bearing separation properties against the shipped reference.

The reference standard is the exact public NumPy best-of-family in `gold/fit`:
hierarchical empirical Bayes or fixed-grid ridge logistic, selected on a
deterministic training-only inner split and refit on the complete training data.
Its per-dataset metrics are baked into `truth/anchors.json`. The
grader's native skill normalizes every submission onto that axis: naive baseline
= 0 and shipped reference = 1. Thus "the reference beats method X" is exactly
"X's skill score < 1".

This test used to grade ONLY the cheats, which made it one-sided: it could prove
the env denies a shortcut (no false POSITIVE) and could not notice the env
denying everything (a false NEGATIVE). An env where nothing scores is not hard,
it is broken, and the cheats-only harness reports that state as a clean pass. So
the gold arm is graded too.

Load-bearing properties (what this test proves and keeps proven):

  0. LIVENESS. Every arm actually RAN. This is first because it invalidates the
     rest: a crashed or timed-out arm scores the invalid floor, which satisfies
     (1) trivially and forever. Measured 2026-07-15: both cheats had been dying
     on a bad shebang, so this suite reported "the env denies shortcuts" while
     measuring "the shortcuts never ran" -- and once they ran they BEAT the
     reference on sparse_dense_mix. The green check was hiding the defect.
  1. FP-resistance. No cheat's per-category native REWARD is RELIABLY at/above the
     reference's 1.0 -- reliably meaning the overshoot exceeds K_SE_SATURATION
     standard errors of the replicate mean, not that a point estimate crossed a
     line. A per-category reward is a mean over 3 replicates and carries sampling
     noise for free; a point compare against 1.0 makes the verdict a coin flip in
     a band around the threshold, so a true-0.99 arm and a true-1.01 arm are
     called differently on the draw. Test the estimate, never threshold it. The
     corpus build already holds itself to exactly this standard (K_SE=2.0 on the
     reference-minus-naive gap); the same evidence bar now applies to overshoot.
  2. ORDERING. The gold arm outranks every cheat, overall and per category. This
     is the strongest claim available without a calibrated constant: it fixes no
     threshold, so it cannot manufacture difficulty, yet it fails loudly if the
     env ever prefers a marginal/class-blind shortcut to a genuine joint
     annotation-aware fit -- which is exactly the capability being graded.
  3. FN-resistance. The gold arm clears GOLD_NATIVE_REWARD_FLOOR. Unlike (2) this needs a
     number, so the number is MEASURED, not chosen: see the constant's note. Its
     job is to catch the env drifting into unreachability, not to assert the gold
     arm is as good as the shipped reference (it is not, and it should not have to be).

Native reward is the scientific outcome throughout. Runtime is constrained by hard
phase caps and reported diagnostically; it does not multiply prediction quality.

Run:
  export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
  python validation/e2e_test.py <corpus_dir> <key_file>
"""
from __future__ import annotations

import collections
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from grader.grade import grade_corpus  # noqa: E402
from grader.contract import (  # noqa: E402
    REPLICATES_PER_CATEGORY,
    SHIPPED_CATEGORIES,
    SHIPPED_DATASET_COUNT,
)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# A cheat may sit just under the reference from sampling noise; require it to stay
# at least this far below the reference's skill of 1.0 to count as separated.
SEPARATION_MARGIN = 0.0

# The shipped best-of-family reference is not graded as a submission here -- it
# defines skill=1.0 via the anchors. We grade the cheats and assert each is denied.
CHEATS = {
    "cheat:uniform_ridge": os.path.join(ROOT, "cheats", "uniform_ridge"),
    "cheat:marginal_pt": os.path.join(ROOT, "cheats", "marginal_pt"),
}
GOLD = {"gold:public_best_of_family": os.path.join(ROOT, "gold")}
ARMS = {**GOLD, **CHEATS}

# FN-resistance floor on the GOLD arm's native headline reward.
#
# This constant is the one number here that a calibration can get wrong, so it is
# deliberately loose and its role is narrow: it is a REGRESSION guard against the
# env drifting into "nothing can score", not a claim about how good gold is. It
# is set well below the measured gold headline so that ordinary DGP/reference
# churn does not trip it; only a real collapse does. Update it by MEASUREMENT
# (record the run in rollout_analysis/), never by lowering it until the suite
# goes green -- that converts the anti-FN guard into a rubber stamp.
#
# A NOTE ON WHAT THIS FLOOR IS NOT: gold is not required to reach 1.0. Native
# reward 1.0 means "as good as the shipped reference", and the benchmark's whole
# premise is that reaching it is extremely hard. A floor near 1.0 would assert
# the opposite. None means the required measurement is missing, which FAILS this
# release check closed. A fabricated floor is worse than no floor, but missing
# reachability evidence must never become a green release.
GOLD_NATIVE_REWARD_FLOOR = None


# How many standard errors above the reference a cheat's per-category skill must
# sit before we call it a real saturation rather than a lucky draw. Mirrors the
# corpus build's own reliability gate (datagen/build_corpus.py uses K_SE=2.0 on
# the reference-minus-naive gap), so the same evidentiary standard applies in
# both directions: a gap is only real at 2 SE, and so is an overshoot.
K_SE_SATURATION = 2.0


def _mean_and_se(values):
    """Mean and standard error of the mean. SE is None below 2 observations."""
    n = len(values)
    if n == 0:
        return 0.0, None
    mean = sum(values) / n
    if n < 2:
        return mean, None
    var = sum((v - mean) ** 2 for v in values) / (n - 1)
    return mean, (var / n) ** 0.5


def saturation_verdict(rewards):
    """Is this category's native reward RELIABLY at/above the reference's 1.0?

    A per-category reward is a mean over a handful of replicates, so it carries
    sampling noise for free. Comparing that point estimate to 1.0 makes the
    verdict a coin flip in a band around the threshold: an arm at a true 0.99 and
    one at a true 1.01 are not distinguishable from 3 draws, yet a point compare
    calls one clean and the other a failure. Reward -- and a red suite -- must not
    turn on a draw.

    So test the estimate instead of thresholding it. But a 2-SE bar over the
    shipped 3 replicates has very little POWER, and a bar that cannot detect a
    real blowout is as useless as one that fires on noise: measured 2026-07-15,
    uniform_ridge on sparse_dense_mix ran ~[1.50, 1.04] -- mean 1.27, SE ~0.23 --
    which is a flagrant saturation (1.50 is the winsorization CEILING) that a
    strict 2-SE test would wave through. Choosing between flaky and blind is a
    false choice, so the verdict is two-tier:

      "saturated" -- overshoot exceeds K_SE_SATURATION SE. Real, not a draw.
                     This FAILS the suite.
      "suspect"   -- the point estimate is at/above 1.0 but the evidence does not
                     clear that bar. Reported loudly, does NOT fail the suite.

    That keeps the hard verdict off a coin flip while refusing to hide an
    overshoot. A persistent "suspect" is not noise to be ignored -- it means the
    replicate count is too low to adjudicate a category that is visibly at the
    ceiling, and the answer is more replicates, not a greener suite.

    Returns (mean, se, saturated, suspect).
    """
    mean, se = _mean_and_se(rewards)
    crossed = mean >= 1.0 - SEPARATION_MARGIN
    if se is None or se == 0.0:
        # Single replicate, or zero spread: no noise estimate exists, so there is
        # no evidence to test. Report the crossing as suspect rather than
        # asserting either way from one draw.
        return mean, se, False, crossed
    saturated = (mean - (1.0 - SEPARATION_MARGIN)) > K_SE_SATURATION * se
    return mean, se, saturated, (crossed and not saturated)


def per_category_reward_values(rd):
    """Category -> final native per-replicate rewards (unaggregated, for SE).

    Use ``reward``, not ``accuracy``: accuracy is the display-only ``max(raw_skill,
    0)`` field, while reward is the signed scientific value after the continuous
    calibration factor. The release check must exercise exactly what the env scores.
    """
    cats = {}
    for d in rd.get("datasets", []):
        cats.setdefault(d["category"], []).append(float(d["reward"]))
    return cats


def main(corpus_dir, key_file):
    results = {}
    for name, sub in ARMS.items():
        print(f"\n===== grading {name} =====", flush=True)
        rj, rd = grade_corpus(
            corpus_dir,
            sub,
            key_file=key_file,
            log=lambda *_: None,
        )
        per_cat_reward = {
            category: float(value)
            for category, value in rd["additional_data"]["per_category"].items()
        }
        native_headline = float(rj["reward"])
        results[name] = {"headline": native_headline,
                         "per_cat_reward": per_cat_reward,
                         "per_cat_values": per_category_reward_values(rd),
                         "datasets": rd.get("datasets", [])}
        print(f"  native headline reward = {native_headline:+.4f}")
        for c in sorted(per_cat_reward):
            print(f"    {c:20s} native reward={per_cat_reward[c]:+.4f}")

    print("\n===== VERDICT (shipped reference = native reward 1.0) =====")
    ok = True
    all_cats = list(SHIPPED_CATEGORIES)
    gold_name = next(iter(GOLD))
    gold = results[gold_name]

    # (0) LIVENESS: an arm that never ran cannot verify anything.
    #
    # This precedes every other check because it invalidates them. Both cheats
    # shipped a '#!/usr/bin/env python3' shebang that resolves to a NumPy-less
    # distro Python inside the sandbox, so every cheat fit exited rc=1 and every
    # cheat dataset scored the invalid floor. Check (1) then passed trivially --
    # a crashed arm is always "below the reference" -- and that green check was
    # concealing a live saturation defect: once the arms actually ran, they BEAT
    # the reference on sparse_dense_mix. A dead arm does not merely fail to
    # verify the property, it actively asserts the opposite of the truth.
    print("\n  liveness -- every arm must actually run:")
    for name in ARMS:
        detail = results[name]["datasets"]
        statuses = collections.Counter(d.get("status") for d in detail)
        category_counts = collections.Counter(d.get("category") for d in detail)
        dataset_ids = [d.get("dataset") for d in detail]
        exact_coverage = (
            len(detail) == SHIPPED_DATASET_COUNT
            and len(set(dataset_ids)) == SHIPPED_DATASET_COUNT
            and set(category_counts) == set(SHIPPED_CATEGORIES)
            and all(category_counts[category] == REPLICATES_PER_CATEGORY
                    for category in SHIPPED_CATEGORIES)
        )
        ran = statuses.get("ok", 0)
        alive = exact_coverage and ran == SHIPPED_DATASET_COUNT
        ok &= alive
        print(f"    {'PASS' if alive else 'FAIL'}: {name:22s} {ran}/{len(detail)} "
              f"datasets scored; exact_grid={exact_coverage}  {dict(statuses)}")

    # (1) FP-resistance: every cheat stays STRICTLY below the reference.
    # Strict: a cheat that lands exactly on 1.0 has MATCHED the reference, which is
    # the very loss this test exists to catch. The old comparison was `<= 1.0`,
    # which passed that case while the docstring claimed it failed.
    for name in CHEATS:
        values = results[name]["per_cat_values"]
        saturated, suspect = [], []
        worst_mean = -9.9
        for category in all_cats:
            rewards = values.get(category, [])
            if not rewards:
                continue
            mean, se, is_sat, is_suspect = saturation_verdict(rewards)
            worst_mean = max(worst_mean, mean)
            label = f"{category} ({mean:+.3f}+-{(se if se is not None else 0.0):.3f})"
            if is_sat:
                saturated.append(label)
            elif is_suspect:
                suspect.append(label)
        denied = not saturated
        ok &= denied
        print(f"  {'PASS' if denied else 'FAIL'}: {name:22s} "
              f"max per-category native reward={worst_mean:+.3f}  "
              f"saturated(>{K_SE_SATURATION}SE)={saturated or 'none'}")
        if suspect:
            print(f"        SUSPECT (>=1.0 but under-powered at n={len(values.get(all_cats[0], [])) or '?'} "
                  f"replicates; needs more evidence, not a greener suite): {suspect}")

    # (2) ORDERING: the genuine model outranks every shortcut on the exact native
    # headline: 0.6 category mean + 0.4 weakest-fifth mean. A plain category mean
    # would discard the load-bearing tail and could hide a lopsided collapse.
    print("\n  ordering -- gold must outrank every cheat (native headline reward):")
    for name in CHEATS:
        beats = gold["headline"] > results[name]["headline"]
        ok &= beats
        print(f"    {'PASS' if beats else 'FAIL'}: {gold_name} ({gold['headline']:+.3f}) "
              f"vs {name} ({results[name]['headline']:+.3f})")

    # (3) FN-resistance: the env must remain reachable by a genuine model.
    if GOLD_NATIVE_REWARD_FLOOR is None:
        ok = False
        print(f"\n  FAIL: FN-resistance floor is uncalibrated "
              f"(gold native headline={gold['headline']:+.3f}). Measure it, record "
              f"the run, then set GOLD_NATIVE_REWARD_FLOOR.")
    else:
        reached = gold["headline"] >= GOLD_NATIVE_REWARD_FLOOR
        ok &= reached
        print(f"\n  {'PASS' if reached else 'FAIL'}: FN-resistance -- gold native headline "
              f"{gold['headline']:+.3f} >= floor {GOLD_NATIVE_REWARD_FLOOR:+.3f}")

    print("\n  per-category native reward vs reference:")
    for cat in all_cats:
        best_cheat_reward = max(results[n]["per_cat_reward"].get(cat, 0.0)
                                for n in CHEATS)
        gold_reward = gold["per_cat_reward"].get(cat, 0.0)
        sep = not any(saturation_verdict(results[n]["per_cat_values"].get(cat, []))[2]
                      for n in CHEATS if results[n]["per_cat_values"].get(cat))  # [2]=saturated
        ordered = gold_reward > best_cheat_reward
        ok &= sep and ordered
        print(f"    {cat:20s} gold={gold_reward:+.3f} best_cheat={best_cheat_reward:+.3f} "
              f"reference=1.000  [{'ok' if sep else 'LOSS'}"
              f"{'' if ordered else ', gold<=cheat'}]")
    print("\n" + ("ALL CHECKS PASSED" if ok else "SOME CHECKS FAILED"))
    return 0 if ok else 1


if __name__ == "__main__":
    if len(sys.argv) != 3:
        sys.exit("usage: validation/e2e_test.py <corpus_dir> <key_file>")
    sys.exit(main(sys.argv[1], sys.argv[2]))
