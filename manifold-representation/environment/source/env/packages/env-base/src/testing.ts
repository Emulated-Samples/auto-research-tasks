/**
 * Testing utilities
 */

import type { BatchTest, Logger, SimpleTest, TestResult } from "./types.js";

/**
 * Patterns that indicate environment/infrastructure errors
 * These are issues outside the agent's control
 */
const ENVIRONMENT_ERROR_PATTERNS = [
  // AWS capacity/quota issues
  /insufficient.*capacity/i,
  /quota.*exceeded/i,
  /limit.*exceeded/i,
  /vcpu.*limit/i,
  /resource.*limit/i,
  // Timeouts
  /timeout/i,
  /timed out/i,
  // Service availability
  /service.*unavailable/i,
  /internal.*error/i,
  /throttl/i,
  /rate.*exceeded/i,
  /too many requests/i,
  // Network issues (excluding connection refused - often cascade failures from build errors)
  /network.*error/i,
  /etimedout/i,
  /enotfound/i,
  /socket hang up/i,
  // AWS-specific environment errors
  /no.*available.*capacity/i,
  /insufficient.*instance.*capacity/i,
  /request.*limit.*exceeded/i,
];

/**
 * Check if an error is an environment/infrastructure error
 * 
 * Environment errors are issues outside the agent's control:
 * - AWS quota/capacity limits
 * - Network timeouts
 * - Service unavailable
 * - Rate limiting
 * 
 * These should be marked as "errored" rather than "failed"
 */
export function isEnvironmentError(error: Error | string): boolean {
  const message = typeof error === "string" ? error : error.message;
  return ENVIRONMENT_ERROR_PATTERNS.some(pattern => pattern.test(message));
}

/**
 * Determine status from a SimpleTestResult, handling both scored (rubric or
 * signed-score) and binary (legacy) test results.
 *
 * Priority:
 * 1. Skipped -> "skipped" (excluded from scoring)
 * 2. Environment errors -> "errored"
 * 3. Score present -> map to passed/partially_passed/failed
 * 4. Legacy binary -> passed/failed
 *
 * Score-to-status mapping for scored tests:
 *   success === true  -> "passed" (test author signaled threshold pass;
 *                        eg. a rubric with score >= passThreshold)
 *   score >= 1.0      -> "passed"
 *   0 < score < 1.0   -> "partially_passed"
 *   score <= 0        -> "failed" (includes negative penalty scores; the
 *                        numerical sign and magnitude are preserved on the
 *                        TestResult.score field)
 *
 * TODO: Consider adding a "penalized" status for score < 0 to distinguish
 * "test failed normally" from "test detected a penalty condition (anti-cheat
 * fired, banned pattern observed, etc)." For now, score < 0 maps to "failed"
 * and the magnitude lives in the score field.
 */
function determineStatus(
  result: { success: boolean; error?: string; errored?: boolean; skipped?: boolean; score?: number }
): "passed" | "failed" | "errored" | "partially_passed" | "skipped" {
  if (result.skipped) {
    return "skipped";
  }

  // Environment errors take highest priority
  const isEnvError =
    result.errored || (result.error && isEnvironmentError(result.error));
  if (isEnvError) {
    return "errored";
  }

  // Score-based status mapping (scored tests, incl. rubric and penalty tests).
  // When the test author has explicitly set success=true alongside a score
  // (eg. a rubric where score >= passThreshold), honor that — the configured
  // threshold IS the pass/fail contract for that test. Without this, any
  // rubric scoring < 1.0 (essentially all of them) is forced to
  // "partially_passed" and the orchestrator treats it as a rollout failure
  // even when the score sat well above its threshold.
  if (result.score !== undefined) {
    if (result.success) return "passed";
    if (result.score >= 1.0) return "passed";
    if (result.score > 0) return "partially_passed";
    return "failed";
  }

  // Legacy binary path
  return result.success ? "passed" : "failed";
}

/**
 * Determine the score for a test result.
 *
 * Scored tests (incl. rubric and penalty tests) provide an explicit score,
 * which may be signed. Legacy binary tests get 1.0 (success) or 0.0 (fail).
 * Errored and skipped tests get 0 (and are excluded from aggregation anyway).
 */
function determineScore(
  result: { success: boolean; score?: number },
  status: "passed" | "failed" | "errored" | "partially_passed" | "skipped"
): number {
  if (result.score !== undefined) return result.score;
  if (status === "errored" || status === "skipped") return 0;
  return result.success ? 1.0 : 0.0;
}

/**
 * Run a set of tests and return results.
 *
 * Accepts both SimpleTest (single result per test) and BatchTest
 * (multiple results from one execution, e.g., subprocess test suites).
 *
 * Supports binary (pass/fail), scored (rubric), and batch tests.
 * All tests produce a universal score (0.0-1.0).
 */
export async function runSimpleTests(
  tests: (SimpleTest | BatchTest)[],
  logger: Logger
): Promise<TestResult[]> {
  const results: TestResult[] = [];

  for (const test of tests) {
    // BatchTest — returns multiple TestResults from a single execution
    if ("runBatch" in test) {
      logger.info(`\nRunning batch: ${test.name}`);
      const startTime = Date.now();

      try {
        const batchResults = await test.runBatch(logger);
        results.push(...batchResults);

        const batchDuration = Date.now() - startTime;
        const passed = batchResults.filter((r) => r.status === "passed").length;
        const failed = batchResults.filter(
          (r) => r.status === "failed" || r.status === "errored"
        ).length;

        if (failed > 0) {
          logger.error(
            `   ${passed} passed, ${failed} failed (${batchDuration}ms)`
          );
        } else {
          logger.info(
            `   ${passed} passed, ${failed} failed (${batchDuration}ms)`
          );
        }
      } catch (error) {
        const duration = Date.now() - startTime;
        const errorMessage =
          error instanceof Error ? error.message : String(error);
        const isEnvError = isEnvironmentError(errorMessage);
        const status = isEnvError ? ("errored" as const) : ("failed" as const);
        const statusLabel = isEnvError ? "ERROR" : "FAIL";

        logger.error(`${statusLabel} ${test.name} (${duration}ms)`);
        logger.error(`   Error: ${errorMessage}`);

        results.push({
          id: test.id,
          name: test.name,
          description: test.description,
          status,
          duration,
          score: 0,
          error: errorMessage,
        });
      }

      continue;
    }

    // SimpleTest — single result per test (existing behavior)
    logger.info(`\nRunning: ${test.name}`);
    const startTime = Date.now();

    try {
      const result = await test.run(logger);
      const duration = Date.now() - startTime;
      const status = determineStatus(result);
      const score = determineScore(result, status);

      if (status === "skipped") {
        logger.warn(`SKIP ${test.name} (${duration}ms)`);
        if (result.error) {
          logger.warn(`   Reason: ${result.error}`);
        }
      } else if (status === "passed") {
        logger.info(`PASS ${test.name} (${duration}ms)`);
      } else if (status === "partially_passed") {
        const pct = Math.round(score * 100);
        logger.info(`PARTIAL ${test.name}: ${pct}% (${duration}ms)`);
      } else if (status === "errored") {
        logger.error(`ERROR ${test.name} (${duration}ms)`);
        if (result.error) {
          logger.error(`   Error: ${result.error}`);
        }
      } else {
        logger.error(`FAIL ${test.name} (${duration}ms)`);
        if (result.error) {
          logger.error(`   Error: ${result.error}`);
        }
      }

      results.push({
        id: test.id,
        name: test.name,
        description: test.description,
        status,
        duration,
        score,
        // Runtime weight override (result.weight) takes precedence over
        // declaration-time SimpleTest.weight — lets severity-escalated
        // tests bump their own weight at run time (eg. a downtime test
        // that's mildly over threshold uses base weight, but escalates
        // when downtime is >> threshold).
        weight: result.weight ?? test.weight,
        error: result.error,
        rationale: result.rationale,
        criteriaResults: result.criteriaResults,
      });
    } catch (error) {
      const duration = Date.now() - startTime;
      const errorMessage = error instanceof Error ? error.message : String(error);

      // Caught exceptions are often environment errors (network, timeout, etc.)
      const isEnvError = isEnvironmentError(errorMessage);
      const status = isEnvError ? "errored" as const : "failed" as const;
      const statusLabel = isEnvError ? "ERROR" : "FAIL";

      logger.error(`${statusLabel} ${test.name} (${duration}ms)`);
      logger.error(`   Error: ${errorMessage}`);

      results.push({
        id: test.id,
        name: test.name,
        description: test.description,
        status,
        duration,
        score: 0,
        weight: test.weight,
        error: errorMessage,
      });
    }
  }

  return results;
}

/**
 * Per-test breakdown emitted by aggregateTestResults.
 */
export interface AggregatedScoreContribution {
  id: string;
  name: string;
  /** Effective weight used in aggregation (defaults to 1 when absent). */
  weight: number;
  /** Effective score used in aggregation (defaults to 0 when absent). */
  score: number;
  /** weight * score; the test's signed contribution to the numerator. */
  contribution: number;
}

/**
 * Aggregate score over a set of test results.
 *
 * Pure function — does not run tests, only scores already-computed TestResults.
 * The orchestrator's rollout path delegates to this; Tier-3 environments that
 * implement custom logic in their own runTests can call it directly.
 */
export interface AggregatedScore {
  /**
   * Signed weighted mean: sum(weight * score) / sum(weight). Can be < 0
   * when penalty tests fire. Not clamped to [0, 1] — the persisted value
   * carries the full signal; the categorical rolloutStatus is the
   * success/failure source of truth.
   */
  rolloutScore: number;
  /** Sum of weights of non-skipped, non-errored results. */
  weightTotal: number;
  /** Per-test breakdown for debugging / display. */
  contributions: AggregatedScoreContribution[];
}

/**
 * Compute the rollout score from a set of TestResults.
 *
 * Semantics:
 * - Results with status "skipped" or "errored" are excluded from both
 *   numerator and denominator. They appear in `contributions` with their
 *   effective weight/score so callers can see exclusion explicitly.
 * - For each remaining result:
 *     weight = result.weight ?? 1   (missing weight defaults to 1)
 *     score  = result.score  ?? 0   (missing score defaults to 0)
 *     contribution = weight * score
 *     weightTotal += weight
 * - rolloutScore = weightTotal > 0 ? sum(contribution) / weightTotal : 0.
 * - No clamping. rolloutScore can be negative when penalty tests dominate.
 *
 * Note: a "clean" anti-cheat-style test (e.g. weight: 5, score: 0) still
 * adds its weight to the denominator. This is intentional — the test
 * occupies importance in the suite regardless of outcome. Authors who want
 * an anti-cheat to be neutral when clean can either pick a small weight, or
 * model the test as a reward (score: 1 clean, score: -1 fired).
 */
export function aggregateTestResults(results: TestResult[]): AggregatedScore {
  const contributions: AggregatedScoreContribution[] = [];
  let numerator = 0;
  let weightTotal = 0;
  let statusOnlyPassIncluded = false;

  for (const result of results) {
    const isExcluded = result.status === "skipped" || result.status === "errored";
    const weight = result.weight ?? 1;
    // Defensive NaN coercion. A test that returns `score: NaN` (typically
    // from `parseInt('non-numeric', 10)` slipping past a missing isFinite
    // guard) would otherwise contaminate the rollup: `weight * NaN = NaN`
    // propagates through `numerator += contribution` and the final
    // `rolloutScore` is NaN — surfacing in JSON as `score: NaN%` which
    // skews score-distribution analysis and can hide real signal. The
    // upstream emitter is the right place to validate, but the aggregator
    // is the chokepoint; treating NaN as 0 here gives belt-and-suspenders
    // protection without affecting any well-formed test result.
    const hasExplicitScore =
      result.score !== undefined && Number.isFinite(result.score);
    // Status-only results (no score field — the normal output of batch /
    // subprocess runners) are scored from their status: passed counts as 1,
    // everything else as 0. `score ?? 0` here previously zeroed entire
    // rollouts that had hundreds of passing tests (the historical bug this
    // fork branch exists to keep dead).
    const score = hasExplicitScore
      ? (result.score as number)
      : result.status === "passed"
        ? 1
        : 0;
    const contribution = weight * score;

    contributions.push({
      id: result.id,
      name: result.name,
      weight,
      score,
      contribution,
    });

    if (isExcluded) continue;
    numerator += contribution;
    if (!hasExplicitScore && result.status === "passed" && weight > 0 && !isExcluded) {
      statusOnlyPassIncluded = true;
    }
    weightTotal += weight;
  }


  // Belt-and-braces guard against the status-only zeroing bug regressing:
  // a status-only passed result with weight > 0 contributes weight * 1 > 0,
  // so the numerator cannot be exactly 0 when one was included. Throw loudly
  // instead of persisting a false-negative rolloutScore of 0.
  if (statusOnlyPassIncluded && numerator === 0) {
    throw new Error(
      "aggregateTestResults: numerator 0 despite an included status-only " +
        "passed test — the status-only zeroing bug is back; refusing to " +
        "report a false 0."
    );
  }

  const rolloutScore = weightTotal > 0 ? numerator / weightTotal : 0;
  return { rolloutScore, weightTotal, contributions };
}
