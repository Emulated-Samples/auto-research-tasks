import type { BatchTest, SimpleTest } from "@hyperfocal/env-base";
import { pytestSuite } from "./pytest-suite.js";
import { antiCheatTests } from "./anti-cheat.js";

/**
 * Assemble the grader item list for a problem: the hidden host-side pytest suite plus the
 * anti-cheat catastrophe gates (weight-0 `cat-*` items; applyCatastropheMultiplier scales the
 * suite score when one fires). problemId is accepted for future per-tier composition
 * (guided / standard / minimal).
 */
export function getTestsForProblem(
  _problemId: string
): Array<SimpleTest | BatchTest> {
  return [pytestSuite, ...antiCheatTests];
}
