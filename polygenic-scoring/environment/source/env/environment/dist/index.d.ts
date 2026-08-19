/**
 * svpgsbench Hyperfocal environment.
 *
 * The solver agent builds a from-scratch polygenic-scoring library (./fit,
 * ./predict, optional ./build.sh) in its workspace. runTests grades it with the
 * hidden Python grader (repo grader/) over the dataset corpus, and reports one
 * continuous-score TestResult per capability category. The grader supplies the
 * exact category coefficients for its mean/lower-tail headline, and Hyperfocal's
 * weighted aggregation therefore reproduces the hardened reward exactly.
 *
 * Implements EnvironmentDefinition from @hyperfocal/env-base.
 */
import { EnvironmentDefinition, Logger, TestResult } from "@hyperfocal/env-base";
declare class Environment implements EnvironmentDefinition {
    listProblems(): Promise<import("@hyperfocal/env-base").Problem[]>;
    setupProblem(problemId: string): Promise<void>;
    runTests(problemId: string, logger: Logger): Promise<TestResult[]>;
    private gradeInto;
}
declare const _default: Environment;
export default _default;
