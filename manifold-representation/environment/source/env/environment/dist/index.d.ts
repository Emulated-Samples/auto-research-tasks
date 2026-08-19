/** Hyperfocal adapter for manifold-bench's full nine-suite problem. */
import type { EnvironmentDefinition, Logger, Problem, TestResult } from "@hyperfocal/env-base";
declare class Environment implements EnvironmentDefinition {
    listProblems(): Promise<Problem[]>;
    setupProblem(problemId: string): Promise<void>;
    runTests(problemId: string, logger: Logger): Promise<TestResult[]>;
}
declare const _default: Environment;
export default _default;
//# sourceMappingURL=index.d.ts.map