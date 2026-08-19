/**
 * The "should we grade?" policy — the agent-exit-to-grading decision the old
 * environments.ts rollout handler carried inline, now a named seam (split
 * per TODO(maintainability), no behavior change).
 *
 * Policy: a clean agent exit grades; a TIMED-OUT agent still grades against
 * the current workspace state (partial credit is real signal — exit code
 * 124 is the `timeout(1)` convention and agent runners also flag timedOut
 * explicitly); any other non-zero exit is an agent-infrastructure failure
 * and must NOT grade — it throws upstream so the rollout records an agent
 * error, never a fake 0 score.
 */

export interface AgentExitLike {
  exitCode: number;
  timedOut?: boolean;
  pid?: number;
  exitReason?: string;
}

export interface GradingDecision {
  /** Whether the rollout should proceed to the test/grading phase. */
  runTests: boolean;
  /**
   * Present when grading proceeds despite an agent timeout: the warning
   * message + structured telemetry data the rollout must record on the
   * forced session end and the test session.
   */
  timeoutContinuation?: {
    message: string;
    data: Record<string, unknown>;
  };
}

/** Decide whether (and how) to grade after the agent process exited. */
export function decideGradingAfterAgentExit(
  agentResult: AgentExitLike
): GradingDecision {
  const timedOut = agentResult.timedOut || agentResult.exitCode === 124;
  if (agentResult.exitCode !== 0 && !timedOut) {
    return { runTests: false };
  }
  if (timedOut) {
    return {
      runTests: true,
      timeoutContinuation: {
        message: `Agent timed out with code ${agentResult.exitCode}; running tests against current state`,
        data: {
          agentExitCode: agentResult.exitCode,
          agentPid: agentResult.pid,
          agentExitReason: agentResult.exitReason,
          testsContinuedAfterAgentTimeout: true,
        },
      },
    };
  }
  return { runTests: true };
}
