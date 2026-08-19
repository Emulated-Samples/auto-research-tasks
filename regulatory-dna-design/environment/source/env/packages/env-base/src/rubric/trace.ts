/**
 * Agent trace preprocessing for rubric evaluation.
 *
 * Raw JSONL agent traces are ~87% tool results (full file contents from
 * when the agent reads code). LLM judges conflate "reading code about X"
 * with "reasoning about X", causing systematic false positives.
 *
 * This module provides modes for condensing traces before sending to the judge:
 *
 * - "full": Raw JSONL as-is. Useful for debugging, but causes false positives.
 * - "reasoning": Only agent reasoning text. Ultra-minimal, may miss action context.
 * - "summary": Agent reasoning + tool call summaries + truncated results.
 *              Best balance of signal and noise. Default.
 */

export type TraceMode = "full" | "reasoning" | "summary";

export interface TracePreprocessOptions {
  /**
   * How to process the trace.
   * - "full": Pass raw JSONL unchanged
   * - "reasoning": Extract only agent reasoning text (agent_text events)
   * - "summary": Agent reasoning + tool call descriptions + truncated tool results
   * @default "summary"
   */
  mode?: TraceMode;

  /**
   * Maximum characters to keep from each tool result in "summary" mode.
   * Higher values preserve more test output but risk leaking file contents
   * to the judge. Set to 0 to omit tool results entirely.
   * @default 800
   */
  resultMaxLength?: number;

  /**
   * Maximum characters to keep from each tool call description.
   * @default 300
   */
  actionMaxLength?: number;
}

const DEFAULT_RESULT_MAX_LENGTH = 800;
const DEFAULT_ACTION_MAX_LENGTH = 300;

/**
 * Preprocess a raw JSONL agent trace for LLM judge evaluation.
 *
 * @param rawJsonl - The raw JSONL trace content
 * @param options - Preprocessing options
 * @returns Processed trace string ready for the judge
 */
export function preprocessTrace(
  rawJsonl: string,
  options?: TracePreprocessOptions,
): string {
  const mode = options?.mode ?? "summary";

  if (mode === "full") {
    return rawJsonl;
  }

  const resultMaxLen = options?.resultMaxLength ?? DEFAULT_RESULT_MAX_LENGTH;
  const actionMaxLen = options?.actionMaxLength ?? DEFAULT_ACTION_MAX_LENGTH;
  const lines = rawJsonl.trim().split("\n");
  const sections: string[] = [];

  for (const line of lines) {
    let evt: Record<string, unknown>;
    try {
      evt = JSON.parse(line);
    } catch {
      continue;
    }

    const type = evt.type as string;
    const message = (evt.message as string) || "";
    const data = (evt.data as Record<string, unknown>) || {};

    switch (type) {
      case "agent_text": {
        const text = (data.text as string) || message;
        if (text.trim()) {
          sections.push(`[REASONING] ${text.trim()}`);
        }
        break;
      }
      case "tool_call": {
        if (mode === "summary") {
          const desc = message.slice(0, actionMaxLen);
          sections.push(
            `[ACTION] ${desc}${message.length > actionMaxLen ? "..." : ""}`,
          );
        }
        break;
      }
      case "tool_result": {
        if (mode === "summary" && resultMaxLen > 0) {
          let result: string;
          if (typeof data.result === "string") {
            result = data.result;
          } else if (typeof message === "string" && message.length > 0) {
            result = message;
          } else {
            result = JSON.stringify(data.result ?? "");
          }
          const snippet = result.slice(0, resultMaxLen).replace(/\n/g, " ");
          sections.push(
            `[RESULT] ${snippet}${result.length > resultMaxLen ? "..." : ""}`,
          );
        }
        break;
      }
      case "session_start":
      case "session_end": {
        if (mode === "summary") {
          sections.push(`[${type.toUpperCase()}] ${message}`);
        }
        break;
      }
    }
  }

  return sections.join("\n\n");
}
