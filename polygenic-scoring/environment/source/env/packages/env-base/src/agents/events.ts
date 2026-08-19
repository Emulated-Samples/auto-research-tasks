import type { LogEventType } from "../telemetry/types.js";

export type ToolFamily = "shell" | "file" | "todo" | "mcp" | "generic";

export interface AgentEventPayload {
  type: LogEventType;
  message: string;
  data?: Record<string, unknown>;
}

function previewJson(value: unknown, limit = 200): string {
  const text = JSON.stringify(value);
  if (text.length <= limit) return text;
  return `${text.slice(0, limit)}...`;
}

function previewText(value: string, limit = 200): string {
  if (value.length <= limit) return value;
  return `${value.slice(0, limit)}...`;
}

/**
 * Canonical telemetry builders for coding-agent traces.
 *
 * Provider CLIs expose different protocol envelopes, but downstream trace
 * consumers should see the same stable fields: tool name, top-level
 * tool_use_id for pairing, optional tool_family for semantic routing, and
 * standardized shell output fields. Raw provider payloads are intentionally
 * not copied into product JSONL; enable coding-agent debug logs when the
 * byte-for-byte CLI protocol is needed for investigation.
 */
export function toolCallEvent(args: {
  tool: string;
  toolUseId?: string;
  input?: Record<string, unknown>;
  family?: ToolFamily;
  message?: string;
}): AgentEventPayload {
  const message =
    args.message ?? `${args.tool}: ${previewJson(args.input ?? {})}`;
  return {
    type: "tool_call",
    message,
    data: {
      tool: args.tool,
      tool_use_id: args.toolUseId,
      tool_family: args.family,
      input: args.input,
    },
  };
}

export function shellResultEvent(args: {
  toolUseId?: string;
  stdout: string;
  stderr?: string;
  interrupted?: boolean;
  exitCode?: number | null;
  status?: string;
  tool?: string;
  message?: string;
}): AgentEventPayload {
  const stdoutPreview = previewText(args.stdout);
  // TODO(trace-schema): consider renaming stdout/stderr to
  // output/error_output. Keep the current wire names for historical traces
  // and existing renderers until we can do the migration deliberately.
  return {
    type: "tool_result",
    message: args.message ?? `Result: ${stdoutPreview}`,
    data: {
      tool: args.tool,
      tool_use_id: args.toolUseId,
      stdout: args.stdout,
      stderr: args.stderr ?? "",
      interrupted: args.interrupted,
      exit_code: args.exitCode,
      status: args.status,
    },
  };
}

export function structuredResultEvent(args: {
  toolUseId?: string;
  result: unknown;
  tool?: string;
  message?: string;
}): AgentEventPayload {
  return {
    type: "tool_result",
    message: args.message ?? `Result: ${previewJson(args.result)}`,
    data: {
      tool: args.tool,
      tool_use_id: args.toolUseId,
      result: args.result,
    },
  };
}

export function agentTextEvent(text: string): AgentEventPayload {
  return {
    type: "agent_text",
    message: text,
    data: { text },
  };
}

export function agentReasoningEvent(text: string): AgentEventPayload {
  return {
    type: "agent_reasoning",
    message: text,
    data: { text },
  };
}
