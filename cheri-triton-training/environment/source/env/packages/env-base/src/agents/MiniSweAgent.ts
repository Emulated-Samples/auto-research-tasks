/**
 * Mini SWE Agent
 *
 * Wraps mini-swe-agent's local CLI in headless yolo mode. The orchestrator
 * owns credential allowlisting and process isolation before this class runs.
 */

import { ChildProcess, execSync } from "child_process";
import * as fs from "fs";
import * as path from "path";
import type { AgentRunOptions, MiniSweAgentConfiguration } from "../types.js";
import type { TelemetrySession } from "../telemetry/index.js";
import type { LogEventType } from "../telemetry/types.js";
import {
  agentTextEvent,
  shellResultEvent,
  toolCallEvent,
  type AgentEventPayload,
} from "./events.js";
import { runStreamingCli } from "./streaming-cli.js";

type JsonRecord = Record<string, unknown>;

interface MiniAction {
  command: string;
  toolCallId?: string;
}

interface PendingToolCall {
  command: string;
  toolCallId: string;
}

interface MiniSweCommand {
  command: string;
  argsPrefix: string[];
  displayName: string;
}

function isJsonRecord(value: unknown): value is JsonRecord {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

function which(command: string): string | undefined {
  try {
    const result = execSync(`which ${JSON.stringify(command)}`, {
      encoding: "utf-8",
      stdio: ["pipe", "pipe", "pipe"],
    }).trim();
    return result || undefined;
  } catch {
    return undefined;
  }
}

function resolveMiniSweCommand(): MiniSweCommand {
  const configured = process.env.HYPERFOCAL_MINI_SWE_AGENT_BIN;
  if (configured) {
    if (configured.includes(path.sep)) {
      if (fs.existsSync(configured)) {
        return { command: configured, argsPrefix: [], displayName: configured };
      }
      throw new Error(`HYPERFOCAL_MINI_SWE_AGENT_BIN does not exist: ${configured}`);
    }
    const resolved = which(configured);
    if (resolved) return { command: resolved, argsPrefix: [], displayName: resolved };
    throw new Error(`HYPERFOCAL_MINI_SWE_AGENT_BIN was not found on PATH: ${configured}`);
  }

  for (const candidate of ["mini-swe-agent", "mini"]) {
    const resolved = which(candidate);
    if (resolved) return { command: resolved, argsPrefix: [], displayName: resolved };
  }

  const uvx = which("uvx");
  if (uvx) {
    return {
      command: uvx,
      argsPrefix: ["mini-swe-agent"],
      displayName: `${uvx} mini-swe-agent`,
    };
  }

  throw new Error(
    "mini-swe-agent CLI not found. Checked: HYPERFOCAL_MINI_SWE_AGENT_BIN, `mini-swe-agent`, `mini`, and `uvx`.\n" +
    "Install with: pip install mini-swe-agent, or install uv and run through uvx."
  );
}

function preview(text: string, length = 200): string {
  return text.length <= length ? text : `${text.slice(0, length)}...`;
}

function contentToText(content: unknown): string {
  if (typeof content === "string") return content;
  if (Array.isArray(content)) {
    return content.map(contentToText).filter(Boolean).join("\n");
  }
  if (isJsonRecord(content)) {
    const text = content.text;
    if (typeof text === "string") return text;
    return JSON.stringify(content);
  }
  return content == null ? "" : String(content);
}

function parseTextActions(content: string): MiniAction[] {
  const actions: MiniAction[] = [];
  const re = /```(?:mswea_bash_command|bash)?\s*\n([\s\S]*?)\n```/g;
  let match: RegExpExecArray | null;
  while ((match = re.exec(content)) != null) {
    const command = match[1]?.trim();
    if (command) actions.push({ command });
  }
  return actions;
}

function extractActions(message: JsonRecord): MiniAction[] {
  const extra = isJsonRecord(message.extra) ? message.extra : {};
  const rawActions = Array.isArray(extra.actions) ? extra.actions : [];
  const actions = rawActions
    .filter(isJsonRecord)
    .map((action) => {
      const command = typeof action.command === "string" ? action.command : "";
      const toolCallId = typeof action.tool_call_id === "string"
        ? action.tool_call_id
        : undefined;
      return { command, toolCallId };
    })
    .filter((action) => action.command);

  if (actions.length > 0) return actions;
  return parseTextActions(contentToText(message.content));
}

function parseTaggedContent(content: string): { stdout: string; exitCode?: number | null } {
  const outputMatch = content.match(/<output>\s*([\s\S]*?)\s*<\/output>/);
  const returnCodeMatch = content.match(/<returncode>\s*(-?\d+)\s*<\/returncode>/);
  const exitCode = returnCodeMatch ? Number(returnCodeMatch[1]) : undefined;
  return {
    stdout: outputMatch ? outputMatch[1] : content,
    exitCode,
  };
}

function extractObservation(message: JsonRecord): {
  stdout: string;
  exitCode?: number | null;
  status?: string;
  toolCallId?: string;
} {
  const extra = isJsonRecord(message.extra) ? message.extra : {};
  const stdout = typeof extra.raw_output === "string"
    ? extra.raw_output
    : parseTaggedContent(contentToText(message.content ?? message.output)).stdout;
  const tagged = parseTaggedContent(contentToText(message.content ?? message.output));
  const rawReturnCode = extra.returncode;
  const exitCode = typeof rawReturnCode === "number" ? rawReturnCode : tagged.exitCode;
  const toolCallId = typeof message.tool_call_id === "string"
    ? message.tool_call_id
    : typeof message.call_id === "string"
      ? message.call_id
      : undefined;
  return {
    stdout,
    exitCode,
    status: typeof extra.exception_info === "string" && extra.exception_info
      ? "error"
      : undefined,
    toolCallId,
  };
}

function loadTrajectoryMessages(trajectoryPath: string): JsonRecord[] {
  const parsed = JSON.parse(fs.readFileSync(trajectoryPath, "utf-8")) as unknown;
  const messages = Array.isArray(parsed)
    ? parsed
    : isJsonRecord(parsed) && Array.isArray(parsed.messages)
      ? parsed.messages
      : [];
  return messages.filter(isJsonRecord);
}

export class MiniSweAgent {
  private config: MiniSweAgentConfiguration;
  private session: TelemetrySession | null = null;
  private cliProcess: ChildProcess | null = null;
  private miniSweCommand: MiniSweCommand;

  constructor(config: MiniSweAgentConfiguration) {
    this.config = config;
    this.miniSweCommand = resolveMiniSweCommand();
    this.log("info", `MiniSweAgent initialized (model: ${config.model}, command: ${this.miniSweCommand.displayName})`);
  }

  setTelemetrySession(session: TelemetrySession): void {
    this.session = session;
  }

  private log(
    type: LogEventType,
    message: string,
    data?: Record<string, unknown>
  ): void {
    if (this.session) {
      this.session.log(type, message, data);
    } else {
      console.log(message);
    }
  }

  private logError(message: string, data?: Record<string, unknown>): void {
    if (this.session) {
      this.session.log("error", message, data);
    } else {
      console.error(message);
    }
  }

  private emit(event: AgentEventPayload): void {
    this.log(event.type, event.message, event.data);
  }

  private runtimeDir(): string {
    const homeDir = process.env.HOME || "/root";
    const dir = path.join(
      homeDir,
      ".local/share/mini-swe-agent/runtime/hyperfocal",
      `${Date.now()}-${process.pid}`,
    );
    fs.mkdirSync(dir, { recursive: true });
    return dir;
  }

  private buildCLIArgs(prompt: string, trajectoryPath: string): string[] {
    const args = [
      "--model", this.config.model,
      "--task", prompt,
      "--yolo",
      "--exit-immediately",
      "--output", trajectoryPath,
      "--config", this.config.configPath || "mini.yaml",
    ];

    if (this.config.costLimit !== undefined) {
      args.push("--cost-limit", String(this.config.costLimit));
    }
    if (this.config.stepLimit !== undefined) {
      args.push("--config", `agent.step_limit=${this.config.stepLimit}`);
    }

    return args;
  }

  async run(
    prompt: string,
    workingDir: string,
    options?: AgentRunOptions
  ): Promise<void> {
    if (!prompt?.trim()) {
      throw new Error("Prompt must be a non-empty string");
    }

    this.log("info", `Using mini-swe-agent CLI (${this.config.model})`);
    this.session?.setResolvedModel(this.config.model);
    this.log(
      "info",
      `Prompt: ${preview(prompt, 100)}`,
      { promptLength: prompt.length }
    );
    this.log("info", `Working directory: ${workingDir}`);

    if (options?.schemaPath) {
      this.log("warn", `mini-swe-agent does not support CLI schema enforcement; relying on prompt/test validation`, {
        schemaPath: options.schemaPath,
      });
    }
    if (this.config.mcpConfigPath && fs.existsSync(this.config.mcpConfigPath)) {
      this.log("warn", `mini-swe-agent does not support MCP config; ignoring ${this.config.mcpConfigPath}`, {
        mcpConfigPath: this.config.mcpConfigPath,
      });
    }

    const runtimeDir = this.runtimeDir();
    const trajectoryPath = path.join(runtimeDir, "trajectory.traj.json");
    const globalConfigDir = path.join(runtimeDir, "config");
    fs.mkdirSync(globalConfigDir, { recursive: true });
    const cliArgs = [
      ...this.miniSweCommand.argsPrefix,
      ...this.buildCLIArgs(prompt, trajectoryPath),
    ];
    const cliEnv: NodeJS.ProcessEnv = {
      ...process.env,
      HOME: process.env.HOME || "/root",
      MSWEA_CONFIGURED: "true",
      MSWEA_SILENT_STARTUP: "1",
      MSWEA_GLOBAL_CONFIG_DIR: globalConfigDir,
      MSWEA_MODEL_NAME: this.config.model,
    };
    if (this.config.model.startsWith("openrouter/") && !cliEnv.MSWEA_COST_TRACKING) {
      cliEnv.MSWEA_COST_TRACKING = "ignore_errors";
    }

    this.log("info", "Starting mini-swe-agent CLI", {
      args: cliArgs,
      binary: this.miniSweCommand.displayName,
      trajectoryPath,
    });

    let result: Awaited<ReturnType<typeof runStreamingCli>>;
    try {
      result = await runStreamingCli({
        command: this.miniSweCommand.command,
        args: cliArgs,
        cwd: workingDir,
        stdin: "ignore",
        env: cliEnv,
        debugLogStream: this.createDebugLogStream(),
        onProcess: (child) => {
          this.cliProcess = child;
        },
        onStdoutLine: (line) => {
          if (line.trim()) this.log("info", line);
        },
        onStderrText: (text) => {
          if (text.includes("Error") || text.includes("error") || text.includes("Traceback")) {
            this.logError(`CLI stderr: ${text.trim()}`);
          }
        },
      });
    } catch (error) {
      const message = `Failed to start mini-swe-agent CLI: ${error instanceof Error ? error.message : String(error)}`;
      this.logError(`Error: ${message}`);
      throw new Error(message);
    }

    if (fs.existsSync(trajectoryPath)) {
      this.replayTrajectory(trajectoryPath);
    }

    if (result.code !== 0 || result.signal) {
      let errorMessage = result.signal
        ? `mini-swe-agent CLI terminated with signal ${result.signal}`
        : `mini-swe-agent CLI exited with code ${result.code}`;
      if (result.stderrChunks.length > 0) {
        errorMessage += `\nStderr: ${result.stderrChunks.join("").slice(-1000)}`;
      }
      this.logError(`Error: ${errorMessage}`);
      throw new Error(errorMessage);
    }

    if (!fs.existsSync(trajectoryPath)) {
      throw new Error(`mini-swe-agent completed without writing trajectory: ${trajectoryPath}`);
    }

    this.log("info", "mini-swe-agent CLI completed", {
      exitCode: result.code,
      completionSource: result.completionSource,
      trajectoryPath,
    });
  }

  private createDebugLogStream(): fs.WriteStream | undefined {
    if (!this.config.codingAgentDebugLogs || !this.session) return undefined;
    return fs.createWriteStream(this.session.getDebugLogPath(), { flags: "a" });
  }

  private replayTrajectory(trajectoryPath: string): void {
    let messages: JsonRecord[];
    try {
      messages = loadTrajectoryMessages(trajectoryPath);
    } catch (error) {
      this.logError(`Failed to parse mini-swe-agent trajectory: ${error instanceof Error ? error.message : String(error)}`, {
        trajectoryPath,
      });
      return;
    }

    const pending: PendingToolCall[] = [];
    let generatedId = 0;

    for (const message of messages) {
      const role = typeof message.role === "string" ? message.role : undefined;
      const type = typeof message.type === "string" ? message.type : undefined;

      if (role === "assistant") {
        const text = contentToText(message.content);
        if (text) {
          this.emit(agentTextEvent(text));
        }
        for (const action of extractActions(message)) {
          const toolCallId = action.toolCallId || `mini-swe-agent-${++generatedId}`;
          pending.push({ command: action.command, toolCallId });
          this.emit(toolCallEvent({
            tool: "bash",
            toolUseId: toolCallId,
            family: "shell",
            input: { command: action.command },
            message: `bash: ${preview(action.command)}`,
          }));
        }
        continue;
      }

      const isObservation =
        (role === "tool" || type === "function_call_output" || role === "user") &&
        pending.length > 0;
      if (!isObservation) continue;

      const observation = extractObservation(message);
      const pendingIndex = observation.toolCallId
        ? pending.findIndex((item) => item.toolCallId === observation.toolCallId)
        : 0;
      const call = pendingIndex >= 0 ? pending.splice(pendingIndex, 1)[0] : pending.shift();
      if (!call) continue;

      this.emit(shellResultEvent({
        tool: "bash",
        toolUseId: call.toolCallId,
        stdout: observation.stdout,
        stderr: "",
        exitCode: observation.exitCode,
        status: observation.status,
        message: `Result: ${preview(observation.stdout)}`,
      }));
    }

    this.log("info", `Replayed mini-swe-agent trajectory`, {
      trajectoryPath,
      messages: messages.length,
    });
  }

  getModel(): string {
    return this.config.model;
  }

  getConfiguration(): MiniSweAgentConfiguration {
    return { ...this.config };
  }
}
