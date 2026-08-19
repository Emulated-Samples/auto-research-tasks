import { spawn, type ChildProcess } from "child_process";
import * as fs from "fs";
import * as readline from "readline";

export type StreamingCliCompletionSource = "close" | "exit";

export interface StreamingCliResult {
  code: number | null;
  signal: NodeJS.Signals | null;
  stderrChunks: string[];
  completionSource: StreamingCliCompletionSource;
}

export interface StreamingCliOptions {
  command: string;
  args: string[];
  cwd: string;
  env: NodeJS.ProcessEnv;
  stdin: "pipe" | "ignore";
  closeGraceMs?: number;
  debugLogStream?: fs.WriteStream;
  onProcess?: (child: ChildProcess) => void;
  onStdoutLine: (line: string) => void;
  onStderrText?: (text: string) => void;
  writeStdin?: (child: ChildProcess) => void;
}

const DEFAULT_CLOSE_GRACE_MS = 2_000;

/**
 * Run a streaming JSONL-style CLI process and return process lifecycle facts.
 *
 * This helper deliberately does not classify success or failure. Callers own
 * provider protocol parsing, retry policy, and exit-code interpretation.
 */
export function runStreamingCli(options: StreamingCliOptions): Promise<StreamingCliResult> {
  const closeGraceMs = options.closeGraceMs ?? DEFAULT_CLOSE_GRACE_MS;
  const stderrChunks: string[] = [];

  return new Promise((resolve, reject) => {
    const child = spawn(options.command, options.args, {
      cwd: options.cwd,
      env: options.env,
      stdio: [options.stdin, "pipe", "pipe"],
    });

    options.onProcess?.(child);

    let settled = false;
    let closeGraceTimer: NodeJS.Timeout | undefined;

    const rl = child.stdout
      ? readline.createInterface({ input: child.stdout, crlfDelay: Infinity })
      : undefined;

    const finishDebugLog = () => {
      options.debugLogStream?.end();
    };

    const cleanupReaders = (source: StreamingCliCompletionSource | "error") => {
      if (closeGraceTimer) clearTimeout(closeGraceTimer);
      rl?.close();
      finishDebugLog();

      if (source === "exit" || source === "error") {
        // Descendant Bash/SSH processes can inherit the provider CLI's stdio
        // fds. Destroy our readers when falling back from `exit` so a leaked
        // descendant cannot keep the wrapper alive forever. Do the same on
        // startup/write errors so partially initialized pipes do not linger.
        child.stdout?.destroy();
        child.stderr?.destroy();
        child.stdin?.destroy();
      }
    };

    const settle = (
      code: number | null,
      signal: NodeJS.Signals | null,
      completionSource: StreamingCliCompletionSource
    ) => {
      if (settled) return;
      settled = true;
      cleanupReaders(completionSource);
      resolve({
        code,
        signal,
        stderrChunks,
        completionSource,
      });
    };

    child.stdout?.on("data", (data) => {
      options.debugLogStream?.write(data);
    });

    child.stderr?.on("data", (data) => {
      options.debugLogStream?.write(data);
      const text = data.toString();
      stderrChunks.push(text);
      options.onStderrText?.(text);
    });

    rl?.on("line", options.onStdoutLine);

    child.on("close", (code, signal) => {
      settle(code, signal, "close");
    });

    child.on("exit", (code, signal) => {
      if (settled) return;
      closeGraceTimer = setTimeout(() => {
        settle(code, signal, "exit");
      }, closeGraceMs);
    });

    child.on("error", (error) => {
      if (settled) return;
      settled = true;
      cleanupReaders("error");
      reject(error);
    });

    try {
      options.writeStdin?.(child);
    } catch (error) {
      if (settled) return;
      settled = true;
      cleanupReaders("error");
      child.kill("SIGTERM");
      reject(error);
    }
  });
}
