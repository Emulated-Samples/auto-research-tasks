/**
 * Command execution utilities
 */

import { spawn, SpawnOptions } from "child_process";

/**
 * Execute options
 */
export interface ExecuteOptions extends SpawnOptions {
  /** Override source for tracking */
  source?: string;
  /** If true, don't track this command */
  silent?: boolean;
  /** Number of retries on failure */
  retries?: number;
  /** Timeout in milliseconds */
  timeout?: number;
}

/**
 * Execution result
 */
export interface ExecutionResult {
  success: boolean;
  output: string;
  exitCode: number;
  error?: Error;
}

/**
 * Execute a command and return full result (doesn't throw)
 */
export async function executeWithExitCode(
  command: string,
  options?: ExecuteOptions
): Promise<ExecutionResult> {
  const retries = options?.retries ?? 0;

  for (let attempt = 0; attempt <= retries; attempt++) {
    const result = await runCommand(command, options);

    if (result.success || attempt >= retries) {
      return result;
    }

    // Log retry
    console.log(`⚠️  Command failed, retrying (${attempt + 1}/${retries})...`);
  }

  // Should never reach here, but TypeScript needs it
  return {
    success: false,
    output: "",
    exitCode: 1,
    error: new Error("Unexpected: exceeded retries"),
  };
}

/**
 * Execute a command and return output (throws on failure)
 */
export async function execute(
  command: string,
  options?: ExecuteOptions
): Promise<string> {
  const result = await executeWithExitCode(command, options);

  if (!result.success) {
    const error = new Error(
      `Command failed with exit code ${result.exitCode}: ${command}\n${result.output}`
    );
    throw error;
  }

  return result.output;
}

/**
 * Run a single command
 */
async function runCommand(
  command: string,
  options?: ExecuteOptions
): Promise<ExecutionResult> {
  return new Promise((resolve) => {
    const spawnOpts: SpawnOptions = {
      cwd: options?.cwd,
      env: options?.env ?? process.env,
      shell: true,
      stdio: "pipe",
    };

    let output = "";
    let resolved = false;
    const child = spawn(command, [], spawnOpts);

    // Setup timeout if specified
    let timeoutId: NodeJS.Timeout | undefined;
    if (options?.timeout) {
      timeoutId = setTimeout(() => {
        if (!resolved) {
          resolved = true;
          child.kill("SIGTERM");
          resolve({
            success: false,
            output,
            exitCode: 124,
            error: new Error(`Command timed out after ${options.timeout}ms`),
          });
        }
      }, options.timeout);
    }

    child.stdout?.on("data", (chunk: Buffer) => {
      const text = chunk.toString();
      output += text;
      if (!options?.silent) {
        process.stdout.write(text);
      }
    });

    child.stderr?.on("data", (chunk: Buffer) => {
      const text = chunk.toString();
      output += text;
      if (!options?.silent) {
        process.stderr.write(text);
      }
    });

    child.on("close", (code) => {
      if (timeoutId) clearTimeout(timeoutId);
      if (resolved) return;
      resolved = true;
      const exitCode = code ?? 1;
      resolve({
        success: exitCode === 0,
        output,
        exitCode,
        error: exitCode !== 0 ? new Error(`Exit code ${exitCode}`) : undefined,
      });
    });

    child.on("error", (err) => {
      if (timeoutId) clearTimeout(timeoutId);
      if (resolved) return;
      resolved = true;
      resolve({
        success: false,
        output,
        exitCode: 1,
        error: err,
      });
    });
  });
}
