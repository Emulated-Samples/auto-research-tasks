/**
 * Subprocess test runner — parses structured test output (JUnit XML, etc.)
 * into individual TestResult entries.
 *
 * Default format: JUnit XML (built-in to pytest, Go, Jest, JUnit).
 * Custom parsers can be provided for other formats.
 */

import * as fs from "fs";
import * as os from "os";
import * as path from "path";
import type { Logger, TestResult } from "./types.js";
import { executeWithExitCode } from "./execute.js";
import type { ExecutionResult } from "./execute.js";

/**
 * Options for running subprocess tests.
 */
export interface SubprocessTestOptions {
  /** Timeout in milliseconds for the subprocess */
  timeout: number;
  /** Working directory for the subprocess */
  cwd?: string;
  /** Environment variables for the subprocess */
  env?: Record<string, string>;
  /** Logger for summary output */
  logger: Logger;
  /**
   * Path where the subprocess writes JUnit XML results.
   * If not provided, a temp file is generated automatically.
   */
  resultsFile?: string;
  /**
   * Custom parser that overrides default JUnit XML parsing.
   * Receives the raw ExecutionResult (stdout, exit code) and returns TestResult[].
   * Use for non-JUnit formats (Go test JSON, Jest JSON, custom protocols).
   */
  parseResults?: (result: ExecutionResult) => TestResult[];
}

/**
 * Run a subprocess test command and parse results into individual TestResult entries.
 *
 * By default, expects the command to produce JUnit XML at `resultsFile`.
 * If `parseResults` is provided, it overrides the default parsing.
 *
 * If the subprocess crashes before writing results, returns a single
 * errored TestResult with the subprocess output as the error message.
 */
export async function runSubprocessTests(
  command: string,
  options: SubprocessTestOptions
): Promise<TestResult[]> {
  const { timeout, cwd, env, logger, parseResults } = options;
  const resultsFile =
    options.resultsFile ||
    path.join(os.tmpdir(), `hyperfocal-test-results-${Date.now()}.xml`);

  // Clean up any stale results file from a previous run
  try {
    if (fs.existsSync(resultsFile)) {
      fs.unlinkSync(resultsFile);
    }
  } catch {
    // Ignore cleanup errors
  }

  const startTime = Date.now();

  // Execute the test command
  const result = await executeWithExitCode(command, {
    cwd,
    env: env ? { ...process.env, ...env } : undefined,
    timeout,
    silent: true,
  });

  const totalDuration = Date.now() - startTime;

  // If a custom parser is provided, use it
  if (parseResults) {
    try {
      const results = parseResults(result);
      logSummary(results, totalDuration, logger);
      return results;
    } catch (parseError) {
      const errorMsg =
        parseError instanceof Error ? parseError.message : String(parseError);
      logger.error(`Failed to parse custom test results: ${errorMsg}`);
      return [
        createErrorResult(
          "parse-error",
          `Custom parser failed: ${errorMsg}`,
          totalDuration,
          result.output
        ),
      ];
    }
  }

  // Default: parse JUnit XML from resultsFile
  let xml: string;
  try {
    xml = fs.readFileSync(resultsFile, "utf-8");
  } catch {
    // JUnit XML file missing — subprocess likely crashed before writing it
    const output = result.output || "";
    let errorMsg = "Test subprocess failed before producing results";

    if (output.includes("ModuleNotFoundError")) {
      errorMsg = "ModuleNotFoundError — test module could not be imported";
    } else if (output.includes("SyntaxError")) {
      errorMsg = "SyntaxError in test code or fixtures";
    } else if (result.exitCode === 124) {
      errorMsg = `Test subprocess timed out after ${timeout}ms`;
    } else if (output.trim()) {
      // Use last meaningful line of output as the error
      const lines = output.trim().split("\n").filter(Boolean);
      errorMsg = lines[lines.length - 1].slice(0, 500);
    }

    logger.error(`⚠️  ${errorMsg}`);
    return [createErrorResult("subprocess-crash", errorMsg, totalDuration, output)];
  }

  if (!xml.trim()) {
    logger.error("⚠️  JUnit XML results file is empty");
    return [
      createErrorResult(
        "empty-results",
        "JUnit XML results file is empty",
        totalDuration,
        result.output
      ),
    ];
  }

  // Parse JUnit XML
  const results = parseJunitXml(xml);

  if (results.length === 0) {
    logger.error("⚠️  No test results found in JUnit XML");
    return [
      createErrorResult(
        "no-results",
        "No test cases found in JUnit XML output",
        totalDuration,
        result.output
      ),
    ];
  }

  // Log summary
  logSummary(results, totalDuration, logger);

  // Clean up temp file
  try {
    fs.unlinkSync(resultsFile);
  } catch {
    // Ignore cleanup errors
  }

  return results;
}

/**
 * Parse JUnit XML into individual TestResult entries.
 *
 * Zero-dependency regex-based parser for the JUnit XML format.
 * Handles: passed (self-closing <testcase/>), failed (<failure>),
 * errored (<error>), and skipped (<skipped>).
 *
 * Can be used standalone if you want to run the command yourself
 * and just parse the output.
 */
export function parseJunitXml(xml: string): TestResult[] {
  const results: TestResult[] = [];

  // Match <testcase> elements — both self-closing and with children
  const testcaseRegex =
    /<testcase\s+classname="([^"]*?)"\s+name="([^"]*?)"\s+time="([^"]*?)"\s*(?:\/>|>([\s\S]*?)<\/testcase>)/g;

  let match;
  while ((match = testcaseRegex.exec(xml)) !== null) {
    const [, , name, timeStr, inner] = match;
    const durationSec = parseFloat(timeStr) || 0;
    const durationMs = Math.round(durationSec * 1000);

    // Determine status from child elements
    if (!inner) {
      // Self-closing <testcase .../> — passed
      results.push({
        id: name,
        name,
        status: "passed",
        duration: durationMs,
        score: 1.0,
      });
    } else if (inner.includes("<skipped")) {
      // <skipped> — intentionally skipped, count as passed
      const msgMatch = inner.match(/<skipped[^>]*message="([^"]*?)"/);
      results.push({
        id: name,
        name,
        status: "passed",
        duration: durationMs,
        score: 1.0,
        output: msgMatch ? decodeXmlEntities(msgMatch[1]) : "skipped",
      });
    } else if (inner.includes("<failure")) {
      // <failure> — test assertion failed
      const msgMatch = inner.match(/<failure[^>]*message="([^"]*?)"/);
      const bodyMatch = inner.match(/<failure[^>]*>([\s\S]*?)<\/failure>/);
      const error = msgMatch ? decodeXmlEntities(msgMatch[1]) : "Test failed";
      const output = bodyMatch ? decodeXmlEntities(bodyMatch[1]).trim() : undefined;
      results.push({
        id: name,
        name,
        status: "failed",
        duration: durationMs,
        score: 0.0,
        error,
        output,
      });
    } else if (inner.includes("<error")) {
      // <error> — test errored (setup/teardown failure)
      const msgMatch = inner.match(/<error[^>]*message="([^"]*?)"/);
      const bodyMatch = inner.match(/<error[^>]*>([\s\S]*?)<\/error>/);
      const error = msgMatch ? decodeXmlEntities(msgMatch[1]) : "Test errored";
      const output = bodyMatch ? decodeXmlEntities(bodyMatch[1]).trim() : undefined;
      results.push({
        id: name,
        name,
        status: "failed",
        duration: durationMs,
        score: 0.0,
        error,
        output,
      });
    } else {
      // Unknown child elements — treat as passed
      results.push({
        id: name,
        name,
        status: "passed",
        duration: durationMs,
        score: 1.0,
      });
    }
  }

  return results;
}

/**
 * Decode common XML entities in attribute values and text content.
 */
function decodeXmlEntities(text: string): string {
  return text
    .replace(/&amp;/g, "&")
    .replace(/&lt;/g, "<")
    .replace(/&gt;/g, ">")
    .replace(/&quot;/g, '"')
    .replace(/&apos;/g, "'")
    .replace(/&#10;/g, "\n")
    .replace(/&#13;/g, "\r")
    .replace(/&#9;/g, "\t");
}

/**
 * Create a single errored TestResult for subprocess crashes.
 */
function createErrorResult(
  id: string,
  error: string,
  duration: number,
  output?: string
): TestResult {
  return {
    id,
    name: id,
    status: "errored",
    duration,
    score: 0,
    error,
    output: output?.slice(0, 5000), // Truncate large outputs
  };
}

/**
 * Log a summary of test results.
 */
function logSummary(
  results: TestResult[],
  totalDuration: number,
  logger: Logger
): void {
  const passed = results.filter((r) => r.status === "passed").length;
  const failed = results.filter(
    (r) => r.status === "failed" || r.status === "errored"
  ).length;
  const skipped = results.filter(
    (r) => r.status === "passed" && r.output?.includes("skipped")
  ).length;

  const parts = [`${passed} passed`];
  if (failed > 0) parts.push(`${failed} failed`);
  if (skipped > 0) parts.push(`${skipped} skipped`);
  parts.push(`(${Math.round(totalDuration / 1000)}s)`);

  if (failed > 0) {
    logger.error(`📊 ${parts.join(", ")}`);
  } else {
    logger.info(`📊 ${parts.join(", ")}`);
  }
}
