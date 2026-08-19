/**
 * Anthropic Coding Agent
 * 
 * Uses Anthropic's API directly with tool use for coding tasks.
 * This is a custom implementation with bash and str_replace_editor tools.
 * 
 * For the official Claude Code CLI wrapper, see ClaudeCodeAgent.
 */

import Anthropic from "@anthropic-ai/sdk";
import * as fs from "fs";
import * as path from "path";
import type { AnthropicCodingConfiguration, AgentRunOptions } from "../types.js";
import { execute } from "../execute.js";
import type { TelemetrySession } from "../telemetry/index.js";
import type { LogEventType } from "../telemetry/types.js";
import {
  validateManifest,
  writeManifest,
  loadSchema,
  generateSchemaDescription,
  generateSchemaExample,
} from "../manifest.js";

/**
 * Anthropic Coding Agent - executes coding tasks using Anthropic API with custom tools
 * 
 * This agent uses the Anthropic SDK directly and implements its own tool loop
 * with bash and str_replace_editor tools. For environments that need the full
 * Claude Code experience with built-in tools, use ClaudeCodeAgent instead.
 */
export class AnthropicCodingAgent {
  private config: AnthropicCodingConfiguration;
  private client: Anthropic;
  private session: TelemetrySession | null = null;
  private schemaPath: string | null = null;
  private workingDir: string = "";

  constructor(config: AnthropicCodingConfiguration) {
    this.config = config;

    if (!config.credentials?.anthropic) {
      throw new Error("Anthropic API key required in credentials.anthropic");
    }

    this.client = new Anthropic({
      apiKey: config.credentials.anthropic,
      maxRetries: 9, // Handle transient 529 overload errors with exponential backoff
    });

    this.log("info", `🔷 AnthropicCodingAgent initialized with model: ${config.model}`);
  }

  /**
   * Set telemetry session for logging
   * When set, all agent activity is logged to persistent files
   */
  setTelemetrySession(session: TelemetrySession): void {
    this.session = session;
  }

  /**
   * Log helper - uses session if available, else console
   */
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

  /**
   * Log error helper
   */
  private logError(message: string, data?: Record<string, unknown>): void {
    if (this.session) {
      this.session.log("error", message, data);
    } else {
      console.error(message);
    }
  }

  /**
   * Format tool input for console display
   * Creates a human-readable summary without exposing full content
   */
  private formatToolInputSummary(toolName: string, input: Record<string, unknown>): string {
    const lines: string[] = [];

    if (toolName === "bash") {
      const cmd = input.command as string;
      if (cmd) {
        // Show first 100 chars of command
        const preview = cmd.length > 100 ? cmd.substring(0, 100) + "..." : cmd;
        lines.push(`   command: ${preview}`);
      } else {
        lines.push(`   command: (empty)`);
      }
    } else if (toolName === "str_replace_editor") {
      const command = input.command as string;
      const filePath = input.path as string;
      lines.push(`   ${command || "(no command)"} → ${filePath || "(no path)"}`);

      switch (command) {
        case "view":
          if (input.view_range) {
            lines.push(`   range: ${JSON.stringify(input.view_range)}`);
          }
          break;
        case "create":
          const fileText = input.file_text as string;
          if (fileText !== undefined) {
            lines.push(`   file_text: (${fileText.length} chars)`);
          } else {
            lines.push(`   file_text: (missing!)`);
          }
          break;
        case "str_replace":
          const oldStr = input.old_str as string;
          const newStr = input.new_str as string;
          if (oldStr !== undefined) {
            const oldPreview = oldStr.length > 50 ? oldStr.substring(0, 50).replace(/\n/g, "\\n") + "..." : oldStr.replace(/\n/g, "\\n");
            lines.push(`   old_str: (${oldStr.length} chars) "${oldPreview}"`);
          } else {
            lines.push(`   old_str: (missing!)`);
          }
          if (newStr !== undefined) {
            lines.push(`   new_str: (${newStr.length} chars)`);
          } else {
            lines.push(`   new_str: (MISSING - will error!)`);
          }
          break;
        case "insert":
          lines.push(`   insert_line: ${input.insert_line}`);
          const insertText = input.insert_text as string;
          if (insertText !== undefined) {
            lines.push(`   insert_text: (${insertText.length} chars)`);
          }
          break;
      }
    } else if (toolName === "submit_output") {
      const manifest = input.manifest as Record<string, unknown>;
      if (manifest) {
        lines.push(`   manifest keys: ${Object.keys(manifest).join(", ")}`);
      }
    } else {
      // Generic fallback: show keys and value lengths
      for (const [key, value] of Object.entries(input)) {
        if (typeof value === "string") {
          lines.push(`   ${key}: (${value.length} chars)`);
        } else if (value === undefined) {
          lines.push(`   ${key}: undefined`);
        } else if (value === null) {
          lines.push(`   ${key}: null`);
        } else {
          lines.push(`   ${key}: ${JSON.stringify(value).substring(0, 50)}`);
        }
      }
    }

    // Flag empty inputs (sign of truncation)
    if (Object.keys(input).length === 0) {
      lines.push(`   ⚠️ EMPTY INPUT (possible truncation!)`);
    }

    return lines.join("\n");
  }

  async run(
    prompt: string,
    workingDir: string,
    options?: AgentRunOptions
  ): Promise<void> {
    if (!prompt?.trim()) {
      throw new Error("Prompt must be a non-empty string");
    }

    // Store for use in tool handlers
    this.workingDir = workingDir;
    this.schemaPath = options?.schemaPath || null;

    this.log("info", `🔷 Using Anthropic API (${this.config.model}) for code execution`);
    this.log(
      "info",
      `📝 Prompt: ${prompt.substring(0, 100)}${prompt.length > 100 ? "..." : ""}`,
      { promptLength: prompt.length }
    );

    if (this.schemaPath) {
      this.log("info", `📋 Output schema configured: ${this.schemaPath}`);
    }

    // Map known short aliases to their dated Anthropic API model ids.
    // Anything not in this map is passed through verbatim — the Anthropic API
    // accepts both the dated form ("claude-sonnet-4-5-20250929") and the modern
    // alias form ("claude-sonnet-4-5"), so newer models like claude-opus-4-7 and
    // claude-sonnet-4-6 work without an entry here. Add an entry only when we
    // observe Anthropic rejecting the alias form for a given model id.
    const modelMap: Record<string, string> = {
      "claude-sonnet-4-0": "claude-sonnet-4-20250514",
      "claude-sonnet-4-5": "claude-sonnet-4-5-20250929",
      "claude-opus-4-0": "claude-opus-4-20250514",
    };

    const apiModel = modelMap[this.config.model] || this.config.model;
    const maxTurns = this.config.options?.maxTurns || 200;

    // Define base tools
    const tools: Anthropic.Tool[] = [
      {
        name: "bash",
        description:
          "Execute bash commands in the working directory. Use this to run shell commands, " +
          "install packages, run tests, use AWS CLI, etc.",
        input_schema: {
          type: "object" as const,
          properties: {
            command: {
              type: "string" as const,
              description: "The bash command to execute",
            },
          },
          required: ["command"],
        },
      },
      {
        name: "str_replace_editor",
        description:
          "View, create, and edit files. Supports viewing files, creating new files, " +
          "replacing text, and inserting text at specific lines.",
        input_schema: {
          type: "object" as const,
          properties: {
            command: {
              type: "string" as const,
              description: "The command: 'view', 'create', 'str_replace', or 'insert'",
              enum: ["view", "create", "str_replace", "insert"],
            },
            path: {
              type: "string" as const,
              description: "Path to the file (relative to working directory)",
            },
            file_text: {
              type: "string" as const,
              description: "Full file content (for 'create' command)",
            },
            view_range: {
              type: "array" as const,
              items: { type: "number" as const },
              description: "Line range for 'view' [start, end] (1-indexed)",
            },
            old_str: {
              type: "string" as const,
              description: "String to replace (for 'str_replace')",
            },
            new_str: {
              type: "string" as const,
              description: "Replacement string (for 'str_replace'). Required - use empty string to delete text.",
            },
            insert_text: {
              type: "string" as const,
              description: "Text to insert (for 'insert')",
            },
            insert_line: {
              type: "number" as const,
              description: "Line number to insert at (1-indexed)",
            },
          },
          required: ["command", "path"],
        },
      },
    ];

    // Add manifest tools if schema is configured
    if (this.schemaPath) {
      tools.push(this.buildSubmitOutputTool());
      tools.push(this.buildGetOutputSchemaTool());
    }

    const messages: Anthropic.MessageParam[] = [{ role: "user", content: prompt }];
    let turnCount = 0;

    try {
      while (turnCount < maxTurns) {
        turnCount++;
        this.log("turn_start", `\n🔄 Turn ${turnCount}/${maxTurns}`, {
          turn: turnCount,
          maxTurns,
        });

        const response = await this.client.messages.create({
          model: apiModel,
          max_tokens: 16384,
          messages,
          tools,
          temperature: this.config.options?.temperature || 0,
        });

        // The API echoes the concrete model it served, which may differ from
        // the requested string (alias map above, or server-side aliasing).
        this.session?.setResolvedModel(response.model);

        // Check for max_tokens truncation
        if (response.stop_reason === "max_tokens") {
          this.log("warn", `⚠️ Response truncated due to max_tokens limit`, {
            stopReason: response.stop_reason,
          });
        }

        // Process text content
        const textBlocks = response.content.filter(
          (b): b is Anthropic.TextBlock => b.type === "text"
        );
        if (textBlocks.length > 0) {
          const agentText = textBlocks.map((b) => b.text).join("\n");
          this.log("agent_text", agentText, { text: agentText });
        }

        // Check for tool use
        const toolUseBlocks = response.content.filter(
          (b): b is Anthropic.ToolUseBlock => b.type === "tool_use"
        );

        if (toolUseBlocks.length === 0) {
          this.log("info", "\n✅ Task completed", { turns: turnCount });
          break;
        }

        // Detect potentially truncated tool calls (from max_tokens)
        if (response.stop_reason === "max_tokens") {
          for (const toolUse of toolUseBlocks) {
            const input = toolUse.input as Record<string, unknown>;
            const isEmpty = Object.keys(input).length === 0;
            if (isEmpty) {
              this.log("warn", `⚠️ Tool call ${toolUse.name} appears truncated (empty input)`, {
                tool: toolUse.name,
                stopReason: response.stop_reason,
              });
            }
          }
        }

        // Process tool calls
        const toolResults: Anthropic.ToolResultBlockParam[] = [];

        for (const toolUse of toolUseBlocks) {
          const toolInput = toolUse.input as Record<string, unknown>;
          const inputSummary = this.formatToolInputSummary(toolUse.name, toolInput);
          this.log("tool_call", `\n🔧 Executing tool: ${toolUse.name}\n${inputSummary}`, {
            tool: toolUse.name,
            tool_use_id: toolUse.id,
            input: toolInput,
          });

          try {
            const result = await this.executeTool(toolUse, workingDir);
            const resultPreview = result.length > 200 
              ? `(${result.length} chars) ${result.substring(0, 200)}...` 
              : result;
            this.log("tool_result", `✅ ${toolUse.name}: ${resultPreview}`, {
              tool: toolUse.name,
              tool_use_id: toolUse.id,
              output: result,
              success: true,
            });
            toolResults.push({
              type: "tool_result",
              tool_use_id: toolUse.id,
              content: result,
            });
          } catch (error) {
            const errorMsg = error instanceof Error ? error.message : String(error);
            this.logError(`❌ Tool error: ${errorMsg}`, {
              tool: toolUse.name,
              error: errorMsg,
            });
            toolResults.push({
              type: "tool_result",
              tool_use_id: toolUse.id,
              content: `Error: ${errorMsg}`,
              is_error: true,
            });
          }
        }

        // Log turn end
        this.log("turn_end", `Turn ${turnCount} completed`, {
          turn: turnCount,
          toolsCalled: toolUseBlocks.length,
        });

        // Add assistant response and tool results to conversation
        messages.push({ role: "assistant", content: response.content });
        messages.push({ role: "user", content: toolResults });
      }

      if (turnCount >= maxTurns) {
        this.log("warn", `\n⚠️  Reached maximum turns (${maxTurns})`, { maxTurns });
      }
    } catch (error) {
      const errorMsg = error instanceof Error ? error.message : String(error);
      this.logError(`❌ Anthropic API execution failed: ${errorMsg}`, { error: errorMsg });
      throw error;
    }
  }

  private async executeTool(
    toolUse: Anthropic.ToolUseBlock,
    workingDir: string
  ): Promise<string> {
    const input = toolUse.input as Record<string, unknown>;

    if (toolUse.name === "bash") {
      const command = input.command as string;
      if (!command) {
        throw new Error("Command is required");
      }

      this.log("command_start", `📝 Command: ${command}`, { command });
      const output = await execute(command, { cwd: workingDir, silent: true });
      this.log("command_end", `Command completed`, {
        command,
        outputLength: output?.length || 0,
      });
      return output || "Command executed successfully";
    }

    if (toolUse.name === "str_replace_editor") {
      return this.executeEditorTool(input, workingDir);
    }

    if (toolUse.name === "submit_output") {
      return this.handleSubmitOutput(input, workingDir);
    }

    if (toolUse.name === "get_output_schema") {
      return this.handleGetOutputSchema();
    }

    throw new Error(`Unknown tool: ${toolUse.name}`);
  }

  private executeEditorTool(
    input: Record<string, unknown>,
    workingDir: string
  ): string {
    const command = input.command as string;
    const filePath = input.path as string;
    const fullPath = path.resolve(workingDir, filePath);

    if (!command || !filePath) {
      throw new Error("command and path are required");
    }

    switch (command) {
      case "view": {
        if (!fs.existsSync(fullPath)) {
          throw new Error(`File not found: ${filePath}`);
        }
        const content = fs.readFileSync(fullPath, "utf-8");
        const lines = content.split("\n");
        const viewRange = input.view_range as number[] | undefined;

        if (viewRange && viewRange.length >= 2) {
          const start = Math.max(0, viewRange[0] - 1);
          const end = Math.min(lines.length, viewRange[1]);
          return lines
            .slice(start, end)
            .map((line, i) => `${start + i + 1}: ${line}`)
            .join("\n");
        }

        return lines.map((line, i) => `${i + 1}: ${line}`).join("\n");
      }

      case "create": {
        const fileText = input.file_text as string;
        if (fileText === undefined) {
          throw new Error("file_text is required for 'create'");
        }
        const dir = path.dirname(fullPath);
        if (!fs.existsSync(dir)) {
          fs.mkdirSync(dir, { recursive: true });
        }
        fs.writeFileSync(fullPath, fileText, "utf-8");
        this.log("info", `✅ Created: ${filePath}`, { file: filePath, action: "create" });
        return `File created: ${filePath}`;
      }

      case "str_replace": {
        const oldStr = input.old_str as string;
        if (oldStr === undefined) {
          throw new Error("old_str is required for 'str_replace'");
        }
        if (input.new_str === undefined) {
          throw new Error("new_str is required for 'str_replace' (use empty string to delete)");
        }
        const newStr = input.new_str as string;

        if (!fs.existsSync(fullPath)) {
          throw new Error(`File not found: ${filePath}`);
        }
        const content = fs.readFileSync(fullPath, "utf-8");

        const matches = content.split(oldStr).length - 1;
        if (matches === 0) {
          throw new Error(`String not found in file: ${filePath}`);
        }
        if (matches > 1) {
          throw new Error(`String appears ${matches} times. Use more specific string.`);
        }

        fs.writeFileSync(fullPath, content.replace(oldStr, newStr), "utf-8");
        this.log("info", `✅ Replaced in: ${filePath}`, {
          file: filePath,
          action: "replace",
        });
        return `Replaced text in ${filePath}`;
      }

      case "insert": {
        const insertText = input.insert_text as string;
        const insertLine = input.insert_line as number;

        if (insertText === undefined || insertLine === undefined) {
          throw new Error("insert_text and insert_line required for 'insert'");
        }

        if (!fs.existsSync(fullPath)) {
          throw new Error(`File not found: ${filePath}`);
        }

        const content = fs.readFileSync(fullPath, "utf-8");
        const lines = content.split("\n");
        const lineIndex = Math.max(0, Math.min(lines.length, insertLine - 1));
        lines.splice(lineIndex, 0, insertText);
        fs.writeFileSync(fullPath, lines.join("\n"), "utf-8");
        this.log("info", `✅ Inserted at line ${insertLine}: ${filePath}`, {
          file: filePath,
          action: "insert",
          line: insertLine,
        });
        return `Inserted at line ${insertLine} in ${filePath}`;
      }

      default:
        throw new Error(`Unknown editor command: ${command}`);
    }
  }

  getModel(): string {
    return this.config.model;
  }

  getConfiguration(): AnthropicCodingConfiguration {
    return { ...this.config };
  }

  /**
   * Build the submit_output tool definition
   */
  private buildSubmitOutputTool(): Anthropic.Tool {
    return {
      name: "submit_output",
      description:
        "Submit the output manifest declaring how your solution should be built, run, and tested. " +
        "The manifest will be validated against the environment's schema. " +
        "On validation error, fix the issues and resubmit. " +
        "On success, the manifest is saved for test execution.",
      input_schema: {
        type: "object" as const,
        properties: {
          manifest: {
            type: "object" as const,
            description: "The output manifest matching the environment's expected schema",
          },
        },
        required: ["manifest"],
      },
    };
  }

  /**
   * Build the get_output_schema tool definition
   */
  private buildGetOutputSchemaTool(): Anthropic.Tool {
    return {
      name: "get_output_schema",
      description:
        "Get the JSON schema that the output manifest must conform to, " +
        "along with a human-readable description and example.",
      input_schema: {
        type: "object" as const,
        properties: {},
      },
    };
  }

  /**
   * Handle the submit_output tool
   */
  private handleSubmitOutput(
    input: Record<string, unknown>,
    workingDir: string
  ): string {
    if (!this.schemaPath) {
      return JSON.stringify({
        status: "error",
        message: "No output schema configured for this environment",
      });
    }

    const manifest = input.manifest as Record<string, unknown>;
    if (!manifest || typeof manifest !== "object") {
      return JSON.stringify({
        status: "error",
        message: "Invalid manifest: must be an object",
        errors: [{ path: "/manifest", message: "must be an object" }],
      });
    }

    try {
      const schema = loadSchema(this.schemaPath);
      const result = validateManifest(manifest, schema);

      if (!result.valid) {
        this.log("info", `❌ Manifest validation failed`, {
          errors: result.errors,
        });
        return JSON.stringify({
          status: "error",
          message: "Output manifest validation failed",
          errors: result.errors,
        });
      }

      const manifestPath = writeManifest(workingDir, manifest);
      this.log("info", `✅ Manifest validated and saved to ${manifestPath}`, {
        manifestPath,
      });
      return JSON.stringify({
        status: "success",
        message: `Output manifest validated and saved to ${manifestPath}`,
        manifest,
      });
    } catch (error) {
      const errorMsg = error instanceof Error ? error.message : String(error);
      return JSON.stringify({
        status: "error",
        message: `Failed to process manifest: ${errorMsg}`,
      });
    }
  }

  /**
   * Handle the get_output_schema tool
   */
  private handleGetOutputSchema(): string {
    if (!this.schemaPath) {
      return JSON.stringify({
        status: "error",
        message: "No output schema configured for this environment",
      });
    }

    try {
      const schema = loadSchema(this.schemaPath);
      return JSON.stringify({
        schema,
        description: generateSchemaDescription(schema),
        example: generateSchemaExample(schema),
      });
    } catch (error) {
      const errorMsg = error instanceof Error ? error.message : String(error);
      return JSON.stringify({
        status: "error",
        message: `Failed to load schema: ${errorMsg}`,
      });
    }
  }
}
