# @hyperfocal/env-base

Base package for Hyperfocal RL environments. Provides core types, utilities, and agent implementations.

## Installation

```bash
npm install @hyperfocal/env-base
```

## Usage

### Types

```typescript
import { EnvironmentDefinition, Problem, TestResult, Logger } from "@hyperfocal/env-base";
```

### Execute Commands

```typescript
import { execute, executeWithExitCode } from "@hyperfocal/env-base";

// Throws on non-zero exit code
await execute("aws s3 ls", { cwd: "/workspace" });

// Returns result without throwing
const result = await executeWithExitCode("aws s3 ls");
if (!result.success) {
  console.error(result.error);
}
```

### Logging

```typescript
import { ConsoleLogger, SilentLogger } from "@hyperfocal/env-base";

const logger = new ConsoleLogger();
logger.info("Hello");
```

### Testing

```typescript
import { runSimpleTests, SimpleTest } from "@hyperfocal/env-base";

const tests: SimpleTest[] = [
  {
    id: "test-1",
    name: "My Test",
    description: "Tests something",
    run: async (logger) => {
      // Test logic
      return { success: true };
    },
  },
];

const results = await runSimpleTests(tests, logger);
```

### Claude Code Agent

```typescript
import { ClaudeCodeAgent, ClaudeCodeConfiguration } from "@hyperfocal/env-base";

const config: ClaudeCodeConfiguration = {
  type: "claude-code",
  model: "claude-sonnet-4-0",
  credentials: {
    anthropic: process.env.ANTHROPIC_API_KEY!,
  },
};

const agent = new ClaudeCodeAgent(config);
await agent.run("Create a CloudFormation template...", "/workspace");
```

#### Claude Code CLI lifecycle notes

`ClaudeCodeAgent` runs the Claude Code CLI in `stream-json` mode. The raw
provider stream is useful for debugging, but its `result` events are not the
same thing as Hyperfocal agent completion.

Important distinctions:

- **Claude `type: "result"`** means Claude finished the current turn or yielded
  while waiting for provider-managed work. It can appear multiple times in one
  CLI process, especially around background Bash tasks and task notifications.
- **Hyperfocal completion** happens when `ClaudeCodeAgent.run()` returns and the
  orchestrator child calls `TelemetrySession.end("completed")`.
- **Process `exit` vs `close`**: `exit` means the Claude CLI process ended;
  `close` means stdio also drained. Descendant Bash/SSH processes can inherit
  file descriptors and delay `close`, so the wrapper settles after `exit` when
  needed.

For practical grading, Claude Code rollouts also use a post-result idle cutoff:

- successful Claude result + no pending background tasks + 5 minutes quiet
- successful Claude result + pending background tasks + 15 minutes quiet

When the cutoff fires, Hyperfocal terminates the Claude CLI and runs tests
against the current state. This is intentionally an operational grading policy,
not a documented Claude Code completion signal. Trace logs should distinguish
natural CLI completion from `claude_result_idle_cutoff`.

Useful logs when debugging:

- `{problem}/agent/*.debug.log`: raw Claude Code stream-json
- `{problem}/agent/*.jsonl`: Hyperfocal structured agent trace
- `{problem}/agent/*.log`: human-readable agent trace
- `{problem}/metadata.json`: current phase and session summaries

## Building

```bash
npm run build
```
