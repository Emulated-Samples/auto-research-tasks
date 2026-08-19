/**
 * CLI help output
 */

export function showHelp(): void {
  console.log(`
env-orchestrator - Hyperfocal RL Environment Orchestrator

Usage:
  env-orchestrator <command> [options]

Configuration Commands:
  env show                        Show current configuration and credentials
  env validate                    Validate configuration and check for issues
  env setup-isolation [--dry-run] Set up workspace isolation (create agent user)
  env init [--name <name>]        Create a new hyperfocal.yaml

Environment Commands:
  problems                        List available problems
  prompt [--problem <id>]         Show prompt for a problem
  setup [--problem <id>]          Run setup for a problem
  test [--problem <id>] [--cleanup]
                                  Run tests for a problem
  cleanup                         Clean up test infrastructure
  check                           Pre-flight check for workspace readiness
  validate-manifest [--manifest <path>]
                                  Validate a manifest file against the schema
  solve [--problem <id>] [--model <model>] [--agent-type <type>] [--permissions-mode <mode>] [--reasoning-mode <mode>]
                                  Have an agent solve a problem
  solve-oracle [--problem <id>]   Run the environment's programmatic reference
                                  solution (the solveProblem() hook) — no
                                  agent involved
  rollout [--problem <id>] [--model <model>] [--agent-type <type>] [--permissions-mode <mode>] [--reasoning-mode <mode>] [--cleanup]
                                  Full cycle: setup -> solve -> test

Options:
  --problem, -p <id>              Problem ID to use (default: first problem)
  --model, -m <model>             Model to use (default: opus)
  --agent-type, -a <type>         Agent type: "claude-code", "anthropic-coding", "opencode", "codex", or "mini-swe-agent"
  --permissions-mode <mode>       Permission mode for the agent:
                                    "linux-user" (default) - unprivileged user, kernel-enforced isolation
                                    "claude-permissions" - agent as root, tool-level (string) restrictions only
  --reasoning-mode <mode>         Provider-specific reasoning/variant selector (for OpenCode)
  --cleanup                       Run cleanup before tests (for test/rollout)
  --dry-run                       Preview changes without applying them
  --help, -h                      Show this help

Credentials (.env file):
  Credentials are loaded from a .env file. Search order:
    1. {config.paths.root}/environment/.env (e.g., /hyperfocal/env/environment/.env)
    2. ./environment/.env (relative to current directory)
    3. ./.env (current directory)
    4. {config.paths.root}/.env (legacy)
    5. ~/.hyperfocal/.env (user home fallback)

  Required in .env file:
    ANTHROPIC_API_KEY               API key for Claude models (not required for opencode/codex; one of several provider keys for mini-swe-agent)

  Optional in .env file:
    AWS_ACCESS_KEY_ID               AWS access key (for environment tests)
    AWS_SECRET_ACCESS_KEY           AWS secret key (for environment tests)
    AWS_SESSION_TOKEN               AWS session token (for assumed roles)
    AWS_REGION                      AWS region (default: us-west-2)
    CODEX_API_KEY                   Optional API key for codex exec automation
    OPENAI_API_KEY                  Optional LiteLLM provider key for mini-swe-agent
    OPENROUTER_API_KEY              Optional LiteLLM provider key for mini-swe-agent
    GOOGLE_API_KEY                  Optional LiteLLM provider key for mini-swe-agent
    XAI_API_KEY                     Optional LiteLLM provider key for mini-swe-agent
    TELEMETRY_S3_BUCKET             S3 bucket for telemetry logs
    RUN_ID                          Run identifier (set by control plane)
    ROLLOUT_ID                      Rollout identifier (set by control plane)

Configuration:
  Configuration is stored in hyperfocal.yaml. Search order:
    1. Current directory
    2. \$HYPERFOCAL_ENV_ROOT/hyperfocal.yaml
    3. /hyperfocal/env/hyperfocal.yaml
    4. /hyperfocal/env/environment/hyperfocal.yaml
  Use 'env init' to create a new configuration file.

Examples:
  env-orchestrator env init --name my-environment
  env-orchestrator env show
  env-orchestrator env validate
  env-orchestrator problems
  env-orchestrator solve --problem basic-ec2 --model opus
  env-orchestrator solve --agent-type claude-code
  env-orchestrator rollout --problem my-problem
  env-orchestrator rollout --permissions-mode linux-user
  env-orchestrator rollout --agent-type opencode --model openrouter/openai/gpt-5.5 --reasoning-mode xhigh
  env-orchestrator rollout --agent-type codex --model gpt-5.6-sol
  env-orchestrator rollout --agent-type mini-swe-agent --model anthropic/claude-opus-4-6
`);
}
