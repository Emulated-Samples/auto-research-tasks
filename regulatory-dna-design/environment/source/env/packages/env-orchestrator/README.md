# @hyperfocal/env-orchestrator

CLI orchestrator for Hyperfocal RL environments. Manages the setup → solve → test cycle for agent training.

## Installation

```bash
npm install @hyperfocal/env-orchestrator
```

## Usage

### Configuration

Configuration is stored in `hyperfocal.yaml`. The orchestrator searches for it in this order:

1. Current directory (`./hyperfocal.yaml`)
2. `$HYPERFOCAL_ENV_ROOT/hyperfocal.yaml`
3. `/hyperfocal/env/hyperfocal.yaml`
4. `/hyperfocal/env/environment/hyperfocal.yaml`

Example `hyperfocal.yaml`:

```yaml
version: "1.0"

environment:
  name: my-training-environment
  description: Description of what this environment teaches

paths:
  root: /hyperfocal/env
  environmentDist: environment/dist
  workspace: workspace

agent:
  awsAccess: false
  defaultModel: claude-sonnet-4-5-20250929
```

### Environment Commands

```bash
# Show current configuration
env-orchestrator env show

# Validate configuration
env-orchestrator env validate

# Create a new hyperfocal.yaml
env-orchestrator env init --name my-environment

# List available problems
env-orchestrator problems

# Show prompt for a problem
env-orchestrator prompt --problem basic-ec2-cfn

# Run setup for a problem
env-orchestrator setup --problem basic-ec2-cfn

# Run tests
env-orchestrator test --problem basic-ec2-cfn

# Have agent solve a problem
env-orchestrator solve --problem basic-ec2-cfn --model claude-sonnet-4-5

# Full rollout: setup → solve → test
env-orchestrator rollout --problem basic-ec2-cfn --model claude-sonnet-4-5
```

### Environment Variables

Required:
- `ANTHROPIC_API_KEY` - API key for Claude models

Optional:
- `AWS_ACCESS_KEY_ID` - AWS access key
- `AWS_SECRET_ACCESS_KEY` - AWS secret key
- `AWS_SESSION_TOKEN` - AWS session token (for assumed roles)
- `AWS_REGION` - AWS region (default: us-west-2)
- `HYPERFOCAL_ENV_ROOT` - Override environment root path
- `HYPERFOCAL_NO_ISOLATION` - Set to "1" to disable agent isolation

## Building

```bash
npm run build
```

## Example Setup

```bash
# Clone/create your environment definition
cd /path/to/environments/my-training-env

# Build the environment
npm install && npm run build

# Create hyperfocal.yaml (or use env init)
cat > hyperfocal.yaml << EOF
version: "1.0"
environment:
  name: my-training-env
  description: My training environment
paths:
  root: /path/to/environments/my-training-env
  environmentDist: environment/dist
  workspace: workspace
agent:
  awsAccess: false
  defaultModel: claude-sonnet-4-5-20250929
EOF

# Run a rollout
export ANTHROPIC_API_KEY=your-key-here
env-orchestrator rollout
```
