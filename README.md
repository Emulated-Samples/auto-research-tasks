# Auto-Research Tasks

Four open-ended research optimization tasks in harbor format, with sample rollouts.
Each task asks the agent to iteratively improve a continuous metric under a budget,
rather than reach a single pass/fail state.

## Layout

```
cheri-triton-training/      task: write a Triton training kernel that matches a reference
manifold-representation/    task: learn a disentangled representation of a signal manifold
polygenic-scoring/          task: build a polygenic scoring method from scratch
regulatory-dna-design/      task: design regulatory DNA sequences against a scoring oracle
trajectory/                 sample rollouts, grouped by task and model
```

Each task directory contains:

```
task.toml           task metadata and resource requirements
instruction.md      the prompt given to the agent
environment/        Dockerfile and sources for the task environment
tests/              verifier that scores a solution
solution/           reference solution
README.md           task description
THIRD_PARTY_NOTICES.md
```

## Usage

Requires [harbor](https://pypi.org/project/harbor) 0.18.0 and Docker.

Run an agent against a task:

```
harbor run -p <task-dir> -a claude-code -m <model>
```

Verify the reference solution with the built-in oracle agent:

```
harbor run -p <task-dir> -a oracle
```

The verifier writes the score to `verifier/reward.json` in the trial directory.
Note that polygenic-scoring uses a native-skill score scale: a naive baseline scores
about 0.33, the reference method scores 1.0, and the ceiling is 1.33.

## Trajectories

`trajectory/<task>/<model>/rollout_<harness>_<n>/` holds complete rollouts from
Claude Opus 5, GPT-5.6 Sol, and Gemini 3.7 Flash: the agent session under `agent/`,
the task and outcome in `config.json` and `result.json`, and the verifier record
under `verifier/`.

## Licensing

See `LICENSING.md`, `THIRD_PARTY_NOTICES.md`, and `LICENSES/`.
