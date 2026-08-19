# Opus 4.8 platform rollout — 2026-07-13

Run `run_019f5a38-226f-770d-8bd8-fdc53fb22272` evaluated commit `4088273` with
one Claude Code / `claude-opus-4-8` rollout. It passed all three capability
categories with a headline reward of **1.1601174410**:

| Category | Reward |
| --- | ---: |
| `svld_class` | 1.1174181998 |
| `svld_rare_poly` | 1.2577797689 |
| `svld_strong` | 1.1051543543 |

The trusted grader reported the following per-dataset outcome components:

| Dataset | Reward | Accuracy | Performance |
| --- | ---: | ---: | ---: |
| `svld_class__s200` | 0.984 | 0.984 | 1.00 |
| `svld_class__s201` | 0.939 | 0.939 | 1.00 |
| `svld_class__s202` | 1.401 | 1.121 | 1.25 |
| `svld_rare_poly__s200` | 0.939 | 0.939 | 1.00 |
| `svld_rare_poly__s201` | 1.386 | 1.109 | 1.25 |
| `svld_rare_poly__s202` | 1.364 | 1.091 | 1.25 |
| `svld_strong__s200` | 1.377 | 1.102 | 1.25 |
| `svld_strong__s201` | 0.961 | 0.961 | 1.00 |
| `svld_strong__s202` | 0.993 | 0.993 | 1.00 |

The sandbox preflight also proved that the submission could neither read its
secret probe nor reach the network. This is valid behavioral evidence of a
strong solution, not evidence for changing the verifier or reward shape.

The rollout did expose an observability defect: category `TestResult` records
showed zero duration and omitted the grader's per-dataset components even though
the console contained them. The follow-up wrapper change preserves the exact
category scores, weights, and `0.6` pass threshold while reporting measured
fit/predict duration and nonce-bound per-dataset reward, accuracy, performance,
and timing in each category result's structured `output` field.

The platform diff collector also reported that it had no baseline tag and
captured zero file changes after the agent committed its work. That is an
orchestrator trace-capture issue; no environment-side workaround was added.
