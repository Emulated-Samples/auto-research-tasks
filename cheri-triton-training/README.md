# Cheri Block Triton training kernels

Hand-write Triton forward and backward kernels for the core block of a genomics sequence-to-function model, without changing a single gradient the block produces, and be scored on how much faster the training step gets.

Every run that ships a kernel the grader can launch clears all six numerical-parity cases — the entire score spread above zero is backward-pass speed. The zeros are files whose training path never runs at all.

## The task

Cherimoya is a compact sequence-to-function model for genomics; its backbone stacks Cheri Blocks — a 3-tap dilated depthwise convolution fused with a per-example layer norm, followed by a channel-mixing MLP and scaled residual. The block ships as plain PyTorch, which is slow on GPU.

The prompt is the message an engineer would send: training is slow, the spatial mixing is the expensive part, move it into hand-written Triton kernels wrapped in a `torch.autograd.Function`, and make the forward and the backward fast at the model's working sizes — batch up to 512, sequence length 2114, 128 channels, dilations 1 through 256, fp32 and reduced precision. Every gradient the block produced before, with respect to the input and every parameter, must still be produced and still be correct. The CPU path and the no-grad CUDA path must keep using the existing PyTorch code, `torch.compile`, `torch.jit`, and `torch.fx` are banned, and only the block's source file may be edited.

The backward dominates training cost and is much harder to fuse than the forward, which is what makes the task discriminate: an honest training-path speedup is the only thing that earns marks.

## Verifier design

The grader recomputes everything itself — parity against an independent fp32 reference that reads only the block's parameters, and timings measured live on the same GPU — so the only numbers that count are the ones its own run produces, never the agent's benchmark tables.

| What we check | How |
| --- | --- |
| Real Triton, no shortcuts | The source is scanned for banned compiler routes (`torch.compile`, JIT, FX) — any hit scores zero — and must launch actual `@triton.jit` kernels |
| The block is still the same block | With the MLP zeroed, the block must be bit-exact identity under grad and pass gradients through unchanged, pinning the residual structure before anything else is scored |
| Forward numerics | Six seeded cases — including single-example, dilation past the sequence length, production shape, bf16, and fp16 — checked against an independent fp32 recomputation that never trusts the agent's code |
| Backward numerics | On the same six cases, per-tensor relative error of the input gradient and every parameter gradient against autograd of the reference; a dropped gradient term fails outright |
| All-or-nothing correctness | Any parity failure zeroes the score; performance is never measured on incorrect kernels |
| Speedup against a live ceiling | Four production shapes, each timed as a full training step (forward plus backward) by CUDA events, median of 30 iterations after warmup; the vanilla block and a tuned reference are re-timed on the same GPU every run, and full marks require beating the reference's speedup by 10 percent |
| No credit for standing still | A correct but unaccelerated training path (around 1.0x) sits below the scoring floor of 1.2x and earns zero performance credit |

## Trace walkthrough

Nine of the fifteen evaluated runs cleared every parity case and scored between 0.86 and 1.0 on speed alone. Not one zero was a slow kernel: every zero shipped a file the grader either rejected on sight or could not run.

### A strong run

1. **Profile before writing a line.** The top run mapped the block, then measured the baseline: a batch-256 fp32 training step at 48.6 ms, with roughly three quarters of it in the conv-and-norm memory traffic. That is where it aimed the first kernel.
2. **Treat the first fusion as a floor, not a finish.** With the fused conv-plus-norm forward and backward landed, it went after the MLP: fp32 `tl.dot` in strict IEEE mode benched at an unusable 207 ms, so it emulated fp32 matmuls by splitting operands into bf16 high and low halves, then merged the residual gradient into the conv backward and folded the norm's backward reductions into the MLP backward — each fusion deleting a full pass over memory.
3. **Let your own harness catch you.** Midway, its parity checks exposed a real bug — when the channels span more than one tile, a partial buffer left rows uninitialized — and it fixed the indexing before ever timing that variant.
4. **Beat the live ceiling everywhere.** The grader measured 6/6 parity and a 2.47x median fp32 training step, with 5.5x on the batch-512 production shape (40 ms against 224 ms) and 5.2x in bf16 — faster than the tuned reference on all four shapes, for a 1.0. About two hours and 150 tool calls.

### A failed run

1. **Bank a real win.** One run fused the conv and norm cleanly and measured it honestly: 1.65x in fp32, 2.7x in bf16 — comfortably above the 1.2x floor, a solid score if shipped as-is.
2. **Trade it for a bigger design.** It confirmed Triton matmuls match cuBLAS on these shapes and began rewriting the entire block — conv, norm, MLP, and residual — as one fully-fused forward and backward.
3. **Ship mid-refactor.** The rewrite changed a backward kernel's signature, and one launch site never got the two new arguments. The very first backward pass the grader ran raised a TypeError, correctness gates everything, and the run scored zero — with a working 1.65x sitting in its own transcript history.

The rest of the zeros never reached the timer either: one left a banned compiler route named in the block's source, which the scan zeroes on sight even next to real Triton kernels, and the others shipped the file with no `@triton.jit` kernel in it at all.

## Failure modes

These are the failure modes we saw across the evaluated runs.

| Failure mode | What goes wrong |
| --- | --- |
| Triton forward, autograd backward | Passes every parity check, then measures near 1.0x on the training step — below the floor on all four shapes, so performance scores zero |
| A backward path never executed | A late refactor leaves one kernel launch stale; the agent's own tests never touch that branch, and the grader's first backward call crashes a run that had a scoring solution hours earlier |
| The shipped file fails the scan | Any trace of the banned compiler routes in the block's source zeroes the run even alongside real kernels, and a file with no `@triton.jit` kernel scores zero before anything is timed |
| The norm's gradient path | The layer-norm statistics depend on every element of the conv output; a backward that drops that coupling, or mishandles zero-padded edges at large dilations, fails per-tensor parity |
| Over-decomposition | Many small kernels pass parity and lose the measurement to launch overhead, landing mid-band instead of near the reference |

Every shipped rollout was audited for reward hacking and is clean.

## Running

Replay the reference solution with the same verifier the agent is scored against:

```bash
harbor run -p delivery/auto-research-tasks/cheri-triton-training --agent oracle -k 1 -o jobs/
```

The reference replays around 0.85 — full marks require beating the live-timed reference itself — against a 0.7 floor. The task needs one L4-class NVIDIA GPU on an x86 host.
