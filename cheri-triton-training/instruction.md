I need help with some training and inference for genomics. I'm working on cherimoya, a compact sequence-to-function model for genomics. Its backbone is a stack of cheri blocks (`cherimoya/cheri.py`). On an input of shape `(N, L, C)`, each block mixes information along the sequence and across channels. The block currently runs as plain PyTorch on every device, which is slow on GPU. You can read the pure-PyTorch version `CheriBlock` in `cheri.py` read it for the exact operations, shapes, and constants.

We need to figure out how to make the Cheri block fast on CUDA during training, in other words when `torch.is_grad_enabled()` is True and the input is a CUDA tensor. Run the block's expensive spatial mixing (the 3-tap dilated depthwise convolution fused with the per-example layer norm) through a hand-written Triton kernel, and make the forward and backward fast, without changing the gradients the block produces.

Because the path must be differentiable, wrap your Triton forward and backward in a `torch.autograd.Function`. You may leave the channel-mixing MLP (the two linears + GELU + scaled residual) as ordinary PyTorch — autograd will handle it — or fuse more of it if you wish. The CPU path, and the CUDA path when gradients are disabled, must keep using the existing PyTorch code unchanged (the no_grad inference path is not what this task grades).

You choose the kernel design. The one hard requirement is that the training path stays a correct autograd node: every gradient the block produced before (w.r.t. the input and w.r.t. every parameter) must still be produced, and still be correct.

Requirements:
- Preserve the block's public behavior, constructor signature, parameter layout, and output semantics.
- Ensure the existing paths (CPU, and CUDA no_grad) stay correct.
- Forward accuracy (grad-enabled, graded vs the pure-PyTorch reference, over many shapes and dtypes):
- fp32 input: forward max-abs error ≤ `2e-4`
- bf16 input: ≤ `2e-2` · fp16 input: ≤ `5e-3`
- Backward accuracy — the gradients from your training path must match autograd of the reference block. Graded as a relative error per gradient tensor (input, conv weight, and both linear weights): ≤ `3e-3` (fp32), with looser budgets in reduced precision. A dropped or miscomputed gradient term fails this.
- Performance: make the grad-enabled CUDA forward+backward as fast as you can. It is scored on how much speedup it achieves over the current PyTorch block on the same GPU, at the model's working size (`N` up to 512, `L=2114`, `C=128`, dilations `1..256`, fp32 and bf16). More speedup scores higher throughout — there is no point past which further optimisation stops counting. The training regime is harder to speed up than inference — the backward dominates — so a real, honest speedup is what earns marks.

Constraints:
- Do not use `torch.compile`, `torch.jit`, or `torch.fx`. Your CUDA path must launch real `@triton.jit` kernels.
- Work in `cherimoya/cheri.py`.

When you are done, leave a short `REPORT.md` in the workspace describing the kernel design you chose, what you measured, and why.

You have 24 hours.