# grade_training.py
# Hidden grader for the cheri-triton-training problem, ITERATION 2 (training).
# Runs in the VERIFIER (root), never shipped to the agent. Invoked by
# environment/src/index.ts runTests against the agent's modified workspace/.
#
# Iteration 1 (grade_inference.py) graded the no_grad forward. Iteration 2 grades
# the DIFFERENTIABLE training path: when torch.is_grad_enabled() is True and the
# input is CUDA, the block must run its conv+norm through a hand-written Triton
# autograd.Function (fwd+bwd) and be fast, WITHOUT breaking gradients. The MLP may
# stay in PyTorch (as gold does). We grade three things:
#   (a) grad-enabled CUDA forward parity vs an independent pure-PyTorch reference,
#   (b) BACKWARD parity — gradients w.r.t. the input, the conv weight, and both
#       linear weights match autograd of the reference (the new iter-2 contract),
#   (c) fwd+bwd speedup vs the vanilla PyTorch block, anchored on LIVE gold.
# Correctness (a)+(b) is a pure GATE over a randomized grid; the score is the
# fp32-weighted fwd+bwd speedup. Emits one `GRADE_JSON: {...}` line.
#
# Anti-cheat posture (iteration 2):
#   - source scan bans torch.compile / torch.jit / torch.fx / dynamo,
#   - requires an actual @triton.jit definition in the block module,
#   - the reference (forward AND gradients) is computed INDEPENDENTLY here via
#     autograd over a pure-PyTorch recomputation that only reads the block's
#     parameters — never the agent's forward — so tampering cannot move the target,
#   - the fwd+bwd perf FLOOR is the primary "did you actually accelerate the
#     training path" check: a pure-PyTorch autograd path scores ~1x and fails it,
#     so a differentiable-but-unaccelerated fallback earns ~0 on perf,
#   - a zero-MLP fixed point (out == X, bit-exact) under grad, which no kernel
#     trick can fake — pins the residual + residual_scale and bans a spurious affine.

import argparse
import importlib.util
import json
import os
import sys

import torch
import torch.nn.functional as F


CONV_NORM_EPS = 1e-3

# Perf anchors (fwd+bwd training regime).
#   FLOOR: agent fwd+bwd speedup over vanilla below which perf score is 0. The
#     vanilla reference is RE-TIMED on the agent's GPU every run, so the floor is
#     hardware-relative. The training regime is intrinsically harder to accelerate
#     than inference (the backward dominates and is memory-bound), so gold itself
#     only reaches ~1.5-2x here — the floor sits at 1.2x: above a pure-torch /
#     unaccelerated autograd path (~1.0x, must score 0) but low enough that a real,
#     modestly-fused fwd+bwd kernel earns credit. The @triton.jit + banned-construct
#     gates and the backward-parity gate remain the primary anti-cheat; the floor
#     is the secondary "did you actually speed the training path up" check.
#   full-marks anchor: gold's OWN fwd+bwd speedup over vanilla, measured LIVE on
#     the same GPU each run (gold_cheri_fixture.py, autotune fixed so this is the
#     TUNED number, not the collapsed-sweep floor). "Full marks = match tuned gold
#     on this hardware." GOLD_FALLBACK is used only if the live measurement is
#     unavailable (fixture missing / gold errored).
#   NOTE: these defaults are seeded from L4 diagnostics (untuned gold fwd+bwd
#     ~1.5x fp32 / ~2.3x bf16, run_019f7100); the live anchor supersedes them, and
#     the FLOOR/GAMMA want one calibration run on the rollout GPU to finalize
#     (as iteration 1's were calibrated over run_019f7162 / run_019f7195).
FLOOR = 1.2
GOLD_FALLBACK = {'fp32': 1.6, 'bf16': 2.3, 'fp16': 2.3}

# Full-marks anchor = ANCHOR_MULT * gold's own live fwd+bwd speedup. The training
# ceiling is low (gold ~1.6x over vanilla), low enough that a good hand-written
# kernel can *match or edge* gold — under a bare 1.0x-gold anchor two of eight
# GPT-5.5 rollouts hit 100% (1.75x / 1.67x vs gold ~1.6x), failing the calibration
# bar of at most one full pass in >=8. Requiring the agent to beat gold by ~10% for
# full marks makes 100% rare (0/8 in the same cohort, max 0.99) while keeping a
# wide discriminating band. Gold-replay then scores ~0.85 (still clears
# minReplayScore 0.7). Set to 1.0 to restore "match gold = full marks".
ANCHOR_MULT = 1.1

# Scoring model — mirrors iteration 1:
#   - Correctness (forward + backward parity) is a pure GATE; any failure -> 0.
#   - fp32 weighted far above reduced precision (the hard, representative path).
#   - Per-case score is a CONCAVE map of the normalized speedup so the low band
#     where agents cluster gets a real gradient; gold still anchors 1.0.
DTYPE_PERF_WEIGHT = {'fp32': 1.0, 'bf16': 0.2, 'fp16': 0.2}
PERF_GAMMA = 0.5

# Forward parity tolerances (grad-enabled), same basis as iteration 1.
FWD_TOL = {'fp32': 2e-4, 'bf16': 2e-2, 'fp16': 5e-3}

# Backward parity tolerances — RELATIVE per grad tensor: ||g_a - g_ref|| /
# (||g_ref|| + eps). Gold's own backward agrees with fp32 autograd to ~1e-7 abs;
# an agent kernel with different accumulation order drifts more, so these are
# generous enough to pass a correct kernel but tight enough to catch a wrong one
# (a dropped term / miscomputed dx sits well above these).
BWD_REL_TOL = {'fp32': 3e-3, 'bf16': 1e-1, 'fp16': 4e-2}

DT = {'fp32': torch.float32, 'bf16': torch.bfloat16, 'fp16': torch.float16}


# Randomized-but-seeded correctness grid (varied widths, non-pow2 C, L%64!=0,
# N=1, dilation>=L). Seed makes it reproducible per run, not memorizable. Every
# case is graded on BOTH forward and backward parity.
def correctness_grid(seed):
	g = torch.Generator().manual_seed(seed)
	def r(lo, hi):
		return int(torch.randint(lo, hi + 1, (1,), generator=g).item())
	cases = [
		dict(N=r(1, 3), L=r(60, 200), C=16 * r(1, 4), dil=r(1, 8), dtype='fp32'),
		dict(N=1, L=r(100, 300), C=16 * r(2, 8), dil=r(1, 4), dtype='fp32'),
		dict(N=2, L=130, C=48, dil=200, dtype='fp32'),          # dilation >= L
		dict(N=2, L=2114, C=128, dil=256, dtype='fp32'),         # production
		dict(N=2, L=2114, C=128, dil=4, dtype='bf16'),
		dict(N=2, L=512, C=128, dil=16, dtype='fp16'),
	]
	return cases


# Perf grid: production-ish shapes, fwd+bwd timed.
PERF_GRID = [
	dict(N=64, L=2114, C=128, dil=1, dtype='fp32'),
	dict(N=64, L=2114, C=128, dil=256, dtype='fp32'),
	dict(N=64, L=2114, C=128, dil=4, dtype='bf16'),
	dict(N=512, L=2114, C=128, dil=4, dtype='fp32'),
]


def load_block_module(workspace):
	path = os.path.join(workspace, 'cherimoya', 'cheri.py')
	if not os.path.exists(path):
		fail(f'cheri.py not found at {path}')
	spec = importlib.util.spec_from_file_location('_agent_cheri', path)
	mod = importlib.util.module_from_spec(spec)
	spec.loader.exec_module(mod)
	return mod, path


def load_gold_module():
	"""Load the gold CheriBlock used as the LIVE fwd+bwd perf anchor.

	`gold_cheri_fixture.py` is a verbatim copy of gold's `cheri.py` (the tuned
	Triton implementation), committed next to this grader — agent-unreadable, same
	trust boundary. Regenerate when gold changes:
	  git show main:workspace/cherimoya/cheri.py > gold_cheri_fixture.py
	Returns None if absent or Triton is unusable, in which case scoring falls back
	to GOLD_FALLBACK."""
	path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
		'gold_cheri_fixture.py')
	if not os.path.exists(path):
		return None
	try:
		spec = importlib.util.spec_from_file_location('_gold_cheri', path)
		mod = importlib.util.module_from_spec(spec)
		spec.loader.exec_module(mod)
		if not getattr(mod, 'HAS_TRITON', False):
			return None
		return mod
	except Exception:
		return None


def conv_weight_c13(block):
	"""Depthwise conv weight as (C, 1, 3) fp32, layout-agnostic.

	Agent block: nn.Conv1d (`block.conv`, weight (C,1,3)). Gold: raw `conv_weight`
	(3, C). Read whichever is present, WITHOUT detaching, so autograd flows back to
	the leaf parameter when this is used inside the reference forward."""
	if hasattr(block, 'conv'):
		return block.conv.weight.float()                  # (C,1,3)
	return block.conv_weight.float().t().unsqueeze(1)     # (3,C) -> (C,1,3)


def conv_grad_c13(block):
	"""The conv weight's .grad as (C, 1, 3), layout-agnostic. None if unset."""
	if hasattr(block, 'conv'):
		g = block.conv.weight.grad
		return g.float() if g is not None else None
	g = block.conv_weight.grad
	return g.float().t().unsqueeze(1).contiguous() if g is not None else None


def reference_forward(block, x):
	"""Independent pure-PyTorch fp32 recomputation using the block's weights.

	Reads parameters only (never any agent function), so the agent cannot move the
	target by editing forward. Differentiable: because it reads the block's leaf
	parameters and `x`, calling .backward() on its output populates the SAME
	parameter .grad tensors the agent path does — that is how we get the reference
	gradients for the backward-parity check, and the vanilla fwd+bwd for perf."""
	C = block.n_filters
	d = block.dilation
	xf = x.float()
	# 3-tap depthwise dilated conv, channels-last (N, L, C).
	weight = conv_weight_c13(block)                       # (C,1,3)
	yt = F.conv1d(xf.transpose(1, 2), weight,
		padding=d, dilation=d, groups=C)
	y = yt.transpose(1, 2)
	# Per-example layer norm over (L, C).
	N = x.shape[0]
	flat = y.reshape(N, -1)
	mean = flat.mean(1, keepdim=True)
	var = flat.var(1, unbiased=False, keepdim=True)
	xhat = ((flat - mean) * (var + CONV_NORM_EPS).rsqrt()).reshape(y.shape)
	# MLP (no bias) + tanh-GELU + scaled residual.
	h = F.linear(xhat, block.linear1.weight.float())
	h = F.gelu(h, approximate='tanh')
	mlp = F.linear(h, block.linear2.weight.float())
	return xf + mlp * block.residual_scale


def collect_grads(block, x_leaf):
	"""Snapshot (clone) the grads we grade: input, conv weight, both linears."""
	cg = conv_grad_c13(block)
	return {
		'x': x_leaf.grad.float().clone() if x_leaf.grad is not None else None,
		'conv': cg.clone() if cg is not None else None,
		'l1': (block.linear1.weight.grad.float().clone()
			if block.linear1.weight.grad is not None else None),
		'l2': (block.linear2.weight.grad.float().clone()
			if block.linear2.weight.grad is not None else None),
	}


def rel_err(a, b):
	"""||a - b|| / (||b|| + eps). If reference grad is ~0, fall back to abs."""
	bn = b.norm().item()
	num = (a - b).norm().item()
	return num / (bn + 1e-12) if bn > 1e-8 else num


def time_ms(fn, warmup=10, iters=30):
	for _ in range(warmup):
		fn()
	torch.cuda.synchronize()
	ts = []
	for _ in range(iters):
		s = torch.cuda.Event(enable_timing=True)
		e = torch.cuda.Event(enable_timing=True)
		s.record(); fn(); e.record(); torch.cuda.synchronize()
		ts.append(s.elapsed_time(e))
	ts.sort()
	return ts[len(ts) // 2]


def fail(msg):
	print('GRADE_JSON: ' + json.dumps({
		'ok': False, 'score': 0.0, 'reason': msg,
		'correctness': None, 'perf': None}))
	sys.exit(0)


def grade_backward(Block, c, seed):
	"""Run one correctness case: grad-enabled forward + backward parity.

	Returns a dict with forward err, per-tensor backward rel errors, and pass/fail.
	Both the agent path and the fp32 reference read/write the SAME leaf parameter
	.grad tensors, so we capture the agent grads, zero, then the reference grads."""
	dt = DT[c['dtype']]
	torch.manual_seed(seed)
	blk = Block(n_filters=c['C'], dilation=c['dil']).cuda().to(dt)
	blk.train()  # grad-enabled training regime

	x = torch.randn(c['N'], c['L'], c['C'], device='cuda', dtype=dt)
	torch.manual_seed(seed + 1)
	g_up = torch.randn(c['N'], c['L'], c['C'], device='cuda')  # fp32 upstream grad

	# Warm up one fwd+bwd on this shape so we grade STEADY-STATE training
	# gradients. A Triton training kernel autotunes on the first call per shape,
	# and that first backward can carry autotune-benchmark residue (gold's own
	# path documents ~7e-2 on the first call, clean thereafter). Every real
	# training run is thousands of steady-state steps, so grading the warmed
	# kernel is the honest contract — and it removes a one-time artifact without
	# masking a genuinely wrong backward (a wrong kernel is wrong on every call).
	xw = x.detach().clone().requires_grad_(True)
	blk.zero_grad(set_to_none=True)
	blk(xw).backward(g_up.to(dt))
	blk.zero_grad(set_to_none=True)
	del xw

	# --- agent path (grad enabled -> must hit the Triton training path) ---
	xa = x.detach().clone().requires_grad_(True)
	blk.zero_grad(set_to_none=True)
	ya = blk(xa)
	if not ya.requires_grad:
		return {'case': c, 'pass': False, 'reason': 'grad-enabled forward has no grad_fn'}
	ya.backward(g_up.to(ya.dtype))
	a = collect_grads(blk, xa)

	# --- reference path (fp32 autograd over the independent recomputation) ---
	xr = x.detach().clone().float().requires_grad_(True)
	blk.zero_grad(set_to_none=True)
	yr = reference_forward(blk, xr)
	yr.backward(g_up.float())
	r = collect_grads(blk, xr)

	fwd_err = (ya.float() - yr).abs().max().item()
	fwd_tol = FWD_TOL[c['dtype']]

	rel = {}
	bwd_tol = BWD_REL_TOL[c['dtype']]
	bwd_ok = True
	for k in ('x', 'conv', 'l1', 'l2'):
		if a[k] is None:
			rel[k] = None
			bwd_ok = False  # a required gradient did not flow
			continue
		rel[k] = rel_err(a[k], r[k])
		if rel[k] > bwd_tol:
			bwd_ok = False

	passed = (fwd_err <= fwd_tol) and bwd_ok
	del blk, x, xa, xr, ya, yr, a, r
	torch.cuda.empty_cache()
	return {
		'case': c, 'fwd_err': fwd_err, 'fwd_tol': fwd_tol,
		'bwd_rel': rel, 'bwd_tol': bwd_tol, 'pass': passed,
	}


def fwd_bwd_step(forward_fn, x_leaf, block):
	"""A fwd+bwd closure: zero grads, run forward_fn(x), backprop a scalar loss."""
	def step():
		if x_leaf.grad is not None:
			x_leaf.grad = None
		block.zero_grad(set_to_none=True)
		y = forward_fn(x_leaf)
		y.float().sum().backward()
	return step


def main():
	ap = argparse.ArgumentParser()
	ap.add_argument('--workspace', default=os.environ.get(
		'WORKSPACE_PATH', '/hyperfocal/env/workspace'))
	ap.add_argument('--seed', type=int, default=0)
	args = ap.parse_args()

	if not torch.cuda.is_available():
		fail('CUDA not available in grader')

	mod, path = load_block_module(args.workspace)
	src = open(path).read()

	# --- Validity gates (any failure -> score 0) ---
	banned = ['torch.compile', 'torch.jit', 'torch.fx', 'torchdynamo', '_dynamo']
	hit = [b for b in banned if b in src]
	if hit:
		fail(f'banned construct(s) in cheri.py: {hit} (hand-written Triton only)')
	if '@triton.jit' not in src and 'triton.jit' not in src:
		fail('no @triton.jit kernel defined in cheri.py')
	if not getattr(mod, 'HAS_TRITON', False):
		fail('HAS_TRITON is False in the grader environment (triton import failed)')

	Block = mod.CheriBlock

	# Zero-MLP fixed point UNDER GRAD: with both linears zeroed the block must be
	# identity, bit-exact, in grad-enabled mode on CUDA. The MLP output is exactly
	# 0, so out == X regardless of the conv+norm numerics — this pins the residual
	# + residual_scale and bans any spurious learnable affine, and no kernel trick
	# can fake it.
	torch.manual_seed(1)
	b0 = Block(n_filters=32, dilation=2).cuda()
	with torch.no_grad():
		b0.linear1.weight.zero_(); b0.linear2.weight.zero_()
	b0.train()
	xz = torch.randn(2, 128, 32, device='cuda', requires_grad=True)
	yz = b0(xz)
	if not torch.equal(yz.detach(), xz.detach()):
		fail(f'zero-MLP block is not identity under grad; '
			f'max|y-x|={(yz - xz).abs().max().item():.2e}')
	# and the identity must be differentiable (dx = upstream grad).
	yz.sum().backward()
	if xz.grad is None or not torch.allclose(xz.grad, torch.ones_like(xz.grad)):
		fail('zero-MLP identity is not differentiable to the input (dx != 1)')
	del b0, xz, yz
	torch.cuda.empty_cache()

	# --- Correctness: grad-enabled forward + backward parity vs reference ---
	corr = []
	for i, c in enumerate(correctness_grid(args.seed)):
		try:
			corr.append(grade_backward(Block, c, seed=100 + i))
		except torch.cuda.OutOfMemoryError:
			torch.cuda.empty_cache()
			corr.append({'case': c, 'pass': False, 'reason': 'OOM'})

	n_pass = sum(r['pass'] for r in corr)
	all_correct = n_pass == len(corr)

	# --- Perf: agent fwd+bwd speedup vs vanilla, and gold's LIVE fwd+bwd speedup ---
	gold_mod = load_gold_module()
	GoldBlock = getattr(gold_mod, 'CheriBlock', None) if gold_mod else None
	perf = []
	if all_correct:
		for c in PERF_GRID:
			dt = DT[c['dtype']]
			try:
				torch.manual_seed(7)
				blk = Block(n_filters=c['C'], dilation=c['dil']).cuda().to(dt)
				blk.train()
				xt = torch.randn(c['N'], c['L'], c['C'], device='cuda', dtype=dt)

				xa = xt.detach().clone().requires_grad_(True)
				xr = xt.detach().clone().float().requires_grad_(True)
				t_a = time_ms(fwd_bwd_step(blk, xa, blk))
				t_r = time_ms(fwd_bwd_step(lambda z: reference_forward(blk, z), xr, blk))

				# Gold's own fwd+bwd speedup on this same GPU/shape is the
				# full-marks anchor. Reference compute is shape-determined; reuse t_r.
				gold_ratio = None
				if GoldBlock is not None:
					try:
						torch.manual_seed(7)
						gblk = GoldBlock(n_filters=c['C'],
							dilation=c['dil']).cuda().to(dt)
						gblk.train()
						gx = torch.randn(c['N'], c['L'], c['C'],
							device='cuda', dtype=dt).requires_grad_(True)
						t_g = time_ms(fwd_bwd_step(gblk, gx, gblk))
						gold_ratio = t_r / t_g
						del gblk, gx
					except Exception:
						gold_ratio = None

				perf.append({'case': c, 'ratio': t_r / t_a,
					'agent_ms': t_a, 'ref_ms': t_r, 'gold_ratio': gold_ratio})
				del blk, xt, xa, xr
				torch.cuda.empty_cache()
			except torch.cuda.OutOfMemoryError:
				torch.cuda.empty_cache()

	# --- Score: correctness gates perf ---
	if not all_correct:
		# Report the first failing case's detail so a debugging run is legible.
		first_fail = next((r for r in corr if not r['pass']), None)
		print('GRADE_JSON: ' + json.dumps({
			'ok': True, 'score': 0.0,
			'reason': f'correctness {n_pass}/{len(corr)} '
				f'(forward+backward parity gate) — perf gated to 0; '
				f'first failure: {first_fail}',
			'correctness': corr, 'perf': None}))
		return

	def anchor_for(p):
		gr = p.get('gold_ratio')
		base = gr if (gr is not None and gr > FLOOR) \
			else GOLD_FALLBACK.get(p['case']['dtype'], 1.6)
		# Full marks = beat gold by ANCHOR_MULT (training ceiling is low; see note).
		return ANCHOR_MULT * base

	if perf:
		def pscore(p):
			anchor = anchor_for(p)
			base = max(0.0, min(1.0,
				(p['ratio'] - FLOOR) / max(anchor - FLOOR, 1e-6)))
			return base ** PERF_GAMMA
		num = sum(DTYPE_PERF_WEIGHT.get(p['case']['dtype'], 1.0) * pscore(p)
			for p in perf)
		den = sum(DTYPE_PERF_WEIGHT.get(p['case']['dtype'], 1.0) for p in perf)
		perf_score = num / den if den else 0.0
	else:
		perf_score = 0.0

	score = perf_score
	fp32 = [p['ratio'] for p in perf if p['case']['dtype'] == 'fp32']
	live = any(p.get('gold_ratio') is not None for p in perf)
	anchor_src = 'live gold on this GPU' if live else 'GOLD_FALLBACK (gold not timed)'
	print('GRADE_JSON: ' + json.dumps({
		'ok': True, 'score': round(score, 4),
		'perf_score': round(perf_score, 4),
		'anchor': anchor_src,
		'reason': f'correct {n_pass}/{len(corr)} (fwd+bwd parity gate); '
			f'fp32 fwd+bwd median {sorted(fp32)[len(fp32)//2] if fp32 else 0:.2f}x; '
			f'full-marks anchor = {anchor_src}',
		'correctness': corr, 'perf': perf}))


if __name__ == '__main__':
	main()
