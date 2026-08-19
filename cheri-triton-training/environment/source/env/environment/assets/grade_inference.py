# grade_inference.py
# Hidden grader for the cheri-triton-kernel problem, iteration 1 (inference).
# Runs in the VERIFIER (root), never shipped to the agent. Invoked by
# environment/src/index.ts runTests against the agent's modified workspace/.
#
# Grades ONE thing: is the block's no_grad CUDA forward (a) still correct vs an
# independent pure-PyTorch reference, and (b) fast, via real Triton. Backward is
# NOT graded in iteration 1.
#
# Emits a single line `GRADE_JSON: {...}` that runTests parses.
#
# Anti-cheat posture (iteration 1):
#   - source scan bans torch.compile / torch.jit / torch.fx / dynamo,
#   - requires an actual @triton.jit definition in the block module,
#   - the reference is computed INDEPENDENTLY here (never via agent code), so
#     tampering with the block's forward cannot move the target,
#   - the perf floor (>= FLOOR x over pure PyTorch) is itself anti-cheat: a
#     pure-torch or unaccelerated path scores ~1x and fails the floor,
#   - a zero-MLP fixed point (out == X, bit-exact) that no precision trick can fake.

import argparse
import importlib.util
import json
import os
import re
import sys

import torch
import torch.nn.functional as F


CONV_NORM_EPS = 1e-3

# Perf anchors.
#   FLOOR: the agent's speedup-over-vanilla below which perf score is 0. This is
#     a fixed multiple of the pure-PyTorch reference, which is RE-TIMED on the
#     agent's GPU every run, so the floor is already hardware-relative. Set at
#     1.5x: comfortably above a fake / pure-torch path (~1.0x, which must score 0),
#     but low enough that a correct-but-modest kernel (~1.7x — real, if unfused
#     work) earns non-trivial credit rather than rounding to zero. The @triton.jit
#     and banned-construct gates remain the primary anti-cheat; the floor is the
#     secondary "did you actually accelerate anything" check.
#   full-marks anchor: gold's OWN speedup-over-vanilla, measured LIVE on the same
#     GPU each run (see load_gold_module + the perf loop; gold_cheri_fixture.py).
#     "Full marks =
#     match gold on this hardware, whatever it is." GOLD_FALLBACK is used only if
#     the live gold measurement is unavailable (fixture missing / gold errored) —
#     the old fixed L4 constants, kept so grading never hard-fails.
FLOOR = 1.5
GOLD_FALLBACK = {'fp32': 7.0, 'bf16': 10.0, 'fp16': 10.0}

# Scoring model (calibrated from the run_019f7162 / run_019f7195 rollouts):
#   - Correctness is a pure GATE. Every case must pass; any failure -> score 0.
#     No correctness credit is folded into the score — a correct-but-slow kernel
#     is not worth a baseline 0.5. The score IS the (weighted) speedup.
#   - fp32 is weighted far above the reduced-precision dtypes. fp32 is the hard,
#     representative path (no precision shortcut); bf16 saturates near ~11x and
#     would otherwise inflate the mean and compress the visible spread.
#   - Per-case score is a CONCAVE map of the normalized speedup:
#     pscore = ((ratio-FLOOR)/(anchor-FLOOR)) ** PERF_GAMMA, clamped to [0,1].
#     Real agent attempts cluster low (fp32 ~1.7-3.5x on L4, well under gold's
#     ~7x). A linear map crushes that whole band into the bottom ~30% of the
#     range with almost no gradient; gamma < 1 stretches the low end so early
#     improvements are rewarded, while gold still anchors 1.0 with headroom above
#     the agent band. gamma=1.0 recovers the original linear scoring. Concavity
#     also pushes gold-replay up, never down, so minReplayScore stays safe.
DTYPE_PERF_WEIGHT = {'fp32': 1.0, 'bf16': 0.2, 'fp16': 0.2}
PERF_GAMMA = 0.5

FWD_TOL = {'fp32': 2e-4, 'bf16': 2e-2, 'fp16': 5e-3}

# Randomized-but-seeded correctness grid (varied widths, non-pow2 C, L%64!=0,
# N=1, dilation>=L). Seed makes it reproducible per run, not memorizable.
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


# Perf grid: production-ish shapes only.
PERF_GRID = [
	dict(N=64, L=2114, C=128, dil=1, dtype='fp32'),
	dict(N=64, L=2114, C=128, dil=256, dtype='fp32'),
	dict(N=64, L=2114, C=128, dil=4, dtype='bf16'),
	dict(N=512, L=2114, C=128, dil=4, dtype='fp32'),
]

DT = {'fp32': torch.float32, 'bf16': torch.bfloat16, 'fp16': torch.float16}


def load_block_module(workspace):
	path = os.path.join(workspace, 'cherimoya', 'cheri.py')
	if not os.path.exists(path):
		fail(f'cheri.py not found at {path}')
	spec = importlib.util.spec_from_file_location('_agent_cheri', path)
	mod = importlib.util.module_from_spec(spec)
	spec.loader.exec_module(mod)
	return mod, path


def load_gold_module():
	"""Load the gold CheriBlock used as the LIVE perf anchor.

	`gold_cheri_fixture.py` is a verbatim copy of gold's `cheri.py` (the tuned
	Triton implementation), committed next to this grader in environment/assets/
	— agent-unreadable, same trust boundary as the grader. Regenerate it when
	gold changes:  git show main:workspace/cherimoya/cheri.py > gold_cheri_fixture.py

	A committed fixture (not a git ref) because the verifier's clone is not
	guaranteed to have the `main` ref at grade time. Returns None if the fixture
	is absent or Triton is unusable, in which case scoring falls back to the
	fixed GOLD_FALLBACK constants."""
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
	"""Return the depthwise conv weight as (C, 1, 3) fp32, layout-agnostic.

	The agent-facing block exposes an nn.Conv1d (`block.conv`, weight already
	(C,1,3)); gold exposes a raw `conv_weight` parameter of shape (3, C). Read
	whichever is present so this grader validates both the agent's solution and
	the gold-replay solution unchanged."""
	if hasattr(block, 'conv'):
		return block.conv.weight.float()                  # (C,1,3)
	return block.conv_weight.float().t().unsqueeze(1).contiguous()   # (3,C)->(C,1,3)


def reference_forward(block, x):
	"""Independent pure-PyTorch fp32 ground truth using the block's weights.

	Deliberately does NOT call any agent function — reads parameters only, so an
	agent cannot move the target by editing the block's forward."""
	C = block.n_filters
	d = block.dilation
	xf = x.float()
	# 3-tap depthwise dilated conv, channels-last (N, L, C).
	weight = conv_weight_c13(block)                       # (C,1,3)
	yt = F.conv1d(xf.transpose(1, 2).contiguous(), weight,
		padding=d, dilation=d, groups=C)
	y = yt.transpose(1, 2).contiguous()
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


def time_ms(fn, warmup=15, iters=40):
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

	# Zero-MLP fixed point: with both linears zeroed the block must be identity,
	# bit-exact, under no_grad on CUDA. No precision trick can fake this.
	# Zero the weights BEFORE .eval() so any eval-time weight cache captures the
	# zeros — otherwise a cache built at eval() holds the pre-zero weights and the
	# MLP is not actually zero (this is exactly the stale-cache trap).
	torch.manual_seed(1)
	b0 = Block(n_filters=32, dilation=2).cuda()
	with torch.no_grad():
		b0.linear1.weight.zero_(); b0.linear2.weight.zero_()
	b0.eval()
	with torch.no_grad():
		xz = torch.randn(2, 128, 32, device='cuda')
		yz = b0(xz)
	if not torch.equal(yz, xz):
		fail(f'zero-MLP block is not identity; max|y-x|={(yz-xz).abs().max().item():.2e}')

	# --- Correctness: no_grad CUDA forward vs independent reference ---
	corr = []
	for c in correctness_grid(args.seed):
		dt = DT[c['dtype']]
		torch.manual_seed(100 + len(corr))
		blk = Block(n_filters=c['C'], dilation=c['dil']).cuda().to(dt).eval()
		x = torch.randn(c['N'], c['L'], c['C'], device='cuda', dtype=dt)
		with torch.no_grad():
			y = blk(x)
		ref = reference_forward(blk, x)
		err = (y.float() - ref).abs().max().item()
		tol = FWD_TOL[c['dtype']]
		corr.append({'case': c, 'err': err, 'tol': tol, 'pass': err <= tol})

	n_pass = sum(r['pass'] for r in corr)
	all_correct = n_pass == len(corr)

	# --- Perf: agent speedup vs vanilla, and gold's LIVE speedup on this GPU ---
	gold_mod = load_gold_module()
	GoldBlock = getattr(gold_mod, 'CheriBlock', None) if gold_mod else None
	perf = []
	if all_correct:
		for c in PERF_GRID:
			dt = DT[c['dtype']]
			try:
				torch.manual_seed(7)
				blk = Block(n_filters=c['C'], dilation=c['dil']).cuda().to(dt).eval()
				x = torch.randn(c['N'], c['L'], c['C'], device='cuda', dtype=dt)

				def agent():
					with torch.no_grad():
						blk(x)

				def ref():
					with torch.no_grad():
						reference_forward(blk, x)

				t_a = time_ms(agent)
				t_r = time_ms(ref)

				# Gold's own speedup on this same GPU/shape is the full-marks
				# anchor. The reference compute is shape-determined, so reuse t_r.
				gold_ratio = None
				if GoldBlock is not None:
					try:
						torch.manual_seed(7)
						gblk = GoldBlock(n_filters=c['C'],
							dilation=c['dil']).cuda().to(dt).eval()
						gx = torch.randn(c['N'], c['L'], c['C'],
							device='cuda', dtype=dt)

						def gold():
							with torch.no_grad():
								gblk(gx)

						t_g = time_ms(gold)
						gold_ratio = t_r / t_g
						del gblk, gx
					except Exception:
						gold_ratio = None

				perf.append({'case': c, 'ratio': t_r / t_a,
					'agent_ms': t_a, 'ref_ms': t_r, 'gold_ratio': gold_ratio})
			except torch.cuda.OutOfMemoryError:
				torch.cuda.empty_cache()

	# --- Score: correctness gates perf ---
	if not all_correct:
		print('GRADE_JSON: ' + json.dumps({
			'ok': True, 'score': 0.0,
			'reason': f'correctness {n_pass}/{len(corr)} — perf gated to 0',
			'correctness': corr, 'perf': None}))
		return

	# Correctness passed (gate). Score IS the fp32-weighted speedup — no
	# correctness baseline. Full marks per case = matching gold's LIVE speedup on
	# this GPU; each case is clamped (ratio - FLOOR)/(anchor - FLOOR) and combined
	# as a weighted mean with fp32 dominant. anchor falls back to GOLD_FALLBACK if
	# gold could not be timed (fixture missing / gold errored / degenerate).
	def anchor_for(p):
		gr = p.get('gold_ratio')
		if gr is not None and gr > FLOOR:
			return gr
		return GOLD_FALLBACK.get(p['case']['dtype'], 7.0)

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
		'reason': f'correct {n_pass}/{len(corr)} (gate); '
			f'fp32 median {sorted(fp32)[len(fp32)//2] if fp32 else 0:.1f}x; '
			f'full-marks anchor = {anchor_src}',
		'correctness': corr, 'perf': perf}))


if __name__ == '__main__':
	main()
