# bench_kernels.py
# Reconstruction of the benchmark script credited in docs/benchmarks.rst:15-18
# ("lives in the repository root but is not tracked in git") but which does not
# exist in the tree. See env-docs/02-gold-state-defects.md section 5.
#
# WRITTEN ON A CPU-ONLY BOX AND NEVER EXECUTED. Expect to debug it on first run.
#
# Two modes, mapping onto env-docs/04-gpu-port-plan.md:
#
#   drift  -- measure the ACTUAL max-abs disagreement between the three forward
#             paths. Answers open question #1: gold claims ~1e-5 in five places
#             and ~1e-2 in four, and two tests pin the same comparison at 1e-4
#             and 1e-2. This settles it with a number.
#
#   bench  -- median latency for fwd / bwd / fwd+bwd at production shapes.
#             Produces the t_gold baseline the verifier scores against. Never
#             hardcode the H200 table from benchmarks.rst; measure in the
#             verifier's own container.
#
# Deliberately NOT under tests/ -- tests/conftest.py:8-9 sets
# TORCH_COMPILE_DISABLE=1 / TORCHDYNAMO_DISABLE=1 for the whole suite, so a
# benchmark living there would silently measure the uncompiled path.
#
# Usage:
#   python bench_kernels.py drift
#   python bench_kernels.py bench --grid prod --json out.json
#   python bench_kernels.py bench --grid published   # cross-check vs the docs

import argparse
import importlib.util
import itertools
import json
import os
import sys

import torch

# Populated by _load_cheri() in main(). Module-level so the helpers below read
# naturally, but bound at runtime because the workspace path is a CLI arg.
CheriBlock = None
HAS_TRITON = False


def _load_cheri(workspace):
	"""Load cherimoya/cheri.py DIRECTLY, bypassing cherimoya/__init__.py.

	Deliberate: the package __init__ imports Cherimoya/EMA/wrappers, which drag
	in modisco, tangermeme, macs3, bam2bw and bpnet-lite (pyproject.toml:20-34)
	-- heavy, compile-happy deps that can fail or take 15+ minutes in a fresh
	sandbox. cheri.py itself imports only torch and triton (cheri.py:34-46), so
	loading the module file directly skips the whole dependency wall and lets a
	diagnostic run with nothing but `pip install torch triton`.

	Consequence: `pip install -e .` is NOT required to run this script.
	"""

	path = os.path.join(workspace, 'cherimoya', 'cheri.py')
	if not os.path.exists(path):
		sys.exit(f"cheri.py not found at {path} (pass --workspace)")

	spec = importlib.util.spec_from_file_location('_cheri_direct', path)
	mod = importlib.util.module_from_spec(spec)
	spec.loader.exec_module(mod)
	return mod


# Production config. Sources:
#   n_filters=128, expansion=2, n_layers=9   -> cherimoya_cli/defaults.py:35-37
#   dilations 2**i for i in 0..8             -> cherimoya.py:177
#   in_window=2114, batch 64 train/512 eval  -> defaults.py:40-41, :77
#   layout (N, L, C) channels-last           -> cherimoya.py:374
# NOTE: every block runs at L=in_window=2114. Trimming to out_window=1000
# happens AFTER the block loop (cherimoya.py:384), so L=1000 is never seen by a
# block. benchmarks.rst's L=1024 is synthetic and not a production shape.
PROD_C = 128
PROD_EXPANSION = 2
PROD_L = 2114
PROD_DILATIONS = [2 ** i for i in range(9)]  # 1, 2, 4, ..., 256

DTYPES = {
	'fp32': torch.float32,
	'bf16': torch.bfloat16,
	'fp16': torch.float16,
}


def _time_ms(fn, warmup, iters):
	"""Median wall-clock ms via CUDA events.

	The warmup is load-bearing, not hygiene: the first call at a given shape
	triggers Triton autotune, which both costs seconds AND (per
	env-docs/02 section 2) corrupts the first backward because
	_bwd_apply_kernel aliases its scratch buffer. benchmarks.rst:120 says the
	original script warmed up for the same reason.
	"""

	for _ in range(warmup):
		fn()
	torch.cuda.synchronize()

	times = []
	for _ in range(iters):
		start = torch.cuda.Event(enable_timing=True)
		end = torch.cuda.Event(enable_timing=True)
		start.record()
		fn()
		end.record()
		torch.cuda.synchronize()
		times.append(start.elapsed_time(end))

	times.sort()
	return times[len(times) // 2]


def _make_block(n_filters, dilation, expansion, dtype, seed=0):
	torch.manual_seed(seed)
	return CheriBlock(n_filters=n_filters, dilation=dilation,
		expansion=expansion).cuda().to(dtype)


# --------------------------------------------------------------------------
# drift mode
# --------------------------------------------------------------------------

def _drift_row(n_filters, expansion, dilation, L, N, dtype_name, eval_mode):
	"""One (config, dtype) drift measurement across the three paths."""

	dtype = DTYPES[dtype_name]
	hidden = n_filters * expansion

	gpu = _make_block(n_filters, dilation, expansion, dtype)
	cpu = CheriBlock(n_filters=n_filters, dilation=dilation,
		expansion=expansion).to(dtype)
	cpu.load_state_dict({k: v.cpu() for k, v in gpu.state_dict().items()})

	if eval_mode:
		gpu.eval()
		cpu.eval()

	torch.manual_seed(1234)
	x_cpu = torch.randn(N, L, n_filters, dtype=dtype)
	x_gpu = x_cpu.detach().cuda()

	# Path 2: training Triton (grad enabled).
	y_train = gpu(x_gpu)

	# Path 3: inference megakernel (no_grad + hidden % 16 == 0).
	with torch.no_grad():
		y_inf = gpu(x_gpu)

	def maxabs(a, b):
		return (a.float() - b.float()).abs().max().item()

	# Path 1: CPU reference. Guarded: CPU conv1d has patchy fp16 support and
	# may raise "not implemented for 'Half'". fp32 is the case that matters --
	# it is the one the contradictory tolerances are about -- so a CPU failure
	# on a reduced-precision dtype must not take the whole run down.
	try:
		with torch.no_grad():
			y_cpu = cpu(x_cpu)
		cpu_vs_train = maxabs(y_cpu, y_train.cpu())
		cpu_vs_inf = maxabs(y_cpu, y_inf.cpu())
	except (RuntimeError, NotImplementedError) as e:
		print(f"  CPU ref unavailable for dtype={dtype_name}: {e}",
			file=sys.stderr)
		cpu_vs_train = float('nan')
		cpu_vs_inf = float('nan')

	return {
		'n_filters': n_filters,
		'expansion': expansion,
		'hidden': hidden,
		'dilation': dilation,
		'L': L,
		'N': N,
		'dtype': dtype_name,
		'eval_mode': eval_mode,
		# True iff the megakernel actually dispatched. hidden % 16 == 0 is the
		# gate (cheri.py:915-926) -- note hidden=16 PASSES it, which is why
		# test_cheri_block_small_hidden_no_grad_matches_grad_cuda[16-1] is
		# mis-parametrized (env-docs/02 section 4).
		'megakernel_dispatched': bool(HAS_TRITON and hidden % 16 == 0),
		# THE headline number: this is what the 1e-4 vs 1e-2 tests disagree on.
		'train_vs_inf': maxabs(y_train, y_inf),
		'cpu_vs_train': cpu_vs_train,
		'cpu_vs_inf': cpu_vs_inf,
	}


def run_drift(args):
	rows = []
	# n_filters=32 mirrors the test suite; 128 is production width.
	# eval_mode toggles the bf16 weight cache (cheri.py:892-913) vs the inline
	# cast -- both should land in the same place numerically.
	for n_filters, dtype_name, eval_mode in itertools.product(
			[32, 128], ['fp32', 'bf16', 'fp16'], [False, True]):
		rows.append(_drift_row(
			n_filters=n_filters, expansion=2, dilation=2,
			L=128, N=2, dtype_name=dtype_name, eval_mode=eval_mode))

	# Large dilation at production L: the masked out-of-range loads are the
	# most likely place a reimplementation diverges, and nothing in the
	# existing suite covers dilation > 4.
	for dilation in [1, 64, 256]:
		rows.append(_drift_row(
			n_filters=PROD_C, expansion=PROD_EXPANSION, dilation=dilation,
			L=PROD_L, N=2, dtype_name='fp32', eval_mode=True))

	_report_drift(rows)
	return rows


def _report_drift(rows):
	print()
	print('DRIFT: max-abs disagreement between forward paths')
	print('  train_vs_inf is the number test_cheri.py:402 pins at 1e-4 and')
	print('  test_cheri.py:583 pins at 1e-2. Both cannot be right.')
	print()
	hdr = (f"{'C':>4} {'dil':>4} {'L':>5} {'dtype':>5} {'eval':>5} "
	       f"{'mega':>5} {'train_vs_inf':>13} {'cpu_vs_train':>13} {'cpu_vs_inf':>11}")
	print(hdr)
	print('-' * len(hdr))
	for r in rows:
		print(f"{r['n_filters']:>4} {r['dilation']:>4} {r['L']:>5} "
		      f"{r['dtype']:>5} {str(r['eval_mode']):>5} "
		      f"{str(r['megakernel_dispatched']):>5} "
		      f"{r['train_vs_inf']:>13.3e} {r['cpu_vs_train']:>13.3e} "
		      f"{r['cpu_vs_inf']:>11.3e}")

	print()
	fp32 = [r for r in rows if r['dtype'] == 'fp32' and r['megakernel_dispatched']]
	if fp32:
		worst = max(r['train_vs_inf'] for r in fp32)
		print(f"worst fp32 train_vs_inf = {worst:.3e}")
		print(f"  vs test_cheri.py:402 tolerance 1e-4 -> "
		      f"{'PASSES' if worst <= 1e-4 else 'FAILS (expected)'}")
		print(f"  vs test_cheri.py:583 tolerance 1e-2 -> "
		      f"{'PASSES' if worst <= 1e-2 else 'FAILS (unexpected!)'}")
		print()
		print('  Whichever survives becomes THE number. Rewrite all nine sites')
		print('  listed in env-docs/02-gold-state-defects.md section 1b.')


# --------------------------------------------------------------------------
# bench mode
# --------------------------------------------------------------------------

GRIDS = {
	# Production: what a real training step actually runs.
	# WARNING: N=512, L=2114, C=128 fp32 is ~554 MB per (N,L,C) tensor and the
	# backward allocates several. Expect OOM below ~40 GB; use --grid quick or
	# trim N if so.
	'prod': [
		dict(N=64, L=PROD_L, C=PROD_C, dilation=d, dtype=dt)
		for d in [1, 4, 16, 64, 256] for dt in ['fp32', 'bf16']
	] + [
		dict(N=512, L=PROD_L, C=PROD_C, dilation=4, dtype=dt)
		for dt in ['fp32', 'bf16']
	],
	# Reproduce docs/benchmarks.rst:22-23 (L=1024, C=128, dilation=4) purely to
	# validate the harness against the published table. If we can't roughly hit
	# training-fwd = 0.101 / 0.199 / 1.337 ms (fp32, N=16/64/512), either this
	# harness or the version story in benchmarks.rst:10-12 is wrong.
	'published': [
		dict(N=n, L=1024, C=128, dilation=4, dtype=dt)
		for n in [16, 64, 512] for dt in ['fp32', 'bf16', 'fp16']
	],
	'quick': [
		dict(N=8, L=256, C=128, dilation=4, dtype='fp32'),
	],
}


def _bench_case(case, warmup, iters):
	dtype = DTYPES[case['dtype']]
	block = _make_block(case['C'], case['dilation'], PROD_EXPANSION, dtype)

	torch.manual_seed(1234)
	x = torch.randn(case['N'], case['L'], case['C'],
		device='cuda', dtype=dtype, requires_grad=True)

	# fwd (grad enabled) -- the training forward path.
	def fwd():
		block(x)

	t_fwd = _time_ms(fwd, warmup, iters)

	# fwd under no_grad -- the inference megakernel.
	block.eval()

	def fwd_inf():
		with torch.no_grad():
			block(x)

	t_fwd_inf = _time_ms(fwd_inf, warmup, iters)
	block.train()

	# bwd only: build the graph once, re-run grad against it.
	y = block(x)
	gy = torch.randn_like(y)
	params = [p for p in block.parameters()] + [x]

	def bwd():
		torch.autograd.grad(y, params, gy, retain_graph=True)

	t_bwd = _time_ms(bwd, warmup, iters)

	# fwd+bwd -- the thing a training step actually pays for. benchmarks.rst
	# reports FORWARD ONLY and concedes at :88-90 that "training step time is
	# dominated by the backward pass" -- which is exactly the half this task
	# asks an agent to rebuild. Grading on fwd alone would grade the wrong work.
	def fwd_bwd():
		block.zero_grad(set_to_none=True)
		x.grad = None
		out = block(x)
		out.backward(gy)

	t_fwd_bwd = _time_ms(fwd_bwd, warmup, iters)

	return {
		**case,
		'fwd_ms': t_fwd,
		'fwd_inf_ms': t_fwd_inf,
		'bwd_ms': t_bwd,
		'fwd_bwd_ms': t_fwd_bwd,
	}


VANILLA_GRID = [
	dict(N=64, L=PROD_L, C=PROD_C, dilation=4, dtype='fp32'),
	dict(N=64, L=PROD_L, C=PROD_C, dilation=256, dtype='fp32'),
	dict(N=64, L=PROD_L, C=PROD_C, dilation=4, dtype='bf16'),
	dict(N=512, L=PROD_L, C=PROD_C, dilation=4, dtype='fp32'),
]


def _time_block(block, x, warmup, iters):
	"""(train_fwd, inf_fwd, fwd_bwd) ms for one block/input at current HAS_TRITON."""
	gy = torch.randn_like(block(x))

	def train_fwd():
		block(x)

	def inf_fwd():
		with torch.no_grad():
			block(x)

	def fwd_bwd():
		block.zero_grad(set_to_none=True)
		x.grad = None
		block(x).backward(gy)

	block.train()
	t_train = _time_ms(train_fwd, warmup, iters)
	t_fb = _time_ms(fwd_bwd, warmup, iters)
	block.eval()
	t_inf = _time_ms(inf_fwd, warmup, iters)
	block.train()
	return t_train, t_inf, t_fb


def run_vanilla(cheri, args):
	"""Gold Triton kernels vs pure-PyTorch, same GPU/shape/weights.

	Toggles the module-global HAS_TRITON that both fused_dilated_conv_norm and
	_can_use_inference_path read at call time, so False forces every path
	through _cheri_conv_norm_cpu. This is THE number that justifies the task:
	how much faster are the hand-written kernels than the vanilla block, and do
	they still agree numerically."""

	rows = []
	for case in VANILLA_GRID:
		dtype = DTYPES[case['dtype']]
		try:
			# One block, two behaviors. Same weights, same input.
			cheri.HAS_TRITON = True
			blk = _make_block(case['C'], case['dilation'], PROD_EXPANSION, dtype)
			torch.manual_seed(1234)
			x = torch.randn(case['N'], case['L'], case['C'],
				device='cuda', dtype=dtype, requires_grad=True)

			t_tri = _time_block(blk, x, args.warmup, args.iters)

			# Accuracy: Triton fwd/bwd vs vanilla fwd/bwd, same weights.
			cheri.HAS_TRITON = True
			yt = blk(x); gt, = torch.autograd.grad(yt.sum(), x, retain_graph=False)
			cheri.HAS_TRITON = False
			yv = blk(x); gv, = torch.autograd.grad(yv.sum(), x, retain_graph=False)
			fwd_err = (yt.float() - yv.float()).abs().max().item()
			grad_err = (gt.float() - gv.float()).abs().max().item()

			cheri.HAS_TRITON = False
			t_van = _time_block(blk, x, args.warmup, args.iters)
		except torch.cuda.OutOfMemoryError:
			print(f"OOM: {case} -- skipping", file=sys.stderr)
			torch.cuda.empty_cache()
			cheri.HAS_TRITON = True
			continue
		finally:
			cheri.HAS_TRITON = True

		rows.append({**case,
			'van': t_van, 'tri': t_tri,
			'fwd_err': fwd_err, 'grad_err': grad_err})

	print()
	print("VANILLA (pure PyTorch) vs GOLD TRITON -- same GPU, shape, weights")
	print(f"  device: {torch.cuda.get_device_name()}  "
	      f"(median of {args.iters}, {args.warmup} warmup)")
	print()
	hdr = (f"{'N':>5} {'dil':>4} {'dtype':>5} | "
	       f"{'train_fwd(v/t=x)':>22} | {'fwd+bwd(v/t=x)':>22} | "
	       f"{'inf_fwd(v/t=x)':>22} | {'fwd_err':>9} {'grad_err':>9}")
	print(hdr)
	print('-' * len(hdr))
	for r in rows:
		v, t = r['van'], r['tri']
		def cell(i):
			return f"{v[i]:6.2f}/{t[i]:6.2f}={v[i]/t[i]:4.1f}x"
		print(f"{r['N']:>5} {r['dilation']:>4} {r['dtype']:>5} | "
		      f"{cell(0):>22} | {cell(2):>22} | {cell(1):>22} | "
		      f"{r['fwd_err']:>9.2e} {r['grad_err']:>9.2e}")
	print()
	print("v/t = vanilla_ms / triton_ms = speedup. fwd_err/grad_err = max-abs")
	print("Triton-vs-vanilla disagreement (fp32 should be ~1e-6; the training")
	print("path uses no bf16, so it matches vanilla to fp32).")
	return rows


def run_bench(args):
	grid = GRIDS[args.grid]
	rows = []
	for case in grid:
		try:
			rows.append(_bench_case(case, args.warmup, args.iters))
		except torch.cuda.OutOfMemoryError:
			print(f"OOM: {case} -- skipping", file=sys.stderr)
			torch.cuda.empty_cache()

	print()
	print(f"BENCH (median of {args.iters}, after {args.warmup} warmup iters)")
	print(f"  device: {torch.cuda.get_device_name()}")
	print()
	hdr = (f"{'N':>5} {'L':>5} {'C':>4} {'dil':>4} {'dtype':>5} "
	       f"{'fwd':>9} {'fwd_inf':>9} {'bwd':>9} {'fwd+bwd':>9}")
	print(hdr)
	print('-' * len(hdr))
	for r in rows:
		print(f"{r['N']:>5} {r['L']:>5} {r['C']:>4} {r['dilation']:>4} "
		      f"{r['dtype']:>5} {r['fwd_ms']:>9.3f} {r['fwd_inf_ms']:>9.3f} "
		      f"{r['bwd_ms']:>9.3f} {r['fwd_bwd_ms']:>9.3f}")
	print()
	print('All times in ms. fwd_bwd_ms is the number to baseline the task on.')
	return rows


# --------------------------------------------------------------------------

def _check_autotune_configs(cheri):
	"""Print the training kernels' autotune configs.

	env-docs/02 section 3: _autotune_configs() (cheri.py:106) puts num_warps /
	num_stages inside the Config KWARGS DICT, while the inference path
	(cheri.py:501) correctly passes them as constructor args. If Config
	.all_kwargs() overrides from the attributes, all 12 configs collapse to the
	default and the four training kernels tune over nothing.

	If every line below prints the SAME num_warps/num_stages, that's confirmed.
	"""

	fn = getattr(cheri, '_autotune_configs', None)
	if fn is None:
		print('_autotune_configs not present (Triton missing? HAS_TRITON False?)')
		return

	print()
	print('AUTOTUNE CONFIG CHECK (env-docs/02 section 3)')
	print('  If num_warps/num_stages are identical on every row, the training')
	print('  kernels sweep 12 copies of one config and tune nothing.')
	print()
	seen = set()
	for c in fn():
		print(f"  kwargs={c.kwargs}  num_warps={c.num_warps}  "
		      f"num_stages={c.num_stages}")
		seen.add((c.num_warps, c.num_stages))

	print()
	if len(seen) == 1:
		print(f"  VERDICT: all configs collapse to {seen.pop()} -- the training")
		print("  kernels autotune over NOTHING. env-docs/02 section 3 CONFIRMED.")
		print("  Fix: triton.Config({}, num_warps=nw, num_stages=ns), as the")
		print("  inference path already does at cheri.py:501.")
	else:
		print(f"  VERDICT: {len(seen)} distinct (num_warps, num_stages) configs")
		print("  -- the sweep is live. env-docs/02 section 3 REFUTED.")


def main():
	global CheriBlock, HAS_TRITON

	p = argparse.ArgumentParser(description=__doc__)
	p.add_argument('mode', choices=['drift', 'bench', 'autotune', 'vanilla'])
	p.add_argument('--workspace', type=str,
		default=os.environ.get('WORKSPACE_PATH', '/hyperfocal/env/workspace'),
		help='dir containing cherimoya/cheri.py')
	p.add_argument('--grid', choices=list(GRIDS), default='prod')
	p.add_argument('--warmup', type=int, default=25,
		help='warmup iters; MUST be >0 to lock Triton autotune before timing')
	p.add_argument('--iters', type=int, default=50,
		help='timed iters; benchmarks.rst:13-14 used 50-100')
	p.add_argument('--json', type=str, default=None)
	args = p.parse_args()

	cheri = _load_cheri(args.workspace)
	CheriBlock = cheri.CheriBlock
	HAS_TRITON = cheri.HAS_TRITON

	print(f"torch={torch.__version__}")
	try:
		import triton
		print(f"triton={triton.__version__}")
	except ImportError:
		print('triton: NOT INSTALLED')
	print(f"cuda_available={torch.cuda.is_available()} HAS_TRITON={HAS_TRITON}")
	if torch.cuda.is_available():
		print(f"device={torch.cuda.get_device_name()}")
		cc = torch.cuda.get_device_capability()
		print(f"compute_capability=sm_{cc[0]}{cc[1]} "
		      f"bf16_supported={torch.cuda.is_bf16_supported()}")
		# T4 is sm_75 and has no bf16. The whole precision question is a bf16
		# question, so a T4 cannot answer it. See env-docs/04 section 0b.
		if not torch.cuda.is_bf16_supported():
			print()
			print('  WARNING: this GPU has NO bf16 support (T4/Turing?).')
			print('  The drift question is fundamentally about the bf16 MLP')
			print('  cast -- these results will NOT settle it. Use L4 or A10.')

	if args.mode == 'autotune':
		_check_autotune_configs(cheri)
		return

	if not torch.cuda.is_available():
		sys.exit('CUDA required. This script measures GPU kernels.')
	if not HAS_TRITON:
		sys.exit('Triton required (HAS_TRITON is False).')

	if args.mode == 'drift':
		rows = run_drift(args)
	elif args.mode == 'vanilla':
		rows = run_vanilla(cheri, args)
	else:
		rows = run_bench(args)

	if args.json:
		with open(args.json, 'w') as f:
			json.dump(rows, f, indent=2)
		print(f"\nwrote {args.json}")


if __name__ == '__main__':
	main()
