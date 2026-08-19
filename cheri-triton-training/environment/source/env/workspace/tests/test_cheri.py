"""Tests for the CheriBlock."""

import pytest
import torch

from cherimoya.cheri import CheriBlock, HAS_TRITON


# --------- CheriBlock ------------------------------------------------------

def test_cheri_block_forward_shape_preserved():
	block = CheriBlock(n_filters=16, dilation=4)
	x = torch.randn(3, 64, 16)
	y = block(x)
	assert y.shape == x.shape


def test_cheri_block_default_expansion_is_two():
	block = CheriBlock(n_filters=16, dilation=1)
	assert block.expansion == 2
	assert block.linear1.out_features == 32
	assert block.linear2.in_features == 32


@pytest.mark.parametrize("expansion", [1, 2, 4])
def test_cheri_block_expansion_configurable(expansion):
	block = CheriBlock(n_filters=8, dilation=1, expansion=expansion)
	assert block.linear1.out_features == 8 * expansion
	assert block.linear2.in_features == 8 * expansion
	# Forward pass must still preserve shape regardless of expansion.
	x = torch.randn(2, 32, 8)
	assert block(x).shape == x.shape


def test_cheri_block_residual_uses_fixed_scale():
	"""Zeroing the inner MLP output (by zeroing both linear weights)
	collapses the block to the identity since X + 0 * residual_scale == X."""

	block = CheriBlock(n_filters=8, dilation=1)
	with torch.no_grad():
		block.linear1.weight.zero_()
		block.linear2.weight.zero_()
	x = torch.randn(2, 32, 8)
	y = block(x)
	assert torch.allclose(y, x, atol=1e-5)


def test_cheri_block_has_no_gamma_parameter():
	"""The block must not register a learnable channel-wise scaling
	parameter — the residual scale is a fixed scalar."""

	block = CheriBlock(n_filters=8, dilation=1)
	param_names = {n for n, _ in block.named_parameters()}
	assert 'gamma' not in param_names
	assert not hasattr(block, 'gamma')


def test_cheri_block_default_residual_scale():
	block = CheriBlock(n_filters=8, dilation=1)
	assert block.residual_scale == 0.15


def test_cheri_block_residual_scale_is_not_a_parameter():
	"""residual_scale is a plain Python float, not a learnable Parameter."""
	block = CheriBlock(n_filters=8, dilation=1, residual_scale=0.3)
	assert isinstance(block.residual_scale, float)
	param_names = {n for n, _ in block.named_parameters()}
	assert 'residual_scale' not in param_names


@pytest.mark.parametrize("residual_scale", [0.0, 0.05, 0.15, 1.0])
def test_cheri_block_residual_scale_controls_output(residual_scale):
	"""The output must be exactly X + residual_scale * (MLP path). Compare a
	block against an otherwise-identical block with residual_scale = 1, whose
	(output - input) is the unscaled MLP path."""

	block = CheriBlock(n_filters=8, dilation=1, residual_scale=residual_scale)
	unit = CheriBlock(n_filters=8, dilation=1, residual_scale=1.0)
	unit.load_state_dict(block.state_dict())

	x = torch.randn(2, 32, 8)
	y = block(x)
	if residual_scale == 0.0:
		# residual path contributes nothing: output equals input exactly.
		assert torch.allclose(y, x, atol=1e-6)
	else:
		mlp_path = unit(x) - x
		expected = x + mlp_path * residual_scale
		assert torch.allclose(y, expected, atol=1e-6)


def test_cheri_block_backward_produces_finite_grads():
	block = CheriBlock(n_filters=8, dilation=2)
	x = torch.randn(2, 32, 8, requires_grad=True)
	y = block(x).sum()
	y.backward()
	for name, p in block.named_parameters():
		assert p.grad is not None, name
		assert torch.isfinite(p.grad).all(), name
	assert torch.isfinite(x.grad).all()


# --------- GPU --------------------------------------------------------------

@pytest.mark.cuda
def test_cheri_block_runs_on_cuda():
	block = CheriBlock(n_filters=16, dilation=2).cuda()
	x = torch.randn(2, 64, 16, device='cuda')
	y = block(x)
	assert y.shape == x.shape
	assert y.is_cuda


def test_has_triton_flag_is_bool():
	"""HAS_TRITON is an informational flag reflecting whether Triton imported."""
	assert isinstance(HAS_TRITON, bool)


# --------- no_grad forward invariants -------------------------------------
#
# Properties the block's forward must satisfy in eval / no_grad mode. They
# hold for the pure-PyTorch block and must keep holding for any faster
# implementation of the forward path.

def _block_forward(block, x, *, no_grad):
	"""Run a block forward optionally under torch.no_grad()."""
	if no_grad:
		with torch.no_grad():
			return block(x)
	return block(x)


@pytest.mark.parametrize("n_filters,expansion", [(8, 1), (16, 2), (32, 4)])
def test_cheri_block_no_grad_matches_grad_cpu(n_filters, expansion):
	"""On CPU, the no_grad and grad-enabled forward paths must produce the
	same output. Any CUDA-only fast path must not change CPU behavior."""

	torch.manual_seed(0)
	block = CheriBlock(n_filters=n_filters, dilation=2, expansion=expansion)
	x = torch.randn(2, 64, n_filters)

	y_grad = _block_forward(block, x, no_grad=False)
	y_nograd = _block_forward(block, x, no_grad=True)

	assert torch.allclose(y_grad.detach(), y_nograd, atol=1e-6, rtol=1e-5)


def test_cheri_block_no_grad_forward_is_stable_across_calls():
	"""Repeated forwards under no_grad must be deterministic — the same input
	must give the same output across calls."""

	torch.manual_seed(0)
	block = CheriBlock(n_filters=16, dilation=2)
	x = torch.randn(2, 64, 16)

	with torch.no_grad():
		y1 = block(x)
		y2 = block(x)
		y3 = block(x.clone())

	assert torch.equal(y1, y2)
	assert torch.equal(y1, y3)


def test_cheri_block_residual_scale_attribute_preserved_after_no_grad_forward():
	"""The public `residual_scale` attribute must remain the original float
	regardless of any internal folding a fast forward path may perform, so
	users can introspect / save the model unchanged."""

	block = CheriBlock(n_filters=16, dilation=1, residual_scale=0.07)
	x = torch.randn(2, 32, 16)
	with torch.no_grad():
		_ = block(x)
	assert block.residual_scale == 0.07
	assert torch.isfinite(block.linear2.weight).all()


def test_cheri_block_load_state_dict_reflects_in_subsequent_no_grad_forward():
	"""After `load_state_dict`, the next no_grad forward must use the newly
	loaded weights. If a fast forward path caches anything derived from the
	weights, that cache must not go stale across a state-dict load."""

	torch.manual_seed(0)
	block_a = CheriBlock(n_filters=16, dilation=2)
	torch.manual_seed(1)
	block_b = CheriBlock(n_filters=16, dilation=2)
	x = torch.randn(2, 64, 16)

	with torch.no_grad():
		y_a_before = block_a(x)
		y_b = block_b(x)

	block_a.load_state_dict(block_b.state_dict())

	with torch.no_grad():
		y_a_after = block_a(x)

	assert torch.allclose(y_a_after, y_b, atol=1e-6, rtol=1e-5)
	assert not torch.allclose(y_a_after, y_a_before, atol=1e-3)


def test_cheri_block_in_place_param_update_reflects_in_subsequent_no_grad_forward():
	"""An optimizer step performs in-place updates on parameters. The same
	cache-invalidation concern as load_state_dict applies. Simulate an
	in-place update and verify the next no_grad forward sees the new weights."""

	torch.manual_seed(0)
	block = CheriBlock(n_filters=16, dilation=2)
	x = torch.randn(2, 64, 16)

	with torch.no_grad():
		y_before = block(x)

	with torch.no_grad():
		block.linear1.weight.mul_(0.5)
		block.linear2.weight.mul_(0.5)

	with torch.no_grad():
		y_after = block(x)

	assert not torch.allclose(y_after, y_before, atol=1e-3)


@pytest.mark.parametrize("n_filters,expansion", [(8, 1), (16, 1), (16, 2)])
def test_cheri_block_small_hidden_works_under_no_grad(n_filters, expansion):
	"""CheriBlock must support any (n_filters, expansion) combination
	regardless of grad mode — including small or awkward hidden widths. A fast
	path with shape constraints must fall back transparently rather than
	raise, so any model configuration keeps working."""

	block = CheriBlock(n_filters=n_filters, dilation=1, expansion=expansion)
	x = torch.randn(2, 32, n_filters)

	with torch.no_grad():
		y = block(x)  # must not raise
	assert y.shape == x.shape
	assert torch.isfinite(y).all()
