"""Tests for the EMA helper."""

import pytest
import torch
import torch.nn as nn

from cherimoya import EMA


class _Tiny(nn.Module):
	def __init__(self):
		super().__init__()
		self.lin = nn.Linear(4, 4)
		self.scalar = nn.Parameter(torch.tensor(0.0))


def _flatten(model):
	return torch.cat([p.detach().flatten() for _, p in model.named_parameters()])


def test_shadow_initialized_from_model_weights():
	model = _Tiny()
	ema = EMA(model, decay=0.9)
	for name, p in model.named_parameters():
		assert name in ema.shadow
		assert torch.equal(ema.shadow[name], p.detach())


def test_update_blends_toward_current_weights():
	"""After many updates with the model held fixed, shadow should converge
	to the current model weights regardless of starting state."""

	model = _Tiny()
	ema = EMA(model, decay=0.5)

	# Start with shadow at zero so the difference is obvious.
	for k in ema.shadow:
		ema.shadow[k].zero_()

	for _ in range(50):
		ema.update(model)

	for name, p in model.named_parameters():
		assert torch.allclose(ema.shadow[name], p.detach(), atol=1e-6)


def test_update_obeys_decay_factor():
	"""One update with decay=d should produce shadow = d*shadow + (1-d)*p."""
	model = _Tiny()
	ema = EMA(model, decay=0.7)

	# Capture initial shadow (= initial weights).
	old_shadow = {k: v.clone() for k, v in ema.shadow.items()}

	# Mutate the model so update has something to blend toward.
	with torch.no_grad():
		for p in model.parameters():
			p.add_(torch.ones_like(p))

	ema.update(model)

	for name, p in model.named_parameters():
		expected = 0.7 * old_shadow[name] + 0.3 * p.detach()
		assert torch.allclose(ema.shadow[name], expected, atol=1e-6)


def test_apply_shadow_then_restore_is_identity():
	model = _Tiny()
	ema = EMA(model)

	# Mutate shadow so apply_shadow is observable.
	for k in ema.shadow:
		ema.shadow[k].fill_(7.0)

	original = _flatten(model)
	ema.apply_shadow(model)
	assert (_flatten(model) == 7.0).all()

	ema.restore(model)
	assert torch.equal(_flatten(model), original)
	# After restore the backup is cleared so apply_shadow can be called again.
	assert ema._backup == {}


def test_apply_shadow_twice_without_restore_raises():
	model = _Tiny()
	ema = EMA(model)
	ema.apply_shadow(model)
	with pytest.raises(AssertionError):
		ema.apply_shadow(model)


def test_non_floating_buffers_are_not_tracked():
	class WithBuffer(nn.Module):
		def __init__(self):
			super().__init__()
			self.lin = nn.Linear(2, 2)
			self.register_buffer("counter", torch.zeros(1, dtype=torch.long))

	model = WithBuffer()
	ema = EMA(model)
	assert all(not k.endswith("counter") for k in ema.shadow)
