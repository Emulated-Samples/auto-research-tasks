"""Tests for the performance-measure utilities."""

import numpy
import pytest
import torch

from cherimoya.performance import (
	calculate_performance_measures,
	jensen_shannon_distance,
	mean_squared_error,
	pearson_corr,
	smooth_gaussian1d,
	spearman_corr,
)


# --------- pearson_corr ----------------------------------------------------

def test_pearson_corr_perfect_positive():
	a = torch.tensor([[1.0, 2.0, 3.0, 4.0]])
	b = torch.tensor([[2.0, 4.0, 6.0, 8.0]])
	assert torch.allclose(pearson_corr(a, b), torch.tensor([1.0]), atol=1e-6)


def test_pearson_corr_perfect_negative():
	a = torch.tensor([[1.0, 2.0, 3.0, 4.0]])
	b = torch.tensor([[8.0, 6.0, 4.0, 2.0]])
	assert torch.allclose(pearson_corr(a, b), torch.tensor([-1.0]), atol=1e-6)


def test_pearson_corr_zero_when_one_input_is_constant():
	"""A constant array has zero variance — the function returns 0
	rather than NaN to keep downstream aggregation safe."""

	a = torch.tensor([[1.0, 2.0, 3.0, 4.0]])
	b = torch.tensor([[5.0, 5.0, 5.0, 5.0]])
	assert torch.allclose(pearson_corr(a, b), torch.tensor([0.0]))


def test_pearson_corr_broadcasts_over_leading_dims():
	a = torch.randn(3, 5, 10)
	b = torch.randn(3, 5, 10)
	out = pearson_corr(a, b)
	assert out.shape == (3, 5)


def test_pearson_corr_matches_numpy_reference():
	g = torch.Generator().manual_seed(0)
	a = torch.randn(4, 16, generator=g)
	b = torch.randn(4, 16, generator=g)

	expected = numpy.array([
		numpy.corrcoef(a[i].numpy(), b[i].numpy())[0, 1] for i in range(4)
	])
	got = pearson_corr(a, b).numpy()
	assert numpy.allclose(got, expected, atol=1e-5)


# --------- spearman_corr --------------------------------------------------

def test_spearman_corr_monotonic_relationship():
	a = torch.tensor([[1.0, 2.0, 3.0, 4.0]])
	b = torch.tensor([[1.0, 4.0, 9.0, 16.0]])  # monotonic but non-linear
	assert torch.allclose(spearman_corr(a, b), torch.tensor([1.0]), atol=1e-6)


# --------- mean_squared_error ---------------------------------------------

def test_mse_zero_when_equal():
	a = torch.randn(2, 5)
	assert torch.allclose(mean_squared_error(a, a),
		torch.zeros(2), atol=1e-7)


def test_mse_known_value():
	a = torch.tensor([[1.0, 2.0, 3.0]])
	b = torch.tensor([[2.0, 4.0, 6.0]])
	# Errors: 1, 4, 9 -> mean 14/3
	assert torch.allclose(mean_squared_error(a, b),
		torch.tensor([14.0 / 3.0]), atol=1e-6)


# --------- jensen_shannon_distance ----------------------------------------

def test_jsd_zero_for_identical_distributions():
	logits = torch.log(torch.tensor([[[0.25, 0.25, 0.25, 0.25]]]))
	counts = torch.tensor([[[1.0, 1.0, 1.0, 1.0]]])
	jsd = jensen_shannon_distance(logits, counts)
	assert torch.allclose(jsd.squeeze(), torch.tensor(0.0), atol=1e-5)


# --------- smooth_gaussian1d ----------------------------------------------

def test_gaussian_smoothing_preserves_total_signal_for_constant_input():
	"""A normalized Gaussian kernel applied to a constant signal returns
	the same constant in the interior; near the edges, conv1d's zero
	padding means the kernel sees fewer non-zero values, so we only
	check positions that are at least kernel_width//2 from each edge."""

	x = torch.ones(2, 1, 64)
	kw = 11
	y = smooth_gaussian1d(x, kernel_sigma=2.0, kernel_width=kw)
	half = kw // 2
	assert torch.allclose(y[..., half:-half], x[..., half:-half], atol=1e-4)


def test_gaussian_smoothing_reduces_variance():
	g = torch.Generator().manual_seed(0)
	x = torch.randn(1, 1, 64, generator=g)
	y = smooth_gaussian1d(x, kernel_sigma=3.0, kernel_width=15)
	assert y.var() < x.var()


# --------- calculate_performance_measures ---------------------------------

def test_calculate_performance_measures_subset_runs():
	"""Restricting to count metrics keeps the test fast and avoids the
	scikit-learn dependency from the labels branch."""

	g = torch.Generator().manual_seed(0)
	logits = torch.randn(2, 1, 16, generator=g)
	true_counts = torch.randint(0, 5, (2, 1, 16), generator=g).float()
	pred_logcounts = torch.randn(2, 1, generator=g)

	measures = calculate_performance_measures(
		logits, true_counts, pred_logcounts,
		measures=['count_pearson', 'count_spearman', 'count_mse'],
	)

	assert set(measures.keys()) == {'count_pearson', 'count_spearman', 'count_mse'}
	for k, v in measures.items():
		assert torch.isfinite(v).all(), k


def test_calculate_performance_measures_per_group_count_target():
	"""When signal_groups=[1, 2] and pred has 2 count outputs (one per
	group), the true target for each group is the SUM of that group's
	channels — not the total across all channels. So correlating
	prediction-equals-truth-per-group should give a perfect (or
	near-perfect) per-group Pearson."""

	g = torch.Generator().manual_seed(0)
	# 4 examples, 3 channels (1 unstranded + 1 stranded pair), length 8.
	logits = torch.randn(4, 3, 8, generator=g)
	true_counts = torch.randint(0, 7, (4, 3, 8), generator=g).float()

	# Build a "perfect" per-group prediction.
	per_channel = true_counts.sum(dim=-1)
	per_group = torch.stack([
		per_channel[:, 0],
		per_channel[:, 1] + per_channel[:, 2],
	], dim=-1)
	perfect_pred = torch.log(per_group + 1)

	measures = calculate_performance_measures(
		logits, true_counts, perfect_pred,
		measures=['count_pearson', 'count_mse'],
		signal_groups=[1, 2],
	)
	# Perfect predictions: MSE per group is ~0, Pearson is ~1.
	assert torch.allclose(measures['count_mse'], torch.zeros(2), atol=1e-5)
	assert torch.allclose(measures['count_pearson'],
		torch.ones(2), atol=1e-4)


def test_calculate_performance_measures_legacy_total_when_no_groups():
	"""When signal_groups=None and there are multiple count outputs, the
	legacy code-path collapses across all channels into a single total
	target. Documenting this fall-through here so the new code-path
	doesn't silently change it."""

	g = torch.Generator().manual_seed(0)
	logits = torch.randn(4, 3, 8, generator=g)
	true_counts = torch.randint(0, 7, (4, 3, 8), generator=g).float()

	# A "perfect" prediction against the TOTAL count, repeated for every
	# output head, must still produce Pearson == 1 in the legacy path.
	total = true_counts.sum(dim=(-1, -2), keepdim=False).unsqueeze(-1)
	pred = torch.log(total + 1).repeat(1, 3)

	measures = calculate_performance_measures(
		logits, true_counts, pred,
		measures=['count_pearson', 'count_mse'],
	)
	assert torch.allclose(measures['count_mse'], torch.zeros(3), atol=1e-5)


def test_calculate_performance_measures_signal_groups_sum_mismatch_raises():
	"""sum(signal_groups) must equal true_counts.shape[1] when groups
	are given and the prediction has one count per group. Otherwise the
	caller has wired the modalities up wrong and we'd silently pool
	channels into the wrong groups."""

	logits = torch.randn(2, 3, 8)
	true_counts = torch.randint(0, 5, (2, 3, 8)).float()
	pred = torch.randn(2, 2)  # 2 group-count outputs

	# signal_groups=[1, 1] sums to 2, but true_counts has 3 channels.
	with pytest.raises(ValueError, match="sum.signal_groups"):
		calculate_performance_measures(
			logits, true_counts, pred,
			measures=['count_pearson'], signal_groups=[1, 1])


def test_calculate_performance_measures_validates_signal_groups():
	"""The shared validator is invoked, so bad signal_groups (empty,
	zero entry, etc.) fail with a clear error rather than silently
	producing a degenerate target tensor."""

	logits = torch.randn(2, 3, 8)
	true_counts = torch.randint(0, 5, (2, 3, 8)).float()
	pred = torch.randn(2, 3)

	with pytest.raises(ValueError, match="positive ints"):
		calculate_performance_measures(
			logits, true_counts, pred,
			measures=['count_pearson'], signal_groups=[0, 3])
