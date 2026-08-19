# losses.py
# Authors: The Cherimoya Authors


"""
This module contains the mixture loss function used for training Cherimoya
models, which is comprised of a multinomial log likelihood component and a
mean-squared error component. These losses are provided independently, so
other code can implement different ways of combining them into a single loss.
"""

import torch

from bpnetlite.losses import MNLLLoss

from .io import _validate_signal_groups


def _mixture_loss(y, y_hat_logits, y_hat_logcounts, labels=None,
		signal_groups=None):
	"""A function that takes in predictions and truth and returns the loss.

	This function takes in the observed integer read counts, the predicted logits,
	and the predicted logcounts, and returns per-*group* profile and count
	losses. Each signal group is scored as a **single multinomial over its
	channels and length jointly**: the group's channels are concatenated,
	one ``log_softmax`` is taken over the flattened ``channels * length``
	axis, and one MNLL is computed against the group's observed counts. This
	matches how :class:`~cherimoya.wrappers.ExpectedCountsWrapper` distributes
	a group's predicted counts across its channels and positions, so for a
	stranded ``(+, -)`` pair the relative magnitude between the two strands
	is part of the trained objective rather than an unconstrained per-channel
	gauge. (A per-channel normalization leaves that relative offset free,
	which lets a trained model collapse nearly all of a group's predicted
	signal onto a single strand at inference — the joint normalization here
	is what prevents that.) A single-channel (unstranded) group reduces
	exactly to a per-length multinomial, so accessibility models are
	unaffected. The count loss is per-group by construction (one prediction
	per group). When ``signal_groups`` is None the function falls back to
	per-channel losses (every channel is its own group), which matches the
	pre-grouping behavior.


	Parameters
	----------
	y: torch.Tensor, shape=(n, n_outputs, length)
		The observed counts for each example across each strand/output and at each
		position. This should likely be sparse integers.

	y_hat_logits: torch.Tensor, shape=(n, n_outputs, length)
		The predicted *logits* for each example across each strand/output and at
		each position. This will be normalized internally, so DO NOT run a softmax
		on your model.

	y_hat_logcounts: torch.Tensor, shape=(n, n_count_outputs)
		The predicted *log counts* for each example. ``n_count_outputs`` is
		``n_groups`` when ``signal_groups`` is given (per-group count head)
		or ``n_outputs`` otherwise (per-channel count head); the truth
		is derived from ``y`` accordingly.


	labels: torch.Tensor, shape=(n,), optional
		Whether the example is from a peak (1) or a non-peak (0). If provided, the
		profile loss will only be calculated on the peak examples. The count loss
		will always be calculated on the entire set of examples. If not provided,
		the profile loss will also be calculated on the entire set of examples.
		Default is None.

	signal_groups: list of int or None, optional
		Group sizes for the channel dimension of ``y``. When given, each
		group's channels are normalized jointly (a single multinomial over
		the group's ``channels * length``) and the true counts are pooled
		per group: a stranded ``(+, -)`` pair contributes one profile-loss
		term and one count-target. ``sum(signal_groups)`` must equal
		``y.shape[1]``. When None, every channel is treated as its own
		group (legacy behavior). Default is None.


	Returns
	-------
	profile_loss: torch.Tensor, shape=(n_groups,) or (n_outputs,)
		The per-group multinomial log likelihood (one joint multinomial
		over the group's channels and length, mean across examples). Falls
		back to per-channel shape ``(n_outputs,)`` when ``signal_groups``
		is None.

	count_loss: torch.Tensor, shape=(n_count_outputs,)
		The per-group (or per-track) mean-squared error on log(count+1),
		averaged across examples.
	"""

	# Per-channel true counts: (n, n_outputs).
	y_per_track = y.sum(dim=-1)

	if signal_groups is not None:
		_validate_signal_groups(signal_groups)
		if sum(signal_groups) != y_per_track.shape[-1]:
			raise ValueError(
				"sum(signal_groups)={} does not match y.shape[1]={}"
				.format(sum(signal_groups), y_per_track.shape[-1]))

	# Restrict the profile loss to peak examples when labels are given;
	# the count loss always runs on the full batch (below).
	if labels is not None:
		y_prof = y[labels == 1]
		logits_prof = y_hat_logits[labels == 1]
	else:
		y_prof = y
		logits_prof = y_hat_logits

	# Profile loss. Each signal group is scored as a single multinomial
	# over its channels and length jointly: the group's logits are
	# flattened to (n, group_channels * length), one log_softmax is
	# applied, and one MNLL is computed over the flattened axis. This
	# couples a stranded (+, -) pair so the relative magnitude between its
	# strands is part of the objective, matching how
	# ExpectedCountsWrapper spreads a group's counts at inference.
	#
	# When every group is size one (the accessibility / unstranded
	# multi-task case, and the signal_groups=None fallback) a joint
	# softmax over a one-channel group is identical to a per-channel
	# softmax over length, so this path is taken via the vectorized
	# branch below and is bit-identical to the pre-grouping code.
	groups = ([1] * y_prof.shape[1] if signal_groups is None
		else list(signal_groups))

	if all(g == 1 for g in groups):
		log_probs = torch.nn.functional.log_softmax(logits_prof, dim=-1)
		profile_loss = MNLLLoss(log_probs, y_prof).mean(dim=0)
	else:
		n = logits_prof.shape[0]
		per_group = []
		offset = 0
		for g in groups:
			logits_g = logits_prof[:, offset:offset+g].reshape(n, -1)
			y_g = y_prof[:, offset:offset+g].reshape(n, -1)
			log_probs_g = torch.nn.functional.log_softmax(logits_g, dim=-1)
			per_group.append(MNLLLoss(log_probs_g, y_g).mean(dim=0))
			offset += g
		profile_loss = torch.stack(per_group)

	# Per-group true counts (sum over each group's channels) so the count
	# target matches the per-group count head. Skipped for the all-size-one
	# case (pure identity) to avoid an unnecessary copy.
	if signal_groups is not None and len(signal_groups) != y_per_track.shape[-1]:
		groups_t = torch.tensor(signal_groups, device=y_per_track.device)
		group_idx = torch.repeat_interleave(
			torch.arange(len(signal_groups), device=y_per_track.device),
			groups_t)

		n = y_per_track.shape[0]
		y_per_group = torch.zeros(n, len(signal_groups),
			device=y_per_track.device, dtype=y_per_track.dtype)
		y_per_group.index_add_(1, group_idx, y_per_track)
		y_per_track = y_per_group

	# Count loss: per-example per-group squared error, then mean over examples.
	count_sq_err = (torch.log(y_per_track + 1) - y_hat_logcounts) ** 2
	count_loss = count_sq_err.mean(dim=0)

	return profile_loss, count_loss
