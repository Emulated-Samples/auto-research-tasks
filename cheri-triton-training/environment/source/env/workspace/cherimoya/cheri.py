# cheri.py
# Author: The Cherimoya Authors

"""The Cheri block — Cherimoya's convolutional block for genomics.

Each block mixes spatial information with a depthwise dilated convolution,
normalizes each example, mixes channels with a small MLP, and adds the result
back through a scaled residual connection. The implementation is plain PyTorch.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


try:
	import triton
	import triton.language as tl
	HAS_TRITON = True
except ImportError:
	HAS_TRITON = False


NORM_EPS = 1e-3


class CheriBlock(torch.nn.Module):
	"""A single Cheri Block.

	The Cheri Block is the core building block of the Cherimoya model. It
	adapts the ConvNeXt block to noisy genomics data, mixing spatial and
	channel information cheaply while remaining stable to train.

	On an input of shape ``(N, L, C)`` it applies, in order:

	1. A 3-tap depthwise dilated convolution over the length axis, mixing
	   spatial information independently within each channel. The kernel reads
	   positions ``(i - dilation, i, i + dilation)`` at each output position
	   ``i``, with zero padding outside the sequence.
	2. A layer normalization applied per example over the entire
	   ``(length, channel)`` plane: each example is normalized by a single mean
	   and standard deviation computed across all of its positions and channels,
	   with no learnable affine term. Statistics are computed in fp32 with an
	   epsilon of ``1e-3``.
	3. A pointwise linear expansion from ``C`` to ``expansion * C`` channels.
	4. A tanh-approximated GELU non-linearity.
	5. A pointwise linear contraction back to ``C`` channels.
	6. A residual connection in which the MLP output is scaled by a fixed
	   constant (``residual_scale``) before being added to the input, keeping
	   the residual path near-identity at initialization.

	Parameters
	----------
	n_filters: int
		The number of channels (the C dimension).

	dilation: int
		Dilation rate for the depthwise convolution. The kernel reads from
		positions ``(i - dilation, i, i + dilation)`` at each output position
		``i``, with zero padding outside the sequence.

	expansion: int, optional
		The factor by which the inner MLP expands the channel dimension. The
		first projection maps ``n_filters -> expansion * n_filters`` and the
		second projects back. Default is 2.

	residual_scale: float, optional
		Fixed scalar applied to the MLP output before it is added back to the
		residual stream. Default is 0.15.
	"""

	def __init__(self, n_filters, dilation, expansion=2, residual_scale=0.15):
		super().__init__()
		self.n_filters = n_filters
		self.dilation = dilation
		self.expansion = expansion
		self.residual_scale = residual_scale

		hidden = expansion * n_filters

		self.conv = torch.nn.Conv1d(n_filters, n_filters, kernel_size=3,
			padding=dilation, dilation=dilation, groups=n_filters, bias=False)
		self.linear1 = torch.nn.Linear(n_filters, hidden, bias=False)
		self.linear2 = torch.nn.Linear(hidden, n_filters, bias=False)
		self.activation = torch.nn.GELU(approximate='tanh')

		torch.nn.init.trunc_normal_(self.conv.weight, std=0.02)
		torch.nn.init.trunc_normal_(self.linear1.weight, std=0.02)
		torch.nn.init.trunc_normal_(self.linear2.weight, std=0.02)

	def forward(self, X):
		"""Run the block on an input of shape (N, L, C)."""

		# Depthwise dilated convolution over the length axis. Conv1d expects
		# (N, C, L), so transpose in and back out.
		Y = self.conv(X.transpose(1, 2)).transpose(1, 2)

		# Per-example layer norm over the whole (L, C) plane: no affine term,
		# statistics accumulated in fp32.
		N = Y.shape[0]
		flat = Y.reshape(N, -1).float()
		mean = flat.mean(dim=1, keepdim=True)
		var = flat.var(dim=1, unbiased=False, keepdim=True)
		flat = (flat - mean) * (var + NORM_EPS).rsqrt()
		Y = flat.reshape(Y.shape).to(Y.dtype)

		# Channel-mixing MLP with a scaled residual connection.
		Y = self.linear2(self.activation(self.linear1(Y)))
		return X + Y * self.residual_scale
