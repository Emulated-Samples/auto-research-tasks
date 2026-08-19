Benchmarks
==========

The Cheri Block currently runs as a single pure-PyTorch path on every
device — the same code executes on CPU and GPU, and the block
participates in autograd automatically. See :doc:`architecture` for a
description of the block.

Performance benchmarks are not currently published. Timing depends
heavily on the device, the PyTorch build, the input shape, and whether
``torch.compile`` is enabled, so any numbers quoted here would be
misleading without the exact environment that produced them. If you
need figures for your own setup, time the model directly on your target
hardware and batch size.
