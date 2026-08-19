Saving and Loading Models
=========================

Cherimoya checkpoints store the constructor arguments next to the
parameter state dict. This format is robust to source-layout changes
and is safe to load with PyTorch's ``weights_only=True`` flag, which
the loader uses by default.


Saving
------

.. code-block:: python

   from cherimoya import Cherimoya

   model = Cherimoya(n_filters=128, n_layers=9)
   # ... train ...
   model.save("my_model.torch")

The saved payload is a dict with two keys:

* ``config`` — the kwargs needed to reconstruct the model (every
  argument to ``Cherimoya.__init__``).
* ``state_dict`` — the parameter tensors.


Loading
-------

.. code-block:: python

   from cherimoya import Cherimoya

   # CPU by default — fine for inspection or CPU-only inference.
   model = Cherimoya.load("my_model.torch")

   # Or load directly onto a GPU:
   model = Cherimoya.load("my_model.torch", device="cuda")

.. important::

   The weights inside a checkpoint are the **EMA snapshot**, not the
   running training weights. ``model.fit`` applies the EMA shadow
   before every save, both for the best-by-validation checkpoint and
   the final-epoch checkpoint. This is what produces the smoothed
   validation numbers reported in the training log; the model you
   load and use at inference is the same one those numbers describe.

The loader:

1. Reads the payload with ``torch.load(..., weights_only=True)``.
2. Reconstructs the module via ``Cherimoya(**payload['config'])``.
3. Calls ``load_state_dict`` with the saved state dict.
4. Moves the module to ``device`` before returning.


Best-of and final checkpoints
-----------------------------

``model.fit(...)`` writes two checkpoint files:

* ``{model.name}.torch`` — the best checkpoint observed during
  training, selected by validation count Pearson correlation. Saved
  after EMA weights are applied.
* ``{model.name}.final.torch`` — the model at the very end of
  training, also with EMA weights applied.

Both are produced via ``model.save(...)`` and are loadable with
``Cherimoya.load(...)``.

The ``.log`` file emitted alongside the checkpoints is a plain TSV
with one row per epoch. It is not used by the loader; it is for your
own inspection.


Compatibility
-------------

Checkpoints saved with the v0.1.0+ ``model.save`` format are forward-
and backward-compatible across point releases as long as the
constructor signature does not change. The version of Cherimoya you
trained with is recorded implicitly by the keys in the ``config``
dict — if you later add or rename a constructor argument, you may
need to map old keys to new ones before loading.

The pinned-version warning in the README is about this: if a future
release renames a constructor argument, an older checkpoint will
fail to load until you do the mapping.
