# Project workspace

Implement the representation-learning system under `submission/`. Depend only on
files under `submission/` and your own temporary development files.

`format_stub/` holds a `train.npz`, an `eval.npz` and a `config.json` with the
same keys and dtypes your programs will receive, filled with structureless
Gaussian noise. Use it to exercise your entry points -- the config keys, the
output schema, and how long your `fit` takes at a realistic shape. It contains no
latent objects and no geometry, so fitting it teaches you nothing about the real
data, which arrives in the run directory when your programs are invoked.
