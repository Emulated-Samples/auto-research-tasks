I have a research question I want you to investigate. `workspace/` is varseq, a library for training and interpreting sequence-to-function models of gene regulation. I have a trained model that maps a DNA sequence to enhancer activity for two promoter contexts, developmental and housekeeping, and I want to design sequences that drive a chosen activity as high as possible.

Implement `varseq.design_task.design`. How you search the sequence space is up to you. The expected interface is as follows:

```python
def design(oracle, track, budget, length, n_designs, seed) -> list[str]:
    ...
```

- `oracle` is a callable, your only access to the model: `oracle(seqs) -> array (len(seqs), n_tracks)`. It is metered (every sequence you score counts against `budget`, and you may call it however you like until then).
- `track` is the output head to drive up. `budget` is the maximum number of sequences you may score through `oracle`, total. `length` is the exact length of every sequence you return. `n_designs` is how many to return. `seed` seeds your randomness.

There is no internet access, but a venv with the dependencies you need is already set up in the workspace. `varseq.oracle.load_oracle()` connects you to the same scoring service that backs the oracle handed to `design()` at evaluation time, so you can develop against it directly. Metering is not on the honour system: the service counts every sequence it is asked to score, and the count is kept outside your process — development queries included. You have 200,000 scored sequences for the whole session; the object `load_oracle()` returns reports `.used`, `.budget` and `.remaining`. That total is measured and enforced when your `design()` is graded, so spend development queries deliberately. A set of real regulatory sequences is available via `varseq.oracle.load_seed_sequences()`.

Your designs will be scored on a different, held-out model you cannot query. A sequence that merely maxes out the oracle by exploiting its quirks won't transfer, so aim for designs whose regulatory signal is real and holds up under a model that never saw your search. `design()` must work for whatever track, budget, and starting condition it is given.

Contracts:

- Deterministic given the arguments and seed.
- Return exactly `n_designs` sequences, each exactly `length` long, over the `A/C/G/T` alphabet.
- Stay within `budget` oracle scores.
- Invalid arguments raise `ValueError` (nonpositive budget/length/n_designs, a budget too small to score every design at least once, an invalid track or length).

Everything runs on CPU.

You have 24 hours.
