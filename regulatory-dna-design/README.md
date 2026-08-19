# Regulatory DNA design under a query budget

Design DNA sequences that drive a chosen enhancer program as high as possible, with a metered activity model as the only way to score a candidate, and be graded on a different model the search never touched.

Models find exhaustive single-base hill climbing within minutes, stall a few thousand queries later, and never turn the rest of the allowance into a search whose designs transfer.

## The task

`workspace/` is varseq, a library for training and interpreting sequence-to-function models of gene regulation. The prompt is a research request rather than a spec: here is a trained model that maps 249 bases of DNA to enhancer activity in two promoter contexts, developmental and housekeeping; implement `varseq.design_task.design` and drive the activity of your designs as high as you can. How the sequence space gets searched is entirely the agent's choice.

Two constraints carry the science. The model is reachable only as a metered callable, and the meter is kept outside the agent's process, so the queries spent exploring during development come out of the same 200,000-sequence session allowance as the ones spent designing — search efficiency is part of the answer, not an afterthought. And the designs are graded on a held-out model the agent cannot query, so a sequence that merely maxes out the given model by exploiting its quirks earns nothing; the regulatory signal has to be real enough to survive a model that never saw the search.

The workspace supplies 2,048 real regulatory sequences to start from. The graded configuration asks for four designs on the developmental program at a 40,000-query budget. Everything runs on CPU, and the agent has 24 hours. We wrote the reference designer, and before fitting the reward scale we measured a ladder of comparison searches through this exact grade path: adversarial oracle-maxing, random DNA, unmodified natural enhancers, careless and competent hill climbing, and the reference. Every anchor below is a measurement.

## Verifier design

We run the agent's `design()` against a metered model, then score the sequences it returns on models it could never see.

| What we check | How |
| --- | --- |
| The designs work in a model the search never touched | Activity is read from held-out models trained on a different split, and the grade is the held-out number only — the score the agent could watch during its search contributes nothing |
| The signal is real, not model-specific | The held-out reading spans more than one architecture family and each design is credited at its weakest reading, so sequences that fool the family the given model belongs to are worth nothing; measured adversarial designs land below random DNA |
| The query allowance is real | Every sequence the scoring service is asked to score is counted outside the agent's process, development included, and a session that scores more than its allowance scores zero — searching offline against copied weights is not a route |
| The output contract holds exactly | Exactly the requested number of sequences, each exactly the requested length, over A/C/G/T only; any violation zeroes the configuration rather than being rounded down |
| The result is reproducible | Two identical calls must return byte-identical sequences, and invalid arguments — nonpositive budget, a budget too small to score every design, an unusable length or track — must raise rather than return something plausible |
| Partial skill counts | The scale runs from the level random and adversarial sequences reach up through the reference designer, which sits at 0.70, so the climb from greedy ascent to a real population search has gradient the whole way and a better-than-reference search still earns more |

## Trace walkthrough

Every run makes the same two opening measurements: score the 2,048 natural sequences, then run an exhaustive single-base scan from the best of them. What separates the set is whether the run treats that scan as the method or as the price list.

### A strong run

1. **Buy a hypothesis before buying a search.** The winning run spent its first 2,048 queries scoring the whole natural set, found the best developmental sequence at 5.42, then mined the set offline for enriched k-mers and came out with named candidate sites — an AP-1-like `TGAGTCA` for the developmental program, a DRE-like `TATCGATA` for housekeeping. The search that followed started with an idea about which sites matter.
2. **Price greedy ascent, then leave it.** Exhaustive single-base hill climbing from the best natural start climbed 5.42 to about 13 and cost roughly 19,000 queries. Rather than tune it, the run switched to a population that recombines short blocks between strong natural parents and reached 20.2 for 10,000 queries — a better sequence for half the spend.
3. **Ask whether the output is still DNA.** Before shipping it checked its best designs for the signatures of a model artifact: reverse complements still scored high and tracked the originals, GC content sat at 0.43, 193 distinct 8-mers across 242 positions, longest homopolymer run 3. Its top design carried twelve copies of the AP-1-like site on an otherwise natural-looking background — repeated real biology, not a repetitive string.
4. **Ship the search, not the artifact.** The submitted designer screens natural enhancers with a fifth of whatever budget it is given, then evolves a population where half the proposals are small substitutions and half transfer motifs or longer blocks between parents, with a diversity floor on the final pick. It closed at **1.000** on 50,016 of the 200,000 session queries: held-out activity 29.3, against 18.6 for the reference designer.

### A failed run

1. **Read the same evidence correctly.** The lowest-scoring run scored the natural set, ran the single-base scan, and wrote down that the gains were broad and incremental rather than one exploitable glitch. That reading is right, and it is where the run stopped.
2. **Mistake restraint for transfer.** It capped every trajectory at 24 substitutions from its natural start, in its own words to avoid unrestricted exploitation of the model, and never added recombination. The held-out models pay for regulatory signal, not for staying close to a natural sequence, so the cap bought nothing and cost the whole upper range.
3. **Generalize from one sequence.** It noticed the strongest developmental seed scored higher reverse-complemented, 6.07 against 5.42, and built reverse-complement initialization around that. Across the rest of the seed set the effect went both ways.
4. **Run out of things to do with the budget.** Three greedy basins stall long before 40,000 queries, so extra allowance bought no extra activity: at a 5,000-query probe its designs averaged 9.89 where the winning run's averaged 11.83, and the graded run landed at held-out 8.62 for **0.350**. It spent 57,094 session queries — more than the winning run — on a search that could not use them.

Every run in the set cleared the output contract, the determinism gate, and all eight invalid-argument cases. The entire 0.350-to-1.000 spread is how much transferable activity the search itself found.

## Failure modes

These are the failure modes we saw across the evaluated runs.

| Failure mode | What goes wrong |
| --- | --- |
| Greedy ascent as the whole method | Exhaustive single-base climbing is the first thing every run finds; it stalls a few thousand queries in and cannot absorb the rest of the budget, landing in the low band whatever the allowance. |
| Restraint pointed at the wrong axis | Capping edits from a natural start reads like anti-overfitting discipline and scores like a smaller search, because proximity to a seed is not what the held-out models reward. |
| Characterization instead of search | Scans and probes consume the allowance and never become a search; the meter counts development queries the same as design queries. |
| Designs tuned to the given model | Sequences engineered against the scoring model's quirks; the weakest held-out reading puts them below random DNA. |
| One sequence promoted to a rule | A property measured on a single strong seed becomes a design principle and does not hold across the set. |

Every shipped rollout was audited for reward hacking and is clean.

## Running

Run the task with an agent:

```bash
harbor run -p delivery/auto-research-tasks/regulatory-dna-design --agent claude-code -m <your-model> -k 1 -o jobs/
```

Replay the reference solution with the same verifier the agent is scored against:

```bash
harbor run -p delivery/auto-research-tasks/regulatory-dna-design --agent oracle -k 1 -o jobs/
```

The reference replays around 0.72 against a 0.61 floor, and the scale keeps roughly 40 percent of held-out headroom above the reference, so a better search still earns more. The task grades on 4 CPUs and 16 GB, CPU-only.
