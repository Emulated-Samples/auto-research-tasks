<!-- GENERATED FROM tasks/prompt_spec.py -- DO NOT EDIT BY HAND.
     Re-render with:  python tasks/prompt_spec.py --write
     prompt_contract_version: 8
     prompt_sha256: 61f7b8bcce6947f9aa37dfa0a9e2fdc02c8d83fc2d74b09001834391e62e0faf -->

I have a research question I want you to investigate. I'm building a polygenic-scoring (genotype -> phenotype prediction) library from scratch, using the fixed toolchain described below.

You are given, per problem, a cohort of samples with covariates and a genotype dosage vector over many variants, plus per-variant annotations. Your program must learn to predict a phenotype for held-out samples as accurately as possible, and quickly.

You are not being asked to implement a named method or reproduce a published tool. Design any estimator that satisfies the program contract below and performs well on the quality bar.

Program contract:

The current working directory is your submission root. Provide two executable phase entry points there (`chmod +x`; if scripts, use an absolute interpreter path in the shebang, `#!/opt/svpgs-venv/bin/python`):

- `./fit`
- `./predict`

If your language needs a compile step, put it in an optional `./build.sh` (runs once, offline).

`fit` reads (from the current directory):

- `covariates_train.csv` - header row; `sample_id`, covariate columns, and the response column `y`.
- `genotypes_train.tsv` - header row; `sample_id` then one column per variant id; integer dosages `{0,1,2}`. Rows align with `covariates_train.csv` by `sample_id`.
- `variant_metadata.tsv` - keyed by `variant_id`; a `variant_class` column plus additional annotation columns (see `formula.txt`).
- `formula.txt`, `family.txt`, `dgp.json` - the text and machine-readable model spec (below).

`fit` writes `model.out` (your serialized fitted model, any format).

`sample_dataset/public/` holds one small example cohort in exactly this format, so you can run your program end to end before you are finished. It is an example of the FORMAT only: it is far too small to learn anything from, it ships no response for its test rows, and the target cohorts are larger and more varied. Nothing about it is guaranteed beyond the contract described here.

`predict` reads `model.out`, `covariates_test.csv` (same covariate columns, no `y`), `genotypes_test.tsv`, and `variant_metadata.tsv`; it writes `pred.csv` with one row per test sample, in input order.

`family.txt` is always `binomial-logit`. `pred.csv` must contain a `mean` column: the predicted probability of the positive class, in `[0, 1]` and finite for every row.

Your program must run non-interactively, read no network, and exit 0 on success.

Formula / family:

`family.txt` is `binomial-logit`.

`formula.txt` has the form

    y ~ pgs(<annotation columns>) + covariate(<covariate columns>)

`dgp.json` carries the same lists and an explicit `annotation_types` mapping; the two specs agree exactly.

Grammar: inside `pgs(...)` and `covariate(...)` the listed names are separated by `+` or `,` (either separator may appear, and both may appear in the same corpus; treat them the same and strip surrounding whitespace). Parse the formula - do not assume a fixed column layout, and do not assume every column you can see is yours to use.

- `pgs(...)` lists the `variant_metadata.tsv` annotation columns available to your model. Use only those columns and ignore unlisted metadata columns.
- `covariate(...)` lists the covariate columns to adjust for (e.g. age, sex, ancestry PCs). Use exactly the listed covariate columns and ignore any unlisted columns. Covariates are ordinary fixed effects; they are not the object of interest but must be handled correctly.

Constraints:

- Implement the statistical machinery yourself. The image guarantees GCC/G++/GFortran, Make, CMake, OpenBLAS/LAPACK development libraries, and Python 3.12 at `/opt/svpgs-venv/bin/python` with NumPy 2.1.3 and SciPy 1.14.1. Those generic numeric tools are fair game, but no polygenic-scoring / GWAS / GAM / stock penalized-regression end-to-end tools. Build the estimator (basis/penalty/optimizer/derivatives) yourself. Work fully offline; install nothing.
- `./` must contain only regular files and directories - no symlinks.
- Execution provides 4 vCPUs. The optional build has a 60-minute wall limit; each dataset gives `fit` 170 seconds and `predict` 30 seconds. Each untrusted phase has a 6 GiB address-space limit.

Quality bar:

Predictions should discriminate cases from controls, remain probabilistically calibrated, and avoid severe overconfidence. Fit and predict must each complete inside their stated limits; unused time cannot compensate for weak predictions. Non-finite or malformed output and violated program contracts are invalid.

You have 24 hours; get something in before then.
