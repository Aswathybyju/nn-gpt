# Hyperparameter-family generation experiment — plan and measurements

Status 2026-09-30.

All numbers measured against the real LEMUR DB (`nn-gpt/db/ab.nn.db`,
~1.01M `stat` rows, 325,846 `train_stat` rows).

## 1. The experiment

Show the LLM **one architecture under k distinct training configurations**, each
with its hyperparameters, its accuracy, and (experimental arm only) its
`train_stat` diagnostics. Ask it to generate a new model. No fine-tuning.
Generated networks are trained for 3 epochs and compared by accuracy.

Both arms show member accuracy, so the question is **whether diagnostics add
value over the accuracy the pipeline already showed** — not whether information
beats no information.

| pair | arms | epoch | grouping | families |
|---|---|---|---|---|
| **primary** | `control`, `experimental` | 10 | architecture only (transform varies per member) | 29 |
| secondary | `control-e1`, `experimental-e1` | 1 | architecture + transform fixed | 183* |

\* 180 of the 183 are GenFractalNet variants — one lineage, not 183 independent
architectures (section 7f). This arm was abandoned.

## 2. Family definition

One architecture (`nn`), one fixed `epoch`, >= 3 distinct `(lr, momentum, batch)`
settings, `max(lr)/min(lr) >= 2`, all diagnostic fields non-NULL, best-accuracy
row per setting, members chosen to span the lr range.

Grouping is on hyperparameter **VALUES**, never on `stat.prm` (the uid): the
triple `(0.01, 0.9, 64)` alone spans **4,138 uids**. `epoch` is inside the
partition key, or a 50-epoch row can appear beside 1-epoch rows.

`group_by_transform` controls the rest:

- `true` — transform constant, members differ only in hyperparameters. Exists
  **only at epoch 1** (183 families); epochs >= 2 yield zero.
- `false` — architecture alone. The only grouping that reaches epochs >= 2
  (29 families at epoch 10). Each member carries its own transform, shown with
  it, so the prompt says "training configurations", not "hyperparameters".
  Defensible in LEMUR's own terms: `transform` is itself a key in `prm`.

## 3. Field selection

Shown: **`train_loss`, `train_accuracy`** — config-driven via
`hp_family.diagnostic_fields`, so a field ablation is a config change.

### The original screening criterion was wrong

Fields were first excluded on Spearman correlation with accuracy. **That was the
wrong quantity and the wrong measurement.** For `gradient_norm` the figure used
was r = -0.13, which is *marginal* (pooled across architectures, so dominated by
between-architecture variation), computed *at epoch 5*, and *monotone-only* (a
correlation cannot see a U-shape). The relevant quantity is the within-family
association at the epoch actually used.

Re-screened properly (7k), `gradient_norm` explains **R^2 = 0.163 at epoch 10 and
0.241 at epoch 20 within families on its own**, and its within-family argmax
disagreement (52-55%) sits well below the 69% chance baseline. It carries real
information; the criterion that excluded it said otherwise.

**The exclusion survives on different grounds:** added to `train_loss` and
`train_accuracy` it contributes +0.012 R^2 at epoch 10 and +0.000 at epoch 20.
It is subsumed by what is already shown, not devoid of signal. The table below
gives the corrected reasons.

### What looked real and did not survive

Two apparent findings from the re-screen dissolved under the within-family test:
a **U-shaped** marginal relation between `gradient_norm` and accuracy (0.707,
0.595, 0.601, 0.663, 0.711 across epoch-20 quintiles), and a **sign-flipping
interaction** with `train_loss` (-3.11 points when loss is low, +5.88 when it is
high). With architecture held fixed the interaction term is worth +0.000 R^2 and
the correlation is negative in both strata with no flip. Both were
between-architecture confounds.

This is the same lesson as the copy-rate collapse (7c): a pattern that looks
clear in the margin can be an artifact of what varies between groups rather than
within them. The check that caught it — holding the family fixed — is the same
check the paired design applies to the main result.

| excluded | reason (corrected) |
|---|---|
| `gradient_norm` | subsumed by `train_loss` + `train_accuracy` (+0.012 / +0.000 R^2). Informative alone (R^2 0.16-0.24); the original r = -0.13 criterion was inadequate |
| `epoch_max` | non-NULL in 7.2% of candidate rows |
| `samples_per_second` | 37% of families mix RTX 4090 and RTX 3090 |
| `test_loss` | within-family rho with accuracy 0.913 even at epoch 10 — redundant with the accuracy both arms already show |
| `loss_gap`, `gen_gap` | within-family spread 0.015 / 0.008 at epoch 1 |
| cpu / ram / gpu fields | describe the machine, not the network |

`train_stat` has **17** diagnostic fields, not 16.

## 4. Why epoch 10 is the primary

The treatment is only meaningful if the diagnostics say something accuracy does
not. Within-family Spearman against accuracy, by configuration:

| epoch (grouping) | families | `train_loss` mean\|rho\| | `train_accuracy` mean\|rho\| |
|---|---|---|---|
| 1 (transform fixed) | 183 | 0.934 (perfect 74%) | 0.963 (85%) |
| 3 (transform free) | 1031 | 0.933 (71%) | 0.980 (87%) |
| 5 (transform free) | 1008 | 0.935 (70%) | 0.969 (81%) |
| **10 (transform free)** | **29** | **0.738 (38%)** | **0.769 (45%)** |
| 20 (transform free) | 29 | 0.686 (38%) | 0.683 (41%) |
| 50 (transform free) | 26 | 0.638 (31%) | 0.588 (19%) |

`test_loss` was 0.944 / 0.913 / 0.753 at epochs 1 / 10 / 50 — redundant
throughout, hence dropped.

### How strong is the treatment, in actionable terms

rho says the diagnostics are not a restatement of accuracy; this says how often
they would change a decision. Over the members actually shown in the prompt
(<= 4, spanning the lr range):

| | epoch 10 (29 fam) | epoch 1 (183 fam) |
|---|---|---|
| argmax(`train_accuracy`) != argmax(accuracy) | **12/29 (41%)** | 15/183 (8%) |
| argmin(`train_loss`) != argmax(accuracy) | **14/29 (48%)** | 17/183 (9%) |
| median within-family spread, `train_loss` | **0.6372** | 0.2149 |
| median within-family spread, `train_accuracy` | **0.2387** | 0.0776 |

In 41% of epoch-10 families the best-fitting configuration is **not** the
best-scoring one — the overfit signal a diagnostics-aware model could act on —
and where they disagree the accuracy difference is large: median 6.5 points,
max 16.6. At epoch 1 that happens in 8% of families and is worth 0.25 points,
which is noise. The spreads confirm the shown numbers differ meaningfully
between settings at epoch 10 (train_accuracy ranges 0.02-0.59 within a family).

**Decoupling and sample size are in direct conflict.** Epochs 3 and 5 have ~1000
families but diagnostics as redundant as epoch 1 (0.93-0.98): at those horizons
train and test still track each other, so there is no high-n configuration where
the treatment is strong. Decoupling begins only at epoch 10, where the corpus
thins to the 50-epoch curve runs (~1740 rows per epoch, 29 families).

`min_lr_ratio` is not the constraint: at epoch 10 the family count is **29 at
every threshold** (1.0, 1.5, 2.0), and at epoch 5 it moves only 1012 -> 1008.
The binding filter is ">= 3 distinct settings". Keeping 2.0 is free.

At epoch 1 the diagnostics are a near-perfect monotone restatement of accuracy,
so the experimental arm would add almost nothing and a null result would be
uninterpretable. At epoch 10 the **train-side** fields decouple: a member can
have high train_accuracy and mediocre test accuracy, which accuracy alone cannot
express. A real example from the assembled prompt:

```
Setting 2: acc=0.7634  train_loss=0.1395  train_accuracy=0.9544   <- overfit
Setting 3: acc=0.8250  train_loss=0.3473  train_accuracy=0.8795   <- better fit
```

**`test_loss` was dropped** (mean|rho| 0.913 even at epoch 10): it measures the
same thing as test accuracy, so it dilutes the treatment with a field that adds
nothing. The two train-side fields that genuinely decouple are what remain.
Re-add it in `diagnostic_fields` if a field ablation wants it.

Cost of epoch 10: n = 29 families per arm per round instead of 183 — though the
183 turned out to be one lineage (section 7f), so the real cost is smaller than
it looked. Use more rounds to even out the per-family draws.

## 5. What the families actually vary

At epoch 1: `lr` varies in 183/183 families, `momentum` in 178/183, **`batch` in
only 1/183**. Median within-family Spearman(accuracy, lr) = **-0.500**
(120 neg / 60 pos); momentum **-0.142** (95/76 — noise). **The families vary
learning rate.** At epoch 10 the transform varies too, by construction.

### Ceiling on what either arm can extract

Family-demeaned OLS of accuracy on log10(lr), momentum, log2(batch) at epoch 1:
**R^2 = 0.094**. The hyperparameters explain under 10% of within-family accuracy
variance; the rest is run-to-run noise (within-family accuracy sd 0.047, 19.8% of
total). This bounds *both* arms: the expected effect is small, and a null result
should be read against that ceiling rather than as proof diagnostics are useless.

## 6. How the generated networks are trained — READ BEFORE THE GPU RUN

`Eval` does **no hyperparameter search**. Its precedence (`Eval.py:394-466`):

1. `hp.json` in the model dir, if present — "LLM recommended prm"
2. otherwise CLI / built-in defaults
3. **`dataframe.df`'s `prm` overrides whatever 1 or 2 produced** (`prm.update`)
4. `--prm_json` overrides
5. `prm["epoch"]` forced to `nn_train_epochs`; `transform` defaulted if missing

Consequences as currently wired:

- The entry point saves `new_nn.py` but **no `hp.json`**, so the `<hp>` block the
  model produces is **discarded**.
- `anchor_row()` writes `prm`, so every generated network is trained with **the
  best family member's hyperparameters** — i.e. from the DB.

**Decision: training settings are held fixed across both arms, by design.**

The hypothesis is that diagnostics help the LLM generate better *architectures*.
With identical training settings in both arms, any accuracy difference is
attributable to the architecture. Letting the model pick its own
hyperparameters would make the arms differ in architecture *and* training
settings, adding a variance source that -- at n = 29 families and R^2 = 0.094 --
could only obscure a small effect.

Accordingly the prompt asks for the **architecture only**:

- the `<hp>` request is gone
- the `<tr>` request is gone too, for the same reason: `anchor_row`'s `prm`
  carries `transform`, so step 3 overrides any transform the model proposes
- the prompt states explicitly that training settings are fixed by the harness

The model still *sees* how different training configurations played out; that is
information about the architecture's behaviour (how it responds to a high
learning rate, whether it overfits easily), which is what it is asked to reason
from. It does not need to choose the settings for that to hold.

## 7. Sample size and yield

**Effective n is 29 families, not 174 attempts.** Rounds re-sample the same
families, so the units are not independent: the family is a blocking factor. A
paired test (per-family mean, control vs experimental) has n = 29 pairs however
many rounds are run. Extra rounds reduce the noise in each family's mean; they
do not add independent observations. The epoch-1 pair (183 families) looked like
the lever for more independent units, at the cost of a weaker treatment
(section 4) — but 180 of those families are one architecture lineage, so it never
offered the independence the count implies (section 7f).

**This machine cannot run the pilot.** No GPU (`nvidia-smi` absent), torch is
`2.14.0+cpu` with `cuda available: False`, and `transformers`, `accelerate` and
`peft` are all missing from the venv. Generation must run on a GPU node.

**Historical yield.** This checkout has no prior run artifacts (no `out/`, no
`B*` dirs, no `eval_info.json`), and the DB persists only successes, so it
cannot supply a denominator (only successes are persisted). The one usable
record is
`results/analog/acc_gap_advantage_pairs_20260606.json` from the analog
experiments:

| field | value |
|---|---|
| `all_details` | 870 |
| `valid_arm_rows_used_before_pairing` | 528 (**60.7%**) |
| `skipped_no_valid` | 306 (35.2%) |
| `skipped_missing_acc` | 36 |

So roughly **60% yield**, with the caveat that the script producing these numbers
is not in the repo (mined from an external path per the README), so "valid"
cannot be verified to mean "compiled and trained". Different prompt, configs and
LLM mix, too. The analog paper tables used `n_per_arm = 32`.

### Power — plan for a directional result, not significance

Paired two-sided t, alpha = 0.05 (normal approximation):

| true effect d | power at n=29 (epoch 10) | power at n=183 (epoch 1) |
|---|---|---|
| 0.10 | 8% | 27% |
| 0.15 | 12% | 53% |
| 0.20 | 19% | 77% |
| 0.25 | 27% | 92% |
| 0.30 | 37% | 98% |
| 0.55 | 84% | 100% |

80% power needs **d = 0.52 at n=29**, or d = 0.21 at n=183.

The mean within-family accuracy sd at epoch 10 is **0.081**. Against that scale:

| gain from diagnostics | d | power at n=29 |
|---|---|---|
| 1 point | 0.12 | 10% |
| 2 points | 0.25 | 26% |
| 3 points | 0.37 | 51% |
| 5 points | 0.62 | 91% |

So if diagnostics buy one or two accuracy points, **the epoch-10 arm will most
likely show a positive direction that does not reach significance**. That is the
expected outcome, not a failure, and the write-up should say so in advance
rather than discover it afterwards. Report the effect size and its confidence
interval, not a p-value verdict.

This was the main reason to run **both** pairs. Epoch 1 (n=183) reaches 77% power
at d = 0.20 *on paper*, so it looked able to detect a small effect while
carrying the weak treatment — but that power calculation assumes 183 independent
families, which section 7f shows it never had;
epoch 10 (n=29) carries the strong treatment but detects only a large one.
Agreement in direction across the two is worth more than either alone.

**Recommendation: measure yield with a pilot** rather than guessing — one round of
one arm (29 prompts) plus `Eval`, then count `B*/new_nn.py` (extraction
succeeded) against `B*/eval_info.json` and `B*/error.txt` (training succeeded or
failed). That gives the real two-stage yield for this exact prompt and model at
a fraction of the full GPU cost, and sizes the rounds properly. At ~60% yield,
5 rounds gives ~88 trained networks per arm across the 29 families.

Counting the pilot once it has run:

```
R=$(python - <<'P'
from ab.gpt.util.Const import epoch_dir, synth_dir
print(synth_dir(epoch_dir(0)))
P
)
echo "prompts sent : 29"
echo "new_nn.py    : $(ls -d $R/B*/new_nn.py 2>/dev/null | wc -l)"     # extraction OK
echo "eval_info    : $(ls -d $R/B*/eval_info.json 2>/dev/null | wc -l)" # trained OK
echo "error.txt    : $(ls -d $R/B*/error.txt 2>/dev/null | wc -l)"      # training failed
cat $R/B*/error.txt 2>/dev/null | sort | uniq -c | sort -rn | head
```

Stage 1 (prompt -> parseable `<nn>`) is `new_nn.py / 29`; stage 2
(code -> trained) is `eval_info.json / new_nn.py`.


## 7a. Pilot results (2026-09-25)

One round of the experimental arm at epoch 10: 29 prompts, then Eval at 3 epochs.

### Yield: 55% end to end, capped by one fixable failure mode

| stage | result |
|---|---|
| prompt -> parseable `<nn>` | 28/29 (97%) |
| code -> trained 3 epochs | 16/28 (57%) |
| **end to end** | **16/29 (55%)** |

Close to the 61% historical proxy. Accuracy of the 16 that trained: mean 0.6905,
median 0.7088, max 0.7879, min 0.4190.

The 12 failures:

| count | error |
|---|---|
| 5 | `mat1 and mat2 shapes cannot be multiplied` |
| 2 | `Expected input batch_size to match target batch_size` |
| 1 | tensor size mismatch at non-singleton dimension |
| 1 | conv channel mismatch |
| 1 | missing required function `learn` |
| 1 | `AttributeError: module has no attribute` |
| 1 | accuracy too low (0.1) |

**9 of 12 are shape errors**: the model edits an architecture and breaks the
tensor dimensions. Yield is limited by one dominant, fixable mode rather than
diffuse noise — a prompt or repair-pass target, and a result in its own right.

**Correction (see 7i): these yield figures are round-level survival, not a
per-model probability.** The evaluator runs one persistent worker per round, and
a single model that half-initialises an import kills every model after it in
that round. Failures are therefore *not* independent events, the binomial
reasoning used to size the rounds (0.55^2 per family) is optimistic, and an
arm's yield depends partly on which round drew a poisoning model. In the final
pooled run the arms ended unequal on exactly this account: **87 trained control
against 105 experimental**, after control lost a whole round of 28.

### Sizing follows from the yield

A family is paired only if *both* arms produce a trained network for it:

| rounds | P(paired) | expected paired families |
|---|---|---|
| 1 | 0.55^2 = 30% | ~9 of 29 |
| 3 | (1-0.45^3)^2 = 83% | ~24 of 29 |

Three rounds is the minimum that keeps the paired design intact.

### The v2 prompt made copying worse

v1 produced 36% near-copies of the reference, so v2 required at least one
structural change plus a one-line `<change>` statement. Measured over a full
round each:

| | v1 | v2 |
|---|---|---|
| identical after normalising | 2/28 | **0/29** |
| similarity >= 0.98 | 7/28 (25%) | 6/29 (21%) |
| **similarity >= 0.95** | **10/28 (36%)** | 17/29 (59%) |
| similarity >= 0.90 | 12/28 (43%) | 20/29 (69%) |
| **median similarity** | **0.880** | **0.959** |
| median generated length | 2,214 chars | 3,604 chars |

Counterintuitive but consistent: telling the model to *change the reference*
anchors it **to** the reference. It starts from the reference and makes a small
edit, where v1 sometimes produced a shorter design written from scratch. v2 did
eliminate exact duplicates and produced fuller code, but on the metric that
matters it is a regression. **Both arms use v1.** A future v3 should try a
different lever (forbid reusing the reference's class structure, or ask for a
design that addresses a named weakness) rather than demanding "a change".

### Correction: generated networks DO get train_stat

Section 6 previously concluded that diagnostics could not be recorded for
generated networks because the container image ships a pre-`train_stat`
`ab.nn`. That no longer holds. The image's `ab.nn` also turned out to be
unusable for a different reason — it queries a `loader` table this DB does not
have (`no such table: loader`) — so jobs now shadow the **local nn-dataset
checkout**, the version that built this DB. `save_train_stat` therefore runs,
and the DB has train_stat rows for generated networks (verified:
`train_loss 0.8371`, `train_accuracy 0.7108` at epoch 3). Follow-up analysis of
how the generated networks trained is available.

### Running the jobs: two environment fixes

1. **Writable `ab/nn`.** `nn_path()` resolves relative to the installed package,
   so nn-dataset materialises metric/nn/transform code into its own directory,
   which is root-owned in the image (`PermissionError` on every model). Jobs copy
   `ab/nn` from the mount into a writable shadow under the run root and set
   `PYTHONPATH` to it; `stat/` (3.1G) is excluded.
2. **Per-run output roots.** `AB_GPT_NNGPT_DIR` redirects `nngpt_dir`, so arms
   run in parallel without the generator's `rmtree` clobbering another arm.

Eval's skip check reads `eval_info.json`, which this path rarely writes (the
success path validates an artifact first and usually bails); results land in
per-epoch `<n>.json` files instead. Reconstructing `eval_info.json` from those
makes relaunches skip completed models. `error.txt` is **not** cleared between
runs, so it is unreliable as a failure signal — judge success by `<n>.json`.


## 7b. Reporting format (fixed before the results were computed)

Written 2026-09-26, after the epoch-10 jobs finished but **before** the paired
comparison was run, so the framing is not chosen in hindsight.

### The headline

> Experimental minus control, paired over families: **<mean difference>
> accuracy points, 95% CI [<lo>, <hi>], d_z = <effect size>, n = <paired
> families> of 29.**

Always reported next to it, in the same breath:

- **Power limit.** 80% power at n=29 needs d = 0.52. A 1-2 point gain is
  d = 0.12-0.25, i.e. 10-26% power. A positive direction that does not reach
  significance is the *expected* outcome, not a disappointment.
- **Yield.** 55% end to end, so each arm contributes ~16 of 29 families per
  round and the paired set is smaller than 29.
- **Ceiling.** Hyperparameters explain R^2 = 0.094 of within-family accuracy
  variance, which bounds what either arm can extract.

### Rules

1. **No significance verdict.** No "significant"/"not significant", no p-value
   as a verdict. Report the interval and let it speak.
2. **Effect size always.** Paired d_z alongside the raw difference in accuracy
   points, so the result is comparable to the power table above.
3. **A negative result is a result.** If control comes out ahead, that is
   reported in exactly the same format, with the same caveats. Given the stated
   limits, the honest reading of a small difference in either direction is
   "consistent with no effect and with a small effect of either sign".
4. **The interval is the finding.** A wide interval that includes zero is
   reported as such, not narrated as a trend.
5. **Secondary analyses are labelled secondary.** The mechanism split and the
   failure-rate comparison are exploratory; they are reported with their own
   intervals and never substituted for the headline if the headline disappoints.

### Pre-specified secondary analyses

- **Failure rate by arm**, with error types. Fewer shape errors in the
  experimental arm would be diagnostics helping the model write *valid* code —
  a separate effect from accuracy.
- **Mechanism split.** In 12 of 29 families the best-fitting configuration is
  not the best-scoring one; those are where the diagnostics say something
  accuracy cannot. The paired difference is reported separately for those 12 and
  for the other 17. If diagnostics help at all, the effect should concentrate in
  the 12; a difference there and not elsewhere is stronger evidence than a flat
  average, even with a wide interval. Small subgroups, so intervals will be wide.
- **Copy rate by arm**, to check that near-copies of the reference distribute
  evenly and are not carrying one arm.


## 7c. Epoch-10 results (2026-09-26)

3 rounds per arm, 87 generations each, evaluated at 3 epochs.

### Headline (pre-registered)

> **Experimental minus control: +1.38 accuracy points, 95% CI [-4.14, +6.91],
> d_z = +0.108, n = 23 paired families of 29.**

Mean accuracy: control 0.6069, experimental 0.6207. Experimental higher in
16/23 families.

Alongside, as required by section 7b: 80% power at n=29 needs d_z = 0.52, so at
d_z = 0.11 this design had roughly 8-10% power; yield 55%; R^2 = 0.094 bounds
what either arm can extract. No significance verdict. The interval spans zero
and is about ten times the point estimate: **consistent with no effect and with
a small effect of either sign**, direction favouring diagnostics.

### Failure rate by arm (pre-specified): a clean null

| | control | experimental |
|---|---|---|
| generated | 87 | 87 |
| produced code | 87 (100%) | 87 (100%) |
| trained | 55 (63%) | 55 (63%)* |
| failed | 32 | 32 |
| **shape-related errors** | **16** | **15** |

Identical. **Diagnostics do not help the model write structurally valid code.**

\* One of the 55 experimental networks stopped at epoch 2 rather than 3, so
under the strict 3-epoch definition the counts are 55 control against 54
experimental (63% vs 62%). The analysis uses each model's last recorded epoch.

Note on what "63%" means: it is round-level survival, not a per-model
probability. Failures within a round are not independent — one model can poison
the shared worker and take the rest of the round with it (section 7i). Pooled
over six rounds the arms ended at 87 trained (control) versus 105
(experimental), the difference being one poisoned control round.
A pre-specified outcome with a clean negative result, worth stating as such.
Composition differed without changing the total (control 13 matmul-shape vs
experimental 9; experimental 5 tensor-size and 3 NameError vs control 0 each).

### Mechanism split (pre-specified, exploratory subgroups)

| subgroup | n | mean diff | 95% CI | d_z | exp higher |
|---|---|---|---|---|---|
| diagnostics informative (best-fitting != best-scoring) | 10 | **+2.34 pts** | [-3.08, +7.76] | +0.309 | 8/10 |
| diagnostics redundant | 13 | +0.65 pts | [-9.00, +10.30] | +0.040 | 8/13 |

The effect **concentrates where predicted** — roughly 3.6x the point estimate
and 8x the effect size in the informative subgroup. Both intervals span zero and
the subgroups are tiny, so this is suggestive, not established. It is the
strongest pattern in the data because it was predicted in advance by the
mechanism rather than found by searching.

### Copy rate by arm (pre-specified) — DID NOT SURVIVE POOLING

| | first 3 rounds | pooled (6 rounds) |
|---|---|---|
| control | 35/87 (40.2%) | 64/172 (**37%**) |
| experimental | 25/87 (28.7%) | 62/173 (**36%**) |
| **difference** | **-11.5 pts** [-25.0, +2.6] | **-1 pt** |

On the first three rounds this looked like the notable secondary result: the
experimental arm appeared to copy the reference 11.5 points less often. **With
three more rounds per arm it is gone** — 37% versus 36%.

The original interval, [-25.0, +2.6], included zero, and the additional data
landed on zero. Nothing was mis-measured; a 95% interval that spans zero is
exactly a statement that the point estimate may be noise, and here it was. This
is the clearest illustration of why section 7b fixed the reporting rules in
advance: had the -11.5 been written up as a finding on the strength of its point
estimate, it would have had to be retracted.

The same fate met the win rate (16/23 = 70% of families favouring experimental
on the first three rounds, 12/26 = 46% pooled). The two results that collapsed
are the two that were **not** predicted in advance; the mechanism split, which
was, is treated in section 7c.

## 7d. Exploratory analyses (NOT pre-registered)

Chosen after the headline was seen. Descriptive intervals; several comparisons
on one dataset, so they are not confirmatory. Reported including the nulls.

### Copying does not explain the accuracy gain — it works against it

| arm | copies | non-copies | non-copy advantage |
|---|---|---|---|
| control | n=32, 0.6446 | n=23, 0.5724 | **-7.22 pts** [-14.14, -0.29] |
| experimental | n=22, 0.6498 | n=33, 0.6195 | -3.04 pts [-8.83, +2.76] |

**In both arms, near-copies scored higher than novel designs.** That is
unsurprising — the reference is a proven architecture and the model's own
designs are usually worse — but it breaks the tempting chain "fewer copies ->
higher accuracy". The experimental arm reached its small gain *despite* copying
less, not because of it.

### Excluding copies sharpens the difference

| set | n families | mean diff | 95% CI | d_z |
|---|---|---|---|---|
| all generations | 23 | +1.38 pts | [-4.14, +6.91] | +0.108 |
| **non-copies only** | **14** | **+3.63 pts** | [-3.49, +10.74] | **+0.294** |

Combined with the previous table this suggests a mechanism: control non-copies
average 0.5724, experimental non-copies 0.6195 — when the model departs from the
reference, diagnostics may help it depart better.

**Downgrade this result before quoting it.** The 0.95 threshold is arbitrary and
section 7e shows it is doing the work: the same quantity ranges from +0.70 to
+10.34 across reasonable thresholds, with paired n from 3 to 22. **+3.63 is a
mid-range point on a continuum, not a stable estimate.** Wide interval at n=14.

**The mechanism split (section 7c) is the stronger result**: it needs no
arbitrary threshold, and the subgroup was predicted in advance from the 41%
measurement of families whose best-fitting member is not their best-scoring one.
Where one of the two has to be quoted, quote that one.

### Nulls

- **Generalisation gap** (`train_accuracy - accuracy`): control +0.0192,
  experimental +0.0215; difference +0.0022, 95% CI [-0.0042, +0.0086]. No effect.
- **Convergence** (epoch 1 -> 3): gains +0.1654 vs +0.1600; difference -0.0054,
  95% CI [-0.0306, +0.0197]. No effect. (Experimental started marginally higher
  at epoch 1: 0.4717 vs 0.4491.)

### Spread: experimental is tighter

| arm | mean | sd | IQR | min | max |
|---|---|---|---|---|---|
| control | 0.6144 | 0.1324 | 0.1619 | 0.1355 | 0.7818 |
| experimental | 0.6316 | **0.1154** | **0.1183** | **0.1978** | 0.7767 |

Same ceiling, higher floor: the experimental arm produces fewer bad outliers.

### For the write-up: two ways to change the output, opposite effects

Instructing the model to change something **raised** similarity to the reference
(v2 test: 0.880 -> 0.959 median); showing it training diagnostics **lowered** it
(full run: control 0.920 -> experimental 0.901). Same goal, opposite directions.
Note the baselines differ - the v2 figure compares one round of v1 against one
round of v2, the 0.920/0.901 pair compares arms within the 3-round run - so the
contrast is directional, not a matched comparison.


## 7e. Robustness of the non-copies result: the threshold is doing the work

The 0.95 similarity threshold was arbitrary. Varying it:

| threshold | ctl non-copies | exp non-copies | paired non-copy diff |
|---|---|---|---|
| 0.85 | 0.5243 (n=9) | 0.5976 (n=15) | +10.34 [-15.1, +35.8] **n=3** |
| 0.90 | 0.5432 (n=11) | 0.6149 (n=19) | +8.24 [-6.7, +23.1] **n=4** |
| **0.95** | **0.5724 (n=23)** | **0.6195 (n=33)** | **+3.63 [-3.5, +10.7] n=14** |
| 0.98 | 0.6067 (n=43) | 0.6197 (n=46) | +0.70 [-5.5, +6.9] n=20 |
| 0.99 | 0.6122 (n=53) | 0.6264 (n=52) | +0.83 [-4.8, +6.5] n=22 |

Control non-copies are **not** stable: 0.52 -> 0.61 across thresholds.
Experimental non-copies are stable at 0.60-0.63. The gap therefore shrinks
monotonically from 7.3 points to 1.4, and the paired estimate collapses from
+10.34 to +0.83 as the exclusion loosens.

**Direction is robust, magnitude is not.** +3.63 at 0.95 is a mid-range point on
a continuum, not a stable quantity; at the strict end n falls to 3-4 families.
There is a coherent dose-response underneath — the more radically the model
departs from the reference, the larger the experimental advantage — but it rests
on 9-15 networks, so report it as a pattern, not an effect size. The non-copies
result should not carry the weight its +3.63 suggests.

### Outlier sensitivity of the headline

| | mean | 95% CI | d_z | median |
|---|---|---|---|---|
| all 23 families | +1.38 | [-4.14, +6.91] | +0.108 | **+5.19** |
| trimmed (drop min and max) | +2.70 | [-0.71, +6.12] | +0.361 | |

One family (`unq-4f03d8f4...`: control 0.696 from 2 runs, experimental 0.252
from 1 run) is a -44.5 point outlier and pulls the mean well below the median.
The pre-registered mean stays the headline; the median and trimmed estimate are
reported as robustness, not substituted for it.

## 7f. Epoch 1 was one architecture lineage, not 183 independent units

**The primary problem with epoch 1 is not its yield — it is that 180 of its 181
families are GenFractalNet variants.** The family count came from distinct
(nn, transform) pairs, and those are 180 distinct *hashes* of one generated
lineage, not 180 independent architectures. Everywhere this document treats
n = 183 as 183 independent units (sections 4, 5, 7, 7b) that figure should be
read as ~1 architecture family sampled 183 ways. Epoch 1 therefore could not
have served as the high-n companion it was chosen to be, regardless of how it
ran: its effective diversity is closer to 1 than to 183, and a paired test over
183 near-siblings does not buy the independent observations the power table
assumed.

The yield failure below is a second, separate finding.

### Generation yield depends strongly on the source architecture family

| arm | source family | generated | trained | yield |
|---|---|---|---|---|
| epoch 10 | `unq-*` (29 families) | 87 | 55 | **63%** |
| epoch 1 | GenFractalNet (180 of 181) | 181 | 6 | **3%** |

`hpf-e1-ctl` crashed mid-run; `hpf-e1-exp` completed with 6 successes and 175
failures. The failures are concentrated in helper-class structure:

| count | error |
|---|---|
| 85 | `Given groups=N, weight of size [...] expected input[...] to have N channels` |
| 46 | missing required function `learn` |
| 11 | `KeyError` |
| 11 | unused hyperparameter |
| 9 | `IndentationError` after a class definition |
| 8 | `NameError` (e.g. `FractalBlock` not defined) |

GenFractalNet references define helper classes (`FractalBlock`) and grouped
convolutions; the model regenerates the main class and drops or mismatches the
scaffolding. So **generation yield is a property of the source architecture, not
just of the model or the prompt** — 63% versus 3% between two corpora on the
same task, dataset and pipeline. As far as we know this has not been measured
before, and it bears on any LEMUR generation experiment that samples references
from a mixed corpus.

Consequence for this experiment: the epoch-1 comparison is abandoned rather than
rescued. Three independent reasons, any one of which is sufficient: the corpus is
one lineage (above), the treatment is weak by design (diagnostics are a
near-perfect restatement of accuracy at epoch 1, section 4), and the yield is 3%.


## 7g. Evaluation does not parallelise by allocating more GPUs

Allocating 4 GPUs to an Eval job and setting
`NNGPT_NNEVAL_USE_ALL_VISIBLE_GPUS=1` does **not** parallelise it. Measured on
`hpf-e10-ctl-r2` (4 GPUs requested, 85 models):

```
pool_size=1
per_gpu_worker_counts=[0, 1, 0, 0]
```

The pool saw all four GPUs and planned **one worker**; three sat idle for the
whole run. Timings confirm it: generation 15:47 / 15:19 / ~15 min per round
(~47 min), then ~6.6 h of serial evaluation at ~4.7 min per model — the same
per-model rate as a single-GPU job. Total 7.4 h against a predicted 2.5 h.

`NNGPT_NNEVAL_USE_ALL_VISIBLE_GPUS` is necessary but not sufficient. The worker
count is set by a workers-per-GPU planner in
`ab/gpt/util/nneval_worker_pool.py` (`_worker_count_for_gpu`,
`min_workers_per_gpu` / `max_workers_per_gpu`) together with
`NNGPT_NNEVAL_GPU_TOKENS`, which selects the GPU tokens the pool may use.

**The pattern that does work: N single-GPU Eval jobs over disjoint rounds**
(`--only_epoch 0`, `--only_epoch 1`, ...). Each job is a plain serial evaluator,
the split is explicit, and with the `eval_info.json` skip shims a job that is
restarted resumes rather than repeats. Budget ~4.7 min per trained model per
job and divide the rounds accordingly.

A related caution from the same relaunch: an earlier speedup was attributed to
the extra GPUs when it actually came from the skip shims carrying over 84 of 174
already-trained models. Check `pool_size` in the log before assuming a job is
parallel.


## 7h. FINAL pooled results (6 rounds per arm)

Supersedes the 3-round numbers in 7c/7d. Control 87 trained networks,
experimental 105. One caveat: control's batch-2 round A0 (28 models) is missing —
see 7i — so control has fewer draws per family than experimental.

### Headline

> **+1.10 accuracy points, 95% CI [-2.11, +4.31], d_z = +0.139, n = 26 of 29
> paired families.** Control 0.6174, experimental 0.6284. Experimental higher in
> **12/26 (46%)**.

Against the 3-round read (+1.38 [-4.14, +6.91], 16/23 = 70%): the interval is
**42% narrower**, the point estimate barely moved, and the win rate fell below
half. Power, yield and the R^2 ceiling apply as in 7b. No significance verdict.

### What survived pooling and what did not

| result | 3 rounds | 6 rounds | verdict |
|---|---|---|---|
| headline difference | +1.38 [-4.1, +6.9] | +1.10 [-2.1, +4.3] | stable, still spans zero |
| win rate | 16/23 (70%) | 12/26 (46%) | **collapsed** |
| copy rate difference | -11.5 pts | -1.4 pts [-11.4, +8.7] | **collapsed** |
| mechanism split (informative - redundant) | 1.69 pts | 1.00 pt | **weakened, order held** |
| non-copies only | +3.63 (n=14) | +3.80 (n=19) | strengthened, but see 7e |
| failure-rate null | 16 vs 15 | 25 vs 28 | **null holds** |
| spread (sd) | 0.132 vs 0.115 | 0.116 vs 0.110 | narrowed, direction held |

The two results that collapsed are the two that were not predicted in advance.

### Mechanism split (the decisive pre-specified secondary)

| subgroup | n | mean diff | 95% CI | d_z | exp higher |
|---|---|---|---|---|---|
| diagnostics informative | 12 | +1.64 pts | [-3.67, +6.96] | +0.196 | 7/12 |
| diagnostics redundant | 14 | +0.64 pts | [-3.90, +5.18] | +0.081 | 5/14 |

**It did not collapse, but it weakened.** The ordering predicted from the 41%
measurement held — informative above redundant, both positive — but the gap
between subgroups halved (1.69 -> 1.00 points), the effect-size ratio fell from
8x to 2.4x, and the informative arm's win rate dropped from 8/10 to 7/12. Both
intervals comfortably span zero.

Read honestly: the *direction* of the mechanism prediction survived doubling the
data, which the two unpredicted results did not. The *magnitude* is small and
indistinguishable from zero at this n. The right claim is "the pattern is
consistent with the mechanism and did not vanish under more data", not "the
diagnostics help where predicted".

### Exploratory, pooled

- **Copies still outscore non-copies** in both arms: control -4.80 pts
  [-9.69, +0.10], experimental -3.30 [-7.50, +0.89]. The reference remains a
  better architecture than what the model invents.
- **Non-copies only**: +3.80 [-1.36, +8.96], d_z = +0.355, n=19 — but the
  threshold sweep now **flips sign at 0.98** (-0.88), so this is not robust; see
  the updated table in 7e.
- **Generalisation gap**: -0.0043 [-0.0185, +0.0099]. Null (direction flipped
  from the 3-round read, which is itself evidence it is noise).
- **Convergence**: gain -0.0040 [-0.0219, +0.0140]. Null.
- **Spread**: control sd 0.1161, IQR 0.1376, min 0.1355; experimental sd 0.1100,
  IQR 0.1165, min 0.1978. Experimental still tighter with a higher floor.

### Updated threshold sweep (pooled)

| threshold | ctl non-copies | exp non-copies | paired diff |
|---|---|---|---|
| 0.85 | 0.5715 (n=16) | 0.5892 (n=21) | +5.95 (n=6) |
| 0.90 | 0.5819 (n=19) | 0.6063 (n=28) | +4.37 (n=8) |
| 0.95 | 0.6031 (n=38) | 0.6264 (n=54) | +3.80 (n=19) |
| 0.98 | 0.6244 (n=69) | 0.6251 (n=81) | **-0.88 (n=24)** |
| 0.99 | 0.6281 (n=83) | 0.6370 (n=96) | +0.92 (n=26) |

Non-monotone with a sign flip: the earlier "dose-response" reading does not
survive. Threshold choice determines the answer.

## 7i. A persistent worker can be poisoned by one model

Control batch-2 round A0 lost all 28 models to:

```
AttributeError: partially initialized module 'torchvision' has no attribute
'extension'   (torchvision/_meta_registrations.py, @register_meta("roi_align"))
```

The evaluator runs a **persistent serial worker** (`serial_worker_pool`,
`pool_size=1`, one long-lived pid). Once torchvision's import is left
half-initialised in that process, every subsequent model in the round fails with
the same error. A0 was poisoned first, so all 28 failed; the worker was
recreated for A1 and A2, which then ran normally (13/29 and 19/28).

Two re-run attempts reproduced it exactly — it is deterministic for that round,
not a transient race. An initial hypothesis that a stray cwd-relative `ab/nn`
caused it was wrong; that directory is a real nuisance (it shadows the installed
`ab.nn` and broke two analysis runs, now cleaned up by the job template) but it
is not this failure.

Practical consequences: check for a single repeated error filling a whole round
before trusting a yield number, and prefer one Eval job per round so a poisoned
worker costs one round rather than the run. The 28 models were abandoned.


## 7j. Epoch 10 was not optimal, and k>=2 is a real lever

Both are corrections to choices made on incomplete measurement.

### Epoch 20 decouples better than epoch 10 at the same family count

| epoch (k>=3, transform free) | families | mean\|rho\| train_loss | train_accuracy |
|---|---|---|---|
| 8 | 26 | 0.750 | 0.791 |
| **10 (chosen as primary)** | **29** | **0.738** | **0.769** |
| 12 | 26 | 0.842 | 0.865 |
| 15 | 29 | 0.802 | 0.785 |
| **20** | **29** | **0.686** | **0.683** |
| 50 | 26 | 0.638 | 0.588 |

**Epoch 10 was chosen on a scan of 1, 3, 5, 10, 20, 50 and was not the best
option available.** Epoch 20 gives the same 29 families with better decoupling
on both fields. Epoch 50 decouples further still (26 families). Nothing between
5 and 10 widens the pool: epoch 8 has fewer families and no better decoupling.
The primary arm should have run at epoch 20.

### k>=2 roughly triples the pool, and at epoch 20 keeps the treatment

Spearman rho cannot be used to judge k=2 families: with two members it is +/-1 by
construction, so the 0.92-0.95 values in the earlier sweep are an artifact, not
evidence of redundancy. The right measure is the one that produced the 41%
figure — how often the best-fitting member is not the best-scoring one.

| configuration | families | argmax disagreement | accuracy gap when they disagree |
|---|---|---|---|
| epoch 10, k>=3 (current primary) | 29 | **41%** | median +6.49 pts |
| epoch 10, k>=2 | 99 | 24% | median +3.87 pts |
| epoch 15, k>=2 | 98 | 34% | median +4.09 pts |
| **epoch 20, k>=2** | **98** | **40%** | median +2.79 pts |

**Epoch 20 with k>=2 holds the treatment strength of the current primary (40% vs
41%) with 3.4x the families.** At n=98 paired families, 80% power arrives at
d = 0.28 instead of d = 0.52 — still above the d = 0.12-0.25 a 1-2 point gain
implies, but far closer than the present design. The disagreements are smaller
when they occur (+2.79 vs +6.49 points), so the treatment is broader but
shallower.

This is the one remaining design lever. It does not rescue the current result;
it defines what a better-powered replication would look like.

### A second question the replication should answer: the field ablation

At n = 98 the design can also test *which* diagnostics matter, which n = 29
cannot: differences between field sets are necessarily smaller than the
treatment-control difference, and that is already indistinguishable from zero.

Three arms, identical but for `hp_family.diagnostic_fields` (a config value, so
no code changes):

| arm | fields | question |
|---|---|---|
| control | none | baseline |
| primary | `train_loss`, `train_accuracy` | the current treatment |
| + gradient | `train_loss`, `train_accuracy`, `gradient_norm` | does a field that is informative alone but redundant in regression (7k) still help a language model, which does not fit a regression? |

The third arm is the interesting one. The +0.012 R^2 increment says
`gradient_norm` adds nothing *to a linear model that already has the other two*.
It does not follow that it adds nothing to an LLM reading the numbers as text,
and 7k's caveat (102 members, increment within noise) leaves the question open.


## 7k. The field-exclusion criterion was inadequate (the decision survives)

Fields were excluded on Spearman correlation with accuracy, which only detects
monotone relationships. Re-screened three ways on epoch-10 and epoch-20 rows.

### 1. Non-monotone shape (mean accuracy by quintile)

| field | epoch 20 quintiles (low -> high) | shape |
|---|---|---|
| `gradient_norm` | 0.707  0.595  0.601  0.663  0.711 | **U-shaped** |
| `samples_per_second` | 0.686  0.728  0.667  0.620  0.577 | inverted-U at the low end |
| `train_loss` | 0.809  0.802  0.734  0.558  0.374 | monotone |
| `test_loss` | 0.834  0.778  0.696  0.572  0.397 | monotone |

`gradient_norm` is **not** monotone in the margin: both extremes score ~0.71
while the middle dips to ~0.60. A correlation coefficient reports that as
approximately nothing, exactly the failure mode anticipated.

### 2. The interaction does not survive within families

Marginally, `gradient_norm` appears to flip sign on `train_loss`:

| | low gradient | high gradient | difference |
|---|---|---|---|
| low `train_loss` (epoch 10) | 0.789 | 0.758 | -3.11 pts |
| high `train_loss` (epoch 10) | 0.448 | 0.507 | **+5.88 pts** |

But with architecture held fixed (family-demeaned OLS, 29 families, 102
members), the interaction term is +0.0006 at epoch 10 and +0.0024 at epoch 20,
worth +0.000 and +0.010 R^2. Within-family correlation is negative in *both*
strata (-0.27 / -0.55 at epoch 10) with no sign flip. **The marginal interaction
is a between-architecture confound**, and so is the U-shape above.

### 3. Within-family argmax disagreement, against a chance baseline

The 41% figure needs a reference: a pure-noise field with k members disagrees
1 - 1/k of the time, which is ~69% here. Below that is evidence of information.

| field (direction) | epoch 10 | epoch 20 | reading |
|---|---|---|---|
| `train_accuracy` (max) | 41% | 48% | informative (shown) |
| `train_loss` (min) | 48% | 41% | informative (shown) |
| `test_loss` (min) | 14% | 21% | near-duplicate of accuracy |
| **`gradient_norm` (min)** | **52%** | **55%** | **informative, below chance** |
| `samples_per_second` (min) | 45% | 55% | informative, below chance |

### Verdict: right decision, wrong reason

`gradient_norm` carries real within-family signal — **R^2 = 0.163 (epoch 10) and
0.241 (epoch 20) on its own** — far more than the marginal r = -0.13 at epoch 5
that justified excluding it. That criterion was measured on the wrong quantity
(across architectures, at a different epoch) and understated the field.

But added to the fields already shown, it contributes almost nothing:

| added to `train_loss` + `train_accuracy` | epoch 10 | epoch 20 |
|---|---|---|
| baseline R^2 | 0.874 | 0.813 |
| + `gradient_norm` | +0.012 | +0.000 |
| + `samples_per_second` | +0.001 | +0.005 |
| + `test_loss` | +0.112 | +0.147 |

**`gradient_norm` is redundant in context, not uninformative in itself.** The
exclusion stands; the stated reason ("no signal") was wrong and should be
"subsumed by train_loss and train_accuracy".

`test_loss` is the largest incremental contributor, but that is not a reason to
add it: it is a monotone restatement of the test accuracy the prompt already
shows for every member (argmax disagreement 14-21%). Its high R^2 is the outcome
re-entering the model, not new information.

`samples_per_second` adds nothing here and remains hardware-confounded (37% of
families mix RTX 3090 and 4090).

Caveat: 102 members across 29 families. An R^2 gain of +0.012 is within noise at
this size; the claim supported is "no evidence of a useful increment", not "zero
increment". The proper test belongs in the epoch-20, n=98 replication (7j).


## 7l. No other dataset has a usable family pool

CIFAR-10 was chosen because it was specified, not because the alternatives were
measured. They have now been measured, and CIFAR-10 turns out to be effectively
the only option.

### train_stat coverage by dataset

| task | dataset | metric | stat rows | with train_stat |
|---|---|---|---|---|
| img-classification | cifar-10 | acc | 302,132 | **139,663** |
| img-classification | cifar-100 | acc | 120,774 | 47,804 |
| img-classification | svhn | acc | 130,802 | 40,275 |
| img-classification | imagenette | acc | 113,468 | 39,346 |
| img-classification | mnist | acc | 113,858 | 32,409 |
| img-classification | celeba-gender | acc | 75,467 | 15,998 |
| img-super-resolution | div2k | psnr | 7,274 | 7,274 |
| img-denoising | denoise | psnr | 6,494 | 2,730 |
| coco (seg/det), places365 | | | | < 500 |
| coco captioning, wikitext, txt-image | | | 0 | **0** |

Coverage follows the instrumentation timeline: only runs recorded after
per-epoch `train_stat` landed carry diagnostics.

### Families, same definition as CIFAR-10

One architecture, one fixed epoch, k distinct `(lr, momentum, batch)` **values**
(never `prm` uid), `train_stat` present on every member, `max(lr)/min(lr) >= 2`.

| dataset | epoch | k>=3 | k>=2 | disagreement (k>=3) | median gap | distinct architectures |
|---|---|---|---|---|---|---|
| **cifar-10** | 10 | **29** | 99 | **12/29 = 41%** | +0.0649 | 29 |
| **cifar-10** | 20 | **29** | 98 | **14/29 = 48%** | +0.0680 | 29 |
| cifar-100 | 10 / 20 | 1 / 1 | 5 / 4 | 0/1 | – | 1 |
| svhn | 10 / 20 | 0 / 0 | 3 / 1 | – | – | – |
| imagenette | 10 / 20 | 2 / 2 | 22 / 21 | 0/2 | – | 2 |
| mnist | 10 / 20 | 2 / 2 | 9 / 8 | 0/2 | – | 2 |
| celeba-gender | 10 / 20 | 0 / 0 | 1 / 1 | – | – | – |
| div2k, denoise | – | 0 | 0 | – | – | `train_stat` only at epochs 1-7 |

**CIFAR-10 has 29 families; the next best has 2.**

### CIFAR-10's diversity is real (unlike epoch 1's)

All 29 share the `unq` prefix — the same surface signature that made the epoch-1
pool suspect. The code says they are genuinely different architectures:

| pool | median pairwise similarity | pairs >= 0.95 |
|---|---|---|
| **cifar-10 epoch 10 (29 families)** | **0.610** | **7%** |
| epoch-1 GenFractalNet (25 sampled) | 0.867 | 47% |

p10 similarity is 0.206 and normalised code length runs 1,447-3,250 characters.
A shared prefix is not a shared lineage; the check has to be on the code.

### The only alternative worth naming

At k>=2, **imagenette has 22 families** with 27% disagreement at epoch 10
(chance 52%) and median pairwise similarity 0.356 — real diversity. It is the
only non-CIFAR pool above trivial size, but 22 is fewer than the current 29, so
it adds no power. Its value would be as a **replication on a second dataset**,
which is a different argument from extending the sample.

Note the k>=2 caveat for CIFAR-10 itself: disagreement falls to 24% at epoch 10
with two-member families, but **holds at 40% at epoch 20** — which is why the
replication design (7j) specifies epoch 20 rather than 10.

### Pooling across datasets

Raw accuracies are not comparable: MNIST sits near a ceiling, CIFAR-10
mid-range, psnr is a different unit entirely. A naive merge would be dominated
by between-dataset variation.

**The paired design is structurally immune to the level problem** — the unit is
the within-family difference between arms for the same architecture on the same
dataset, so dataset level cancels before the test sees it. What does not cancel
is **scale**: one accuracy point near a ceiling is not one point mid-range.
Pooling raw point-differences would therefore still be wrong; pooling
*standardised* per-family differences (each divided by that dataset's
within-family sd, i.e. the quantity `d_z` already expresses) is valid.

So the design survives with per-dataset standardisation. Empirically the
question is moot: with 29 families against 2, there is no second dataset to pool
with. The available levers remain epoch 20 with k>=2 inside CIFAR-10 (7j), or
generating new multi-setting runs.

Raw output: `results/verification/dataset_family_survey.json`.


## 7m. Hyperparameter-optimisation extension: assessed and not pursued

A proposed follow-up kept the family prompt but had the LLM output hyperparameters
instead of architecture code, with the diagnostics again as the treatment. Four
feasibility checks were run before any design work. Two are disqualifying.

### Prior work: HPGPT is a different task

`ab/gpt/act/tune/Hyperparameters.py` + `ab/gpt/util/lemur_dataset_preparation.py`
LoRA-fine-tune a model on LEMUR question/answer pairs of the form *"generate the
hyperparameters ... so that the model achieves accuracy = X with epochs = N"*.
That is **inverse modelling conditioned on a target accuracy**, not optimisation.
Its corpus comes from `api.data()`, which carries no `train_stat`, so it shows no
diagnostics; and `generate_model_responses` writes the model's text to JSON
without ever training the proposed settings, so it reports no accuracy at all.
An extension is therefore genuinely new work, not a duplicate.

### Disqualifying finding 1: there is no headroom

| | |
|---|---|
| settings per architecture at epoch 10 | median **3**, max 6 |
| shown in the prompt | up to 4 |
| **families whose shown set already contains the global best** | **27/29 (93%)** |
| families with >= 1 accuracy point of headroom | **0/29** |

"Shown" is effectively "all", so *beat the best shown* and *beat Optuna's best*
are the same target, and that target is already in the prompt. Optuna is also a
weaker opponent than assumed — a median of **10** distinct settings per
architecture across all epochs, not hundreds.

### Disqualifying finding 2: the diagnostics carry no direction

Per-setting (67 non-best members across 29 families):

| | epoch 10 | epoch 20 |
|---|---|---|
| corr(gen_gap, \|log10 lr/lr_best\|) | **-0.46** | **-0.47** |
| mean gap, lr above best | +0.081 | +0.133 |
| mean gap, lr below best | +0.090 | +0.126 |

The distance correlation is **negative** — a large train-test gap means the
setting is *close* to the best learning rate, because far-off settings underfit
and so have small gaps. The gap is also near-identical above and below the best
lr, so it gives no directional signal.

Across the family (the shape the prompt actually presents):

| | epoch 10 | epoch 20 | chance |
|---|---|---|---|
| gap peak interior in lr order | 17/29 (59%) | 13/29 (45%) | 39% |
| strict inverted-U | 16/29 (55%) | 12/29 (41%) | 39% |
| **peak is the best-scoring member** | **9/29 (31%)** | **6/29 (21%)** | **31%** |

There is a weak tendency toward an interior peak at epoch 10, but it **does not
localise the better setting** — at chance, and below chance at epoch 20. The
interpolation mechanism requires localisation, so it is not available here.

### Mechanics, for the record

- The file Eval reads is **`hp.txt`**, not `hp.json`, containing JSON. Writing it
  *and* omitting `prm` from `anchor_row` is what makes the model's values bind;
  either alone fails silently.
- All 29 reference architectures declare the same
  `supported_hyperparameters() == {'lr', 'momentum'}`, so one prompt serves all.
- `batch` and `transform` are also honoured from `hp.txt` (`Train` indexes
  `prm['batch']` directly; Eval only defaults `transform` when it is missing).
- **No range validation on this path.** Optuna's bounds (lr 1e-5..1.0 log,
  momentum 0..1, dropout 0..0.5, batch 2^0..2^12) apply only to Optuna's own
  sampling. A value from `hp.txt` goes straight to training: out-of-range is
  accepted silently, a missing `batch` raises `KeyError`, and an unused declared
  param raises in `ab/gpt/util/Eval.py:85`.
- Arbitrary epochs work: `prm["epoch"] = int(nn_train_epochs)` overrides
  unconditionally. `epoch_limit_minutes` (30) is **per epoch** and would not
  bite; the slowest of these architectures runs ~9.6 min/epoch at 20 epochs.
- Cost at 98 families x 3 rounds x 2 arms (~353 trained, measured means):
  **~38 GPU-h at 3 epochs, ~84 at 10, ~163 at 20.**

### Attribution limit on `transform`

`transform` cannot be credited or discounted from this corpus. Across the 102
epoch-10 members there are 76 distinct transform names, and 18 of those names
are reused **across** families — but **within every one of the 29 families there
is exactly one transform per member** (29/29). It is that within-family
uniqueness that matters: one-hot transform becomes a member identifier inside
the unit of analysis, so the model saturates and returns R^2 = 1.000 by
construction, regardless of the global count. Among the estimable
factors, within family: log10(lr) 0.36, log2(batch) 0.12-0.15, momentum 0.00.

## 7n. Recurring hazard: marginal patterns in this corpus dissolve under control

Three times now a clear-looking pattern has disappeared once the right thing was
held fixed:

| pattern | how it looked | what killed it |
|---|---|---|
| experimental arm copies the reference less | -11.5 points | three more rounds: -1.4 points (7c) |
| `gradient_norm` U-shape and its interaction with `train_loss` | +5.88 points at high loss | holding the family fixed: interaction +0.000 R^2 (7k) |
| `transform` explains nearly all within-family accuracy | R^2 = 1.000 | one transform per member *within every family* (29/29), so one-hot is a member identifier (7m) |

The corpus invites this: architectures differ enormously, families are small,
and several fields are near-unique per row. **Any marginal statistic here should
be re-computed with the family held fixed before it is believed**, and a
suspiciously perfect fit should be read as a collinearity warning rather than a
result. The paired design applies this discipline to the main outcome; these
three cases show it is equally needed for every supporting claim.


## 7o. Trajectories do not rescue the mechanism either

The snapshot tests (7m) used one number per setting at a fixed epoch. A
trajectory is a different object — two settings can show the same gap at epoch
10 while one is still improving and the other has plateaued. The corpus already
holds 50-epoch curves, so this was free to test.

**25 of 29 families** have dense trajectories for every member (>= 40 epochs,
reaching >= 45); 89 members, median k = 3, chance of picking the best by
guessing 30%.

### Divergence onset and plateau measure distance, not direction

`onset(theta)` = the first epoch from which the train-test gap stays above
theta. `plateau` = the first epoch reaching 99% of that member's best accuracy.

| feature | corr with signed log10(lr/lr_best) | corr with \|distance\| | lr above best | lr below best |
|---|---|---|---|---|
| `onset` (0.05) | -0.281 | **+0.595** | median epoch 8.5 | 5.0 |
| `onset` (0.02) | -0.181 | +0.367 | 4.0 | 3.0 |
| `onset` (0.10) | +0.236 | +0.484 | 14.5 | 7.0 |
| `plateau` | -0.161 | +0.393 | 34.0 | 35.0 |

The signed correlations are weak and inconsistent in sign across thresholds; the
distance correlations are strong and consistent. **This is the same conclusion
as the snapshot tests, reached from a richer object: the diagnostics encode how
far a setting is from optimal, not which way to move.**

### The distance relationship is real — it survives the 7n control

Applying the rule from 7n, the strongest number here was re-computed
family-demeaned rather than pooled:

| | pooled | family-demeaned |
|---|---|---|
| `onset(0.05)` vs \|distance from best lr\| | +0.595 | **+0.540** |
| `plateau` vs \|distance from best lr\| | +0.393 | **+0.439** |

**This is the first pattern in the project to survive that check.** Divergence
onset genuinely tracks how far a setting sits from its family's best learning
rate, within the family. It is a real property of the data — it is simply not
the property the experiment needs.

### Trajectory shape does not identify the best member

| rule (chance ~30%) | hit rate |
|---|---|
| latest divergence onset | 7/22 = **32%** |
| latest plateau | 3/25 = 12% |
| steepest slope 5->10 | 6/25 = 24% |
| *highest accuracy at epoch 10 (not a trajectory feature)* | *19/25 = 76%* |

### Nor does it predict the epoch-50 winner better than accuracy

Using only epochs <= 10:

| rule | predicts the epoch-50 winner |
|---|---|
| **accuracy at epoch 10 (baseline)** | **19/25 = 76%** |
| latest divergence onset | 10/25 = 40% |
| steepest slope 5->10 | 6/25 = 24% |
| latest plateau | 3/25 = 12% |

Epoch-10 accuracy is a strong early predictor of the epoch-50 outcome, and no
trajectory feature approaches it. Accuracy is already shown in both arms, so
the trajectory adds nothing a control prompt lacks.

### Consequence

Option A (multi-epoch trajectories in the prompt) is viable on coverage — 28/29
families have epochs 1, 3, 5, 10 for every member, and the median member has all
50 — but it has **no mechanism**. Option B (generating a fresh 50-trial search)
fixes the headroom problem from 7m but not this one: a new search gives the model
something real to beat, it does not give the diagnostics something to say.

Measured before spending the 40-141 GPU-hours Option B would cost.
Raw output: `results/verification/trajectory_mechanism.json`.


## 7p. What train_stat actually contains: proximity, not direction

Three independent measurements now converge on one characterisation, and it is
the conclusion a reader should leave with.

| measurement | result | what it shows |
|---|---|---|
| snapshot gap vs distance from the best lr (7m) | **-0.46** | a large train-test gap means a setting is *close* to optimal |
| divergence onset vs distance, family-demeaned (7o) | **+0.540** | trajectories track distance from optimal, and this survives the 7n control |
| trajectory shape identifying the best member (7o) | **32%** vs 30% chance | shape does not say *which* setting is best |

They differ in object (single number vs 50-epoch curve), in epoch (10, 20, and
1-50), and in method (correlation, family-demeaned correlation, selection
accuracy against a chance baseline). They agree:

> **The diagnostics encode proximity to optimal, not direction toward it.**

A setting far from the best learning rate underfits, so its gap is small and its
divergence is late; a setting near the best fits hard, so its gap is large and
its divergence early. The signal is real — the onset relationship is the one
pattern in this project that survived family demeaning — but it is a measure of
*how close*, symmetric about the optimum and therefore silent on *which way*.

### Why this explains the main result

This is also the best available account of why the architecture experiment
produced +1.10 points with an interval spanning zero. Information about how
close a configuration is cannot drive a choice between configurations: to pick,
a model needs to know which direction is better, and that is precisely what is
absent. The diagnostics were a real signal attached to the wrong question.

It also predicts, correctly, where the effect did show up: the mechanism split
(7c) found the largest difference in the families where the best-fitting member
is not the best-scoring one — the cases where proximity information *does*
distinguish something accuracy alone does not. That subgroup effect was small
and its interval spanned zero, which is what a weak-but-real signal at n = 26
should look like.

### What would carry direction

Nothing in `train_stat` as currently recorded is asymmetric about the optimum.
A directional signal would need a quantity whose sign differs above and below
the best setting — for example the sign of the loss curvature, the ratio of
gradient norm to update size, or a comparison against a reference run at a known
learning rate. Recording one of those is a change to the instrumentation, not a
change to the prompt, and it is the precondition for any future version of this
experiment.

## 8. Files (all new; no existing file modified)

| file | role |
|---|---|
| `ab/gpt/util/hp_family.py` | family query, block renderer, `anchor_row` |
| `ab/gpt/conf/prompt/test/NN_gen_hp_family_control.json` | `hp_control` (primary) |
| `ab/gpt/conf/prompt/test/NN_gen_hp_family_experimental.json` | `hp_experimental` (primary) |
| `ab/gpt/conf/prompt/test/NN_gen_hp_family_control_epoch1.json` | `hp_control_e1` (secondary) |
| `ab/gpt/conf/prompt/test/NN_gen_hp_family_experimental_epoch1.json` | `hp_experimental_e1` (secondary) |
| `ab/gpt/act/alter/hp_family.py` | generation-only entry point (`--dry-run`) |
| `ab/gpt/util/hp_family_analysis.py` | pre-registered paired analysis |
| `ab/gpt/util/hp_family_secondary.py` | exploratory analyses (not pre-registered) |

Within each pair the configs differ in exactly one key,
`hp_family.show_diagnostics`; prompt bodies and all selection keys are
identical, so both arms see the same families in the same order.
`diagnostic_fields` drives selection as well as display, so both arms must pass
the same list.

Assembled prompts at epoch 10: the diff between arms is exactly the
`Training diagnostics:` line per family member; everything else is identical.

## 9. Running it on the cluster

Generation needs a GPU node; this host has none. Job specs (validated with
`kubectl apply --dry-run=server`):

| file | job | command |
|---|---|---|
| `k8s/hpf-pilot-gen.json` | `hpf-pilot-gen-exp` | `hp_family --arm experimental --rounds 1` |
| `k8s/hpf-pilot-eval.json` | `hpf-pilot-eval-exp` | `Eval --nn_train_epochs 3 --only_epoch 0 --nn_name_prefix hpf-exp-pilot` |

Both: uid 1062 / gid 1376 (numeric, not quoted — quoted `runAsUser` is rejected),
repo hostPath mounted at `/a/mm`, `cd /a/mm` so the LEMUR root resolves there and
`db/ab.nn.db` is found, 1 GPU, `HF_HOME=/a/mm/.hf` so the 7B download is cached
between jobs, `backoffLimit: 0` so a failure does not silently retry, output
tee'd to `/a/mm/out/hpf-pilot-*.log`.

**Memory.** Requests are what the scheduler matches against; limits are not. A
120Gi *limit* therefore cannot cause Pending — only the 30Gi *request* can, and
against ~35Gi single-GPU allocations that is tight. These specs request **24Gi**
and limit **64Gi**, which schedules more easily and still leaves burst room. The
namespace quota (`compute-resources`) caps `limits.memory` at 560Gi with nothing
currently in use, so neither value is near the ceiling.

**Order matters:** generation clears the epoch root at startup, so run and
evaluate one arm fully before generating the next, or point them at separate
roots.

## 10. Running it locally

```
# primary pair (epoch 10, 29 families per round)
python -m ab.gpt.act.alter.hp_family --arm control      --dry-run   # inspect
python -m ab.gpt.act.alter.hp_family --arm control      --rounds 1  # PILOT first
python -m ab.gpt.act.alter.hp_family --arm control      --rounds 5  # then the run
python -m ab.gpt.act.eval.Eval --nn_train_epochs 3 --nn_name_prefix hpf-ctl
python -m ab.gpt.act.alter.hp_family --arm experimental --rounds 5
python -m ab.gpt.act.eval.Eval --nn_train_epochs 3 --nn_name_prefix hpf-exp

# secondary pair (epoch 1, 183 families per round)
python -m ab.gpt.act.alter.hp_family --arm control-e1      --rounds 1
python -m ab.gpt.act.alter.hp_family --arm experimental-e1 --rounds 1
```

Generation clears the epoch root at startup, so evaluate one arm before
generating the next, or point them at separate roots.

## 11. Entry-point findings (why this path)

- `act/alter/*` -> `AlterNN.alter_delta`: generation-only (no LoRA/fine-tuning),
  builds prompts inline, never imports `NNGenPrompt`.
- `Tune.nn_gen`: builds prompts inline, no tall-mode support.
- `Tune_Curriculum.nn_gen`: the only path reaching upstream tall mode — but tall
  mode is not in this checkout (`origin/main` is 45 commits ahead) and yields k
  *distinct* architectures, not one architecture under k settings.
- `AlterNN`'s fetch cannot express the family: `only_best_accuracy=True`
  collapses to one row per `(task, dataset, metric, nn, epoch)`, and supporting
  models are sampled with `nn != row['nn']`. Hence the new fetch.

## 12. Removed

Deleted 2026-09-21 with the fine-tuning plan they belonged to:
`conf/prompt/{test,train}/NN_gen_delta_train_stat.json`,
`act/tune/delta_train_stat.py`, `ab/gpt/util/train_stat_enrichment.py`.
Also removed: the `*_with_accuracy.json` pair, superseded now that member
accuracy is shown in the primary configuration.
