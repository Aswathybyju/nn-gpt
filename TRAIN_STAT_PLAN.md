# Hyperparameter-family generation experiment — plan and measurements

Written for: the author of this experiment and anyone reviewing it.
Status 2026-09-21. Supersedes the earlier delta / fine-tuning plan.

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
| secondary | `control-e1`, `experimental-e1` | 1 | architecture + transform fixed | 183 |

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

| excluded | evidence |
|---|---|
| `gradient_norm` | r = -0.13 with final accuracy (1,573 curves) |
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

Cost of epoch 10: n = 29 families per arm per round instead of 183. Use more
rounds to reach a usable sample.

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
do not add independent observations. The epoch-1 pair (183 families) is the
lever for more independent units, at the cost of a weaker treatment (section 4).

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

This is the main reason to run **both** pairs. Epoch 1 (n=183) reaches 77% power
at d = 0.20, so it can detect a small effect but carries the weak treatment;
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

## 8. Files (all new; no existing file modified)

| file | role |
|---|---|
| `ab/gpt/util/hp_family.py` | family query, block renderer, `anchor_row` |
| `ab/gpt/conf/prompt/test/NN_gen_hp_family_control.json` | `hp_control` (primary) |
| `ab/gpt/conf/prompt/test/NN_gen_hp_family_experimental.json` | `hp_experimental` (primary) |
| `ab/gpt/conf/prompt/test/NN_gen_hp_family_control_epoch1.json` | `hp_control_e1` (secondary) |
| `ab/gpt/conf/prompt/test/NN_gen_hp_family_experimental_epoch1.json` | `hp_experimental_e1` (secondary) |
| `ab/gpt/act/alter/hp_family.py` | generation-only entry point (`--dry-run`) |

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
