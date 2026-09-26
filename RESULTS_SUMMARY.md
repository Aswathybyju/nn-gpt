# Do training diagnostics help an LLM generate better architectures?

CIFAR-10, generation-only, two arms. Full detail and every measurement:
`TRAIN_STAT_PLAN.md`. Figures: `results/hp_family_e10_pooled/`.

## First, a result that does not depend on the experiment

Across CIFAR-10 architectures trained under several hyperparameter settings at
epoch 10:

> **In 41% of families (12 of 29), the configuration that fits the training data
> best is not the configuration that scores best on test — and where they
> disagree, the difference is a median of 6.5 accuracy points (max 16.6).**

This is a direct measurement of the `train_stat` table, not an outcome of the
generation experiment. It says that **the diagnostics carry information accuracy
alone cannot express**: a setting can look strongest by `train_accuracy` while
being mediocre by test accuracy, and nothing in the accuracy column reveals that.

It is also the reason the generation experiment was worth running, and the basis
on which the one surviving secondary result was specified in advance. For
contrast, at epoch 1 the same disagreement occurs in 8% of families and is worth
0.25 points — which is why epoch 1 was rejected as the primary configuration.

## The experiment

The prompt shows the LLM **one architecture trained under k >= 3 different
hyperparameter settings**, each with its hyperparameters and its accuracy. The
two arms differ in exactly one config key:

- **control** — hyperparameters and accuracy per setting
- **experimental** — the same, plus `train_loss` and `train_accuracy` per setting

The model is asked for a new architecture only; training settings are fixed by
the harness, so any accuracy difference is attributable to the architecture. No
fine-tuning. Generated networks are trained for 3 epochs and compared.

6 rounds per arm, 29 candidate families. **191 networks trained to 3 epochs**
(87 control, 104 experimental); one further experimental network stopped at
epoch 2 and is included at its epoch-2 accuracy.

## Headline

> **Experimental minus control: +1.10 accuracy points, 95% CI [-2.11, +4.31],
> d_z = +0.139, over 26 paired families.**
> Control 0.6174, experimental 0.6284. Experimental higher in 12 of 26 families.

**This is consistent with no effect and with a small effect of either sign.** The
direction favours diagnostics; the interval does not exclude zero.

One data-quality note: a single experimental network stopped at epoch 2 and is
counted at its epoch-2 accuracy, which understates that arm. Excluding it — the
protocol-conformant set — gives **+1.28 points [-1.90, +4.46], d_z = 0.163,
13/26 families**. Both readings span zero and the conclusion is unchanged; the
more conservative one is quoted above.

The design could not have shown much else. 80% power at n = 29 families requires
d_z = 0.52; a realistic 1-2 point gain is d_z = 0.12-0.25, i.e. 10-26% power.
This was stated in the plan **before** the result was computed, along with the
rule that no significance verdict would be reported. The outcome is the one the
plan predicted.

A ceiling was also known in advance: across this corpus, hyperparameters explain
only **R^2 = 0.094** of the accuracy variation *within* a family. There is not
much signal available for either arm to exploit.

## The concrete ask: a properly powered replication

The present design is the weakest configuration that was available, chosen on an
incomplete scan of epochs. Two corrections make a small effect detectable:

| | current | proposed |
|---|---|---|
| epoch | 10 | **20** (better separation of diagnostics from accuracy, same family count) |
| minimum settings per family | 3 | **2** |
| families | 29 | **98** |
| families where diagnostics are informative | 41% | **40%** — treatment strength preserved |
| effect detectable at 80% power | d = 0.52 | **d = 0.28** |

Allowing two-setting families triples the pool, and at epoch 20 it does so
without diluting the treatment (40% informative against the current 41%; at
epoch 10 the same relaxation would drop it to 24%). The disagreements are
shallower when they occur (+2.79 against +6.49 points), so the treatment is
broader but weaker.

This does not change the conclusion below. It is what would be needed to decide
the question rather than bound it.

## What survived more data, and what did not

Three rounds per arm were run first, then three more. This is the most
informative table in the study.

| result | 3 rounds | 6 rounds | outcome |
|---|---|---|---|
| headline difference | +1.38 [-4.1, +6.9] | +1.10 [-2.1, +4.3] | stable, interval 42% narrower |
| families favouring experimental | 16/23 (70%) | 12/26 (46%) | **collapsed** |
| experimental copies the reference less | -11.5 pts | -1.4 pts | **collapsed** |
| effect larger where diagnostics are informative | gap 1.69 pts | gap 1.00 pt | weakened, ordering held |
| failure rates equal | 16 vs 15 | 25 vs 28 | null confirmed |

**The two results that collapsed are the two that were not predicted in
advance.** The one that survived — a larger effect in families where the
best-fitting configuration is not the best-scoring one — was specified before
the data were seen, from a prior measurement that 41% of families have that
property. Even so it is weak: +1.64 points [-3.67, +6.96] in those 12 families
against +0.64 [-3.90, +5.18] in the other 14.

The copy-rate result is the cautionary one. At three rounds it looked like the
notable secondary finding; its interval was [-25.0, +2.6], which included zero,
and three more rounds landed it on zero.

## Clean nulls

- **Structural validity.** Identical failure rates and identical shape-error
  counts. Diagnostics do not help the model write code that runs.
- **Generalisation gap.** -0.0043 [-0.0185, +0.0099]; the direction even flipped
  between the two halves of the data.
- **Convergence speed** (epoch 1 to 3). -0.0040 [-0.0219, +0.0140].

One descriptive difference persisted: the experimental arm is tighter
(sd 0.110 vs 0.116; worst network 0.198 vs 0.136). Same ceiling, higher floor.

## Limitations

**The high-n companion arm was unusable.** An epoch-1 configuration with 183
families was planned as a better-powered comparison. It failed twice over: 180 of
its 181 families are variants of a single architecture lineage, so it was never
183 independent units; and its generation yield was 3% against 63% for the
primary corpus.

**Yield figures are round-level survival, not per-model probability.** The
evaluator uses one persistent worker per round, and a single model that
half-initialises an import kills every model after it in that round. Failures are
not independent. One control round was lost this way, leaving the arms unequal at
87 trained networks against 105.

**A third of generated networks are near-copies of the reference**, and in both
arms copies score *higher* than novel designs — the reference is a proven
architecture and the model's own designs are usually worse. This compresses any
difference the prompt can produce.

## Two further findings from the same runs

**Generation yield depends strongly on the source architecture family.** 63% on
`unq-` models against 3% on GenFractalNet, same task, dataset and pipeline. The
GenFractalNet failures concentrate in helper-class scaffolding the model drops
when regenerating. This bears on any experiment that samples references from a
mixed corpus.

**Instructing the model to change the reference makes it copy more.** A prompt
variant requiring "at least one structural change" raised median similarity to
the reference from 0.880 to 0.959 — it anchors the model to the reference instead
of freeing it. Showing diagnostics moved similarity the other way.

## Bottom line

No detectable effect of training diagnostics on the accuracy of generated
architectures, at a sample size that could only have detected a large one. The
mechanism-based prediction survived a doubling of the data; two post-hoc
observations did not.

**Two things stand independently of that null.** The per-epoch instrumentation
itself — the `train_stat` table, 325,846 rows at the time of this analysis — and
the measurement it makes
possible: in 41% of families the best-fitting configuration is not the
best-scoring one, worth a median 6.5 accuracy points. **The diagnostics
demonstrably contain information that accuracy does not.** What this experiment
leaves undetermined is a narrower question: whether a 7B code model can exploit
that information when it is placed in a prompt, at a sample size of 29 families.

The infrastructure, the analysis and the reporting rules are in place, and the
design that would answer it is specified above.
