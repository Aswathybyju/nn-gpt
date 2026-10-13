"""Fork-paired blocks for the markov loop: checkpoint its state, restart from it.

Rationale (TRAIN_STAT_PLAN.md section 7r). Seed-pairing two runs of
``ab.gpt.act.markov.pipeline`` holds the sampling stream constant only at
iteration 1; after that the arms diverge and the pair is no longer a controlled
comparison. Forking both arms from one *identical* checkpoint gives the stronger
control and many more pairs per GPU-hour.

Nothing here is imported by the markov pipeline or by any existing entry point.
Every file under ``ab/gpt/act/markov/`` is left untouched: the state is read and
written from outside through plain attribute access, and the treatment (showing
``train_history`` in the feedback the prompt improver sees) is installed by
wrapping ``EvaluationResult.get_feedback`` in the calling process only.

Why a checkpoint is exact here, and not approximate:

* ``Pipeline``'s entire mutable state is the eleven JSON-serialisable attributes
  in ``PIPELINE_STATE`` (``pipeline.py:48-61``) -- no tensors, no handles.
* ``CodeGenerator`` holds mirrors of four of them, restored through its own
  ``update_*`` setters.
* The RNG is described by two integers: ``LLMClient`` seeds every call as
  ``self.seed + self._call_counter`` (``llm_client.py:58-69``).

So a restored run must reproduce a continuous run exactly, which
``fork_pair.py --self-test`` checks by comparing generated code byte for byte.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

# pipeline.py:48-61 -- the whole of Pipeline's mutable state
PIPELINE_STATE = (
    'current_prompt_template',
    'best_code',
    'best_accuracy',
    'best_iteration',
    'current_improvement_suggestions',
    'last_generated_code',
    'last_accuracy',
    'iteration',
    'results_history',
    'improvement_history',
    'pending_history_entry',
)

# code_generator.py:125-128 -- mirrors of pipeline state, restored via setters
_GENERATOR_STATE = ('prompt_template', 'improvement_suggestions',
                    'current_code', 'current_accuracy')


def save_state(pipeline, path) -> dict:
    """Write ``pipeline``'s state to ``path`` as JSON and return it.

    Reads only attributes that already exist; the lazy ``llm_client`` and
    ``code_generator`` properties are deliberately *not* touched through their
    property accessors, so calling this on a pipeline that has not generated
    anything yet does not load a 7B model.
    """
    state = {k: getattr(pipeline, k) for k in PIPELINE_STATE}
    llm = getattr(pipeline, '_llm_client', None)
    gen = getattr(pipeline, '_code_generator', None)
    blob = {
        'pipeline': state,
        'rng': None if llm is None else {'seed': llm.seed,
                                        'call_counter': llm._call_counter},
        'generator': None if gen is None else {k: getattr(gen, k)
                                               for k in _GENERATOR_STATE},
        'config': {'dataset': pipeline.config.dataset,
                   'model_name': pipeline.config.model_name,
                   'train_epochs': pipeline.config.train_epochs,
                   'batch_size': pipeline.config.batch_size,
                   'learning_rate': pipeline.config.learning_rate,
                   'history_size': pipeline.config.history_size},
    }
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(blob, indent=2))
    return blob


def load_state(pipeline, path, *, strict_config: bool = True) -> dict:
    """Restore a checkpoint written by :func:`save_state` onto ``pipeline``.

    ``strict_config`` refuses a checkpoint whose dataset / model / epoch count
    differs from the live config: forking across those would silently compare
    runs that are not comparable.
    """
    blob = json.loads(Path(path).read_text())
    if strict_config:
        for k in ('dataset', 'model_name', 'train_epochs'):
            want, got = blob['config'][k], getattr(pipeline.config, k)
            if want != got:
                raise ValueError(f'checkpoint {k}={want!r} but config {k}={got!r}')

    for k, v in blob['pipeline'].items():
        setattr(pipeline, k, v)

    rng = blob.get('rng')
    gen = blob.get('generator')
    if rng is not None or gen is not None:
        # touching .code_generator instantiates the LLM client (loads the model)
        cg = pipeline.code_generator
        if gen is not None:
            cg.update_prompt_template(gen['prompt_template'])
            cg.update_improvement_suggestions(gen['improvement_suggestions'])
            cg.update_current_code_and_accuracy(gen['current_code'],
                                                gen['current_accuracy'])
        if rng is not None:
            client = pipeline.llm_client
            client.seed = rng['seed']
            client._call_counter = rng['call_counter']
    return blob


# --------------------------------------------------------------------------
# the treatment: show the per-epoch trajectory in the improver's feedback
# --------------------------------------------------------------------------

def render_train_history(history, fields=('train_loss', 'train_acc', 'test_acc')) -> str:
    """One line per epoch, in the order the epochs ran.

    Deliberately plain: no interpretation, no "loss is still falling, keep
    going" guidance. The experiment asks whether the model can use the numbers,
    not whether it can follow an instruction about them.
    """
    if not history:
        return ''
    lines = ['Per-epoch training history:']
    for row in history:
        parts = [f'epoch {row.get("epoch")}']
        parts += [f'{f}={row[f]:.4f}' for f in fields if isinstance(row.get(f), (int, float))]
        lines.append('  ' + ', '.join(parts))
    return '\n'.join(lines)


def install_treatment(fields=('train_loss', 'train_acc', 'test_acc')):
    """Append the per-epoch trajectory to ``EvaluationResult.get_feedback()``.

    Applied in the calling process only, by wrapping the bound method on the
    class: no file under ``ab/gpt/act/markov/`` changes, and the control arm is
    the unmodified code path. Returns a callable that undoes it.

    The production form of this is a flag-gated branch inside
    ``evaluator.get_feedback`` next to the existing ``use_history`` /
    ``use_reference_code`` ablation flags; that touches an existing file and is
    held pending approval.
    """
    from ab.gpt.act.markov.evaluator import EvaluationResult

    original = EvaluationResult.get_feedback

    def get_feedback(self) -> str:
        base = original(self)
        if not self.success:
            return base
        extra = render_train_history(self.train_history, fields)
        return f'{base}\n{extra}' if extra else base

    EvaluationResult.get_feedback = get_feedback
    return lambda: setattr(EvaluationResult, 'get_feedback', original)
