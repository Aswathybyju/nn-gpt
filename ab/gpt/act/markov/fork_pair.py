"""Fork-paired blocks for the markov loop (TRAIN_STAT_PLAN.md section 7r).

Runs a control *prefix*, checkpoints it at the pre-registered fork points, and
from each checkpoint runs two short blocks -- control and treatment -- that start
from identical state. The only difference between the two is whether the prompt
improver sees the per-epoch training trajectory.

Pre-registered and fixed before the first run (see section 7r):

  prefixes P   3, seeds 43/44/45, each run to iteration 400
  fork points  25, 50, 100, 150, 200, 275, 350, 400
  horizon H    25 iterations per arm
  pairs        P * 8 = 24
  primary      max accuracy reached inside the horizon
  secondaries  mean accuracy of trainable proposals; near-duplicate
               consecutive-proposal rate (>= 0.95 and byte-identical)

Nothing under ``ab/gpt/act/markov/`` is modified: state is read and written from
outside, and the treatment wraps ``EvaluationResult.get_feedback`` in this
process only.

    # exactness check -- a restored run must reproduce a continuous one
    python3 -m ab.gpt.act.markov.fork_pair --self-test --iterations 4 --fork-at 2 \
        --output-dir out/fork/selftest

    # one prefix, then its forks
    python3 -m ab.gpt.act.markov.fork_pair --prefix --seed 43 \
        --output-dir out/fork/p43
    python3 -m ab.gpt.act.markov.fork_pair --forks --seed 43 \
        --output-dir out/fork/p43
"""
from __future__ import annotations

import argparse
import difflib
import json
import random
import re
from pathlib import Path

import numpy as np
import torch

from ab.gpt.act.markov.config import PipelineConfig
from ab.gpt.act.markov.pipeline import Pipeline
from ab.gpt.util.markov_fork import install_treatment, load_state, save_state

FORK_POINTS = (25, 50, 100, 150, 200, 275, 350, 400)
HORIZON = 25
PREFIX_SEEDS = (43, 44, 45)
PREFIX_LENGTH = 400


def _determinism(seed: int):
    """The settings pipeline.py's own __main__ applies, so a restored run is
    bit-exact. cudnn.enabled=False costs training throughput; section 7r folds
    the measured penalty into the budget."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.enabled = False
    np.random.seed(seed)
    random.seed(seed)


def _config(args, *, max_iterations: int, output_dir) -> PipelineConfig:
    return PipelineConfig(
        model_name=args.model, dataset=args.dataset,
        train_epochs=args.epochs, batch_size=args.batch_size,
        learning_rate=args.lr, data_dir=args.data_dir,
        max_iterations=max_iterations, target_accuracy=1.0,  # never stop early
        output_dir=str(output_dir), history_size=args.history_size)


def _run_to(pipeline, stop: int):
    """Advance the pipeline to iteration ``stop``."""
    pipeline.config.max_iterations = stop
    pipeline.run()


def _models(output_dir) -> dict:
    d = Path(output_dir) / 'generated_models'
    if not d.is_dir():
        return {}
    return {int(re.search(r'(\d+)', f.name).group(1)): f.read_text()
            for f in d.glob('model_iter_*.py')}


def _accuracies(output_dir) -> dict:
    log = Path(output_dir) / 'results.log'
    if not log.is_file():
        return {}
    out = {}
    for line in log.read_text().splitlines():
        m = re.match(r'iteration: (\d+), accuracy: ([\d.]+)', line)
        if m:
            out[int(m.group(1))] = float(m.group(2))
    return out


# --------------------------------------------------------------------------
# exactness check
# --------------------------------------------------------------------------

def self_test(args) -> int:
    """A run restored at iteration f must generate byte-identical code from f+1.

    Exact rather than approximate because the sampling stream is described by
    ``seed + _call_counter`` and both are checkpointed.
    """
    root = Path(args.output_dir)
    f, n = args.fork_at, args.iterations
    assert 0 < f < n, '--fork-at must be inside --iterations'

    _determinism(args.seed)
    cont = Pipeline(_config(args, max_iterations=n, output_dir=root / 'continuous'))
    _run_to(cont, n)
    cont_models = _models(root / 'continuous')

    _determinism(args.seed)
    part = Pipeline(_config(args, max_iterations=f, output_dir=root / 'partial'))
    _run_to(part, f)
    ckpt = root / 'checkpoint.json'
    save_state(part, ckpt)
    del part

    _determinism(args.seed)
    rest = Pipeline(_config(args, max_iterations=n, output_dir=root / 'restored'))
    load_state(rest, ckpt)
    _run_to(rest, n)
    rest_models = _models(root / 'restored')

    checked, mismatched = [], []
    for i in range(f + 1, n + 1):
        a, b = cont_models.get(i), rest_models.get(i)
        if a is None and b is None:
            continue
        checked.append(i)
        if a != b:
            ratio = difflib.SequenceMatcher(None, a or '', b or '').ratio()
            mismatched.append({'iteration': i, 'similarity': round(ratio, 4)})

    verdict = {'fork_at': f, 'iterations': n, 'compared': checked,
               'mismatched': mismatched,
               'exact': bool(checked) and not mismatched}
    (root / 'self_test.json').write_text(json.dumps(verdict, indent=2))
    print(json.dumps(verdict, indent=2))
    if not checked:
        print('INCONCLUSIVE: no post-fork model files to compare')
        return 2
    print('EXACT' if verdict['exact'] else 'NOT EXACT')
    return 0 if verdict['exact'] else 1


# --------------------------------------------------------------------------
# prefix and forks
# --------------------------------------------------------------------------

def run_prefix(args) -> int:
    """Control-arm prefix, checkpointed at each fork point."""
    root = Path(args.output_dir)
    _determinism(args.seed)
    pipe = Pipeline(_config(args, max_iterations=PREFIX_LENGTH,
                            output_dir=root / 'prefix'))
    for point in FORK_POINTS:
        if point <= pipe.iteration:
            continue
        _run_to(pipe, point)
        save_state(pipe, root / 'checkpoints' / f'iter_{point}.json')
        print(f'[fork_pair] checkpointed at iteration {point}', flush=True)
    return 0


def run_forks(args) -> int:
    """Both arms from every checkpoint, over the pre-registered horizon."""
    root = Path(args.output_dir)
    results = []
    for point in FORK_POINTS:
        ckpt = root / 'checkpoints' / f'iter_{point}.json'
        if not ckpt.is_file():
            print(f'[fork_pair] no checkpoint at {point}, skipping', flush=True)
            continue
        for arm in ('control', 'treatment'):
            out = root / 'forks' / f'iter_{point}' / arm
            if (out / 'summary.json').is_file():
                print(f'[fork_pair] {point}/{arm} already done', flush=True)
                continue
            undo = install_treatment() if arm == 'treatment' else None
            try:
                _determinism(args.seed)
                pipe = Pipeline(_config(args, max_iterations=point + HORIZON,
                                        output_dir=out))
                load_state(pipe, ckpt)
                _run_to(pipe, point + HORIZON)
            finally:
                if undo is not None:
                    undo()
            results.append({'fork_point': point, 'arm': arm,
                            'output_dir': str(out)})
            print(f'[fork_pair] finished {point}/{arm}', flush=True)
    (root / 'forks_index.json').write_text(json.dumps(results, indent=2))
    return 0


def main() -> int:
    p = argparse.ArgumentParser()
    mode = p.add_mutually_exclusive_group(required=True)
    mode.add_argument('--self-test', action='store_true')
    mode.add_argument('--prefix', action='store_true')
    mode.add_argument('--forks', action='store_true')
    p.add_argument('--model', default='Qwen/Qwen2.5-7B-Instruct')
    p.add_argument('--dataset', default='cifar10',
                   choices=['imagenette', 'cifar10', 'cifar100'])
    p.add_argument('--epochs', type=int, default=3)
    p.add_argument('--batch-size', type=int, default=128)
    p.add_argument('--lr', type=float, default=0.01)
    p.add_argument('--data-dir', default='./data')
    p.add_argument('--history-size', type=int, default=5)
    p.add_argument('--seed', type=int, default=43,
                   help=f'one of the pre-registered prefixes {PREFIX_SEEDS}')
    p.add_argument('--output-dir', required=True)
    p.add_argument('--iterations', type=int, default=4, help='--self-test only')
    p.add_argument('--fork-at', type=int, default=2, help='--self-test only')
    args = p.parse_args()
    if args.self_test:
        return self_test(args)
    return run_prefix(args) if args.prefix else run_forks(args)


if __name__ == '__main__':
    raise SystemExit(main())
