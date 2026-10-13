"""Train a FIXED architecture under a proposed (lr, momentum).

A parallel entry point rather than a change to
``ab/gpt/act/markov/train_script.py``, whose optimiser line hardcodes
``momentum=0.9``. Everything that does the actual work -- the loaders, the epoch
loop, the evaluation, the seeding -- is imported from that module and reused, so
timings and results stay comparable with the architecture loop and no existing
file is touched.

Writes the same result JSON as ``train_script.py``, including ``train_history``,
so the same feedback renderer and the same evaluator contract apply.

usage: python3 train_hp.py --model_file M.py --lr 0.0234 --momentum 0.87 \
         --epochs 3 --output_file R.json --dataset cifar10 --data_dir ./data
"""
from __future__ import annotations

import argparse
import json
import sys
import traceback
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim

from ab.gpt.act.markov.train_script import (SEED, evaluate, load_cifar10,
                                            load_cifar100, load_imagenette,
                                            set_all_seeds, train_epoch)

_LOADERS = {'cifar10': load_cifar10, 'cifar100': load_cifar100,
            'imagenette': load_imagenette}


def build_model(model_file: str):
    """Instantiate ``Net`` from a markov-style model file (no-arg constructor)."""
    ns = {'torch': torch, 'nn': nn}
    exec('import torch.nn.functional as F', ns)
    exec('from torch import Tensor', ns)
    exec(Path(model_file).read_text(), ns)
    if 'Net' not in ns:
        raise ValueError("model file does not define a 'Net' class")
    return ns['Net']()


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument('--model_file', required=True)
    p.add_argument('--output_file', required=True)
    p.add_argument('--lr', type=float, required=True)
    p.add_argument('--momentum', type=float, required=True)
    p.add_argument('--weight_decay', type=float, default=5e-4)
    p.add_argument('--epochs', type=int, default=3)
    p.add_argument('--batch_size', type=int, default=128)
    p.add_argument('--data_dir', default='./data')
    p.add_argument('--dataset', default='cifar10', choices=sorted(_LOADERS))
    p.add_argument('--scheduler', default='cosine', choices=('cosine', 'none'))
    a = p.parse_args()

    result = {'success': False, 'accuracy': 0.0, 'error': None,
              'train_history': [],
              'hyperparameters': {'lr': a.lr, 'momentum': a.momentum,
                                  'weight_decay': a.weight_decay,
                                  'batch_size': a.batch_size,
                                  'epochs': a.epochs}}
    try:
        if not 0.0 <= a.momentum < 1.0:
            raise ValueError(f'momentum must be in [0, 1), got {a.momentum}')
        if a.lr <= 0:
            raise ValueError(f'lr must be positive, got {a.lr}')

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = build_model(a.model_file).to(device)
        print(f'Model parameters: {sum(q.numel() for q in model.parameters()):,}')

        # seeds reset after construction, exactly as train_script.main does
        set_all_seeds(SEED)
        train_loader, test_loader = _LOADERS[a.dataset](
            batch_size=a.batch_size, data_dir=a.data_dir)

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(model.parameters(), lr=a.lr, momentum=a.momentum,
                              weight_decay=a.weight_decay)
        scheduler = (optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=a.epochs)
                     if a.scheduler == 'cosine' else None)

        for epoch in range(a.epochs):
            train_loss, train_acc = train_epoch(model, train_loader, criterion,
                                                optimizer, device)
            test_acc = evaluate(model, test_loader, device)
            if scheduler is not None:
                scheduler.step()
            result['train_history'].append(
                {'epoch': epoch + 1, 'train_loss': train_loss,
                 'train_acc': train_acc, 'test_acc': test_acc * 100})
            print(f'Epoch {epoch + 1}/{a.epochs}: Train Loss={train_loss:.4f}, '
                  f'Train Acc={train_acc:.2f}%, Test Acc={test_acc * 100:.2f}%',
                  flush=True)

        final = evaluate(model, test_loader, device)
        result['success'] = True
        result['accuracy'] = final
        print(f'\nFinal Test Accuracy: {final * 100:.2f}%')
    except Exception as e:
        result['success'] = False
        result['error'] = f'{type(e).__name__}: {e}\n{traceback.format_exc()}'
        print(f'Error during training: {result["error"]}', file=sys.stderr)

    out = Path(a.output_file)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2))
    return 0 if result['success'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
