"""Pre-registered analysis for the fork-paired markov experiment (section 7r).

Written before any fork has run, so the statistics are fixed in advance:

  primary    max accuracy reached inside the horizon
  secondary  mean accuracy of trainable proposals in the horizon
  mechanism  consecutive near-duplicate proposal rate (>= 0.95; byte-identical),
             against the committed baseline's 64.8% and 32.5%

Drop rule, also fixed in advance: a pair in which either arm trained nothing is
dropped from the primary and the number dropped is reported.

Three intervals are reported for every statistic, because the eight pairs inside
one prefix are not independent of each other:

  naive      paired t over all pairs -- too narrow, reported for comparability
  prefix     paired t over the P prefix-level mean differences -- too few df
  clustered  cluster-robust (CR1) standard error with prefix as the cluster

The effective sample size lies between P and the pair count; no single one of
these three is the answer on its own, which is why all three are printed.

usage: python3 -m ab.gpt.util.markov_fork_analysis out/fork/p43 out/fork/p44 ...
"""
from __future__ import annotations

import difflib
import itertools
import json
import re
import statistics
import sys
from pathlib import Path

BASELINE_NEAR_DUP = 0.648      # similarity >= 0.95, committed 1-epoch run
BASELINE_IDENTICAL = 0.325     # byte-identical consecutive proposals

try:
    from scipy import stats as _st
except Exception:                                    # analysis still runs
    _st = None


def _t_crit(df: int) -> float:
    if _st is not None:
        return float(_st.t.ppf(0.975, df))
    return 1.96 + 2.4 / max(df, 1)                   # crude fallback


def _accuracies(run_dir: Path) -> dict:
    log = run_dir / 'results.log'
    out = {}
    if log.is_file():
        for line in log.read_text().splitlines():
            m = re.match(r'iteration: (\d+), accuracy: ([\d.]+)', line)
            if m:
                out[int(m.group(1))] = float(m.group(2))
    return out


def _models(run_dir: Path) -> dict:
    d = run_dir / 'generated_models'
    if not d.is_dir():
        return {}
    return {int(re.search(r'(\d+)', f.name).group(1)): f.read_text()
            for f in d.glob('model_iter_*.py')}


def _duplication(models: dict) -> dict:
    """Consecutive-proposal similarity, over adjacent iteration numbers only."""
    it = sorted(models)
    adj = [(a, b) for a, b in zip(it, it[1:]) if b == a + 1]
    if not adj:
        return {'pairs': 0, 'near_dup': None, 'identical': None, 'median': None}
    r = [difflib.SequenceMatcher(None, models[a], models[b]).ratio() for a, b in adj]
    return {'pairs': len(adj),
            'near_dup': round(sum(x >= 0.95 for x in r) / len(r), 4),
            'identical': round(sum(x > 0.9999 for x in r) / len(r), 4),
            'median': round(statistics.median(r), 4)}


def _arm(run_dir: Path) -> dict:
    acc = _accuracies(run_dir)
    mods = _models(run_dir)
    return {'trainable': len(acc),
            'max_accuracy': max(acc.values()) if acc else None,
            'mean_accuracy': round(statistics.mean(acc.values()), 6) if acc else None,
            'duplication': _duplication(mods)}


def collect(roots) -> list:
    """One record per (prefix, fork point) pair."""
    pairs = []
    for root in roots:
        root = Path(root)
        for fork in sorted((root / 'forks').glob('iter_*'),
                           key=lambda p: int(p.name.split('_')[1])):
            c, t = fork / 'control', fork / 'treatment'
            if not (c.is_dir() and t.is_dir()):
                continue
            pairs.append({'prefix': root.name,
                          'fork_point': int(fork.name.split('_')[1]),
                          'control': _arm(c), 'treatment': _arm(t)})
    return pairs


def _paired(diffs) -> dict:
    n = len(diffs)
    if n < 2:
        return {'n': n}
    m = statistics.mean(diffs)
    sd = statistics.stdev(diffs)
    se = sd / n ** 0.5
    tc = _t_crit(n - 1)
    return {'n': n, 'mean': round(m, 6), 'sd': round(sd, 6),
            'ci95': [round(m - tc * se, 6), round(m + tc * se, 6)],
            'd_z': round(m / sd, 4) if sd else None}


def _clustered(diffs, clusters) -> dict:
    """CR1 cluster-robust SE for the mean, prefix as the cluster."""
    groups = {}
    for d, c in zip(diffs, clusters):
        groups.setdefault(c, []).append(d)
    n, g = len(diffs), len(groups)
    if g < 2 or n <= g:
        return {'clusters': g}
    m = statistics.mean(diffs)
    # sum over clusters of (sum of within-cluster residuals)^2, with the CR1
    # small-sample correction g/(g-1) * (n-1)/(n-k); k = 1 here (just a mean),
    # so the second factor is exactly 1 and is omitted.
    s = sum(sum(d - m for d in v) ** 2 for v in groups.values())
    se = ((g / (g - 1)) * s) ** 0.5 / n
    tc = _t_crit(g - 1)
    return {'clusters': g, 'se': round(se, 6),
            'ci95': [round(m - tc * se, 6), round(m + tc * se, 6)],
            'note': f't critical taken on {g - 1} df, not on the pair count'}


def analyse(pairs) -> dict:
    kept, dropped = [], []
    for p in pairs:
        if p['control']['trainable'] and p['treatment']['trainable']:
            kept.append(p)
        else:
            dropped.append({'prefix': p['prefix'], 'fork_point': p['fork_point'],
                            'control_trainable': p['control']['trainable'],
                            'treatment_trainable': p['treatment']['trainable']})

    clusters = [p['prefix'] for p in kept]
    out = {'pairs_found': len(pairs), 'pairs_kept': len(kept),
           'pairs_dropped': dropped}

    for name, key in (('primary_max_accuracy', 'max_accuracy'),
                      ('secondary_mean_accuracy', 'mean_accuracy')):
        d = [p['treatment'][key] - p['control'][key] for p in kept]
        out[name] = {'naive': _paired(d),
                     'prefix_level': _paired([
                         statistics.mean([x for x, c in zip(d, clusters) if c == g])
                         for g in sorted(set(clusters))]),
                     'clustered': _clustered(d, clusters)}

    # mechanism: all pairs with duplication measurable, no drop rule
    for key, base in (('near_dup', BASELINE_NEAR_DUP),
                      ('identical', BASELINE_IDENTICAL)):
        usable = [p for p in pairs
                  if p['control']['duplication'][key] is not None
                  and p['treatment']['duplication'][key] is not None]
        d = [p['treatment']['duplication'][key] - p['control']['duplication'][key]
             for p in usable]
        ctl = [p['control']['duplication'][key] for p in usable]
        out[f'mechanism_{key}'] = {
            'baseline_committed_run': base,
            'control_arm_mean': round(statistics.mean(ctl), 4) if ctl else None,
            'paired': _paired(d),
            'clustered': _clustered(d, [p['prefix'] for p in usable]),
        }
    return out


def main() -> int:
    roots = sys.argv[1:]
    if not roots:
        print(__doc__)
        return 2
    pairs = collect(roots)
    if not pairs:
        print('no fork pairs found under: ' + ', '.join(roots))
        return 2
    result = {'roots': roots, 'pairs': pairs, 'analysis': analyse(pairs)}
    print(json.dumps(result['analysis'], indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
