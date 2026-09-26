"""
EXPLORATORY secondary analyses of the hyperparameter-family experiment.

NOT pre-registered. These were chosen after the headline result was seen
(TRAIN_STAT_PLAN.md section 7b lists the pre-specified outcomes). Intervals here
are descriptive; with several comparisons on the same data they should not be
read as confirmatory.

    python -m ab.gpt.util.hp_family_secondary \
        --control out/hpf/e10ctl --experimental out/hpf/e10exp
"""
from __future__ import annotations

import argparse, ast, glob, json, math, os, re, statistics as st
from collections import defaultdict
from pathlib import Path

import pandas as pd

from ab.gpt.util.hp_family_analysis import t95, ci, _norm

COPY = 0.95  # normalised-similarity threshold for "near-copy of the reference"


def wilson(k: int, n: int, z: float = 1.959964):
    if n == 0: return (float('nan'),) * 3
    p = k / n; d = 1 + z*z/n
    c = (p + z*z/(2*n)) / d
    h = z * math.sqrt(p*(1-p)/n + z*z/(4*n*n)) / d
    return p, c - h, c + h


def newcombe(k1, n1, k2, n2):
    """Difference in proportions (arm1 - arm2), Newcombe hybrid-score interval."""
    p1, l1, u1 = wilson(k1, n1); p2, l2, u2 = wilson(k2, n2)
    d = p1 - p2
    return d, d - math.sqrt((p1-l1)**2 + (u2-p2)**2), d + math.sqrt((u1-p1)**2 + (p2-l2)**2)


def welch(a, b):
    """mean difference (a - b) with a 95% Welch interval."""
    if len(a) < 2 or len(b) < 2: return float('nan'), float('nan'), float('nan')
    ma, mb = st.mean(a), st.mean(b)
    va, vb = st.variance(a)/len(a), st.variance(b)/len(b)
    se = math.sqrt(va + vb)
    df = (va+vb)**2 / (va**2/(len(a)-1) + vb**2/(len(b)-1)) if se else 1
    h = t95(max(1, int(round(df)))) * se
    return ma - mb, (ma-mb) - h, (ma-mb) + h


def collect(root: str):
    """One record per generated network."""
    recs = []
    bdirs = []
    for one in str(root).split(','):
        bdirs += glob.glob(os.path.join(one.strip(), 'llm', 'epoch', 'A*', 'synth_nn', 'B*'))
    for bdir in sorted(bdirs):
        gen = os.path.join(bdir, 'new_nn.py')
        orig = glob.glob(os.path.join(bdir, 'original_*.py'))
        dfp = os.path.join(bdir, 'dataframe.df')
        if not (os.path.exists(gen) and orig): continue
        try:
            sim = __import__('difflib').SequenceMatcher(
                None, _norm(open(gen).read()), _norm(open(orig[0]).read())).ratio()
        except Exception:
            sim = None
        family = None
        if os.path.exists(dfp):
            try: family = pd.read_pickle(dfp).get('nn')
            except Exception: pass
        per_epoch = {}
        for f in glob.glob(os.path.join(bdir, '[0-9]*.json')):
            try:
                rec = json.load(open(f)); rec = rec[0] if isinstance(rec, list) else rec
                per_epoch[int(Path(f).stem)] = rec
            except Exception: pass
        if not per_epoch: continue
        last = per_epoch[max(per_epoch)]
        ts = last.get('train_stat') or {}
        recs.append({'family': family, 'sim': sim, 'copy': (sim is not None and sim >= COPY),
                     'acc': float(last['accuracy']),
                     'acc_e1': float(per_epoch[1]['accuracy']) if 1 in per_epoch else None,
                     'train_accuracy': ts.get('train_accuracy'),
                     'train_loss': ts.get('train_loss')})
    return recs


def paired(c_recs, e_recs, predicate=lambda r: True):
    cf, ef = defaultdict(list), defaultdict(list)
    for r in c_recs:
        if r['family'] and predicate(r): cf[r['family']].append(r['acc'])
    for r in e_recs:
        if r['family'] and predicate(r): ef[r['family']].append(r['acc'])
    fams = sorted(set(cf) & set(ef))
    return [st.mean(ef[f]) - st.mean(cf[f]) for f in fams], len(fams)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--control', required=True); ap.add_argument('--experimental', required=True)
    a = ap.parse_args()
    C, E = collect(a.control), collect(a.experimental)
    print('EXPLORATORY — not pre-registered; chosen after the headline was seen.\n')
    print(f'trained networks: control {len(C)}, experimental {len(E)}\n')

    print('=== 1. copy rate: difference in proportions ===')
    kc = sum(1 for r in C if r['copy']); ke = sum(1 for r in E if r['copy'])
    # denominators are all generations, taken from the same roots
    def _bdirs(spec):
        out = []
        for one in str(spec).split(','):
            out += glob.glob(os.path.join(one.strip(), 'llm', 'epoch', 'A*', 'synth_nn', 'B*'))
        return out
    cb, eb = _bdirs(a.control), _bdirs(a.experimental)
    nc = sum(1 for b in cb if os.path.exists(os.path.join(b, 'new_nn.py')))
    ne = sum(1 for b in eb if os.path.exists(os.path.join(b, 'new_nn.py')))
    kc_all = sum(1 for b in cb if _is_copy(b))
    ke_all = sum(1 for b in eb if _is_copy(b))
    d, lo, hi = newcombe(ke_all, ne, kc_all, nc)
    print(f'  control      {kc_all}/{nc} ({kc_all/nc:.1%})')
    print(f'  experimental {ke_all}/{ne} ({ke_all/ne:.1%})')
    print(f'  difference (exp - ctl): {d*100:+.1f} pts, 95% CI [{lo*100:+.1f}, {hi*100:+.1f}]')

    print('\n=== 2. does copying explain the accuracy gain? ===')
    for lab, R in (('control', C), ('experimental', E)):
        cp = [r['acc'] for r in R if r['copy']]; nn = [r['acc'] for r in R if not r['copy']]
        if cp and nn:
            dd, l, h = welch(nn, cp)
            print(f'  {lab:12} copies n={len(cp):2} mean {st.mean(cp):.4f} | '
                  f'non-copies n={len(nn):2} mean {st.mean(nn):.4f} | '
                  f'non-copy advantage {dd*100:+.2f} pts 95% CI [{l*100:+.2f}, {h*100:+.2f}]')
        else:
            print(f'  {lab:12} insufficient split (copies {len(cp)}, non-copies {len(nn)})')

    print('\n=== 3. paired accuracy, non-copies only ===')
    diffs, n = paired(C, E, lambda r: not r['copy'])
    if diffs:
        m, lo, hi, dz = ci(diffs)
        print(f'  n={n} families  mean {m*100:+.2f} pts  95% CI [{lo*100:+.2f}, {hi*100:+.2f}]  d_z={dz:+.3f}')
    else:
        print('  no paired families after excluding copies')
    allд, nall = paired(C, E)
    m2, lo2, hi2, dz2 = ci(allд)
    print(f'  (all generations, for comparison: n={nall}  {m2*100:+.2f} pts  '
          f'[{lo2*100:+.2f}, {hi2*100:+.2f}]  d_z={dz2:+.3f})')

    print('\n=== 4. generalisation gap (train_accuracy - accuracy) ===')
    for lab, R in (('control', C), ('experimental', E)):
        g = [r['train_accuracy'] - r['acc'] for r in R if r['train_accuracy'] is not None]
        print(f'  {lab:12} n={len(g):2} mean {st.mean(g):+.4f} median {st.median(g):+.4f}' if g
              else f'  {lab:12} no train_stat')
    gc = [r['train_accuracy']-r['acc'] for r in C if r['train_accuracy'] is not None]
    ge = [r['train_accuracy']-r['acc'] for r in E if r['train_accuracy'] is not None]
    if gc and ge:
        d, l, h = welch(ge, gc)
        print(f'  difference (exp - ctl): {d:+.4f}  95% CI [{l:+.4f}, {h:+.4f}]')

    print('\n=== 5. convergence: epoch 1 -> epoch 3 ===')
    for lab, R in (('control', C), ('experimental', E)):
        e1 = [r['acc_e1'] for r in R if r['acc_e1'] is not None]
        gain = [r['acc'] - r['acc_e1'] for r in R if r['acc_e1'] is not None]
        if e1:
            print(f'  {lab:12} epoch1 mean {st.mean(e1):.4f} | epoch3 mean '
                  f'{st.mean([r["acc"] for r in R if r["acc_e1"] is not None]):.4f} | '
                  f'gain {st.mean(gain):+.4f}')
    g1 = [r['acc']-r['acc_e1'] for r in C if r['acc_e1'] is not None]
    g2 = [r['acc']-r['acc_e1'] for r in E if r['acc_e1'] is not None]
    if g1 and g2:
        d, l, h = welch(g2, g1)
        print(f'  gain difference (exp - ctl): {d:+.4f}  95% CI [{l:+.4f}, {h:+.4f}]')

    print('\n=== 6. spread of accuracy within each arm ===')
    for lab, R in (('control', C), ('experimental', E)):
        v = sorted(r['acc'] for r in R)
        if len(v) > 3:
            q1, q3 = v[len(v)//4], v[3*len(v)//4]
            print(f'  {lab:12} n={len(v):2} mean {st.mean(v):.4f} sd {st.stdev(v):.4f} '
                  f'IQR [{q1:.4f}, {q3:.4f}] = {q3-q1:.4f}  min {v[0]:.4f} max {v[-1]:.4f}')
    return 0


def _is_copy(bdir):
    gen = os.path.join(bdir, 'new_nn.py'); orig = glob.glob(os.path.join(bdir, 'original_*.py'))
    if not (os.path.exists(gen) and orig): return False
    try:
        import difflib
        return difflib.SequenceMatcher(None, _norm(open(gen).read()),
                                       _norm(open(orig[0]).read())).ratio() >= COPY
    except Exception:
        return False


if __name__ == '__main__':
    raise SystemExit(main())
