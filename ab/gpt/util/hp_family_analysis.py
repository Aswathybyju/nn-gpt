"""
Paired per-family analysis of the hyperparameter-family experiment.

The family is the unit of analysis: rounds are repeated draws within a family,
not independent observations, so arms are compared as paired per-family means.

    python -m ab.gpt.util.hp_family_analysis \
        --control out/hpf/e10ctl --experimental out/hpf/e10exp \
        --out results/hp_family_e10

Reporting follows TRAIN_STAT_PLAN.md section 7b, fixed before results were
computed: effect size with a 95% interval, no significance verdict, power limit
and yield stated alongside the headline.
"""
from __future__ import annotations

import argparse, ast, csv, difflib, glob, json, math, os, re, statistics as st
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd

# Models are trained for this many epochs; a record whose last <n>.json is not
# this is a protocol deviation (the run stopped early) and is counted separately.
_EXPECTED_EPOCHS = 3

_T95 = {1:12.706,2:4.303,3:3.182,4:2.776,5:2.571,6:2.447,7:2.365,8:2.306,9:2.262,
        10:2.228,11:2.201,12:2.179,13:2.160,14:2.145,15:2.131,16:2.120,17:2.110,
        18:2.101,19:2.093,20:2.086,22:2.074,25:2.060,28:2.048,30:2.042,40:2.021,
        60:2.000,120:1.980}


def t95(df: int) -> float:
    try:
        from scipy import stats
        return float(stats.t.ppf(0.975, df))
    except Exception:
        pass
    if df in _T95: return _T95[df]
    ks = sorted(_T95)
    if df < ks[0]: return _T95[ks[0]]
    if df > ks[-1]: return 1.960
    lo = max(k for k in ks if k <= df); hi = min(k for k in ks if k >= df)
    return _T95[lo] + (df-lo)/(hi-lo)*(_T95[hi]-_T95[lo]) if hi != lo else _T95[lo]


def ci(vals):
    """mean, lo, hi, d_z for a set of paired differences."""
    n = len(vals)
    if n < 2: return (vals[0] if vals else float('nan')), float('nan'), float('nan'), float('nan')
    m, sd = st.mean(vals), st.stdev(vals)
    h = t95(n-1) * sd / math.sqrt(n)
    return m, m-h, m+h, (m/sd if sd else float('nan'))


def _norm(src: str) -> str:
    try:
        t = ast.parse(src)
        for n in ast.walk(t):
            if isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef,ast.ClassDef,ast.Module)) and n.body \
               and isinstance(n.body[0],ast.Expr) and isinstance(n.body[0].value,ast.Constant) \
               and isinstance(n.body[0].value.value,str):
                n.body.pop(0)
        src = ast.unparse(t)
    except Exception:
        pass
    src = re.sub(r'#.*','',src)
    return '\n'.join(l.strip() for l in src.splitlines() if l.strip())


def collect(root: str) -> dict:
    """Walk an arm's output root. Success is judged by the per-epoch <n>.json
    files Eval writes; error.txt is NOT cleared between runs, so it only counts
    as a failure when no result JSON exists."""
    acc = defaultdict(list); sim = []; errors = Counter()
    n_dirs = n_code = n_trained = n_failed = n_partial = 0
    # Targeted, not recursive: each run root also holds the abnn/ shadow copy
    # (23k files), which a '**' glob would walk on every call. `root` may be a
    # comma-separated list so several runs of the same arm pool together.
    bdirs = []
    for one in str(root).split(','):
        bdirs += glob.glob(os.path.join(one.strip(), 'llm', 'epoch', 'A*', 'synth_nn', 'B*'))
    for bdir in sorted(bdirs):
        n_dirs += 1
        gen = os.path.join(bdir, 'new_nn.py')
        if os.path.exists(gen): n_code += 1
        eps = sorted((int(Path(f).stem), f) for f in glob.glob(os.path.join(bdir, '[0-9]*.json')))
        dfp = os.path.join(bdir, 'dataframe.df')
        family = None
        if os.path.exists(dfp):
            try: family = pd.read_pickle(dfp).get('nn')
            except Exception: pass
        if eps:
            n_trained += 1
            if eps[-1][0] != _EXPECTED_EPOCHS:
                n_partial += 1
            try:
                rec = json.load(open(eps[-1][1])); rec = rec[0] if isinstance(rec, list) else rec
                if family: acc[family].append(float(rec['accuracy']))
            except Exception: pass
        else:
            err = os.path.join(bdir, 'error.txt')
            if os.path.exists(err):
                n_failed += 1
                first = open(err, errors='replace').readline().strip()
                errors[re.sub(r"'[^']*'", "'X'", re.sub(r'\d+', 'N', first))[:88]] += 1
        orig = glob.glob(os.path.join(bdir, 'original_*.py'))
        if os.path.exists(gen) and orig:
            try:
                sim.append(difflib.SequenceMatcher(
                    None, _norm(open(gen).read()), _norm(open(orig[0]).read())).ratio())
            except Exception: pass
    return {'acc': dict(acc), 'sim': sim, 'errors': errors,
            'dirs': n_dirs, 'code': n_code, 'trained': n_trained, 'failed': n_failed,
            'partial': n_partial}


def overfit_signal_families(config_path: str) -> set:
    """Families where the best-fitting member is not the best-scoring one —
    where the diagnostics say something accuracy cannot."""
    from ab.gpt.util.hp_family import fetch_families, DEFAULT_DIAGNOSTIC_FIELDS
    cfg = list(json.load(open(config_path)).values())[0]
    h = cfg.get('hp_family', {})
    fams = fetch_families(
        task=cfg.get('task','img-classification'), dataset=cfg.get('dataset','cifar-10'),
        metric=cfg.get('metric','acc'), epoch=h.get('epoch',1),
        min_settings=h.get('min_settings',3), max_members=h.get('max_members',4),
        min_lr_ratio=h.get('min_lr_ratio',2.0),
        diagnostic_fields=tuple(h.get('diagnostic_fields') or DEFAULT_DIAGNOSTIC_FIELDS),
        nn_prefixes=tuple(h.get('nn_prefixes') or ()) or None,
        max_families=h.get('max_families'),
        group_by_transform=h.get('group_by_transform', True))
    out = set()
    for f in fams:
        if 'train_accuracy' not in f.members[0]: continue
        if max(f.members, key=lambda m: m['train_accuracy']) is not max(f.members, key=lambda m: m['accuracy']):
            out.add(f.nn)
    return out


def report_block(label, diffs, indent='  '):
    if not diffs:
        print(f'{indent}{label}: no paired families'); return
    m, lo, hi, dz = ci(diffs)
    wins = sum(1 for d in diffs if d > 0)
    print(f'{indent}{label}: n={len(diffs):2}  mean {m*100:+.2f} pts  '
          f'95% CI [{lo*100:+.2f}, {hi*100:+.2f}]  d_z={dz:+.3f}  '
          f'experimental higher in {wins}/{len(diffs)}')


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--control', required=True, help='output root, or several comma-separated')
    ap.add_argument('--experimental', required=True, help='output root, or several comma-separated')
    ap.add_argument('--config', help='experimental config, for the mechanism split')
    ap.add_argument('--out', default='results/hp_family')
    args = ap.parse_args()

    c, e = collect(args.control), collect(args.experimental)

    print('=== yield and failures by arm ===')
    for lab, a in (('control', c), ('experimental', e)):
        y1 = a['code']/a['dirs'] if a['dirs'] else 0
        y2 = a['trained']/a['code'] if a['code'] else 0
        print(f'  {lab:12} generated {a["dirs"]:3}  code {a["code"]:3} ({y1:.0%})  '
              f'trained {a["trained"]:3} ({y2:.0%})  failed {a["failed"]:3}  '
              f'families {len(a["acc"]):2}')
        if a.get('partial'):
            print(f'  {"":12} WARNING: {a["partial"]} of those stopped before epoch '
                  f'{_EXPECTED_EPOCHS}; the last recorded epoch is used for them')
    print('\n  error types:')
    keys = set(c['errors']) | set(e['errors'])
    shape_re = re.compile(r'shapes cannot be multiplied|batch_size|must match the size|channels', re.I)
    for k in sorted(keys, key=lambda k: -(c['errors'][k]+e['errors'][k])):
        print(f'    ctl {c["errors"][k]:2}  exp {e["errors"][k]:2}   {k}')
    cs = sum(v for k,v in c['errors'].items() if shape_re.search(k))
    es = sum(v for k,v in e['errors'].items() if shape_re.search(k))
    print(f'    shape-related total: control {cs}, experimental {es}')

    print('\n=== copy rate by arm (normalised similarity to the reference) ===')
    for lab, a in (('control', c), ('experimental', e)):
        s = a['sim']
        if not s: print(f'  {lab}: none'); continue
        print(f'  {lab:12} n={len(s):3}  median {st.median(s):.3f}  '
              f'>=0.95 {sum(1 for x in s if x>=0.95)}/{len(s)} ({sum(1 for x in s if x>=0.95)/len(s):.0%})')

    shared = sorted(set(c['acc']) & set(e['acc']))
    rows = [{'family': f, 'n_control': len(c['acc'][f]), 'n_experimental': len(e['acc'][f]),
             'control_mean': st.mean(c['acc'][f]), 'experimental_mean': st.mean(e['acc'][f]),
             'diff': st.mean(e['acc'][f]) - st.mean(c['acc'][f])} for f in shared]
    print(f'\n=== HEADLINE: paired over {len(rows)} families '
          f'(control-only {len(set(c["acc"])-set(e["acc"]))}, '
          f'experimental-only {len(set(e["acc"])-set(c["acc"]))}) ===')
    if not rows:
        print('  no paired families'); return 1
    print(f'  mean accuracy   control {st.mean([r["control_mean"] for r in rows]):.4f}   '
          f'experimental {st.mean([r["experimental_mean"] for r in rows]):.4f}')
    report_block('experimental - control', [r['diff'] for r in rows])
    print('  Power limit: 80% at n=29 needs d_z=0.52; a 1-2 point gain is d_z=0.12-0.25')
    print('  (10-26% power). Yield 55%. R^2=0.094 bounds what either arm can extract.')
    print('  No significance verdict: read the interval.')

    if args.config:
        try:
            sig = overfit_signal_families(args.config)
            a = [r['diff'] for r in rows if r['family'] in sig]
            b = [r['diff'] for r in rows if r['family'] not in sig]
            print(f'\n=== SECONDARY: mechanism split (exploratory) ===')
            print(f'  {len(sig)} families flagged: best-fitting member != best-scoring member')
            report_block('diagnostics informative ', a)
            report_block('diagnostics redundant   ', b)
            for r in rows: r['overfit_signal'] = r['family'] in sig
        except Exception as exc:
            print(f'\n(mechanism split skipped: {exc})')

    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    with open(out/'per_family.csv','w',newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    m, lo, hi, dz = ci([r['diff'] for r in rows])
    json.dump({'n_paired': len(rows), 'mean_diff': m, 'ci95': [lo, hi], 'd_z': dz,
               'control': {k: v for k, v in c.items() if k not in ('acc','sim','errors')},
               'experimental': {k: v for k, v in e.items() if k not in ('acc','sim','errors')},
               'control_errors': dict(c['errors']), 'experimental_errors': dict(e['errors'])},
              open(out/'summary.json','w'), indent=2)
    print(f'\nwrote {out}/per_family.csv and summary.json')

    try:
        import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
        xs=[r['control_mean'] for r in rows]; ys=[r['experimental_mean'] for r in rows]
        flag=[r.get('overfit_signal') for r in rows]
        lim=[min(xs+ys)-0.02, max(xs+ys)+0.02]
        fig,ax=plt.subplots(figsize=(5.6,5.6)); ax.plot(lim,lim,ls='--',lw=1,color='#888')
        for f,mk,lb in ((True,'o','diagnostics informative'),(False,'^','diagnostics redundant')):
            px=[x for x,g in zip(xs,flag) if g is f]; py=[y for y,g in zip(ys,flag) if g is f]
            if px: ax.scatter(px,py,s=38,alpha=.8,marker=mk,label=lb)
        if any(f is not None for f in flag): ax.legend(fontsize=8)
        ax.set_xlim(lim); ax.set_ylim(lim)
        ax.set_xlabel('control: mean accuracy'); ax.set_ylabel('experimental: mean accuracy')
        ax.set_title(f'Per-family paired accuracy (n={len(rows)})\n'
                     f'mean {m*100:+.2f} pts, 95% CI [{lo*100:+.2f}, {hi*100:+.2f}]', fontsize=10)
        fig.tight_layout(); fig.savefig(out/'scatter.png',dpi=150); print(f'wrote {out}/scatter.png')
    except Exception as exc:
        print(f'(scatter skipped: {exc})')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
