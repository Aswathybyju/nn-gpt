"""Does an expanded hyperparameter space have more dimensions than one?

Section 7s found that accuracy depends on `lr` and `momentum` almost only
through the effective learning rate lr/(1-momentum), in an inverted U whose peak
the hardcoded default already sits on. Section 7t asks whether a larger space is
genuinely larger, or whether the new parameters collapse the same way.

Four steps, run per architecture and never pooled (the section 7n rule):

1. WHICH COLLAPSE FITS BEST. A quadratic in log10(eta) for each candidate
   collapse, compared by R^2. Quadratic because the established shape is an
   inverted U -- a monotone or linear fit would miss it, which is the section 7k
   trap (a Spearman of +0.48 on a curve that peaks in the middle).

2. WHAT EACH PARAMETER ADDS BEYOND IT. Nested R^2 and adjusted R^2, adding each
   remaining parameter to the best collapse baseline, then all together. The
   delta is the variance a parameter explains that the collapse does not.

3. PARTIAL CORRELATION. Spearman of the residuals from the collapse fit against
   each parameter, as a rank-based check on step 2.

4. CROSS-CHECK. Optuna's own fANOVA importances on the TPE study, when
   available.

Steps 1-3 run on a UNIFORM RANDOM design. TPE concentrates sampling where
accuracy is good, so a parameter it stopped varying looks unimportant whether or
not it is; variance decomposition on TPE trials is not interpretable, and the
TPE set is reported only as a cross-check.

usage: python3 -m ab.gpt.util.hp_dimensionality <results.json> [--self-test]
"""
from __future__ import annotations

import argparse
import json
import math
import statistics
import sys

# candidate collapses: name -> f(trial) -> log10 of the collapsed quantity
COLLAPSES = {
    'log10_lr': lambda t: math.log10(t['lr']),
    'log10_eff_lr': lambda t: math.log10(t['lr'] / max(1 - t['momentum'], 1e-9)),
    'log10_lr_over_batch': lambda t: math.log10(t['lr'] / t['batch_size']),
    'log10_eff_lr_over_batch':
        lambda t: math.log10(t['lr'] / (max(1 - t['momentum'], 1e-9) * t['batch_size'])),
}

# extra parameters tested for variance beyond the collapse
EXTRAS = {
    'log10_batch': lambda t: math.log10(t['batch_size']),
    'log10_weight_decay': lambda t: math.log10(max(t['weight_decay'], 1e-12)),
    'label_smoothing': lambda t: float(t.get('label_smoothing', 0.0)),
    'nesterov': lambda t: 1.0 if t.get('nesterov') else 0.0,
    'cosine': lambda t: 1.0 if t.get('scheduler', 'cosine') == 'cosine' else 0.0,
}


# ---------------------------------------------------------------- regression
def _ols(X, y):
    """Least squares by normal equations with Gaussian elimination."""
    n, k = len(X), len(X[0])
    xtx = [[sum(X[i][a] * X[i][b] for i in range(n)) for b in range(k)] for a in range(k)]
    xty = [sum(X[i][a] * y[i] for i in range(n)) for a in range(k)]
    for c in range(k):                                   # elimination w/ pivot
        p = max(range(c, k), key=lambda r: abs(xtx[r][c]))
        if abs(xtx[p][c]) < 1e-12:
            return None
        xtx[c], xtx[p] = xtx[p], xtx[c]
        xty[c], xty[p] = xty[p], xty[c]
        for r in range(k):
            if r == c:
                continue
            f = xtx[r][c] / xtx[c][c]
            for b in range(c, k):
                xtx[r][b] -= f * xtx[c][b]
            xty[r] -= f * xty[c]
    return [xty[c] / xtx[c][c] for c in range(k)]


def r2(X, y):
    """R^2 and adjusted R^2 of y on design matrix X (intercept column included)."""
    beta = _ols(X, y)
    if beta is None:
        return None
    n, k = len(y), len(X[0])
    my = sum(y) / n
    sst = sum((v - my) ** 2 for v in y)
    sse = sum((y[i] - sum(b * x for b, x in zip(beta, X[i]))) ** 2 for i in range(n))
    if sst <= 0:
        return None
    val = 1 - sse / sst
    adj = 1 - (1 - val) * (n - 1) / (n - k) if n > k else None
    return {'r2': round(val, 4),
            'adj_r2': round(adj, 4) if adj is not None else None,
            'n': n, 'k': k, 'residuals': [y[i] - sum(b * x for b, x in zip(beta, X[i]))
                                          for i in range(n)]}


def _design(specs, n):
    """Intercept, then each (column, add_square) pair.

    Every continuous term enters squared as well as linearly. The self-test
    caught the reason: a parameter whose real effect is an inverted U with its
    peak inside the sampled range has almost no LINEAR component, so a
    linear-only term reports delta R^2 ~ 0 for an effect that is plainly there.
    That is the section 7k trap -- the same one that made a Spearman of +0.48
    look monotone on a curve peaking in the middle -- and an instrument built to
    detect collapses must not fall into it. Binary columns are not squared,
    since x^2 == x makes the design singular.
    """
    X = []
    for i in range(n):
        row = [1.0]
        for col, degree in specs:
            for d in range(1, degree + 1):
                row.append(col[i] ** d)
        X.append(row)
    return X


def _spec(col, binary=False, degree=2):
    """(column, highest power). 1 for binary, 2 for an added continuous term,
    3 for the collapse baseline -- a quadratic underfits an inverted U and the
    leftover curvature reappears as spurious delta R^2 on inert parameters."""
    return (col, 1 if binary else degree)


def _is_binary(col):
    return len(set(col)) <= 2


# ------------------------------------------------------------------- ranking
def rank(xs):
    order = sorted(range(len(xs)), key=lambda i: xs[i])
    r = [0.0] * len(xs)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and xs[order[j + 1]] == xs[order[i]]:
            j += 1
        for k in range(i, j + 1):
            r[order[k]] = (i + j) / 2
        i = j + 1
    return r


def pearson(a, b):
    n = len(a)
    if n < 3:
        return None
    ma, mb = sum(a) / n, sum(b) / n
    num = sum((x - ma) * (y - mb) for x, y in zip(a, b))
    da = math.sqrt(sum((x - ma) ** 2 for x in a))
    db = math.sqrt(sum((y - mb) ** 2 for y in b))
    return num / (da * db) if da and db else None


def spearman(a, b):
    return pearson(rank(a), rank(b))


def _permutation_null(base_col, col, y, n, binary, reps=500, seed=11):
    """Distribution of delta R^2 when the parameter is inert by construction.

    Shuffling the parameter breaks its association with accuracy while keeping
    its marginal distribution and the model's degrees of freedom, so the
    observed delta gets a calibrated p-value instead of an arbitrary threshold
    -- the mistake section 7e records about a cutoff chosen after the fact.
    """
    import random as _r
    rng = _r.Random(seed)
    base = r2(_design([_spec(base_col, degree=3)], n), y)
    obs = r2(_design([_spec(base_col, degree=3), _spec(col, binary)], n), y)
    if not base or not obs:
        return {'p95': None, 'p': None}
    d_obs = obs['r2'] - base['r2']
    draws = []
    shuf = list(col)
    for _ in range(reps):
        rng.shuffle(shuf)
        got = r2(_design([_spec(base_col, degree=3), _spec(shuf, binary)], n), y)
        if got:
            draws.append(got['r2'] - base['r2'])
    if not draws:
        return {'p95': None, 'p': None}
    draws.sort()
    p95 = draws[int(0.95 * (len(draws) - 1))]
    p = (1 + sum(1 for d in draws if d >= d_obs)) / (len(draws) + 1)
    return {'p95': round(p95, 4), 'p': round(p, 4)}


# ------------------------------------------------------------------ analysis
def analyse(trials, label=''):
    ok = [t for t in trials if t.get('success') and t.get('accuracy')]
    if len(ok) < 12:
        return {'label': label, 'usable_trials': len(ok),
                'note': 'too few usable trials for a variance decomposition'}
    y = [t['accuracy'] for t in ok]
    n = len(y)

    # 1. which collapse
    fits = {}
    for name, f in COLLAPSES.items():
        try:
            col = [f(t) for t in ok]
        except Exception:
            continue
        got = r2(_design([_spec(col)], n), y)
        if got:
            fits[name] = {'r2': got['r2'], 'adj_r2': got['adj_r2']}
    if not fits:
        return {'label': label, 'usable_trials': n, 'note': 'no collapse fit'}
    best = max(fits, key=lambda k: fits[k]['r2'])
    base_col = [COLLAPSES[best](t) for t in ok]
    base = r2(_design([_spec(base_col, degree=3)], n), y)

    # 2. nested deltas, and 3. partial correlations on the baseline residuals
    added, partial = {}, {}
    usable_extras = []
    for name, f in EXTRAS.items():
        try:
            col = [f(t) for t in ok]
        except Exception:
            continue
        if len(set(col)) < 2:
            added[name] = {'note': 'constant in this design'}
            continue
        usable_extras.append((name, col))
        got = r2(_design([_spec(base_col, degree=3),
                          _spec(col, _is_binary(col))], n), y)
        if got:
            d_obs = got['r2'] - base['r2']
            null = _permutation_null(base_col, col, y, n, _is_binary(col))
            added[name] = {
                'r2': got['r2'], 'delta_r2': round(d_obs, 4),
                'delta_adj_r2': round((got['adj_r2'] or 0) - (base['adj_r2'] or 0), 4),
                'null_p95_delta_r2': null['p95'], 'p_value': null['p'],
                'verdict': 'beyond the collapse' if null['p'] < 0.05 else 'explained by the collapse'}
        # Spearman is blind to a symmetric U, so report it alongside the
        # correlation of |residual| with distance from the column's midpoint,
        # which a U-shaped effect does move. delta R^2 above remains primary.
        rho = spearman(col, base['residuals'])
        mid = (max(col) + min(col)) / 2
        curv = spearman([abs(v - mid) for v in col],
                        [abs(r) for r in base['residuals']])
        partial[name] = {'spearman': round(rho, 4) if rho is not None else None,
                         'curvature': round(curv, 4) if curv is not None else None}

    full = None
    if usable_extras:
        full = r2(_design([_spec(base_col, degree=3)] +
                          [_spec(c, _is_binary(c)) for _, c in usable_extras], n), y)

    return {
        'label': label, 'usable_trials': n,
        'collapse_fits': fits, 'best_collapse': best,
        'baseline_r2': base['r2'], 'baseline_adj_r2': base['adj_r2'],
        'added_beyond_collapse': added,
        'partial_spearman_with_residuals': partial,
        'all_parameters_r2': full['r2'] if full else None,
        'all_parameters_adj_r2': full['adj_r2'] if full else None,
        'accuracy_spread_pts': round(100 * (max(y) - min(y)), 2),
    }


# ----------------------------------------------------------------- self-test
def self_test():
    """Validate the instrument on data whose dimensionality is known.

    A: accuracy is a pure inverted U in log10 eff_lr; batch and weight_decay are
       sampled but inert. The nested deltas must be ~0 for both.
    B: weight_decay genuinely matters. Its delta must be clearly positive.
    """
    import random
    random.seed(7)
    rng = random.Random(7)

    def make(n, wd_effect):
        """Shaped to match the measured data, not to flatter the instrument.

        A first attempt used an unbounded quadratic penalty; with momentum up to
        0.999 the effective lr reaches 10^9, accuracy went far negative and the
        floor clipped most points, giving a 71-point spread and swamping the
        planted effect. Real trials span roughly 0.10 to 0.82 with an inverted U
        peaking at effective lr 0.1 (section 7s quintile means 0.436, 0.746,
        0.813, 0.769, 0.568), so the generator is a bounded bump of that shape.
        """
        out = []
        for _ in range(n):
            lr = 10 ** rng.uniform(-5, 0)
            mom = rng.uniform(0.0, 0.999)
            batch = 2 ** rng.randint(5, 10)
            wd = 10 ** rng.uniform(-6, -1)
            eff = math.log10(lr / max(1 - mom, 1e-9))
            acc = 0.10 + 0.72 * math.exp(-0.5 * ((eff + 1.0) / 1.0) ** 2)
            if wd_effect:
                penalty = 1 - math.exp(-0.5 * ((math.log10(wd) + 3.5) / 1.0) ** 2)
                acc *= 1 - 0.25 * penalty
            acc += rng.gauss(0, 0.004)
            out.append({'lr': lr, 'momentum': mom, 'batch_size': batch,
                        'weight_decay': wd, 'label_smoothing': 0.0,
                        'nesterov': False, 'scheduler': 'cosine',
                        'accuracy': max(acc, 0.095), 'success': True})
        return out

    a = analyse(make(60, 0.0), 'A: one real dimension')
    b = analyse(make(60, 1.0), 'B: weight_decay also real')
    checks = {
        'A_best_collapse_is_eff_lr': a['best_collapse'] == 'log10_eff_lr',
        'A_batch_not_significant':
            a['added_beyond_collapse']['log10_batch']['p_value'] > 0.05,
        'A_weight_decay_not_significant':
            a['added_beyond_collapse']['log10_weight_decay']['p_value'] > 0.05,
        'B_weight_decay_detected':
            b['added_beyond_collapse']['log10_weight_decay']['p_value'] < 0.01,
        'B_batch_still_not_significant':
            b['added_beyond_collapse']['log10_batch']['p_value'] > 0.05,
    }
    print(json.dumps({'A': a, 'B': b, 'checks': checks}, indent=2))
    ok = all(checks.values())
    print('SELF-TEST PASSED' if ok else 'SELF-TEST FAILED')
    return 0 if ok else 1


def main():
    p = argparse.ArgumentParser()
    p.add_argument('results', nargs='?', help='JSON written by the headroom driver')
    p.add_argument('--self-test', action='store_true')
    a = p.parse_args()
    if a.self_test:
        return self_test()
    if not a.results:
        print(__doc__); return 2
    studies = json.load(open(a.results))
    out = []
    for s in studies:
        rec = {'model': s.get('model')}
        if s.get('random_trials'):
            rec['random_design'] = analyse(s['random_trials'], 'uniform random (primary)')
        if s.get('trials'):
            rec['tpe_design'] = analyse(s['trials'], 'TPE (cross-check, biased)')
        out.append(rec)
    print(json.dumps(out, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
