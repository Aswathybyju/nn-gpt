"""
Hyperparameter families: one architecture shown under several training settings.

A *family* is one architecture (``nn``) with one ``transform``, at one fixed
``epoch``, that was trained under >= ``min_settings`` distinct hyperparameter
settings, each carrying its own ``train_stat`` diagnostics.

Grouping is on hyperparameter VALUES, never on ``stat.prm`` (the uid): the same
(lr, momentum, batch) triple appears under 4,138 distinct uids for CIFAR-10
alone, so grouping by uid would shatter every family. Per setting we keep the
best-accuracy row, and ``epoch`` is part of the partition key so a 50-epoch row
can never be shown next to 1-epoch rows.

Used only by prompt configs carrying an ``hp_family`` block. Nothing here is
imported by existing generation or fine-tuning paths.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import pandas as pd

from ab.nn.util.db.Init import sql_conn, close_conn

# Excluded by measurement, see TRAIN_STAT_PLAN.md:
#   gradient_norm       r = -0.13 with final accuracy
#   epoch_max           non-NULL in 7.2% of candidate rows
#   samples_per_second  37% of families mix RTX 4090 and RTX 3090
#   loss_gap/gen_gap    within-family spread 0.015 / 0.008 at epoch 1
#   cpu/ram/gpu fields  describe the machine, not the network
DEFAULT_DIAGNOSTIC_FIELDS = ('train_loss', 'test_loss', 'train_accuracy')

_HP_COLUMNS = ('lr', 'momentum', 'batch')


@dataclass
class Family:
    nn: str
    transform: Optional[str]          # None when members carry their own
    task: str
    dataset: str
    metric: str
    epoch: int
    nn_code: str
    transform_code: Optional[str]     # None when members carry their own
    members: list  # dicts: lr, momentum, batch, accuracy, + diagnostics
                   # (+ transform, transform_code when group_by_transform=False)

    @property
    def shared_transform(self) -> bool:
        return self.transform is not None

    @property
    def k(self) -> int:
        return len(self.members)


def _diagnostic_columns(cur) -> set:
    return {r[1] for r in cur.execute('PRAGMA table_info(train_stat)')} - {'stat_id'}


def _validate_fields(cur, fields: Sequence[str]) -> tuple:
    known = _diagnostic_columns(cur)
    unknown = [f for f in fields if f not in known]
    if unknown:
        raise ValueError(f'unknown train_stat field(s): {unknown}; available: {sorted(known)}')
    return tuple(fields)


def _span(values: list, n: int) -> list:
    """Evenly spaced picks across an ordered list — spans the lr range instead of
    clustering on near-duplicate top-accuracy settings."""
    if n >= len(values):
        return values
    idx = sorted({round(i * (len(values) - 1) / (n - 1)) for i in range(n)})
    return [values[i] for i in idx]


def fetch_families(
    task: str = 'img-classification',
    dataset: str = 'cifar-10',
    metric: str = 'acc',
    epoch: int = 1,
    min_settings: int = 3,
    max_members: int = 4,
    min_lr_ratio: float = 2.0,
    diagnostic_fields: Sequence[str] = DEFAULT_DIAGNOSTIC_FIELDS,
    nn_prefixes: Optional[Sequence[str]] = None,
    max_families: Optional[int] = None,
    group_by_transform: bool = True,
) -> list:
    """
    Families matching the constraint set, ordered by architecture name so both
    arms of an experiment see exactly the same families in the same order.

    ``diagnostic_fields`` drives SELECTION as well as display: rows must have all
    of them non-NULL. Both arms must therefore pass the same list, or the control
    would be built from a different set of families.

    ``group_by_transform=True`` holds the transform constant inside a family, so
    members differ only in (lr, momentum, batch); this exists only at epoch 1
    (183 families). ``False`` groups by architecture alone, which is the only way
    to reach epochs >= 2 — members then each carry their own transform and the
    prompt must show it.
    """
    conn, cur = sql_conn()
    try:
        fields = _validate_fields(cur, diagnostic_fields)
        ts_cols = ''.join(f', ts.{f}' for f in fields)
        non_null = ''.join(f' AND ts.{f} IS NOT NULL' for f in fields)
        hp_pivot = ''.join(
            f", MAX(CASE WHEN p.name='{c}' THEN p.value END) {c}" for c in _HP_COLUMNS)
        prefix_sql = ''
        params = [task, dataset, metric]
        if nn_prefixes:
            prefix_sql = ' AND (' + ' OR '.join("s.nn LIKE ?" for _ in nn_prefixes) + ')'
            params += [f'{p}%' for p in nn_prefixes]

        cur.execute(f"""
            CREATE TEMP TABLE IF NOT EXISTS _hpf_v AS
            SELECT s.id, s.nn, s.transform, s.epoch, s.accuracy{ts_cols}{hp_pivot}
            FROM stat s
            JOIN train_stat ts ON ts.stat_id = s.id
            JOIN prm p         ON p.uid = s.prm
            WHERE s.task = ? AND s.dataset = ? AND s.metric = ?{prefix_sql}
            GROUP BY s.id
            HAVING lr IS NOT NULL AND momentum IS NOT NULL AND batch IS NOT NULL{non_null}
        """, params)

        # best-accuracy row per setting; epoch inside the partition key
        cur.execute("""
            CREATE TEMP TABLE IF NOT EXISTS _hpf_best AS
            SELECT * FROM (
              SELECT *, ROW_NUMBER() OVER (
                  PARTITION BY nn, transform, epoch, lr, momentum, batch
                  ORDER BY accuracy DESC, id) rn
              FROM _hpf_v)
            WHERE rn = 1
        """)

        group_cols = 'nn, transform' if group_by_transform else 'nn'
        groups = cur.execute(f"""
            SELECT {group_cols} FROM _hpf_best
            WHERE epoch = ?
            GROUP BY {group_cols}
            HAVING COUNT(*) >= ? AND MAX(lr) / MIN(lr) >= ?
            ORDER BY {group_cols}
        """, (epoch, min_settings, min_lr_ratio)).fetchall()
        if max_families:
            groups = groups[:max_families]

        transform_code = {}

        def _transform_code(name):
            if name not in transform_code:
                row = cur.execute('SELECT code FROM transform WHERE name = ?', (name,)).fetchone()
                transform_code[name] = row[0] if row else None
            return transform_code[name]

        families = []
        for group in groups:
            nn = group[0]
            transform = group[1] if group_by_transform else None
            where, args = ('nn = ? AND transform = ?', (nn, transform)) if group_by_transform \
                else ('nn = ?', (nn,))
            rows = cur.execute(f"""
                SELECT lr, momentum, batch, accuracy, transform{ts_cols.replace('ts.', '')}
                FROM _hpf_best WHERE epoch = ? AND {where}
                ORDER BY lr, momentum, batch
            """, (epoch, *args)).fetchall()
            cols = ['lr', 'momentum', 'batch', 'accuracy', 'transform', *fields]
            members = _span([dict(zip(cols, r)) for r in rows], max_members)

            nn_code = cur.execute('SELECT code FROM nn WHERE name = ?', (nn,)).fetchone()
            if not nn_code:
                continue
            if group_by_transform:
                shared = _transform_code(transform)
                if not shared:
                    continue
            else:
                shared = None
                for m in members:
                    m['transform_code'] = _transform_code(m['transform'])
                if any(not m['transform_code'] for m in members):
                    continue
            families.append(Family(
                nn=nn, transform=transform, task=task, dataset=dataset, metric=metric,
                epoch=epoch, nn_code=nn_code[0], transform_code=shared, members=members))
        return families
    finally:
        close_conn(conn)


def _fmt(v) -> str:
    return f'{v:.4f}' if isinstance(v, float) else str(v)


def render_family_block(
    family: Family,
    diagnostic_fields: Sequence[str] = DEFAULT_DIAGNOSTIC_FIELDS,
    show_diagnostics: bool = True,
    show_member_accuracy: bool = True,
) -> str:
    """The repeated per-setting section of the prompt.

    The two arms differ only by ``show_diagnostics``; ``show_member_accuracy``
    exists because at epoch 1 the diagnostics are a near-perfect monotone
    restatement of accuracy (median within-family Spearman |rho| = 1.0), so a
    variant that withholds accuracy from both arms makes the diagnostics the
    only outcome signal.
    """
    out = []
    for i, m in enumerate(family.members, 1):
        out.append(f'--- Setting {i} ---')
        hp = ', '.join(f'"{c}": {m[c]}' for c in _HP_COLUMNS)
        if not family.shared_transform:
            hp += f', "transform": "{m["transform"]}"'
        out.append(f'<hp>{{{hp}}}</hp>')
        if not family.shared_transform and m.get('transform_code'):
            out.append('<tr>')
            out.append(m['transform_code'].rstrip())
            out.append('</tr>')
        if show_member_accuracy:
            out.append(f'Result: {family.metric}={_fmt(m["accuracy"])} '
                       f'after {family.epoch} epoch(s)')
        if show_diagnostics:
            diag = ', '.join(f'{f}={_fmt(m[f])}' for f in diagnostic_fields if f in m)
            out.append(f'Training diagnostics: {diag}')
        out.append('')
    return '\n'.join(out).rstrip('\n')


def anchor_row(family: Family) -> pd.Series:
    """Row pickled as dataframe.df so Eval can train the generated model."""
    best = max(family.members, key=lambda m: m['accuracy'])
    transform = family.transform or best.get('transform')
    return pd.Series({
        'nn': family.nn,
        'nn_code': family.nn_code,
        'task': family.task,
        'dataset': family.dataset,
        'metric': family.metric,
        'epoch': family.epoch,
        'accuracy': best['accuracy'],
        'transform': transform,
        'transform_code': family.transform_code or best.get('transform_code'),
        'prm': {c: best[c] for c in _HP_COLUMNS} | {'transform': transform},
    })
