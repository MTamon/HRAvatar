"""Add a pose-anchor regularizer to DECA's ``optimize.py`` (idempotent).

Why
---
Strongly regularizing the FLAME shape vector (via
``apply_deca_optimize_regularizer.py`` + a high ``--lambda_shape``) fixes
the alien-looking enlarged head in ``optimize_vis.jpg`` but pushes the
optimizer to over-rotate the global pose to keep landmarks in fit:
landmarks must still be matched, and with shape now constrained the
rigid transform is forced to do the work. Visually this manifests as the
fitted head turning *more* than the source video shows.

This patcher anchors the optimized ``pose`` to its **per-frame DECA
initial value** (the per-frame ``pose`` predicted by DECA's regression
network and read from ``code.json``) with a tunable weight. This
preserves DECA's per-frame estimate as a soft prior so per-frame pose
cannot drift far from it just to compensate for shape regularization.

What this script does
---------------------
* Adds ``--lambda_pose_anchor`` CLI arg (default ``0.0``: no change).
* Snapshots ``pose_init`` right after ``pose = nn.Parameter(pose)`` and
  before optimization starts.
* Adds ``+ mean((pose - pose_init)**2) * args.lambda_pose_anchor``
  to the per-iteration ``total_loss``.

Default ``0.0`` means behaviour is unchanged unless the caller opts in.

Usage
-----
From the HRAvatar repo root::

    python tools/patches/apply_deca_pose_anchor.py

Or against a custom DECA checkout::

    python tools/patches/apply_deca_pose_anchor.py /path/to/DECA
"""
from __future__ import annotations

import argparse
from pathlib import Path


MARKER = '# HRAVATAR_POSE_ANCHOR'


# --- Edit 1: snapshot pose_init right after `pose = nn.Parameter(pose)` ----
#
# The line is unique in optimize.py. We insert immediately after it so
# pose_init is in scope for the optimization loop below.

POSE_PARAM_ANCHOR = 'pose = nn.Parameter(pose)'
POSE_INIT_PAYLOAD = (
    f'{MARKER} BEGIN: snapshot DECA per-frame pose for soft anchor\n'
    'pose_init = pose.detach().clone()\n'
    f'{MARKER} END\n'
)


# --- Edit 2: add the anchor term to total_loss -----------------------------
#
# Anchor on the existing per-frame `*args.lambda_pose_diff` line which is
# unique. We append the anchor term right after it.

POSE_DIFF_ANCHOR = (
    'total_loss += torch.mean(torch.square(pose[1:] - pose[:-1])) '
    '*args.lambda_pose_diff'
)
POSE_ANCHOR_LOSS_PAYLOAD = (
    f'{MARKER} BEGIN: keep optimized pose near DECA per-frame initial\n'
    'total_loss += torch.mean(torch.square(pose - pose_init)) '
    '* args.lambda_pose_anchor\n'
    f'{MARKER} END\n'
)


# --- Edit 3: argparse -------------------------------------------------------

ARGPARSE_ANCHOR = (
    "parser.add_argument('--lambda_pose_diff', type=float,default=10)"
)
ARGPARSE_PAYLOAD = (
    f"{MARKER} BEGIN: pose-anchor regularizer weight\n"
    "parser.add_argument('--lambda_pose_anchor', type=float, default=0.0,\n"
    "                    help='Weight on (pose - pose_init)**2 anchor in optimize().\\n"
    "Defaults to 0.0 (off, original behaviour). Use a small value (0.05-0.5)\\n"
    "when --lambda_shape is high to prevent the optimizer from over-rotating\\n"
    "the global pose to compensate for the constrained shape vector.')\n"
    f"{MARKER} END\n"
)


# --- Helpers (mirrors apply_deca_stable_bbox.py) ---------------------------

def _line_indent_at(text: str, idx: int) -> str:
    line_start = text.rfind('\n', 0, idx) + 1
    indent = []
    i = line_start
    while i < len(text) and text[i] in (' ', '\t'):
        indent.append(text[i])
        i += 1
    return ''.join(indent)


def _reindent(payload: str, indent: str) -> str:
    out = []
    for line in payload.split('\n'):
        if line:
            out.append(indent + line)
        else:
            out.append('')
    joined = '\n'.join(out)
    if not joined.endswith('\n'):
        joined += '\n'
    return joined


def _insert_after(text: str, anchor: str, payload: str, label: str,
                  path: Path) -> str:
    if anchor not in text:
        raise SystemExit(
            f'[error] anchor missing in {path} ({label}):\n'
            f'        could not find: {anchor!r}\n'
            f'        DECA fork may have drifted. Re-base or hand-merge.')
    if text.count(anchor) != 1:
        raise SystemExit(
            f'[error] anchor not unique in {path} ({label}): '
            f'{anchor!r} occurs {text.count(anchor)} times.')
    idx = text.index(anchor)
    indent = _line_indent_at(text, idx)
    line_end = text.index('\n', idx) + 1
    return text[:line_end] + _reindent(payload, indent) + text[line_end:]


def patch_optimize(optimize_py: Path) -> bool:
    """Apply all 3 edits to optimize.py. Returns True iff the file was changed."""
    text = optimize_py.read_text()
    if MARKER in text:
        print(f'[skip] {optimize_py} already patched ({MARKER} present)')
        return False

    text = _insert_after(text, POSE_PARAM_ANCHOR, POSE_INIT_PAYLOAD,
                         'pose_init snapshot', optimize_py)
    text = _insert_after(text, POSE_DIFF_ANCHOR, POSE_ANCHOR_LOSS_PAYLOAD,
                         'anchor loss term', optimize_py)
    text = _insert_after(text, ARGPARSE_ANCHOR, ARGPARSE_PAYLOAD,
                         'argparse', optimize_py)

    optimize_py.write_text(text)
    print(f'[ok]   {optimize_py}')
    return True


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description='Add --lambda_pose_anchor regularizer to DECA '
                    "optimize.py (idempotent).")
    default_deca = (Path(__file__).resolve().parent.parent.parent
                    / 'preprocess' / 'submodules' / 'DECA')
    parser.add_argument(
        'deca_root', nargs='?', default=str(default_deca),
        help=f'Path to the DECA checkout (default: {default_deca}).')
    args = parser.parse_args(argv)

    deca_root = Path(args.deca_root).resolve()
    optimize_py = deca_root / 'optimize.py'
    if not optimize_py.is_file():
        raise SystemExit(
            f'[error] expected DECA file not found: {optimize_py}\n'
            f'        is {deca_root} the DECA root?')

    changed = patch_optimize(optimize_py)
    if not changed:
        print('[done] DECA optimize.py already exposes --lambda_pose_anchor.')
    else:
        print('[done] DECA optimize.py patched. Verify with:')
        print(f'         grep -n {MARKER} {optimize_py}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
