"""Add configurable shape / expression regularizer weights to DECA's
``optimize.py`` (idempotent).

Why
---
The HRAvatar DECA fork's ``optimize.py`` uses a hardcoded ``1e-2`` weight
for both ``torch.square(shape).mean()`` and ``torch.square(exp).mean()``
in its multi-frame photometric refinement loop. With the per-frame
landmark loss summed over thousands of frames, that weak prior lets the
optimizer warp the FLAME shape vector to extreme values to fit landmarks
- which manifests in ``optimize_vis.jpg`` as an *alien-looking* enlarged
crown / collapsed mid-face.

This patcher adds two CLI args - ``--lambda_shape`` and ``--lambda_exp``
- to ``optimize.py`` and replaces the hardcoded weights with them. The
defaults remain ``1e-2`` so behaviour is **unchanged** unless the caller
opts in. ``demos/_preprocess_subject.sh`` exposes ``--lambda-shape`` /
``--lambda-exp`` flags that forward stronger values into DECA.

Idempotency
-----------
Re-running the script after a successful apply prints ``[skip]`` and
exits. If the marker is present but anchors changed (DECA fork drift),
the script aborts with a diagnostic so the operator can hand-merge.

Usage
-----
From the HRAvatar repo root::

    python tools/patches/apply_deca_optimize_regularizer.py

Or against a custom DECA checkout::

    python tools/patches/apply_deca_optimize_regularizer.py /path/to/DECA
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path


MARKER = '# HRAVATAR_OPTIMIZE_REGULARIZER'


# --- Edit 1: replace hardcoded shape/exp regularizer weights ---------------
#
# Original (HRAvatar DECA fork, optimize.py around line 202):
#
#     total_loss = landmark_loss2 + torch.mean(torch.square(shape)) * 1e-2 + torch.mean(torch.square(exp)) * 1e-2
#
# Replacement uses args.lambda_shape / args.lambda_exp and tags the line
# with the marker so we can detect and skip on re-run.

OPTIMIZE_TOTALLOSS_OLD = (
    'total_loss = landmark_loss2 + torch.mean(torch.square(shape)) * 1e-2 '
    '+ torch.mean(torch.square(exp)) * 1e-2'
)
OPTIMIZE_TOTALLOSS_NEW = (
    'total_loss = landmark_loss2 + torch.mean(torch.square(shape)) '
    '* args.lambda_shape + torch.mean(torch.square(exp)) '
    '* args.lambda_exp  ' + MARKER
)


# --- Edit 2: add CLI args to argparse --------------------------------------
#
# Anchor: the existing ``--lambda_pose_diff`` line (unique in the file).
# We insert a single payload block right after that line. The payload is
# pre-tagged with the marker so the patcher's idempotency check works
# without separate detection logic.

OPTIMIZE_ARGPARSE_ANCHOR = (
    "parser.add_argument('--lambda_pose_diff', type=float,default=10)"
)
OPTIMIZE_ARGPARSE_PAYLOAD = (
    f"{MARKER} BEGIN: configurable shape / expression regularizer weights\n"
    "parser.add_argument('--lambda_shape', type=float, default=1e-2,\n"
    "                    help='Weight on shape**2 regularizer in optimize().\\n"
    "Defaults to 1e-2 to match the original HRAvatar fork. Increase to\\n"
    "1.0-5.0 if optimize_vis.jpg shows an alien-looking enlarged head /\\n"
    "collapsed mid-face (the unregularized shape vector overfits landmarks).')\n"
    "parser.add_argument('--lambda_exp', type=float, default=1e-2,\n"
    "                    help='Weight on exp**2 regularizer in optimize(). "
    "Defaults to 1e-2.')\n"
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


def patch_optimize(optimize_py: Path) -> bool:
    """Apply both edits to optimize.py. Returns True iff the file was changed."""
    text = optimize_py.read_text()
    if MARKER in text:
        print(f'[skip] {optimize_py} already patched ({MARKER} present)')
        return False

    if OPTIMIZE_TOTALLOSS_OLD not in text:
        raise SystemExit(
            f'[error] anchor missing in {optimize_py}:\n'
            f'        could not find: {OPTIMIZE_TOTALLOSS_OLD!r}\n'
            f'        DECA fork may have drifted. Re-base or hand-merge.')
    if OPTIMIZE_ARGPARSE_ANCHOR not in text:
        raise SystemExit(
            f'[error] anchor missing in {optimize_py}:\n'
            f'        could not find: {OPTIMIZE_ARGPARSE_ANCHOR!r}\n'
            f'        DECA fork may have drifted. Re-base or hand-merge.')
    if text.count(OPTIMIZE_TOTALLOSS_OLD) != 1:
        raise SystemExit(
            f'[error] total_loss anchor not unique in {optimize_py}: '
            f'occurs {text.count(OPTIMIZE_TOTALLOSS_OLD)} times.')
    if text.count(OPTIMIZE_ARGPARSE_ANCHOR) != 1:
        raise SystemExit(
            f'[error] argparse anchor not unique in {optimize_py}: '
            f'occurs {text.count(OPTIMIZE_ARGPARSE_ANCHOR)} times.')

    text = text.replace(OPTIMIZE_TOTALLOSS_OLD, OPTIMIZE_TOTALLOSS_NEW, 1)

    idx = text.index(OPTIMIZE_ARGPARSE_ANCHOR)
    indent = _line_indent_at(text, idx)
    line_end = text.index('\n', idx) + 1
    text = text[:line_end] + _reindent(OPTIMIZE_ARGPARSE_PAYLOAD, indent) + text[line_end:]

    optimize_py.write_text(text)
    print(f'[ok]   {optimize_py}')
    return True


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description='Add --lambda_shape / --lambda_exp CLI args to DECA '
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
        print('[done] DECA optimize.py already exposes the regularizer flags.')
    else:
        print('[done] DECA optimize.py patched. Verify with:')
        print(f'         grep -n {MARKER} {optimize_py}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
