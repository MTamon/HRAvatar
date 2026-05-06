"""Add configurable iter caps + iris-only early-stop to DECA's ``optimize.py`` (idempotent).

Why
---
The HRAvatar DECA fork's ``optimize.py`` runs the multi-frame photometric
refinement loop for a hardcoded ``1000`` iterations and the iris-only
refinement for ``500``, with no early-stop / plateau detection. Inspection
of typical training-video logs shows the iris loss is usually flat from
~iter 300 onwards (4th-decimal noise), so the last 200 iris iters are
pure waste.

The main loop is left WITHOUT early-stop on purpose: empirically it can
keep refining beyond a 100-iter plateau, and inputs that look converged
early sometimes rebound. We only expose ``--max_iters`` to cap it (default
1000 = original behaviour) so users can extend it if needed; the rel-tol
short-circuit applies only to the iris loop.

This patcher exposes four CLI args on ``optimize.py``:

* ``--max_iters`` (default 1000) - hard cap for the main optimize loop
  (no early-stop; runs to the cap).
* ``--max_iris_iters`` (default 500) - hard cap for the iris loop.
* ``--early_stop_rel_tol`` (default 0.0 = disabled) - relative-improvement
  tolerance on ``landmark_loss``, sampled every 100 iters. **Iris loop only.**
* ``--early_stop_patience`` (default 2) - consecutive 100-iter windows
  without a tolerated improvement before breaking the iris loop.

Defaults preserve the original HRAvatar fork behaviour exactly. The
short-circuit only fires when the caller opts in via
``--early_stop_rel_tol > 0``, so existing pipelines are unaffected.

``demos/_preprocess_subject.sh`` exposes ``--max-iters`` /
``--max-iris-iters`` / ``--early-stop-rel-tol`` / ``--early-stop-patience``
flags that forward into DECA.

Idempotency
-----------
Re-running after a successful apply prints ``[skip]`` and exits. Anchor
drift (DECA fork updates) raises a diagnostic so the operator can
hand-merge.

Usage
-----
From the HRAvatar repo root::

    python tools/patches/apply_deca_optimize_iters.py

Or against a custom DECA checkout::

    python tools/patches/apply_deca_optimize_iters.py /path/to/DECA
"""
from __future__ import annotations

import argparse
from pathlib import Path


MARKER = '# HRAVATAR_OPTIMIZE_ITERS'


# --- Edit 1: cap the main loop iter count via args.max_iters ---------------
#
# Original (HRAvatar DECA fork, optimize.py line 157):
#
#     for k in range(1,1001):
#
# We hoist the iter count into a local bound by ``args.max_iters``. No
# early-stop bookkeeping is initialised here on purpose; the main loop
# always runs to the cap (see module docstring for rationale).

MAIN_RANGE_OLD = '        for k in range(1,1001):'
MAIN_RANGE_NEW = (
    f'        _HRAV_MAIN_ITERS = args.max_iters  {MARKER}\n'
    f'        for k in range(1, _HRAV_MAIN_ITERS + 1):  {MARKER}'
)


# --- Edit 2: cap the iris loop iter count via args.max_iris_iters ----------
#
# Mirror of Edit 1 for the iris-only refinement at optimize.py line 296.

IRIS_RANGE_OLD = '            for k in range(1,501):'
IRIS_RANGE_NEW = (
    f'            _HRAV_IRIS_ITERS = args.max_iris_iters  {MARKER}\n'
    f'            _HRAV_BEST_LOSS_IRIS = float(\'inf\')  {MARKER}\n'
    f'            _HRAV_BAD_WINDOWS_IRIS = 0  {MARKER}\n'
    f'            for k in range(1, _HRAV_IRIS_ITERS + 1):  {MARKER}'
)


# --- Edit 3: early-stop on plateau in iris loop ----------------------------
#
# Inserted just after ``avg_lmk_loss+=landmark_loss2.item()`` in the iris
# loop (matching the existing logging cadence). Bookkeeping vars are
# suffixed _IRIS for clarity even though the main loop no longer defines
# the unsuffixed counterparts.

IRIS_EARLYSTOP_OLD = (
    '                opt_p.zero_grad()\n'
    '                total_loss.backward()\n'
    '                opt_p.step()\n'
    '                avg_lmk_loss+=landmark_loss2.item()\n'
    '                # visualize\n'
    '                if k % 100 == 0:'
)
IRIS_EARLYSTOP_NEW = (
    '                opt_p.zero_grad()\n'
    '                total_loss.backward()\n'
    '                opt_p.step()\n'
    '                avg_lmk_loss+=landmark_loss2.item()\n'
    f'                {MARKER} BEGIN: early-stop on plateau (iris loop)\n'
    '                if args.early_stop_rel_tol > 0 and k % 100 == 0:\n'
    '                    _hrav_cur = landmark_loss2.item()\n'
    '                    if _hrav_cur < _HRAV_BEST_LOSS_IRIS * (1.0 - args.early_stop_rel_tol):\n'
    '                        _HRAV_BEST_LOSS_IRIS = _hrav_cur\n'
    '                        _HRAV_BAD_WINDOWS_IRIS = 0\n'
    '                    else:\n'
    '                        _HRAV_BAD_WINDOWS_IRIS += 1\n'
    '                        if _HRAV_BAD_WINDOWS_IRIS >= args.early_stop_patience:\n'
    '                            print(f\'[early-stop] iris optimize plateau at iter={k} \'\n'
    '                                  f\'(best={_HRAV_BEST_LOSS_IRIS:.6f}, cur={_hrav_cur:.6f})\')\n'
    '                            break\n'
    f'                {MARKER} END\n'
    '                # visualize\n'
    '                if k % 100 == 0:'
)


# --- Edit 4: add CLI args to argparse --------------------------------------
#
# Anchor: ``args = parser.parse_args()`` (unique, last line of the
# argparse setup block). We insert immediately *before* this line so the
# new add_argument calls run with the parser still accepting registrations.
# This anchor does not depend on apply_deca_optimize_regularizer.py having
# been applied first - the patches are independent in either order.

ARGPARSE_ANCHOR = '    args = parser.parse_args()'
ARGPARSE_PAYLOAD = (
    f'{MARKER} BEGIN: configurable iter caps + early-stop knobs\n'
    "parser.add_argument('--max_iters', type=int, default=1000,\n"
    "                    help='Max iterations for the main DECA optimize loop. '\n"
    "                         'Default 1000 matches the original HRAvatar fork.')\n"
    "parser.add_argument('--max_iris_iters', type=int, default=500,\n"
    "                    help='Max iterations for the iris-only optimize loop. '\n"
    "                         'Default 500 matches the original HRAvatar fork.')\n"
    "parser.add_argument('--early_stop_rel_tol', type=float, default=0.0,\n"
    "                    help='Plateau tolerance on landmark_loss for the '\n"
    "                         'IRIS-ONLY loop (checked every 100 iter). 0.0 '\n"
    "                         'disables. 0.005-0.01 typical. The main loop is '\n"
    "                         'NOT early-stopped (must run full --max_iters).')\n"
    "parser.add_argument('--early_stop_patience', type=int, default=2,\n"
    "                    help='Consecutive 100-iter windows without rel_tol '\n"
    "                         'improvement before breaking the iris loop. '\n"
    "                         'Only used if early_stop_rel_tol > 0.')\n"
    f'{MARKER} END\n'
)


# --- Helpers (mirrors apply_deca_optimize_regularizer.py) ------------------

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


_ANCHORS = (
    ('main range', MAIN_RANGE_OLD),
    ('iris range', IRIS_RANGE_OLD),
    ('iris early-stop', IRIS_EARLYSTOP_OLD),
    ('argparse anchor', ARGPARSE_ANCHOR),
)


def patch_optimize(optimize_py: Path) -> bool:
    """Apply all four edits to optimize.py. Returns True iff the file was changed."""
    text = optimize_py.read_text()
    if MARKER in text:
        print(f'[skip] {optimize_py} already patched ({MARKER} present)')
        return False

    for label, anchor in _ANCHORS:
        n = text.count(anchor)
        if n != 1:
            raise SystemExit(
                f'[error] {label} anchor count={n} (expected 1) in {optimize_py}.\n'
                f'        DECA fork may have drifted. Re-base or hand-merge.\n'
                f'        Anchor head: {anchor.splitlines()[0]!r}')

    text = text.replace(MAIN_RANGE_OLD, MAIN_RANGE_NEW, 1)
    text = text.replace(IRIS_EARLYSTOP_OLD, IRIS_EARLYSTOP_NEW, 1)
    text = text.replace(IRIS_RANGE_OLD, IRIS_RANGE_NEW, 1)

    idx = text.index(ARGPARSE_ANCHOR)
    line_start = text.rfind('\n', 0, idx) + 1
    indent = _line_indent_at(text, idx)
    payload = _reindent(ARGPARSE_PAYLOAD, indent)
    text = text[:line_start] + payload + text[line_start:]

    optimize_py.write_text(text)
    print(f'[ok]   {optimize_py}')
    return True


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description='Add --max_iters / --max_iris_iters / --early_stop_* '
                    'CLI args to DECA optimize.py (idempotent).')
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
        print('[done] DECA optimize.py already exposes the iter / early-stop flags.')
    else:
        print('[done] DECA optimize.py patched. Verify with:')
        print(f'         grep -n {MARKER} {optimize_py}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
