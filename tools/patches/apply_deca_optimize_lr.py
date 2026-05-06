"""Add configurable lr + opt-in step decay to DECA's ``optimize.py`` (idempotent).

Why
---
The HRAvatar DECA fork's ``optimize.py`` uses a hardcoded Adam ``lr=1e-2``
for the main param group (pose / exp / shape) and has a step-decay
scheduler that is **commented out**::

    # if k%300==0:
    #     lr_opt/=2
    #     for param_group in opt_p.param_groups:
    #         param_group['lr'] = lr_opt

The original commented form forces *all* param groups to a single shared
lr, which silently overrides the carefully chosen per-group lrs for
``eyelid`` (1e-3), ``translation`` (1e-4), and ``translation_p`` (1e-2)
- almost certainly the reason it was disabled.

This patcher exposes three CLI args on ``optimize.py``:

* ``--main_lr`` (default 1e-2) - initial Adam lr for pose / exp / shape.
  The eyelid / translation / translation_p lrs stay at their original
  values; only the main group is touched.
* ``--main_lr_decay_step`` (default 0 = disabled) - apply step decay every
  N iter to *every* param group via a multiplicative factor (so the
  relative ratios between groups are preserved, unlike the original
  commented form).
* ``--main_lr_decay_factor`` (default 0.5) - the multiplier.

Defaults preserve the original HRAvatar fork behaviour exactly. Existing
pipelines are unaffected unless the caller opts in.

``demos/_preprocess_subject.sh`` exposes ``--main-lr`` /
``--main-lr-decay-step`` / ``--main-lr-decay-factor`` flags that forward
into DECA.

Idempotency
-----------
Re-running after a successful apply prints ``[skip]`` and exits. Anchor
drift raises a diagnostic so the operator can hand-merge.

Usage
-----
From the HRAvatar repo root::

    python tools/patches/apply_deca_optimize_lr.py
"""
from __future__ import annotations

import argparse
from pathlib import Path


MARKER = '# HRAVATAR_OPTIMIZE_LR'


# --- Edit 1: route lr_opt through args.main_lr -----------------------------
#
# Original (HRAvatar DECA fork, optimize.py line 127):
#
#     lr_opt=1e-2
#
# The bound name lr_opt is reused in two Adam(...) constructors below, so
# we keep that name and just source it from args.main_lr.

LR_INIT_OLD = '        lr_opt=1e-2'
LR_INIT_NEW = f'        lr_opt = args.main_lr  {MARKER}'


# --- Edit 2: replace the dead commented-out step decay with an opt-in form
#
# The original block (optimize.py line 174-177) is commented out and
# would force all param groups to the same lr. We replace the whole
# block with a multiplicative form that preserves per-group ratios and
# is gated by args.main_lr_decay_step > 0.

DECAY_OLD = (
    '            # if k%300==0:\n'
    '            #     lr_opt/=2\n'
    '            #     for param_group in opt_p.param_groups:\n'
    "            #         param_group['lr'] = lr_opt"
)
DECAY_NEW = (
    f'            {MARKER} BEGIN: opt-in step decay (per-group multiplicative)\n'
    '            if args.main_lr_decay_step > 0 and k % args.main_lr_decay_step == 0:\n'
    '                for _hrav_pg in opt_p.param_groups:\n'
    "                    _hrav_pg['lr'] *= args.main_lr_decay_factor\n"
    f'            {MARKER} END'
)


# --- Edit 3: add CLI args to argparse --------------------------------------
#
# Anchor: ``args = parser.parse_args()`` (unique). Insert immediately
# *before* this line, mirroring apply_deca_optimize_iters.py. Independent
# of patch ordering.

ARGPARSE_ANCHOR = '    args = parser.parse_args()'
ARGPARSE_PAYLOAD = (
    f'{MARKER} BEGIN: configurable main-loop lr + opt-in step decay\n'
    "parser.add_argument('--main_lr', type=float, default=1e-2,\n"
    "                    help='Initial Adam lr for main optimize loop pose/exp/shape '\n"
    "                         'param group. Default 1e-2 matches the original '\n"
    "                         'HRAvatar fork. eyelid/translation/translation_p lrs '\n"
    "                         'are not exposed (kept at fork defaults).')\n"
    "parser.add_argument('--main_lr_decay_step', type=int, default=0,\n"
    "                    help='If >0, multiply every param-group lr by '\n"
    "                         '--main_lr_decay_factor every N iter of the main '\n"
    "                         'loop. 0 disables (default). Per-group ratios are '\n"
    "                         'preserved (unlike the HRAvatar fork commented form).')\n"
    "parser.add_argument('--main_lr_decay_factor', type=float, default=0.5,\n"
    "                    help='Multiplier applied to every param-group lr at each '\n"
    "                         'decay step. Only used if --main_lr_decay_step > 0.')\n"
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
    ('lr_opt init', LR_INIT_OLD),
    ('decay block', DECAY_OLD),
    ('argparse anchor', ARGPARSE_ANCHOR),
)


def patch_optimize(optimize_py: Path) -> bool:
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

    text = text.replace(LR_INIT_OLD, LR_INIT_NEW, 1)
    text = text.replace(DECAY_OLD, DECAY_NEW, 1)

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
        description='Add --main_lr / --main_lr_decay_* CLI args to DECA '
                    'optimize.py (idempotent).')
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
        print('[done] DECA optimize.py already exposes the lr / decay flags.')
    else:
        print('[done] DECA optimize.py patched. Verify with:')
        print(f'         grep -n {MARKER} {optimize_py}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
