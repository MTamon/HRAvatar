"""Apply the HRAvatar `stable_bbox` integration to a local DECA checkout.

Designed as an idempotent string-based patcher rather than a unified diff,
because (a) the HRAvatar DECA fork has drifted from yfeng95/DECA so anchored
context lines are unreliable, and (b) blank context lines in unified diffs
are fragile to whitespace-stripping editors and silently corrupt the patch.

What this script does
---------------------
Given the DECA root (defaults to ``preprocess/submodules/DECA`` relative to
this script's parent's parent), it locates the two files we need to touch:

  1. ``decalib/datasets/datasets.py``  — TestData crop pipeline
  2. ``demos/demo_reconstruct.py``     — CLI entry point

In each file it inserts a clearly-marked ``# HRAVATAR_STABLE_BBOX BEGIN ...``
block. If the marker already exists, the file is left alone (so re-running
this script after a partial / completed apply is a no-op).

If the expected anchor strings cannot be found, the script aborts with a
diagnostic pointing at the file + the missing anchor, so the operator can
either rebase the DECA fork or hand-merge the change. Nothing is written
in that error path.

Usage
-----
From the HRAvatar repo root::

    python tools/patches/apply_deca_stable_bbox.py

Or against a custom DECA checkout::

    python tools/patches/apply_deca_stable_bbox.py /path/to/DECA
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path


MARKER = '# HRAVATAR_STABLE_BBOX'


# All payloads below are written *without* leading indentation. The patcher
# computes the per-anchor line indent at insertion time and re-indents each
# non-blank line of the payload to match. This makes the script tolerant of
# code-style differences (tabs vs 4 spaces vs 2 spaces) across forks.

# --- Edit 1: TestData.__init__ signature gains an optional kwarg ----------

DATASETS_INIT_OLD = (
    "def __init__(self, testpath, iscrop=True, crop_size=224, scale=1.25, "
    "face_detector='fan', sample_step=10):"
)
DATASETS_INIT_NEW = (
    "def __init__(self, testpath, iscrop=True, crop_size=224, scale=1.25, "
    "face_detector='fan', sample_step=10, precomputed_tforms_path=None):"
)


# --- Edit 2: __init__ body — load the npz once and stash a basename->tform map.
#
# Anchor: `self.iscrop = iscrop`. This is at method-body indent in standard
# yfeng95/DECA TestData, and the HRAvatar fork preserves it. We insert our
# initialisation right *before* this line, so `self._precomputed_tforms` is
# defined as long as __init__ reaches that point (the FAN detector setup
# above us must have completed without raising).

DATASETS_INIT_ANCHOR = "self.iscrop = iscrop"
DATASETS_INIT_PAYLOAD = (
    f'{MARKER}_INIT BEGIN — load smoothed (basename -> tform) map.\n'
    '# When `precomputed_tforms_path` points at a `.npz` written by\n'
    "# HRAvatar's `preprocess/stable_bbox.py`, we use those tforms instead\n"
    '# of running FAN per frame. Removes the mouth/blink leak that\n'
    '# otherwise propagates into bbox `size` -> `cam[s]` -> renderer.\n'
    '# Falls back to FAN per-frame for any frame whose basename is\n'
    '# missing from the npz, so partial coverage is safe.\n'
    'self._precomputed_tforms = None\n'
    'if precomputed_tforms_path is not None and os.path.isfile(precomputed_tforms_path):\n'
    '    _hravatar_npz = np.load(precomputed_tforms_path, allow_pickle=False)\n'
    "    _hravatar_names = [str(_b) for _b in _hravatar_npz['frame_basenames']]\n"
    "    _hravatar_tforms = _hravatar_npz['tform']\n"
    '    self._precomputed_tforms = {\n'
    '        _name: _hravatar_tforms[_i]\n'
    '        for _i, _name in enumerate(_hravatar_names)\n'
    '    }\n'
    "    print(f'[deca/TestData] using precomputed stable bbox '\n"
    "          f'({len(self._precomputed_tforms)} frames) from '\n"
    "          f'{precomputed_tforms_path}')\n"
    f'{MARKER}_INIT END\n'
)


# --- Edit 3: __getitem__ — short-circuit FAN when a precomputed tform exists.
#
# Anchor: the comment line that starts the FAN-detection block in the
# `if self.iscrop:` branch. yfeng95/DECA has it as
# `# provide kpt as txt file, or mat file (for AFLW2000)` and the HRAvatar
# fork preserves it. We insert our short-circuit immediately *before* that
# comment so the FAN path is bypassed when a precomputed tform is available.

DATASETS_GETITEM_ANCHOR = "# provide kpt as txt file, or mat file (for AFLW2000)"
DATASETS_GETITEM_PAYLOAD = (
    f'{MARKER}_GETITEM BEGIN — short-circuit FAN with precomputed tform.\n'
    "# If a smoothed similarity transform exists for this frame's basename,\n"
    '# skip detection and warp directly. Downstream `code.json` `tform`\n'
    '# then reflects the stabilised crop, so `optimize.py` and HRAvatar\n'
    '# training all see the same (center, size) sequence frame-to-frame.\n'
    '_hravatar_basename = os.path.basename(imagepath)\n'
    'if (self._precomputed_tforms is not None\n'
    '        and _hravatar_basename in self._precomputed_tforms):\n'
    '    _hravatar_params = self._precomputed_tforms[_hravatar_basename]\n'
    "    tform = estimate_transform('similarity',\n"
    '                               np.eye(3)[:2, :2],\n'
    '                               np.eye(3)[:2, :2])\n'
    '    tform.params[:] = _hravatar_params\n'
    '    dst_image = warp(image, tform.inverse,\n'
    '                     output_shape=(self.crop_size, self.crop_size))\n'
    '    dst_image = dst_image.transpose(2, 0, 1)\n'
    "    return {'image': torch.tensor(dst_image).float(),\n"
    "            'imagename': os.path.splitext(_hravatar_basename)[0],\n"
    "            'tform': torch.tensor(tform.params).float(),\n"
    "            'original_image': torch.tensor(image.transpose(2, 0, 1)).float()}\n"
    f'{MARKER}_GETITEM END\n'
)


# --- Edit 4: demo_reconstruct.py — TestData call gains the new kwarg ------

DEMO_TESTDATA_OLD = (
    "datasets.TestData(args.inputpath, iscrop=args.iscrop, "
    "face_detector=args.detector, sample_step=args.sample_step)"
)
DEMO_TESTDATA_NEW = (
    "datasets.TestData(args.inputpath, iscrop=args.iscrop, "
    "face_detector=args.detector, sample_step=args.sample_step, "
    "precomputed_tforms_path=args.precomputed_bbox)"
)


# --- Edit 5: demo_reconstruct.py — argparse gains --precomputed-bbox ------
#
# Anchor: `main(parser.parse_args())`. We use `insert_before` so the new
# `parser.add_argument(...)` block lands inside the `if __name__ == '__main__':`
# block right before the call to `main`.

DEMO_ARGPARSE_ANCHOR = "main(parser.parse_args())"
DEMO_ARGPARSE_PAYLOAD = (
    f'{MARKER}_ARG BEGIN — accept the HRAvatar stable_bbox npz path.\n'
    'parser.add_argument(\n'
    "    '--precomputed-bbox', type=str, default=None,\n"
    "    help=('Path to HRAvatar `stable_bbox.npz`. When supplied, the '\n"
    "          'smoothed similarity transform per frame replaces FAN '\n"
    "          'per-frame face detection, so DECA crops match the bbox '\n"
    "          'sequence consumed by `scene/data_loader.py` at training.'))\n"
    f'{MARKER}_ARG END\n'
)


# --- Driver --------------------------------------------------------------

def _line_indent_at(text: str, idx: int) -> str:
    """Return the leading-whitespace prefix of the line containing ``idx``.

    Pure whitespace only — never returns substring before the anchor when
    the anchor sits mid-line. This is what callers want when they need to
    re-indent an inserted block to match the surrounding code.
    """
    line_start = text.rfind('\n', 0, idx) + 1
    indent = []
    i = line_start
    while i < len(text) and text[i] in (' ', '\t'):
        indent.append(text[i])
        i += 1
    return ''.join(indent)


def _reindent(payload: str, indent: str) -> str:
    """Prefix every non-blank line of ``payload`` with ``indent``. Trailing
    newline is preserved; the result always ends with a single newline so
    inserts splice cleanly between existing lines."""
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


def _patch_file(path: Path, edits: list[tuple[str, str, str]]) -> bool:
    """Apply ``[(op, find, payload_or_replacement), ...]`` to ``path``.

    Operations:

      * ``replace`` — `find` must occur exactly once and is replaced verbatim.
      * ``insert_after`` — payload is reindented to the line indent of the
        line containing `find`, then inserted after that line.
      * ``insert_before`` — payload is reindented to the line indent of the
        line containing `find`, then inserted on a new line immediately
        before that line.

    Idempotent at the file level — if the marker is already present
    anywhere in the file, the edits are skipped wholesale.

    Returns True if the file was modified, False if it was already patched.
    """
    text = path.read_text()
    if MARKER in text:
        print(f'[skip] {path} already patched ({MARKER} present)')
        return False

    for op, find, payload in edits:
        if find not in text:
            raise SystemExit(
                f'[error] anchor missing in {path}:\n'
                f'        could not find: {find!r}\n'
                f'        DECA fork may have drifted; merge by hand or '
                f'rebase the fork onto a known revision.')

        if op == 'replace':
            occurrences = text.count(find)
            if occurrences != 1:
                raise SystemExit(
                    f'[error] anchor not unique in {path}: {find!r} '
                    f'occurs {occurrences} times.')
            text = text.replace(find, payload, 1)
            continue

        idx = text.index(find)
        indent = _line_indent_at(text, idx)
        payload_indented = _reindent(payload, indent)
        if op == 'insert_after':
            line_end = text.index('\n', idx) + 1
            text = text[:line_end] + payload_indented + text[line_end:]
        elif op == 'insert_before':
            line_start = text.rfind('\n', 0, idx) + 1
            text = text[:line_start] + payload_indented + text[line_start:]
        else:
            raise AssertionError(f'unknown op: {op!r}')

    path.write_text(text)
    print(f'[ok]   {path}')
    return True


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description='Apply the HRAvatar stable_bbox integration to a DECA '
                    'checkout (idempotent).')
    default_deca = (Path(__file__).resolve().parent.parent.parent
                    / 'preprocess' / 'submodules' / 'DECA')
    parser.add_argument(
        'deca_root', nargs='?', default=str(default_deca),
        help=f'Path to the DECA checkout (default: {default_deca}).')
    args = parser.parse_args(argv)

    deca_root = Path(args.deca_root).resolve()
    datasets_py = deca_root / 'decalib' / 'datasets' / 'datasets.py'
    demo_py = deca_root / 'demos' / 'demo_reconstruct.py'
    for p in (datasets_py, demo_py):
        if not p.is_file():
            raise SystemExit(
                f'[error] expected DECA file not found: {p}\n'
                f'        is {deca_root} the DECA root?')

    any_changed = False
    any_changed |= _patch_file(datasets_py, [
        ('replace', DATASETS_INIT_OLD, DATASETS_INIT_NEW),
        ('insert_before', DATASETS_INIT_ANCHOR, DATASETS_INIT_PAYLOAD),
        ('insert_before', DATASETS_GETITEM_ANCHOR, DATASETS_GETITEM_PAYLOAD),
    ])
    any_changed |= _patch_file(demo_py, [
        ('replace', DEMO_TESTDATA_OLD, DEMO_TESTDATA_NEW),
        ('insert_before', DEMO_ARGPARSE_ANCHOR, DEMO_ARGPARSE_PAYLOAD),
    ])

    if not any_changed:
        print('[done] DECA already carries the stable_bbox integration.')
    else:
        print('[done] DECA patched. Verify with:')
        print(f'         grep -n HRAVATAR_STABLE_BBOX '
              f'{datasets_py} {demo_py}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
