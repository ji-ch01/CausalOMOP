"""Build an anonymized source archive for a double-blind workshop submission.

WHY THIS EXISTS
---------------
The paper appendix promises a supplementary code archive that carries "no
credentials, patient-level data, institutional paths, author identities, or
repository URLs that reveal authorship" and that still runs the
artifact-integrity test suite. Doing that by hand is error-prone. This script
produces the archive deterministically from a clean ``git archive`` export
and then *verifies* that no known identifying token survived.

WHAT IT DOES
------------
1. ``git archive`` the given ref (default ``HEAD``) into a staging tree --
   tracked files only, so ``.env``, ``*-key.json``, ``artifacts/runs/``,
   ``.venv/``, ``paper/`` and every other untracked/ignored path are absent
   by construction.
2. Rewrite the identifying tokens listed in ``SUBSTITUTIONS`` across every
   text file, and rename ``scripts/snuh_etl`` -> ``scripts/onprem_etl``
   (nothing imports it as a package; only doc/path references point at it).
3. Keep ``.gitmodules`` but anonymize its URLs -- the submodule *contents*
   are never in a ``git archive`` anyway, so the file then reveals nothing.
4. Walk the result and fail loudly if any token in ``DENYLIST`` still
   appears, printing every offending ``path:line``.
5. Write ``ANONYMIZATION.md`` (what changed + how to run the tests) at the
   archive root and emit ``dist/causalomop-submission-<ref>.tar.gz``.

WHAT IT IS NOT
--------------
Not a substitute for reading the archive. The submodule source
(``agno_fastomop``, ``omcp``, ``foem``) is intentionally excluded -- the test
suite does not import it (both wrappers shell out at runtime) -- so the
archive is a code-read + ``pytest`` artifact, not a clone-and-run-everything
bundle. Review the emitted file list and ``ANONYMIZATION.md`` before upload.

USAGE
-----
    python scripts/build_submission_archive.py
    python scripts/build_submission_archive.py --ref main --out dist/
    python scripts/build_submission_archive.py --keep-staging   # for inspection
"""

from __future__ import annotations

import argparse
import re
import shutil
import subprocess
import sys
import tarfile
import tempfile
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Ordered: apply the most specific pattern first so a later, broader rule
# never rewrites a token an earlier rule already anonymized.
SUBSTITUTIONS: list[tuple[str, str]] = [
    ("https://github.com/ji-ch01/", "https://github.com/ANON-OMOP/"),
    ("github.com/ji-ch01", "github.com/ANON-OMOP"),
    ("ji-ch01", "ANON-OMOP"),
    ("Seoul National University Hospital", "the on-premise partner hospital"),
    ("SYN-SNUH", "SYN-HOSP"),
    ("snuh_etl", "onprem_etl"),
    ("SNUH", "SYN-HOSP"),
    ("Snuh", "Synhosp"),
    ("snuh", "synhosp"),
]

# After substitution, none of these may appear anywhere in the tree.
DENYLIST: list[str] = [
    r"ji-ch01",
    r"jiwoo",
    r"yellowsmob",
    r"\bchoi\b",
    r"118937400",
    r"@users\.noreply\.github\.com",
    r"gmail\.com",
    r"[Ee]frei",
    r"(?i)snuh",
    r"(?i)seoul",
]

DIR_RENAMES: list[tuple[str, str]] = [
    ("scripts/snuh_etl", "scripts/onprem_etl"),
]

# Extensions treated as text for substitution and denylist scanning.
_TEXT_SUFFIXES = {
    ".py", ".md", ".txt", ".toml", ".ini", ".cfg", ".yml", ".yaml", ".json",
    ".sql", ".sh", ".env", ".tex", ".bib", ".gitignore", ".gitattributes",
    ".gitmodules", ".dockerfile", "",
}

_ANON_NOTE = """\
# Anonymized submission archive

This is an anonymized export of the CausalOMOP source for double-blind
review. It was produced by `scripts/build_submission_archive.py` from a
clean `git archive` of a single revision.

## What is included

- `causal/` -- the governed causal-study workflow, agents, and companion.
- `tests/` -- the artifact-integrity suite referenced in the paper.
- `docs/`, `studies/`, `scripts/`, requirements files, `README.md`.

## What was removed or changed

- No `.git/` history, no `.env`, no service-account keys, no
  `artifacts/runs/` or `artifacts/reviewers/` state, no local datasets, no
  virtual environment, no manuscript source -- none of these are tracked
  files, so `git archive` never included them.
- Author- and institution-identifying tokens (repository org, on-premise
  hospital name and its ETL directory) were replaced with neutral
  placeholders. The archive is scanned after rewriting to confirm none
  remain.
- Submodule source (`agno_fastomop/`, `omcp/`, `foem/`) is not included.
  The test suite does not import it: `causal.mcp_client.OMCPClient` and
  `causal.fastomop_bridge` both invoke those tools as out-of-process
  commands and are covered by stubs.

## Running the artifact-integrity tests

    python -m venv .venv && . .venv/bin/activate      # or .venv\\Scripts\\activate
    pip install -r requirements.txt \\
        -r requirements-bigquery.txt \\
        -r requirements-llm.txt \\
        -r requirements-ui.txt \\
        -r requirements-mcp.txt \\
        -r requirements-causal-discovery.txt \\
        -r requirements-knowledge-graph.txt \\
        -r requirements-estimation.txt
    python -m pytest tests -q

No network, database, or LLM credentials are required for the suite.
"""


def _run(*args: str) -> str:
    result = subprocess.run(
        args, cwd=_PROJECT_ROOT, check=True, capture_output=True, text=True, encoding="utf-8"
    )
    return result.stdout


def _export_tree(ref: str, dest: Path) -> None:
    """`git archive <ref>` -> extract into `dest` (tracked files only)."""
    dest.mkdir(parents=True, exist_ok=True)
    archive = dest.parent / "_export.tar"
    with archive.open("wb") as fh:
        subprocess.run(
            ["git", "archive", "--format=tar", ref],
            cwd=_PROJECT_ROOT,
            check=True,
            stdout=fh,
        )
    with tarfile.open(archive) as tf:
        tf.extractall(dest, filter="data")
    archive.unlink()


def _is_text(path: Path) -> bool:
    if path.suffix.lower() in _TEXT_SUFFIXES or path.name in _TEXT_SUFFIXES:
        return True
    try:
        path.read_text(encoding="utf-8")
    except (UnicodeDecodeError, OSError):
        return False
    return True


def _apply_substitutions(root: Path) -> int:
    changed = 0
    for path in sorted(root.rglob("*")):
        if not path.is_file() or not _is_text(path):
            continue
        original = path.read_text(encoding="utf-8")
        text = original
        for needle, replacement in SUBSTITUTIONS:
            text = text.replace(needle, replacement)
        if text != original:
            path.write_text(text, encoding="utf-8")
            changed += 1
    return changed


def _apply_dir_renames(root: Path) -> None:
    for src_rel, dst_rel in DIR_RENAMES:
        src, dst = root / src_rel, root / dst_rel
        if src.is_dir():
            dst.parent.mkdir(parents=True, exist_ok=True)
            src.rename(dst)


def _scan_denylist(root: Path) -> list[str]:
    patterns = [re.compile(p) for p in DENYLIST]
    hits: list[str] = []
    for path in sorted(root.rglob("*")):
        if not path.is_file() or not _is_text(path):
            continue
        rel = path.relative_to(root).as_posix()
        for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            for pattern in patterns:
                if pattern.search(line):
                    hits.append(f"{rel}:{lineno}: {line.strip()[:160]}")
    return hits


def _make_tarball(root: Path, out_path: Path, top_name: str) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(out_path, "w:gz") as tf:
        tf.add(root, arcname=top_name)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ref", default="HEAD", help="git ref to export (default HEAD)")
    parser.add_argument(
        "--out",
        type=Path,
        default=_PROJECT_ROOT / "dist",
        help="output directory for the tarball (default dist/)",
    )
    parser.add_argument(
        "--keep-staging",
        action="store_true",
        help="leave the unpacked staging tree next to the tarball for inspection",
    )
    args = parser.parse_args(argv)

    short = _run("git", "rev-parse", "--short", args.ref).strip()
    top_name = f"causalomop-submission-{short}"
    staging_parent = Path(tempfile.mkdtemp(prefix="causalomop-anon-"))
    staging = staging_parent / top_name

    try:
        print(f"[1/5] git archive {args.ref} ({short}) -> staging")
        _export_tree(args.ref, staging)

        print("[2/5] renaming institutional directories")
        _apply_dir_renames(staging)

        print("[3/5] rewriting identifying tokens")
        n_changed = _apply_substitutions(staging)
        print(f"      {n_changed} file(s) rewritten")

        (staging / "ANONYMIZATION.md").write_text(_ANON_NOTE, encoding="utf-8")

        print("[4/5] scanning for surviving identifiers")
        hits = _scan_denylist(staging)
        if hits:
            print(f"\nFAILED: {len(hits)} identifying token(s) survived:\n", file=sys.stderr)
            for hit in hits:
                print(f"  {hit}", file=sys.stderr)
            print(
                "\nAdd a rule to SUBSTITUTIONS (or fix the source) and re-run.",
                file=sys.stderr,
            )
            return 1
        print("      clean")

        out_path = args.out / f"{top_name}.tar.gz"
        print(f"[5/5] writing {out_path}")
        _make_tarball(staging, out_path, top_name)

        file_count = sum(1 for p in staging.rglob("*") if p.is_file())
        size_mb = out_path.stat().st_size / 1_048_576
        print(f"\nOK  {file_count} files, {size_mb:.2f} MB")
        print(f"    {out_path}")
        if args.keep_staging:
            kept = args.out / top_name
            if kept.exists():
                shutil.rmtree(kept)
            shutil.copytree(staging, kept)
            print(f"    staging kept at {kept}")
        return 0
    finally:
        shutil.rmtree(staging_parent, ignore_errors=True)


if __name__ == "__main__":
    raise SystemExit(main())
