"""List the license of every package installed in the current environment.

Reads only local package metadata (importlib.metadata); flags GPL/AGPL.
Exit code 1 if any GPL/AGPL package is found.
Usage: python tools/check_licenses.py [--markdown]
"""
from __future__ import annotations

import argparse
import re
import site
import sys
from importlib.metadata import distributions
from pathlib import Path

ALLOWED_HINT = re.compile(r"MIT|BSD|Apache|PSF|Python Software Foundation|ISC|Unlicense|HPND|MIT-CMU", re.I)
# word-boundary match so "LGPL" is reported separately from GPL/AGPL
FORBIDDEN = re.compile(r"(?<![L])\b(A?GPL)(?!\s*-?\s*compatible)|GNU (Affero )?General Public", re.I)
LGPL = re.compile(r"LGPL|Lesser General Public", re.I)
SKIP = {"pip", "setuptools", "wheel"}


def license_of(dist) -> str:
    """Best-effort license string from License-Expression, License, or classifiers."""
    md = dist.metadata
    expr = md.get("License-Expression")
    if expr:
        return expr.strip()
    classifiers = [c.split("::")[-1].strip() for c in md.get_all("Classifier") or []
                   if c.startswith("License ::")]
    lic = (md.get("License") or "").strip()
    if classifiers:
        return "; ".join(sorted(set(classifiers)))
    if lic and len(lic) < 80 and "\n" not in lic:
        return lic
    return (lic.splitlines()[0][:80] + " ...") if lic else "UNKNOWN"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--markdown", action="store_true", help="print a markdown table")
    args = parser.parse_args()

    # only the venv's own site-packages (ignore anything leaking in via PYTHONPATH)
    site_dirs = [Path(p) for p in site.getsitepackages()]
    rows = []
    for d in distributions(path=[str(p) for p in site_dirs]):
        name = d.metadata["Name"]
        if not name or name.lower() in SKIP:
            continue
        lic = license_of(d)
        if FORBIDDEN.search(lic) and not LGPL.search(lic) and not ALLOWED_HINT.search(lic):
            flag = "FORBIDDEN"
        elif LGPL.search(lic):
            flag = "LGPL-check"
        elif ALLOWED_HINT.search(lic):
            flag = "ok"
        else:
            flag = "REVIEW"
        rows.append((name, d.version, lic, flag))
    rows.sort(key=lambda r: r[0].lower())

    if args.markdown:
        print("| Package | Version | License | Status |\n|---|---|---|---|")
        for r in rows:
            print(f"| {r[0]} | {r[1]} | {r[2]} | {r[3]} |")
    else:
        w = max(len(r[0]) for r in rows)
        for r in rows:
            print(f"{r[3]:10s} {r[0]:<{w}} {r[1]:<14} {r[2]}")
    bad = [r for r in rows if r[3] == "FORBIDDEN"]
    review = [r for r in rows if r[3] in ("REVIEW", "LGPL-check")]
    print(f"\n{len(rows)} packages, {len(bad)} forbidden, {len(review)} need manual review", file=sys.stderr)
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
