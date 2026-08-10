#!/usr/bin/env python3
"""Flatten ffpopt packages back to single meaningful root modules."""

from __future__ import annotations

import shutil
from pathlib import Path

FF = Path("src/ffpopt")


def flatten(pkg: str, dest: str, *, extra_keep: list[str] | None = None) -> None:
    d = FF / pkg
    if not d.is_dir():
        print("skip", pkg)
        return
    # find largest .py that isn't __init__
    cands = [p for p in d.glob("*.py") if p.name != "__init__.py"]
    if extra_keep:
        # move extras to root first
        for name in extra_keep:
            src = d / name
            if src.is_file():
                target = FF / name
                if name == "mixins.py":
                    target = FF / "wavefront_mixins.py"
                text = src.read_text(encoding="utf-8")
                # rewrite internal imports if needed
                if name == "mixins.py":
                    pass
                target.write_text(text, encoding="utf-8", newline="\n")
                print("  moved", src, "->", target)
    cands = [p for p in d.glob("*.py") if p.name != "__init__.py"]
    cands = [p for p in cands if p.name not in (extra_keep or [])]
    cands.sort(key=lambda p: p.stat().st_size, reverse=True)
    impl = cands[0]
    text = impl.read_text(encoding="utf-8")
    if pkg.startswith("wavefront") and "wavefront.mixins" in text:
        text = text.replace(
            "from ffpopt.wavefront.mixins import",
            "from ffpopt.wavefront_mixins import",
        )
    (FF / dest).write_text(text, encoding="utf-8", newline="\n")
    shutil.rmtree(d)
    print(f"flattened {pkg}/ -> {dest}")


def main() -> None:
    flatten("geomopt", "GeomOpt.py")
    flatten("wavefront", "WaveFront.py", extra_keep=["mixins.py"])
    flatten("wavefront_nd", "WaveFrontND.py")
    flatten("workflows", "Workflows.py")
    flatten("dihedrals", "Dihedrals.py")
    print("done")


if __name__ == "__main__":
    main()
