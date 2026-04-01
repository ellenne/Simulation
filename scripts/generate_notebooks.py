"""
Assemble submission notebooks from scripts/notebook_sources/*.nb.txt
Run from project root: python scripts/generate_notebooks.py

Cell delimiter:
  # %% [markdown]   -> markdown cell
  # %%               -> code cell
"""
from __future__ import annotations

import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "scripts" / "notebook_sources"
OUT_DIR = ROOT / "notebooks"


def parse_nb_source(text: str) -> list:
    lines = text.splitlines(keepends=True)
    cells: list = []
    current: list[str] = []
    mode = "code"

    def flush():
        nonlocal current
        if not current:
            return
        body = "".join(current).rstrip("\n") + "\n"
        if mode == "markdown":
            cells.append({"cell_type": "markdown", "metadata": {}, "source": body.splitlines(keepends=True)})
        else:
            cells.append(
                {
                    "cell_type": "code",
                    "metadata": {},
                    "execution_count": None,
                    "outputs": [],
                    "source": body.splitlines(keepends=True),
                }
            )
        current = []

    for line in lines:
        if re.match(r"^# %%\s*\[markdown\]\s*$", line.strip()):
            flush()
            mode = "markdown"
            continue
        if re.match(r"^# %%\s*$", line.strip()):
            flush()
            mode = "code"
            continue
        current.append(line)
    flush()
    return cells


def write_ipynb(stem: str, cells: list) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    nb = {
        "nbformat": 4,
        "nbformat_minor": 5,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.10.0"},
        },
        "cells": cells,
    }
    out = OUT_DIR / f"{stem}.ipynb"
    out.write_text(json.dumps(nb, indent=1), encoding="utf-8")
    print("Wrote", out)


def main() -> None:
    for stem in ["candidate_success", "complaint_nlp", "hotel_demand", "medical_risk"]:
        p = SRC / f"{stem}.nb.txt"
        if not p.exists():
            print("Missing", p)
            continue
        cells = parse_nb_source(p.read_text(encoding="utf-8"))
        write_ipynb(stem, cells)


if __name__ == "__main__":
    main()
