"""Compara un csv recién generado contra el de referencia.

Uso:
    python scripts/check_reference.py REFERENCIA.csv NUEVO.csv [--tol 1e-9]

La regla del proyecto es que con los parámetros por defecto la corrida de
referencia no cambia: las columnas que ya existían tienen que estar, en el mismo
orden y con los mismos valores. Las columnas nuevas sólo pueden ir **al final**.
Sale con código 1 si algo no coincide.
"""

from __future__ import annotations

import argparse
import sys

import numpy as np
import pandas as pd


def compare(reference: pd.DataFrame, new: pd.DataFrame, tol: float) -> list[str]:
    problems = []

    if len(reference) != len(new):
        problems.append(f"rows: reference {len(reference)} vs new {len(new)}")
        return problems

    prefix = list(new.columns[: len(reference.columns)])
    if prefix != list(reference.columns):
        missing = [c for c in reference.columns if c not in new.columns]
        problems.append(
            f"the reference columns are not a prefix of the new ones "
            f"(missing: {missing}; new prefix: {prefix})"
        )
        return problems

    for column in reference.columns:
        a, b = reference[column], new[column]
        if pd.api.types.is_float_dtype(a) and pd.api.types.is_float_dtype(b):
            same = np.allclose(a, b, rtol=0, atol=tol, equal_nan=True)
        else:
            same = (a.fillna("<NA>").astype(str) == b.fillna("<NA>").astype(str)).all()
        if not same:
            problems.append(f"column {column!r} differs")

    return problems


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reference")
    parser.add_argument("new")
    parser.add_argument("--tol", type=float, default=1e-9)
    args = parser.parse_args()

    reference = pd.read_csv(args.reference)
    new = pd.read_csv(args.new)
    problems = compare(reference, new, args.tol)

    extra = [c for c in new.columns if c not in reference.columns]
    print(f"reference: {reference.shape} | new: {new.shape} | new columns: {extra}")

    if problems:
        print("DIFFERENCES:")
        for p in problems:
            print("  -", p)
        return 1

    print("OK: the reference run is intact.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
