"""Refinado de la capa silver: el catálogo limpio y tipado.

Silver es sólo higiene de datos — parseo, descarte de filas inservibles,
deduplicado y orden cronológico. Todo lo que sea propio del método de Zaliapin
& Ben-Zion (magnitud de completitud, b, d, eta, thinning) queda para las capas
siguientes, así este archivo se reutiliza sin recalcular con cualquier
parametrización del algoritmo.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from sismos import OUTPUT_DIR, partition

log = logging.getLogger(__name__)

SILVER_DIR = OUTPUT_DIR / "silver"

# Si el csv no trae esto, no es un catálogo del FDSN y no tiene sentido seguir.
REQUIRED_COLUMNS = (
    "id",
    "time",
    "updated",
    "latitude",
    "longitude",
    "depth",
    "mag",
    "type",
)

NUMERIC_COLUMNS = (
    "latitude",
    "longitude",
    "depth",
    "mag",
    "horizontalError",
    "depthError",
    "magError",
)#TODO: Veo que falta algunas columnas como "magNst", "nst" - REVISAR

# Sin epicentro, tiempo o magnitud no se puede calcular eta: la fila no sirve.
# La profundidad puede faltar porque la distancia del método es epicentral.
NON_NULL_COLUMNS = ("id", "time", "latitude", "longitude", "mag")

# Lo que se lleva silver: lo que necesita el vecino más cercano, más lo
# mínimo para poder auditar después qué tan homogéneo es el catálogo.
SILVER_COLUMNS = (
    "id",
    "time",
    "latitude",
    "longitude",
    "depth",
    "mag",
    "magType",
    "net",
    "status",
    "place",
    "horizontalError",
    "depthError",
    "magError",
    "updated",
)


def silver_path(**query) -> Path:
    """Misma partición que bronze, pero en parquet.

    Parquet y no csv porque silver ya está tipado: el paso del vecino más
    cercano necesita `time` como datetime y las coordenadas como float, y no
    tiene por qué volver a parsear ni arriesgarse a que pandas infiera otra
    cosa.
    """
    return SILVER_DIR / f"{partition(**query)}.parquet"


def refine(raw: pd.DataFrame) -> pd.DataFrame:
    """Convierte el csv crudo de bronze en el catálogo limpio de silver."""
    missing = [c for c in REQUIRED_COLUMNS if c not in raw.columns]
    if missing:
        raise ValueError(f"The bronze csv is missing the columns {missing}.")

    df = raw.copy()

    df["time"] = pd.to_datetime(df["time"], format="ISO8601", utc=True, errors="coerce")
    df["updated"] = pd.to_datetime(
        df["updated"], format="ISO8601", utc=True, errors="coerce"
    )

    for column in NUMERIC_COLUMNS:
        if column in df.columns:
            df[column] = pd.to_numeric(df[column], errors="coerce")

    n_in = len(df)

    df = df.dropna(subset=list(NON_NULL_COLUMNS))
    df = df.sort_values("updated").drop_duplicates("id", keep="last")
    df = df.sort_values("time").reset_index(drop=True)

    log.info(
        "Silver: %d raw rows -> %d clean (%d discarded).",
        n_in,
        len(df),
        n_in - len(df),
    )

    if df.empty:
        raise ValueError(
            "No rows left after cleaning; check the query parameters."
        )

    return df[[c for c in SILVER_COLUMNS if c in df.columns]]


def silver_write(destination: Path, df: pd.DataFrame) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(destination, index=False)


def silver_read(path: Path) -> pd.DataFrame:
    return pd.read_parquet(path)
