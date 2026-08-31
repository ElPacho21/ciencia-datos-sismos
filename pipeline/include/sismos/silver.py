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

from sismos import OUTPUT_DIR, particion

log = logging.getLogger(__name__)

SILVER_DIR = OUTPUT_DIR / "silver"

# Si el csv no trae esto, no es un catálogo del FDSN y no tiene sentido seguir.
COLUMNAS_REQUERIDAS = (
    "id",
    "time",
    "updated",
    "latitude",
    "longitude",
    "depth",
    "mag",
    "type",
)

COLUMNAS_NUMERICAS = (
    "latitude",
    "longitude",
    "depth",
    "mag",
    "horizontalError",
    "depthError",
    "magError",
)

# Sin epicentro, tiempo o magnitud no se puede calcular eta: la fila no sirve.
# La profundidad puede faltar porque la distancia del método es epicentral.
COLUMNAS_SIN_NULOS = ("id", "time", "latitude", "longitude", "mag")

# Lo que se lleva silver: lo que necesita el vecino más cercano, más lo
# mínimo para poder auditar después qué tan homogéneo es el catálogo.
COLUMNAS_SILVER = (
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


def silver_path(starttime, endtime, minmagnitude) -> Path:
    """Misma partición que bronze, pero en parquet.

    Parquet y no csv porque silver ya está tipado: el paso del vecino más
    cercano necesita `time` como datetime y las coordenadas como float, y no
    tiene por qué volver a parsear ni arriesgarse a que pandas infiera otra
    cosa.
    """
    return SILVER_DIR / f"{particion(starttime, endtime, minmagnitude)}.parquet"


def refine(crudo: pd.DataFrame) -> pd.DataFrame:
    """Convierte el csv crudo de bronze en el catálogo limpio de silver."""
    faltantes = [c for c in COLUMNAS_REQUERIDAS if c not in crudo.columns]
    if faltantes:
        raise ValueError(f"El csv de bronze no trae las columnas {faltantes}.")

    df = crudo.copy()

    df["time"] = pd.to_datetime(df["time"], format="ISO8601", utc=True, errors="coerce")
    df["updated"] = pd.to_datetime(
        df["updated"], format="ISO8601", utc=True, errors="coerce"
    )

    for columna in COLUMNAS_NUMERICAS:
        if columna in df.columns:
            df[columna] = pd.to_numeric(df[columna], errors="coerce")

    entraron = len(df)

    # La API ya filtra por eventtype, pero el csv de bronze puede venir de una
    # corrida vieja hecha con otros parámetros.
    df = df[df["type"] == "earthquake"]

    df = df.dropna(subset=list(COLUMNAS_SIN_NULOS))

    # Un mismo evento puede repetirse si la red lo revisó: nos quedamos con la
    # revisión más nueva.
    df = df.sort_values("updated").drop_duplicates("id", keep="last")

    # El vecino más cercano recorre el catálogo hacia atrás en el tiempo, así
    # que el orden cronológico es parte del contrato de silver.
    df = df.sort_values("time").reset_index(drop=True)

    log.info(
        "Silver: %d filas crudas -> %d limpias (%d descartadas).",
        entraron,
        len(df),
        entraron - len(df),
    )

    if df.empty:
        raise ValueError(
            "No quedó ninguna fila después de limpiar; revisá los parámetros "
            "de la consulta."
        )

    return df[[c for c in COLUMNAS_SILVER if c in df.columns]]


def silver_write(destino: Path, df: pd.DataFrame) -> None:
    destino.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(destino, index=False)


def silver_read(ruta: Path) -> pd.DataFrame:
    return pd.read_parquet(ruta)
