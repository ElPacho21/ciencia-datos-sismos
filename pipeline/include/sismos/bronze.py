"""Rutas y lectura/escritura de la capa bronze.

Vive fuera del DAG para que se pueda importar desde los notebooks y desde los
tests sin levantar Airflow.
"""

from __future__ import annotations

import gzip
from pathlib import Path

import pandas as pd

from sismos import OUTPUT_DIR, particion

BRONZE_DIR = OUTPUT_DIR / "bronze"


def bronze_path(starttime, endtime, minmagnitude) -> Path:
    """Particionado por los parámetros de la consulta, al estilo Hive.

    Así dos corridas con los mismos parámetros caen en el mismo archivo y se
    puede reutilizar lo ya descargado en vez de volver a pegarle a la API.
    """
    return BRONZE_DIR / f"{particion(starttime, endtime, minmagnitude)}.csv.gz"


def bronze_write(destino: Path, csv) -> None:
    destino.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(destino, "wb") as f:
        f.write(csv)


def bronze_load(ruta: Path) -> pd.DataFrame:
    """Lee el csv crudo tal cual vino de la API, sin castear nada.

    Todo entra como texto a propósito: castear y descartar filas es
    responsabilidad de silver, así bronze sigue siendo un espejo de la fuente.
    """
    return pd.read_csv(ruta, dtype=str, encoding="utf-8")
