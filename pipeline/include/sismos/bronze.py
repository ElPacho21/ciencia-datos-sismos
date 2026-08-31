"""Rutas y lectura/escritura de la capa bronze.

Vive fuera del DAG para que se pueda importar desde los notebooks y desde los
tests sin levantar Airflow.
"""
from __future__ import annotations

import gzip
from pathlib import Path

OUTPUT_DIR = Path("/usr/local/airflow/include/output/detector_replicas_api")

BRONZE_DIR = OUTPUT_DIR / "bronze"
PLATA_DIR = OUTPUT_DIR / "silver"


def bronze_path(starttime, endtime, minmagnitude) -> Path:
    """Particionado por los parámetros de la consulta, al estilo Hive.

    Así dos corridas con los mismos parámetros caen en el mismo archivo y se
    puede reutilizar lo ya descargado en vez de volver a pegarle a la API.
    """
    return (BRONZE_DIR / f"starttime={starttime}" / f"endtime={endtime}" / f"minmagnitude={minmagnitude}.csv")


def bronze_write(destino: Path, csv: str) -> None:
    with gzip.open(destino, "wt", encoding="utf-8") as f:
        f.write(csv)


def bronze_read(ruta: Path) -> str:
    with gzip.open(ruta, "rt", encoding="utf-8") as f:
        return f.read()
