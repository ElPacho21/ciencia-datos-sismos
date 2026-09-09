"""Capa de entrega: el CSV que sale del pipeline, y su chequeo de calidad.

Las capas anteriores escriben parquet porque están tipadas y se leen entre
tareas. Ésta escribe **csv** porque el entregable se abre a mano, se mira en
una planilla y se defiende delante de alguien: parquet no se abre con doble
clic.

Se publican dos tablas, que son las dos unidades de análisis posibles del
proyecto y conviene tener las dos a mano:

- `clusters_*.csv` — **el dataset principal**. Una fila es *una secuencia
  sísmica*: un sismo principal con todas sus réplicas. Clave `cluster_id`,
  columna objetivo `n_aftershocks`.
- `events_*.csv` — el catálogo evento por evento, con a qué cluster cayó cada
  sismo. Una fila es *un terremoto*. Clave `id`, columna objetivo
  `is_aftershock`.

El chequeo no es decorativo: si la clave repite o una columna quedó entera en
nulo, la tarea **falla**. Publicar un dataset roto es peor que no publicar
nada, porque el error se descubre recién cuando alguien construye algo encima.
Es la misma idea que una tarea de validación en cualquier pipeline de
producción: el último paso antes de exponer el dato es dudar de él.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from sismos import OUTPUT_DIR, partition

log = logging.getLogger(__name__)

DELIVERY_DIR = OUTPUT_DIR / "delivery"

# Los pisos de la consigna. Están acá y no repartidos por el código para que se
# vea de un vistazo contra qué se está midiendo la salida.
MIN_ROWS = 1000
MIN_COLUMNS = 5


def dataset_path(seed, mainshock, **query) -> Path:
    """El entregable: una fila por cluster.

    `mainshock` entra en el nombre igual que la semilla: es una decisión de
    método que cambia los números, y las dos versiones se quieren lado a lado
    para poder compararlas, no una encima de la otra.
    """
    return DELIVERY_DIR / (
        f"clusters_{partition(**query)}_mainshock={mainshock}_seed={seed}.csv"
    )


def events_path(seed, mainshock, **query) -> Path:
    """El catálogo evento por evento, con su cluster."""
    return DELIVERY_DIR / (
        f"events_{partition(**query)}_mainshock={mainshock}_seed={seed}.csv"
    )


def write_csv(destination: Path, df: pd.DataFrame) -> None:
    """CSV plano, sin índice y con las fechas en ISO 8601.

    `date_format` explícito porque, sin él, pandas serializa los timestamps con
    el `repr` de cada valor y una columna con nulos termina mezclando formatos.
    """
    destination.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(destination, index=False, date_format="%Y-%m-%dT%H:%M:%S%z")


def check(df: pd.DataFrame, key: str, name: str) -> tuple[dict, list[str]]:
    """Los criterios medibles de la consigna, sobre una tabla.

    Devuelve `(report, problems)`. El informe lleva los números aunque estén
    bien —son los que hay que saber decir en la defensa, no sólo el veredicto—
    y `problems` sólo lo que viola un piso.

    Los nulos **no** son un problema: un dataset real los tiene y lo que se
    pide es saber cuáles y por qué. Por eso se reportan, no se penalizan. Lo
    que sí se penaliza es una columna entera en nulo, que no es un nulo sino
    una columna que no se calculó.
    """
    problems = []

    if key not in df.columns:
        raise ValueError(f"{name}: the key column {key!r} does not exist.")

    duplicates = int(df[key].duplicated().sum())
    if duplicates:
        problems.append(
            f"the key {key!r} repeats in {duplicates} rows: either the unit of "
            f"analysis is ill-defined or the pipeline is duplicating rows"
        )

    if len(df) < MIN_ROWS:
        problems.append(
            f"{len(df)} rows, fewer than the {MIN_ROWS} required: widen the "
            f"date window or lower minmagnitude"
        )

    if df.shape[1] < MIN_COLUMNS:
        problems.append(f"{df.shape[1]} columns, fewer than the {MIN_COLUMNS} required")

    empty = [c for c in df.columns if df[c].isna().all()]
    if empty:
        problems.append(f"entirely null columns: {empty}")

    nulls = df.isna().mean().sort_values(ascending=False)
    with_nulls = nulls[nulls > 0]

    report = {
        "table": name,
        "key": key,
        "key_without_duplicates": duplicates == 0,
        "rows": int(len(df)),
        "columns": int(df.shape[1]),
        "types": {str(t): int(n) for t, n in df.dtypes.value_counts().items()},
        "columns_with_nulls": {c: round(float(p), 4) for c, p in with_nulls.items()},
        "empty_columns": empty,
    }

    log.info(
        "%s: %d rows x %d columns | key %r %s | types %s | %d columns with "
        "nulls (the worst, %s)",
        name,
        report["rows"],
        report["columns"],
        key,
        "unique" if duplicates == 0 else f"REPEATS in {duplicates}",
        report["types"],
        len(with_nulls),
        f"{with_nulls.index[0]} at {100 * with_nulls.iloc[0]:.1f}%"
        if len(with_nulls)
        else "none",
    )

    return report, problems


def publish(events: pd.DataFrame, summary: pd.DataFrame, destinations: dict) -> dict:
    """Escribe los dos csv y los valida. Si alguno no pasa, levanta.

    Se escribe **antes** de validar a propósito: si el chequeo falla, querés
    poder abrir el archivo y ver por qué falló, no quedarte sin nada.
    """
    write_csv(Path(destinations["dataset"]), summary)
    write_csv(Path(destinations["events"]), events)

    report_clusters, problems_clusters = check(summary, "cluster_id", "clusters")
    report_events, problems_events = check(events, "id", "events")

    problems = [f"clusters: {p}" for p in problems_clusters]
    problems += [f"events: {p}" for p in problems_events]

    if problems:
        raise ValueError(
            "The published dataset does not meet the quality criteria:\n  - "
            + "\n  - ".join(problems)
        )

    return {"clusters": report_clusters, "events": report_events}
