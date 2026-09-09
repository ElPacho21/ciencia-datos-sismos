"""Capa de entrega: el CSV que sale del pipeline, y su chequeo de calidad.

Las capas anteriores escriben parquet porque están tipadas y se leen entre
tareas. Ésta escribe **csv** porque el entregable se abre a mano, se mira en
una planilla y se defiende delante de alguien: parquet no se abre con doble
clic.

Se publican dos tablas, que son las dos unidades de análisis posibles del
proyecto y conviene tener las dos a mano:

- `clusters_*.csv` — **el dataset principal**. Una fila es *una secuencia
  sísmica*: un sismo principal con todas sus réplicas. Clave `cluster_id`,
  columna objetivo `n_replicas`.
- `eventos_*.csv` — el catálogo evento por evento, con a qué cluster cayó cada
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

from sismos import OUTPUT_DIR, particion

log = logging.getLogger(__name__)

ENTREGA_DIR = OUTPUT_DIR / "entrega"

# Los pisos de la consigna. Están acá y no repartidos por el código para que se
# vea de un vistazo contra qué se está midiendo la salida.
MIN_FILAS = 1000
MIN_COLUMNAS = 5


def dataset_path(seed, mainshock, **consulta) -> Path:
    """El entregable: una fila por cluster.

    `mainshock` entra en el nombre igual que la semilla: es una decisión de
    método que cambia los números, y las dos versiones se quieren lado a lado
    para poder compararlas, no una encima de la otra.
    """
    return ENTREGA_DIR / (
        f"clusters_{particion(**consulta)}_mainshock={mainshock}_seed={seed}.csv"
    )


def eventos_path(seed, mainshock, **consulta) -> Path:
    """El catálogo evento por evento, con su cluster."""
    return ENTREGA_DIR / (
        f"eventos_{particion(**consulta)}_mainshock={mainshock}_seed={seed}.csv"
    )


def escribir_csv(destino: Path, df: pd.DataFrame) -> None:
    """CSV plano, sin índice y con las fechas en ISO 8601.

    `date_format` explícito porque, sin él, pandas serializa los timestamps con
    el `repr` de cada valor y una columna con nulos termina mezclando formatos.
    """
    destino.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(destino, index=False, date_format="%Y-%m-%dT%H:%M:%S%z")


def chequear(df: pd.DataFrame, clave: str, nombre: str) -> tuple[dict, list[str]]:
    """Los criterios medibles de la consigna, sobre una tabla.

    Devuelve `(informe, problemas)`. El informe lleva los números aunque estén
    bien —son los que hay que saber decir en la defensa, no sólo el veredicto—
    y `problemas` sólo lo que viola un piso.

    Los nulos **no** son un problema: un dataset real los tiene y lo que se
    pide es saber cuáles y por qué. Por eso se reportan, no se penalizan. Lo
    que sí se penaliza es una columna entera en nulo, que no es un nulo sino
    una columna que no se calculó.
    """
    problemas = []

    if clave not in df.columns:
        raise ValueError(f"{nombre}: no existe la columna clave {clave!r}.")

    duplicados = int(df[clave].duplicated().sum())
    if duplicados:
        problemas.append(
            f"la clave {clave!r} repite en {duplicados} filas: o la unidad de "
            f"análisis está mal definida o el pipeline está duplicando filas"
        )

    if len(df) < MIN_FILAS:
        problemas.append(
            f"{len(df)} filas, menos de las {MIN_FILAS} pedidas: ampliá la "
            f"ventana de fechas o bajá minmagnitude"
        )

    if df.shape[1] < MIN_COLUMNAS:
        problemas.append(f"{df.shape[1]} columnas, menos de las {MIN_COLUMNAS} pedidas")

    vacias = [c for c in df.columns if df[c].isna().all()]
    if vacias:
        problemas.append(f"columnas enteras en nulo: {vacias}")

    nulos = df.isna().mean().sort_values(ascending=False)
    con_nulos = nulos[nulos > 0]

    informe = {
        "tabla": nombre,
        "clave": clave,
        "clave_sin_duplicados": duplicados == 0,
        "filas": int(len(df)),
        "columnas": int(df.shape[1]),
        "tipos": {str(t): int(n) for t, n in df.dtypes.value_counts().items()},
        "columnas_con_nulos": {c: round(float(p), 4) for c, p in con_nulos.items()},
        "columnas_vacias": vacias,
    }

    log.info(
        "%s: %d filas x %d columnas | clave %r %s | tipos %s | %d columnas con "
        "nulos (la que más, %s)",
        nombre,
        informe["filas"],
        informe["columnas"],
        clave,
        "única" if duplicados == 0 else f"REPITE en {duplicados}",
        informe["tipos"],
        len(con_nulos),
        f"{con_nulos.index[0]} al {100 * con_nulos.iloc[0]:.1f}%"
        if len(con_nulos)
        else "ninguna",
    )

    return informe, problemas


def publicar(eventos: pd.DataFrame, resumen: pd.DataFrame, destinos: dict) -> dict:
    """Escribe los dos csv y los valida. Si alguno no pasa, levanta.

    Se escribe **antes** de validar a propósito: si el chequeo falla, querés
    poder abrir el archivo y ver por qué falló, no quedarte sin nada.
    """
    escribir_csv(Path(destinos["dataset"]), resumen)
    escribir_csv(Path(destinos["eventos"]), eventos)

    informe_clusters, problemas_clusters = chequear(resumen, "cluster_id", "clusters")
    informe_eventos, problemas_eventos = chequear(eventos, "id", "eventos")

    problemas = [f"clusters: {p}" for p in problemas_clusters]
    problemas += [f"eventos: {p}" for p in problemas_eventos]

    if problemas:
        raise ValueError(
            "El dataset publicado no cumple los criterios de calidad:\n  - "
            + "\n  - ".join(problemas)
        )

    return {"clusters": informe_clusters, "eventos": informe_eventos}
