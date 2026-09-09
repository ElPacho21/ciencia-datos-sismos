"""Armado de los clusters y conteo de réplicas por sismo principal.

Después del thinning cada evento tiene un `parent_id` y un `is_aftershock`.
Quedarse con los enlaces aceptados deja un **bosque**: cada evento independiente
es la raíz de un árbol y las réplicas cuelgan de él, posiblemente en cadena.

Esa cadena es la razón de ser de este paso. Agrupar por `parent_id` cuenta sólo
los hijos directos, y las réplicas también tienen réplicas:

    A (independiente)
    ├── B  (réplica de A)
    │   └── D  (réplica de B, pero también del cluster de A)
    └── C  (réplica de A)

`groupby("parent_id")` diría "A tiene 2, B tiene 1". Lo cierto es que el
cluster de A tiene tres réplicas y B no es sismo principal de nada. Con el
decaimiento de Omori las cadenas largas son la norma, no la excepción.

Recorrer el bosque sale sorprendentemente barato gracias al contrato de silver:
como el catálogo está ordenado por tiempo y todo padre es anterior a su hijo,
una sola pasada hacia adelante alcanza. Cuando se llega a un evento, el cluster
de su padre ya está resuelto. Sin recursión, sin union-find, en O(N).
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from sismos import OUTPUT_DIR, particion
from sismos.geo import distancias_a_punto

log = logging.getLogger(__name__)

CLUSTERS_DIR = OUTPUT_DIR / "clusters"

# Rango en el que cae el exponente de productividad de Utsu en casi todo el
# mundo. Fuera de acá el problema no está en este paso sino aguas arriba.
ALPHA_PLAUSIBLE = (0.5, 1.5)

# La zona de réplicas escala con el largo de la ruptura: incluso un M9 queda
# por debajo de los mil kilómetros. Un cluster mucho más extendido que esto no
# es una secuencia, es el árbol encadenando generación tras generación
# sismicidad de regiones que no tienen nada que ver entre sí.
EXTENSION_MAXIMA_KM = 1500.0


def eventos_path(seed, mainshock, **consulta) -> Path:
    """El catálogo evento por evento, con a qué cluster pertenece cada uno."""
    return CLUSTERS_DIR / (
        f"eventos_{particion(**consulta)}"
        f"_mainshock={mainshock}_seed={seed}.parquet"
    )


def resumen_path(seed, mainshock, **consulta) -> Path:
    """Una fila por cluster: el entregable del pipeline.

    `mainshock` va en el nombre por el mismo motivo que la semilla: cambia los
    números de la salida, así que dos corridas que difieren en él no pueden
    compartir archivo. Y acá no sería sólo pisarse: esta tarea reutiliza lo que
    ya está en disco si es más nuevo que su fuente, de modo que sin el nombre
    distinto la segunda corrida devolvería los clusters de la primera.
    """
    return CLUSTERS_DIR / (
        f"resumen_{particion(**consulta)}"
        f"_mainshock={mainshock}_seed={seed}.parquet"
    )


def clusters_write(destino: Path, df: pd.DataFrame) -> None:
    destino.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(destino, index=False)


def clusters_read(ruta: Path) -> pd.DataFrame:
    return pd.read_parquet(ruta)


def _asignar_clusters(eventos: pd.DataFrame):
    """Una pasada cronológica: cada evento hereda el cluster de su padre."""
    ids = eventos["id"].to_numpy()
    padres = eventos["parent_id"].to_numpy()
    sin_padre = eventos["parent_id"].isna().to_numpy()
    es_replica = eventos["is_aftershock"].to_numpy(dtype=bool)

    posicion = {evento: i for i, evento in enumerate(ids)}

    cluster = np.empty(len(eventos), dtype=object)
    generacion = np.zeros(len(eventos), dtype=int)

    for i in range(len(eventos)):
        if sin_padre[i] or not es_replica[i]:
            # Enlace rechazado por el thinning (o evento sin vecino): es raíz.
            cluster[i] = ids[i]
            continue

        j = posicion[padres[i]]
        if j >= i:
            raise ValueError(
                f"El padre de {ids[i]!r} no es anterior a él. El catálogo tiene "
                "que venir ordenado por tiempo para poder recorrerlo de una pasada."
            )

        cluster[i] = cluster[j]
        generacion[i] = generacion[j] + 1

    return cluster, generacion


def build_clusters(clasificados: pd.DataFrame, definicion_mainshock: str = "mayor"):
    """Arma los clusters y devuelve (eventos, resumen).

    `definicion_mainshock` es una decisión de método, no un detalle:

    - `"mayor"`: el evento de mayor magnitud del cluster, que es lo que usan
      Zaliapin & Ben-Zion.
    - `"raiz"`: el primero del cluster, el que disparó el árbol.

    Difieren cuando la secuencia arranca con un premonitor: un M4.5 abre el
    árbol y tres horas después llega el M7. La raíz es el M4.5, pero el sismo
    principal es el otro. El resumen guarda las dos: `cluster_id` **es** el id
    de la raíz —así se arma en `_asignar_clusters`— y `mainshock_id` el del
    principal, de modo que `raiz_es_mainshock` dice en cuántos discrepan sin
    repetir una columna.
    """
    if definicion_mainshock not in ("mayor", "raiz"):
        raise ValueError(
            f"Definición de sismo principal desconocida: {definicion_mainshock!r}. "
            "Usá 'mayor' o 'raiz'."
        )

    eventos = clasificados.sort_values("time").reset_index(drop=True)
    cluster, generacion = _asignar_clusters(eventos)

    eventos["cluster_id"] = cluster
    eventos["generacion"] = generacion
    eventos["orden_en_cluster"] = eventos.groupby("cluster_id").cumcount()

    if definicion_mainshock == "mayor":
        # idxmax se queda con el primero ante empates, o sea el más temprano.
        principal = eventos.groupby("cluster_id")["mag"].idxmax()
    else:
        raices = eventos.index[eventos["generacion"] == 0]
        principal = pd.Series(raices, index=eventos.loc[raices, "cluster_id"])

    eventos["is_mainshock"] = eventos.index.isin(principal.to_numpy())

    lat_rad = np.radians(eventos["latitude"].to_numpy(dtype=float))
    lon_rad = np.radians(eventos["longitude"].to_numpy(dtype=float))

    hay_place = "place" in eventos.columns

    filas = []
    for cluster_id, grupo in eventos.groupby("cluster_id", sort=False):
        k = int(principal[cluster_id])
        miembros = grupo.index.to_numpy()

        momento = eventos["time"].iloc[k]
        premonitores = int((grupo["time"] < momento).sum())

        distancias = distancias_a_punto(
            lat_rad[miembros], lon_rad[miembros], lat_rad[k], lon_rad[k]
        )

        # La raíz sólo se usa para saber si coincide con el sismo principal: su
        # id no se publica porque `cluster_id` ya *es* el id de la raíz.
        raiz = grupo.loc[grupo["generacion"] == 0, "id"].iloc[0]

        fila = {
            "cluster_id": cluster_id,
            "mainshock_id": eventos["id"].iloc[k],
        }

        # Lo primero que mira una persona al abrir el csv. Va acá arriba, al lado
        # de la clave, y no al final entre las métricas.
        if hay_place:
            fila["mainshock_place"] = eventos["place"].iloc[k]

        fila.update(
            {
                "mainshock_mag": float(eventos["mag"].iloc[k]),
                "mainshock_time": momento,
                "mainshock_lat": float(eventos["latitude"].iloc[k]),
                "mainshock_lon": float(eventos["longitude"].iloc[k]),
                # Predictor legítimo y hasta ahora ausente: la sismicidad
                # superficial andina y la del slab profundo son poblaciones
                # distintas, y producen réplicas de forma distinta.
                "mainshock_depth": float(eventos["depth"].iloc[k]),
                "n_eventos": int(len(grupo)),
                # Todo lo que no es el sismo principal ni le antecede.
                "n_replicas": int(len(grupo) - 1 - premonitores),
                "n_premonitores": premonitores,
                "duracion_dias": float(
                    (grupo["time"].max() - grupo["time"].min()).total_seconds() / 86400
                ),
                "extension_km": float(distancias.max()),
                "generacion_max": int(grupo["generacion"].max()),
                "raiz_es_mainshock": bool(raiz == eventos["id"].iloc[k]),
            }
        )
        filas.append(fila)

    resumen = (
        pd.DataFrame(filas)
        .sort_values("n_replicas", ascending=False)
        .reset_index(drop=True)
    )

    # El conteo de réplicas se calcula por cluster, pero la pregunta "¿cuántas
    # réplicas produjo *este* sismo?" se hace evento por evento. Sin estas dos
    # columnas hay que cruzar las dos tablas a mano por `cluster_id`, que es
    # justo el paso donde se cuelan los errores.
    replicas_por_cluster = resumen.set_index("cluster_id")["n_replicas"]

    # Las de la secuencia entera: igual para todos los miembros del cluster.
    eventos["n_replicas_secuencia"] = (
        eventos["cluster_id"].map(replicas_por_cluster).astype(int)
    )

    # Las que produjo este evento: sólo el sismo principal las "produce", así
    # que para las réplicas y los premonitores vale 0.
    eventos["n_replicas"] = np.where(
        eventos["is_mainshock"], eventos["n_replicas_secuencia"], 0
    ).astype(int)

    con_replicas = resumen[resumen["n_replicas"] > 0]
    discrepan = int((~resumen["raiz_es_mainshock"]).sum())

    log.info(
        "Clusters (%s): %d clusters sobre %d eventos | %d con al menos una "
        "réplica | el más grande tiene %d | %d clusters (%.1f%%) donde la raíz "
        "no es el sismo principal.",
        definicion_mainshock,
        len(resumen),
        len(eventos),
        len(con_replicas),
        int(resumen["n_replicas"].max()),
        discrepan,
        100 * discrepan / len(resumen),
    )

    desparramados = resumen[resumen["extension_km"] > EXTENSION_MAXIMA_KM]
    if len(desparramados):
        log.warning(
            "%d cluster(s) se extienden más de %.0f km (el mayor, %.0f km "
            "alrededor de %s, con %d generaciones). Ninguna secuencia de "
            "réplicas abarca eso: el árbol está encadenando regiones sin "
            "relación, que es lo que pasa cuando el catálogo no está acotado a "
            "una zona. Los conteos de esos clusters no son creíbles.",
            len(desparramados),
            EXTENSION_MAXIMA_KM,
            desparramados["extension_km"].max(),
            desparramados.loc[desparramados["extension_km"].idxmax(), "mainshock_id"],
            int(desparramados["generacion_max"].max()),
        )

    return eventos, resumen


def productividad(
    resumen: pd.DataFrame, ancho_bin: float = 0.5, min_mainshocks: int = 5
):
    """Ley de productividad de Utsu: el número medio de réplicas ~ 10^(alpha·M).

    Es el mejor control que tiene el pipeline, porque es lo único que se
    contrasta contra un hecho externo y no contra un sintético fabricado acá:
    `alpha` cae entre 0.8 y 1.0 en casi todos los catálogos del mundo. Si sale
    muy lejos, el problema está aguas arriba.

    Se ajusta sobre el **promedio por bin de magnitud incluyendo los clusters de
    cero réplicas**. Ajustar sólo sobre los que tuvieron réplicas sesgaría el
    resultado: los sismos chicos que no dispararon nada son justamente parte de
    lo que la ley predice.
    """
    magnitudes = resumen["mainshock_mag"].to_numpy(dtype=float)
    replicas = resumen["n_replicas"].to_numpy(dtype=float)

    bordes = np.arange(
        np.floor(magnitudes.min() / ancho_bin) * ancho_bin,
        np.ceil(magnitudes.max() / ancho_bin) * ancho_bin + ancho_bin / 2,
        ancho_bin,
    )
    indice = np.clip(np.digitize(magnitudes, bordes) - 1, 0, len(bordes) - 2)

    centros, medias, cuentas = [], [], []
    for k in range(len(bordes) - 1):
        seleccion = indice == k
        if seleccion.sum() < min_mainshocks:
            continue
        media = float(replicas[seleccion].mean())
        if media <= 0:
            continue
        centros.append(float(bordes[k] + ancho_bin / 2))
        medias.append(media)
        cuentas.append(int(seleccion.sum()))

    if len(centros) < 3:
        log.warning(
            "Sólo %d bins de magnitud con suficientes sismos principales: no "
            "alcanza para ajustar la productividad. Hace falta una ventana más "
            "larga.",
            len(centros),
        )
        return None

    x = np.array(centros)
    y = np.log10(medias)
    alpha, ordenada = np.polyfit(x, y, 1)
    residuos = y - (alpha * x + ordenada)
    varianza = ((y - y.mean()) ** 2).sum()
    r2 = float(1 - (residuos**2).sum() / varianza) if varianza > 0 else 0.0

    if not ALPHA_PLAUSIBLE[0] <= alpha <= ALPHA_PLAUSIBLE[1]:
        log.warning(
            "El exponente de productividad dio alpha=%.3f, fuera del rango "
            "habitual %s. La ley de Utsu no se está reproduciendo: revisá Mc, "
            "el umbral y el thinning antes de creerle a los conteos.",
            alpha,
            ALPHA_PLAUSIBLE,
        )
    else:
        log.info(
            "Productividad de Utsu: alpha=%.3f (R²=%.3f) sobre %d bins de "
            "magnitud — dentro del rango esperado.",
            alpha,
            r2,
            len(centros),
        )

    return {
        "alpha": float(alpha),
        "ordenada": float(ordenada),
        "r2": r2,
        "n_bins": len(centros),
        "bins": [
            {"magnitud": c, "replicas_promedio": m, "n_mainshocks": q}
            for c, m, q in zip(centros, medias, cuentas)
        ],
    }
