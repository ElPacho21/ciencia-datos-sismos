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

from sismos import OUTPUT_DIR, partition
from sismos.geo import distances_to_point

log = logging.getLogger(__name__)

CLUSTERS_DIR = OUTPUT_DIR / "clusters"

# Rango en el que cae el exponente de productividad de Utsu en casi todo el
# mundo. Fuera de acá el problema no está en este paso sino aguas arriba.
ALPHA_PLAUSIBLE = (0.5, 1.5)

# La zona de réplicas escala con el largo de la ruptura: incluso un M9 queda
# por debajo de los mil kilómetros. Un cluster mucho más extendido que esto no
# es una secuencia, es el árbol encadenando generación tras generación
# sismicidad de regiones que no tienen nada que ver entre sí.
MAX_EXTENT_KM = 1500.0


def events_path(seed, mainshock, **query) -> Path:
    """El catálogo evento por evento, con a qué cluster pertenece cada uno."""
    return CLUSTERS_DIR / (
        f"events_{partition(**query)}"
        f"_mainshock={mainshock}_seed={seed}.parquet"
    )


def summary_path(seed, mainshock, **query) -> Path:
    """Una fila por cluster: el entregable del pipeline.

    `mainshock` va en el nombre por el mismo motivo que la semilla: cambia los
    números de la salida, así que dos corridas que difieren en él no pueden
    compartir archivo. Y acá no sería sólo pisarse: esta tarea reutiliza lo que
    ya está en disco si es más nuevo que su fuente, de modo que sin el nombre
    distinto la segunda corrida devolvería los clusters de la primera.
    """
    return CLUSTERS_DIR / (
        f"summary_{partition(**query)}"
        f"_mainshock={mainshock}_seed={seed}.parquet"
    )


def clusters_write(destination: Path, df: pd.DataFrame) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(destination, index=False)


def clusters_read(path: Path) -> pd.DataFrame:
    return pd.read_parquet(path)


def _assign_clusters(events: pd.DataFrame):
    """Una pasada cronológica: cada evento hereda el cluster de su padre."""
    ids = events["id"].to_numpy()
    parents = events["parent_id"].to_numpy()
    no_parent = events["parent_id"].isna().to_numpy()
    is_after = events["is_aftershock"].to_numpy(dtype=bool)

    position = {event: i for i, event in enumerate(ids)}

    cluster = np.empty(len(events), dtype=object)
    generation = np.zeros(len(events), dtype=int)

    for i in range(len(events)):
        if no_parent[i] or not is_after[i]:
            # Enlace rechazado por el thinning (o evento sin vecino): es raíz.
            cluster[i] = ids[i]
            continue

        j = position[parents[i]]
        if j >= i:
            raise ValueError(
                f"The parent of {ids[i]!r} is not earlier than it. The catalog "
                "must be sorted by time to be traversed in a single pass."
            )

        cluster[i] = cluster[j]
        generation[i] = generation[j] + 1

    return cluster, generation


def build_clusters(classified: pd.DataFrame, mainshock_definition: str = "largest"):
    """Arma los clusters y devuelve (events, summary).

    `mainshock_definition` es una decisión de método, no un detalle:

    - `"largest"`: el evento de mayor magnitud del cluster, que es lo que usan
      Zaliapin & Ben-Zion.
    - `"root"`: el primero del cluster, el que disparó el árbol.

    Difieren cuando la secuencia arranca con un premonitor: un M4.5 abre el
    árbol y tres horas después llega el M7. La raíz es el M4.5, pero el sismo
    principal es el otro. El resumen guarda las dos: `cluster_id` **es** el id
    de la raíz —así se arma en `_assign_clusters`— y `mainshock_id` el del
    principal, de modo que `root_is_mainshock` dice en cuántos discrepan sin
    repetir una columna.
    """
    if mainshock_definition not in ("largest", "root"):
        raise ValueError(
            f"Unknown mainshock definition: {mainshock_definition!r}. "
            "Use 'largest' or 'root'."
        )

    events = classified.sort_values("time").reset_index(drop=True)
    cluster, generation = _assign_clusters(events)

    events["cluster_id"] = cluster
    events["generation"] = generation
    events["order_in_cluster"] = events.groupby("cluster_id").cumcount()

    if mainshock_definition == "largest":
        # idxmax se queda con el primero ante empates, o sea el más temprano.
        main_idx = events.groupby("cluster_id")["mag"].idxmax()
    else:
        roots = events.index[events["generation"] == 0]
        main_idx = pd.Series(roots, index=events.loc[roots, "cluster_id"])

    events["is_mainshock"] = events.index.isin(main_idx.to_numpy())

    lat_rad = np.radians(events["latitude"].to_numpy(dtype=float))
    lon_rad = np.radians(events["longitude"].to_numpy(dtype=float))

    has_place = "place" in events.columns

    rows = []
    for cluster_id, group in events.groupby("cluster_id", sort=False):
        k = int(main_idx[cluster_id])
        members = group.index.to_numpy()

        moment = events["time"].iloc[k]
        foreshocks = int((group["time"] < moment).sum())

        distances = distances_to_point(
            lat_rad[members], lon_rad[members], lat_rad[k], lon_rad[k]
        )

        # La raíz sólo se usa para saber si coincide con el sismo principal: su
        # id no se publica porque `cluster_id` ya *es* el id de la raíz.
        root = group.loc[group["generation"] == 0, "id"].iloc[0]

        row = {
            "cluster_id": cluster_id,
            "mainshock_id": events["id"].iloc[k],
        }

        # Lo primero que mira una persona al abrir el csv. Va acá arriba, al lado
        # de la clave, y no al final entre las métricas.
        if has_place:
            row["mainshock_place"] = events["place"].iloc[k]

        row.update(
            {
                "mainshock_mag": float(events["mag"].iloc[k]),
                "mainshock_time": moment,
                "mainshock_lat": float(events["latitude"].iloc[k]),
                "mainshock_lon": float(events["longitude"].iloc[k]),
                # Predictor legítimo y hasta ahora ausente: la sismicidad
                # superficial andina y la del slab profundo son poblaciones
                # distintas, y producen réplicas de forma distinta.
                "mainshock_depth": float(events["depth"].iloc[k]),
                "n_events": int(len(group)),
                # Todo lo que no es el sismo principal ni le antecede.
                "n_aftershocks": int(len(group) - 1 - foreshocks),
                "n_foreshocks": foreshocks,
                "duration_days": float(
                    (group["time"].max() - group["time"].min()).total_seconds() / 86400
                ),
                "extent_km": float(distances.max()),
                "max_generation": int(group["generation"].max()),
                "root_is_mainshock": bool(root == events["id"].iloc[k]),
            }
        )
        rows.append(row)

    summary = (
        pd.DataFrame(rows)
        .sort_values("n_aftershocks", ascending=False)
        .reset_index(drop=True)
    )

    # El conteo de réplicas se calcula por cluster, pero la pregunta "¿cuántas
    # réplicas produjo *este* sismo?" se hace evento por evento. Sin estas dos
    # columnas hay que cruzar las dos tablas a mano por `cluster_id`, que es
    # justo el paso donde se cuelan los errores.
    aftershocks_per_cluster = summary.set_index("cluster_id")["n_aftershocks"]

    # Las de la secuencia entera: igual para todos los miembros del cluster.
    events["n_aftershocks_sequence"] = (
        events["cluster_id"].map(aftershocks_per_cluster).astype(int)
    )

    # Las que produjo este evento: sólo el sismo principal las "produce", así
    # que para las réplicas y los premonitores vale 0.
    events["n_aftershocks"] = np.where(
        events["is_mainshock"], events["n_aftershocks_sequence"], 0
    ).astype(int)

    with_aftershocks = summary[summary["n_aftershocks"] > 0]
    differ = int((~summary["root_is_mainshock"]).sum())

    log.info(
        "Clusters (%s): %d clusters over %d events | %d with at least one "
        "aftershock | the largest has %d | %d clusters (%.1f%%) where the root "
        "is not the mainshock.",
        mainshock_definition,
        len(summary),
        len(events),
        len(with_aftershocks),
        int(summary["n_aftershocks"].max()),
        differ,
        100 * differ / len(summary),
    )

    scattered = summary[summary["extent_km"] > MAX_EXTENT_KM]
    if len(scattered):
        log.warning(
            "%d cluster(s) span more than %.0f km (the largest, %.0f km "
            "around %s, with %d generations). No aftershock sequence covers "
            "that: the tree is chaining unrelated regions, which is what "
            "happens when the catalog is not bounded to a zone. The counts of "
            "those clusters are not credible.",
            len(scattered),
            MAX_EXTENT_KM,
            scattered["extent_km"].max(),
            scattered.loc[scattered["extent_km"].idxmax(), "mainshock_id"],
            int(scattered["max_generation"].max()),
        )

    return events, summary


def productivity(
    summary: pd.DataFrame, bin_width: float = 0.5, min_mainshocks: int = 5
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
    magnitudes = summary["mainshock_mag"].to_numpy(dtype=float)
    aftershocks = summary["n_aftershocks"].to_numpy(dtype=float)

    edges = np.arange(
        np.floor(magnitudes.min() / bin_width) * bin_width,
        np.ceil(magnitudes.max() / bin_width) * bin_width + bin_width / 2,
        bin_width,
    )
    index = np.clip(np.digitize(magnitudes, edges) - 1, 0, len(edges) - 2)

    centers, means, counts = [], [], []
    for k in range(len(edges) - 1):
        selection = index == k
        if selection.sum() < min_mainshocks:
            continue
        mean = float(aftershocks[selection].mean())
        if mean <= 0:
            continue
        centers.append(float(edges[k] + bin_width / 2))
        means.append(mean)
        counts.append(int(selection.sum()))

    if len(centers) < 3:
        log.warning(
            "Only %d magnitude bins with enough mainshocks: not enough to fit "
            "productivity. A longer window is needed.",
            len(centers),
        )
        return None

    x = np.array(centers)
    y = np.log10(means)
    alpha, intercept = np.polyfit(x, y, 1)
    residuals = y - (alpha * x + intercept)
    variance = ((y - y.mean()) ** 2).sum()
    r2 = float(1 - (residuals**2).sum() / variance) if variance > 0 else 0.0

    if not ALPHA_PLAUSIBLE[0] <= alpha <= ALPHA_PLAUSIBLE[1]:
        log.warning(
            "The productivity exponent came out alpha=%.3f, outside the usual "
            "range %s. Utsu's law is not being reproduced: check Mc, the "
            "threshold and the thinning before trusting the counts.",
            alpha,
            ALPHA_PLAUSIBLE,
        )
    else:
        log.info(
            "Utsu productivity: alpha=%.3f (R²=%.3f) over %d magnitude bins — "
            "within the expected range.",
            alpha,
            r2,
            len(centers),
        )

    return {
        "alpha": float(alpha),
        "intercept": float(intercept),
        "r2": r2,
        "n_bins": len(centers),
        "bins": [
            {"magnitude": c, "mean_aftershocks": m, "n_mainshocks": q}
            for c, m, q in zip(centers, means, counts)
        ],
    }
