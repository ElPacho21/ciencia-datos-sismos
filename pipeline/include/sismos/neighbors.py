"""Vecino más cercano en tiempo-espacio-magnitud (Zaliapin & Ben-Zion).

A cada evento se le busca *un* padre entre todos los que ocurrieron antes: el
que minimiza la proximidad

    eta_ij = t_ij * (r_ij ** d) * 10 ** (-b * m_i)

donde `i` es el evento **anterior** (el candidato a padre) y `j` el posterior,
`t_ij` es el tiempo entre ambos en años y `r_ij` la distancia epicentral en
km. La magnitud que entra es la del padre, no la del hijo: un sismo grande
"alcanza" más lejos en el tiempo y en el espacio, así que eventos que le
siguen a distancias que serían enormes para un sismo chico igual le quedan
cerca en esta métrica.

Zaliapin & Ben-Zion factorizan eta en dos mitades independientes,

    T_ij = t_ij * 10 ** (-q * b * m_i)        (tiempo reescalado)
    R_ij = (r_ij ** d) * 10 ** (-p * b * m_i) (distancia reescalada)

con p + q = 1, de forma que eta = T * R. Es en el plano (log T, log R) donde
la población se ve partida en dos nubes — la de fondo y la de réplicas — y de
ahí sale después el umbral. Por eso se guardan las tres columnas y no sólo
eta.

Este paso no decide todavía qué es réplica: sólo tiende el bosque de padres.
El umbral y el thinning vienen después.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from sismos import OUTPUT_DIR, partition
from sismos.geo import (
    DISTANCE_CHUNK,
    EARTH_RADIUS_KM,
    distances_to_point,
    epicentral_distances,
)
from sismos.parameters import MAGNITUDE_BIN, trim_to_mc

log = logging.getLogger(__name__)

NEIGHBORS_DIR = OUTPUT_DIR / "neighbors"

SECONDS_PER_YEAR = 365.25 * 24 * 3600

# Dos eventos con el mismo epicentro darían r=0 y eta=0, y ese padre ganaría
# siempre. Se pone un piso del orden del error de localización.
R_MIN_KM = 0.01

# `place` no lo usa el método, pero es lo único legible por una persona que
# tiene el catálogo: sin él el entregable se identifica sólo por ids del USGS.
CARRIED_COLUMNS = (
    "id",
    "time",
    "latitude",
    "longitude",
    "depth",
    "mag",
    "place",
)


# Gardner-Knopoff: el radio de la zona de réplicas de un sismo de magnitud M es
# 10^(0.1238·M + 0.983) km. El modo "gk" lo multiplica por `r_max_k`.
GK_SLOPE = 0.1238
GK_INTERCEPT = 0.983


def neighbors_path(method="", **query) -> Path:
    return NEIGHBORS_DIR / f"{partition(**query)}{method}.parquet"


def neighbors_write(destination: Path, df: pd.DataFrame) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(destination, index=False)


def neighbors_read(path: Path) -> pd.DataFrame:
    return pd.read_parquet(path)


def gardner_knopoff_km(magnitude) -> np.ndarray:
    """Radio de la zona de réplicas de Gardner-Knopoff, en km."""
    return 10.0 ** (GK_SLOPE * np.asarray(magnitude, dtype=float) + GK_INTERCEPT)


def _parent_radii(magnitudes, r_max_km, r_max_mode, r_max_k):
    """Radio máximo (km) dentro del que cada evento puede ser padre.

    Devuelve `None` si no hay tope. En modo "fixed" es el mismo para todos; en
    "gk" depende de la magnitud del *padre*, igual que la métrica: un sismo
    grande reclama más lejos. Si además se da `r_max_km` en modo "gk", actúa
    como techo absoluto.
    """
    if r_max_mode not in ("fixed", "gk"):
        raise ValueError(f"Unknown r_max_mode: {r_max_mode!r}. Use 'fixed' or 'gk'.")
    if r_max_km is not None and r_max_km <= 0:
        raise ValueError(f"r_max_km must be positive, got {r_max_km}.")

    if r_max_mode == "gk":
        if r_max_k <= 0:
            raise ValueError(f"r_max_k must be positive, got {r_max_k}.")
        radii = r_max_k * gardner_knopoff_km(magnitudes)
        return radii if r_max_km is None else np.minimum(radii, r_max_km)

    if r_max_km is None:
        return None
    return np.full(len(magnitudes), float(r_max_km))


def _search_within_radius(
    lat_rad, lon_rad, times, parent_weight, radii_km, d, r_min_km, chunk
):
    """Padre de cada evento buscando sólo entre los que caen dentro del radio.

    Un `BallTree` con métrica haversine trae, para cada hijo, los eventos a
    menos del radio máximo; después se filtra el que el padre sí alcanza y se
    minimiza eta igual que en la búsqueda completa. Las distancias de los
    candidatos se recalculan con `distances_to_point` (la misma fórmula que
    `epicentral_distances`) para que, con un radio que lo abarque todo, el
    resultado sea idéntico al de la búsqueda sin tope.
    """
    from sklearn.neighbors import BallTree

    n = len(times)
    parent = np.full(n, -1, dtype=int)
    eta = np.full(n, np.nan)
    t_parent = np.full(n, np.nan)
    r_parent = np.full(n, np.nan)

    coordinates = np.column_stack([lat_rad, lon_rad])
    tree = BallTree(coordinates, metric="haversine")
    # Un poco de margen: el filtro exacto lo hace `radii_km` más abajo, y esto
    # sólo evita que el árbol deje afuera un candidato por redondeo.
    query_radius = float(radii_km.max()) * (1 + 1e-9) / EARTH_RADIUS_KM

    for start in range(0, n, chunk):
        end = min(start + chunk, n)
        found = tree.query_radius(coordinates[start:end], r=query_radius)

        for offset, indices in enumerate(found):
            j = start + offset
            # Ordenados, para que ante un empate gane el primero como en argmin.
            candidates = np.sort(indices)
            dt = times[j] - times[candidates]
            earlier = dt > 0
            candidates, dt = candidates[earlier], dt[earlier]
            if candidates.size == 0:
                continue

            distances = distances_to_point(
                lat_rad[candidates], lon_rad[candidates], lat_rad[j], lon_rad[j]
            )
            reachable = distances <= radii_km[candidates]
            if not reachable.any():
                continue

            candidates, dt, distances = (
                candidates[reachable],
                dt[reachable],
                distances[reachable],
            )
            r = np.maximum(distances, r_min_km)
            values = dt * r**d * parent_weight[candidates]

            best = int(np.argmin(values))
            parent[j] = candidates[best]
            eta[j] = values[best]
            t_parent[j] = dt[best]
            r_parent[j] = r[best]

    return parent, eta, t_parent, r_parent


def _search_everywhere(lat_rad, lon_rad, times, parent_weight, d, r_min_km, chunk):
    """Búsqueda sin tope: cada evento contra todos los anteriores, por bloques."""
    n = len(times)
    parent = np.full(n, -1, dtype=int)
    eta = np.full(n, np.nan)
    t_parent = np.full(n, np.nan)
    r_parent = np.full(n, np.nan)

    for start in range(0, n, chunk):
        end = min(start + chunk, n)

        distances = epicentral_distances(lat_rad, lon_rad, start, end)
        dt = times[start:end, None] - times[None, :]
        earlier = dt > 0

        r = np.maximum(distances, r_min_km)
        candidates = np.where(earlier, dt * r**d * parent_weight[None, :], np.inf)

        chosen = np.argmin(candidates, axis=1)
        rows = np.arange(end - start)
        best = candidates[rows, chosen]

        # Los primeros eventos del catálogo pueden no tener ningún anterior.
        has_parent = np.isfinite(best)
        indices = np.arange(start, end)[has_parent]

        parent[indices] = chosen[has_parent]
        eta[indices] = best[has_parent]
        t_parent[indices] = dt[rows, chosen][has_parent]
        r_parent[indices] = r[rows, chosen][has_parent]

    return parent, eta, t_parent, r_parent


def nearest_neighbor(
    quakes: pd.DataFrame,
    b: float,
    d: float,
    mc: float,
    p: float = 0.5,
    magnitude_bin: float = MAGNITUDE_BIN,
    r_min_km: float = R_MIN_KM,
    chunk: int = DISTANCE_CHUNK,
    r_max_km: float | None = None,
    r_max_mode: str = "fixed",
    r_max_k: float = 3.0,
) -> pd.DataFrame:
    """Le asigna a cada evento el padre que minimiza eta.

    `quakes` tiene que venir de silver (ordenado por tiempo); acá se recorta a
    Mc antes de buscar nada, porque un evento incompleto no puede ser padre
    válido de nadie.

    `p` reparte el peso de la magnitud entre las dos mitades reescaladas:
    p = q = 0.5 es lo que usan Zaliapin & Ben-Zion y hace que las dos nubes del
    plano (T, R) queden a 45 grados.

    `r_max_km` pone un tope a la distancia epicentral padre-hijo: un par más
    lejos que eso no puede ser padre-hijo, y un evento sin ningún candidato
    dentro del radio queda como raíz (`eta` en NaN, igual que los primeros del
    catálogo). Sin tope, un sismo grande puede "reclamar" hijos a miles de km.
    `r_max_mode="gk"` usa un radio que depende de la magnitud del padre,
    `r_max_k · 10^(0.1238·M + 0.983)` km. Con `r_max_km=None` y modo "fixed" se
    busca entre todos, como siempre.
    """
    if not 0.0 <= p <= 1.0:
        raise ValueError(f"p must be between 0 and 1, got {p}.")

    catalog = trim_to_mc(quakes, mc, magnitude_bin).reset_index(drop=True)
    n = len(catalog)
    if n < 2:
        raise ValueError(f"Only {n} events remain above Mc={mc}: no pairs.")

    if not catalog["time"].is_monotonic_increasing:
        raise ValueError("The catalog must be sorted by time.")

    lat_rad = np.radians(catalog["latitude"].to_numpy(dtype=float))
    lon_rad = np.radians(catalog["longitude"].to_numpy(dtype=float))
    magnitudes = catalog["mag"].to_numpy(dtype=float)

    # Tiempo en años desde el primer evento: eta se define con t en años, y
    # trabajar en absoluto evita restar timestamps adentro del bucle.
    times = (
        (catalog["time"] - catalog["time"].iloc[0]).dt.total_seconds().to_numpy()
        / SECONDS_PER_YEAR
    )

    # 10^(-b*m) del *padre*: es constante por evento, así que se calcula una
    # sola vez y se reusa en todos los bloques.
    parent_weight = 10.0 ** (-b * magnitudes)

    radii = _parent_radii(magnitudes, r_max_km, r_max_mode, r_max_k)

    if radii is None:
        parent, eta, t_parent, r_parent = _search_everywhere(
            lat_rad, lon_rad, times, parent_weight, d, r_min_km, chunk
        )
    else:
        parent, eta, t_parent, r_parent = _search_within_radius(
            lat_rad, lon_rad, times, parent_weight, radii, d, r_min_km, chunk
        )

    parented = parent >= 0
    parent_mag = np.where(parented, magnitudes[parent], np.nan)

    # eta = T * R por construcción: se reparte el peso de la magnitud entre las
    # dos mitades en vez de recalcular nada.
    q = 1.0 - p
    scale = 10.0 ** (-b * parent_mag)
    t_rescaled = t_parent * scale**q
    r_rescaled = r_parent**d * scale**p

    # Se filtra por las que existen: `place` es opcional en silver.
    neighbors = catalog[
        [c for c in CARRIED_COLUMNS if c in catalog.columns]
    ].copy()
    neighbors["parent_id"] = np.where(parented, catalog["id"].to_numpy()[parent], None)
    neighbors["parent_mag"] = parent_mag
    neighbors["eta"] = eta
    neighbors["log10_eta"] = np.log10(eta, where=eta > 0, out=np.full(n, np.nan))
    neighbors["t_years"] = t_parent
    neighbors["r_km"] = r_parent
    neighbors["log10_T"] = np.log10(
        t_rescaled, where=t_rescaled > 0, out=np.full(n, np.nan)
    )
    neighbors["log10_R"] = np.log10(
        r_rescaled, where=r_rescaled > 0, out=np.full(n, np.nan)
    )

    orphans = int((~parented).sum())
    log.info(
        "Nearest neighbor over %d events (Mc=%.2f, b=%.3f, d=%.3f): "
        "%d without a parent, log10(eta) between %.2f and %.2f (median %.2f).",
        n,
        mc,
        b,
        d,
        orphans,
        np.nanmin(neighbors["log10_eta"]),
        np.nanmax(neighbors["log10_eta"]),
        np.nanmedian(neighbors["log10_eta"]),
    )

    return neighbors
