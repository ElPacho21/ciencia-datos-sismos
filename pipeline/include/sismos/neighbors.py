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
from sismos.geo import DISTANCE_CHUNK, epicentral_distances
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


def neighbors_path(**query) -> Path:
    return NEIGHBORS_DIR / f"{partition(**query)}.parquet"


def neighbors_write(destination: Path, df: pd.DataFrame) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(destination, index=False)


def neighbors_read(path: Path) -> pd.DataFrame:
    return pd.read_parquet(path)


def nearest_neighbor(
    quakes: pd.DataFrame,
    b: float,
    d: float,
    mc: float,
    p: float = 0.5,
    magnitude_bin: float = MAGNITUDE_BIN,
    r_min_km: float = R_MIN_KM,
    chunk: int = DISTANCE_CHUNK,
) -> pd.DataFrame:
    """Le asigna a cada evento el padre que minimiza eta.

    `quakes` tiene que venir de silver (ordenado por tiempo); acá se recorta a
    Mc antes de buscar nada, porque un evento incompleto no puede ser padre
    válido de nadie.

    `p` reparte el peso de la magnitud entre las dos mitades reescaladas:
    p = q = 0.5 es lo que usan Zaliapin & Ben-Zion y hace que las dos nubes del
    plano (T, R) queden a 45 grados.
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
