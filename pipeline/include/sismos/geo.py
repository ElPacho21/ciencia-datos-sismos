"""Distancias sobre la esfera, compartidas por los pasos que recorren pares.

Tanto la integral de correlación como la búsqueda del vecino más cercano
necesitan la matriz de distancias de todos contra todos, que no entra en
memoria: las dos van por bloques de filas con la misma función.
"""

from __future__ import annotations

import numpy as np

EARTH_RADIUS_KM = 6371.0

# Filas por bloque: la matriz completa de un catálogo de 20000 eventos son
# 4e8 distancias, unos 3 GB como float64.
DISTANCE_CHUNK = 256


def distances_to_point(
    lat_rad: np.ndarray, lon_rad: np.ndarray, lat0: float, lon0: float
) -> np.ndarray:
    """Haversine de un conjunto de epicentros contra uno solo, en km.

    Para medir la extensión de un cluster alrededor de su sismo principal no
    hace falta la matriz de todos contra todos: alcanza con una fila.
    """
    dlat = lat_rad - lat0
    dlon = lon_rad - lon0

    a = (
        np.sin(dlat / 2) ** 2
        + np.cos(lat0) * np.cos(lat_rad) * np.sin(dlon / 2) ** 2
    )
    return 2 * EARTH_RADIUS_KM * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0)))


def epicentral_distances(
    lat_rad: np.ndarray, lon_rad: np.ndarray, start: int, end: int
) -> np.ndarray:
    """Haversine de un bloque de filas contra todo el catálogo, en km.

    Distancia epicentral y no hipocentral: es la que usan Zaliapin & Ben-Zion,
    porque la profundidad tiene un error mucho mayor que el epicentro y
    ensuciaría el escaleo.

    Las latitudes y longitudes entran ya en radianes para no repetir la
    conversión en cada bloque.
    """
    lat_block = lat_rad[start:end, None]
    dlat = lat_block - lat_rad[None, :]
    dlon = lon_rad[start:end, None] - lon_rad[None, :]

    a = (
        np.sin(dlat / 2) ** 2
        + np.cos(lat_block) * np.cos(lat_rad[None, :]) * np.sin(dlon / 2) ** 2
    )
    return 2 * EARTH_RADIUS_KM * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0)))
