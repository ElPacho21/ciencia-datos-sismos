"""Distancias sobre la esfera, compartidas por los pasos que recorren pares.

Tanto la integral de correlación como la búsqueda del vecino más cercano
necesitan la matriz de distancias de todos contra todos, que no entra en
memoria: las dos van por bloques de filas con la misma función.
"""

from __future__ import annotations

import numpy as np

RADIO_TIERRA_KM = 6371.0

# Filas por bloque: la matriz completa de un catálogo de 20000 eventos son
# 4e8 distancias, unos 3 GB como float64.
CHUNK_DISTANCIAS = 256


def distancias_epicentrales(
    lat_rad: np.ndarray, lon_rad: np.ndarray, desde: int, hasta: int
) -> np.ndarray:
    """Haversine de un bloque de filas contra todo el catálogo, en km.

    Distancia epicentral y no hipocentral: es la que usan Zaliapin & Ben-Zion,
    porque la profundidad tiene un error mucho mayor que el epicentro y
    ensuciaría el escaleo.

    Las latitudes y longitudes entran ya en radianes para no repetir la
    conversión en cada bloque.
    """
    lat_bloque = lat_rad[desde:hasta, None]
    dlat = lat_bloque - lat_rad[None, :]
    dlon = lon_rad[desde:hasta, None] - lon_rad[None, :]

    a = (
        np.sin(dlat / 2) ** 2
        + np.cos(lat_bloque) * np.cos(lat_rad[None, :]) * np.sin(dlon / 2) ** 2
    )
    return 2 * RADIO_TIERRA_KM * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0)))
