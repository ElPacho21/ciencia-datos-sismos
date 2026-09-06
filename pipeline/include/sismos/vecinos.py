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

from sismos import OUTPUT_DIR, particion
from sismos.geo import CHUNK_DISTANCIAS, distancias_epicentrales
from sismos.parametros import BIN_MAGNITUD, recortar_a_mc

log = logging.getLogger(__name__)

VECINOS_DIR = OUTPUT_DIR / "vecinos"

SEGUNDOS_POR_ANIO = 365.25 * 24 * 3600

# Dos eventos con el mismo epicentro darían r=0 y eta=0, y ese padre ganaría
# siempre. Se pone un piso del orden del error de localización.
R_MINIMO_KM = 0.01

COLUMNAS_ARRASTRADAS = ("id", "time", "latitude", "longitude", "depth", "mag")


def vecinos_path(**consulta) -> Path:
    return VECINOS_DIR / f"{particion(**consulta)}.parquet"


def vecinos_write(destino: Path, df: pd.DataFrame) -> None:
    destino.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(destino, index=False)


def vecinos_read(ruta: Path) -> pd.DataFrame:
    return pd.read_parquet(ruta)


def nearest_neighbor(
    sismos: pd.DataFrame,
    b: float,
    d: float,
    mc: float,
    p: float = 0.5,
    bin_magnitud: float = BIN_MAGNITUD,
    r_minimo_km: float = R_MINIMO_KM,
    chunk: int = CHUNK_DISTANCIAS,
) -> pd.DataFrame:
    """Le asigna a cada evento el padre que minimiza eta.

    `sismos` tiene que venir de silver (ordenado por tiempo); acá se recorta a
    Mc antes de buscar nada, porque un evento incompleto no puede ser padre
    válido de nadie.

    `p` reparte el peso de la magnitud entre las dos mitades reescaladas:
    p = q = 0.5 es lo que usan Zaliapin & Ben-Zion y hace que las dos nubes del
    plano (T, R) queden a 45 grados.
    """
    if not 0.0 <= p <= 1.0:
        raise ValueError(f"p tiene que estar entre 0 y 1, llegó {p}.")

    catalogo = recortar_a_mc(sismos, mc, bin_magnitud).reset_index(drop=True)
    n = len(catalogo)
    if n < 2:
        raise ValueError(f"Quedaron {n} eventos por encima de Mc={mc}: no hay pares.")

    if not catalogo["time"].is_monotonic_increasing:
        raise ValueError("El catálogo tiene que venir ordenado por tiempo.")

    lat_rad = np.radians(catalogo["latitude"].to_numpy(dtype=float))
    lon_rad = np.radians(catalogo["longitude"].to_numpy(dtype=float))
    magnitudes = catalogo["mag"].to_numpy(dtype=float)

    # Tiempo en años desde el primer evento: eta se define con t en años, y
    # trabajar en absoluto evita restar timestamps adentro del bucle.
    tiempos = (
        (catalogo["time"] - catalogo["time"].iloc[0]).dt.total_seconds().to_numpy()
        / SEGUNDOS_POR_ANIO
    )

    # 10^(-b*m) del *padre*: es constante por evento, así que se calcula una
    # sola vez y se reusa en todos los bloques.
    peso_padre = 10.0 ** (-b * magnitudes)

    padre = np.full(n, -1, dtype=int)
    eta = np.full(n, np.nan)
    t_padre = np.full(n, np.nan)
    r_padre = np.full(n, np.nan)

    for desde in range(0, n, chunk):
        hasta = min(desde + chunk, n)

        distancias = distancias_epicentrales(lat_rad, lon_rad, desde, hasta)
        dt = tiempos[desde:hasta, None] - tiempos[None, :]
        anteriores = dt > 0

        r = np.maximum(distancias, r_minimo_km)
        candidatos = np.where(anteriores, dt * r**d * peso_padre[None, :], np.inf)

        elegido = np.argmin(candidatos, axis=1)
        filas = np.arange(hasta - desde)
        mejor = candidatos[filas, elegido]

        # Los primeros eventos del catálogo pueden no tener ningún anterior.
        tiene_padre = np.isfinite(mejor)
        indices = np.arange(desde, hasta)[tiene_padre]

        padre[indices] = elegido[tiene_padre]
        eta[indices] = mejor[tiene_padre]
        t_padre[indices] = dt[filas, elegido][tiene_padre]
        r_padre[indices] = r[filas, elegido][tiene_padre]

    hay_padre = padre >= 0
    mag_padre = np.where(hay_padre, magnitudes[padre], np.nan)

    # eta = T * R por construcción: se reparte el peso de la magnitud entre las
    # dos mitades en vez de recalcular nada.
    q = 1.0 - p
    escala = 10.0 ** (-b * mag_padre)
    t_reescalado = t_padre * escala**q
    r_reescalado = r_padre**d * escala**p

    vecinos = catalogo[list(COLUMNAS_ARRASTRADAS)].copy()
    vecinos["parent_id"] = np.where(hay_padre, catalogo["id"].to_numpy()[padre], None)
    vecinos["parent_mag"] = mag_padre
    vecinos["eta"] = eta
    vecinos["log10_eta"] = np.log10(eta, where=eta > 0, out=np.full(n, np.nan))
    vecinos["t_anios"] = t_padre
    vecinos["r_km"] = r_padre
    vecinos["log10_T"] = np.log10(
        t_reescalado, where=t_reescalado > 0, out=np.full(n, np.nan)
    )
    vecinos["log10_R"] = np.log10(
        r_reescalado, where=r_reescalado > 0, out=np.full(n, np.nan)
    )

    huerfanos = int((~hay_padre).sum())
    log.info(
        "Vecino más cercano sobre %d eventos (Mc=%.2f, b=%.3f, d=%.3f): "
        "%d sin padre, log10(eta) entre %.2f y %.2f (mediana %.2f).",
        n,
        mc,
        b,
        d,
        huerfanos,
        np.nanmin(vecinos["log10_eta"]),
        np.nanmax(vecinos["log10_eta"]),
        np.nanmedian(vecinos["log10_eta"]),
    )

    return vecinos
