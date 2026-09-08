"""Los tres parámetros que necesita la distancia de vecino más cercano.

    eta_ij = t_ij * (r_ij ** d) * 10 ** (-b * m_i)

De los tres, acá se estima **uno solo**:

- `mc`, la magnitud de completitud, por máxima curvatura sobre este catálogo.
  No se puede tomar de tabla porque depende de qué red registró la zona y en
  qué época, y errarle sesga todo lo que venga después.
- `b` y `d` se toman en sus valores estándar (1.0 y 1.5). Son los que usa la
  literatura cuando no se los ajusta, y para `d` es incluso más defendible que
  ajustarlo: el ajuste por integral de correlación necesita elegir un rango de
  escaleo, y esa elección es frágil — sobre California devolvía 0.9 cuando el
  valor publicado para esa región ronda 1.6.

Que `b` y `d` sean constantes no invalida el método: entran en eta como una
reescala, y el umbral que separa réplicas de fondo se ajusta después sobre la
distribución de eta que salga. Lo que sí se pierde es poder decir que los
parámetros son "de estos datos".
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

# El USGS publica las magnitudes en una grilla de 0.1 (con excepciones que se
# redondean), y el binning entra en el cálculo de Mc.
BIN_MAGNITUD = 0.1

# Valores estándar de la literatura. b ronda 1.0 en casi todo el mundo; d es el
# que usan Zaliapin & Ben-Zion como dimensión fractal de los epicentros.
B_ESTANDAR = 1.0
D_ESTANDAR = 1.5

# Máxima curvatura sabe subestimar Mc; ésta es la corrección empírica de
# Woessner & Wiemer (2005).
CORRECCION_MAXC = 0.2

# Por debajo de esto no hay catálogo con el que trabajar.
MIN_EVENTOS = 50  
# TODO: Considero que hay que aumentar este valor, porque piden mínimo más de 1000 filas


def recortar_a_mc(
    sismos: pd.DataFrame, mc: float, bin_magnitud: float = BIN_MAGNITUD
) -> pd.DataFrame:
    """Se queda con los eventos completos, los de magnitud >= Mc."""
    binneadas = np.round(sismos["mag"].to_numpy(dtype=float) / bin_magnitud)
    return sismos[binneadas * bin_magnitud >= mc - bin_magnitud / 4]


def fmd(magnitudes: np.ndarray, bin_magnitud: float = BIN_MAGNITUD):
    """Distribución de frecuencia-magnitud, no acumulada y acumulada.

    Devuelve los centros de bin, cuántos eventos caen en cada uno y cuántos hay
    de esa magnitud para arriba (que es la forma en que se escribe
    Gutenberg-Richter).
    """
    m = np.round(magnitudes / bin_magnitud)
    indices = (m - m.min()).astype(int)
    no_acumulada = np.bincount(indices)
    centros = np.round((m.min() + np.arange(len(no_acumulada))) * bin_magnitud, 4)

    # Acumulada "de acá para arriba": se suma desde el final hacia atrás.
    acumulada = np.cumsum(no_acumulada[::-1])[::-1]

    return centros, no_acumulada, acumulada


def mc_maxc(
    centros: np.ndarray,
    no_acumulada: np.ndarray,
    correccion: float = CORRECCION_MAXC,
) -> float:
    """Máxima curvatura: el bin más poblado de la FMD no acumulada.

    Es donde el catálogo deja de crecer y empieza a perder eventos: por encima
    de esa magnitud los registra a todos, por debajo se le escapan.
    """
    return float(centros[int(np.argmax(no_acumulada))] + correccion)


def estimate(
    sismos: pd.DataFrame,
    bin_magnitud: float = BIN_MAGNITUD,
    correccion_maxc: float = CORRECCION_MAXC,
    min_eventos: int = MIN_EVENTOS,
) -> dict:
    """Devuelve {mc, b, d} para el catálogo dado."""
    magnitudes = sismos["mag"].to_numpy(dtype=float)
    centros, no_acumulada, _ = fmd(magnitudes, bin_magnitud)

    mc = mc_maxc(centros, no_acumulada, correccion_maxc)
    completos = recortar_a_mc(sismos, mc, bin_magnitud)

    if len(completos) < min_eventos:
        raise ValueError(
            f"Sólo quedan {len(completos)} eventos por encima de Mc={mc:.2f} "
            f"(mínimo {min_eventos}). Ampliá la ventana o bajá minmagnitude."
        )

    log.info(
        "Mc=%.2f por máxima curvatura | %d eventos completos de %d | "
        "b=%.1f y d=%.1f (estándar, no ajustados).",
        mc,
        len(completos),
        len(sismos),
        B_ESTANDAR,
        D_ESTANDAR,
    )

    return {
        "mc": mc,
        "b": B_ESTANDAR,
        "d": D_ESTANDAR,
        "n_eventos_catalogo": len(sismos),
        "n_eventos_completos": len(completos),
    }
