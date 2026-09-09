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
MAGNITUDE_BIN = 0.1

# Valores estándar de la literatura. b ronda 1.0 en casi todo el mundo; d es el
# que usan Zaliapin & Ben-Zion como dimensión fractal de los epicentros.
B_STANDARD = 1.0
D_STANDARD = 1.5

# Máxima curvatura sabe subestimar Mc; ésta es la corrección empírica de
# Woessner & Wiemer (2005).
MAXC_CORRECTION = 0.2

# Por debajo de esto no hay catálogo con el que trabajar.
MIN_EVENTS = 50



def trim_to_mc(
    quakes: pd.DataFrame, mc: float, magnitude_bin: float = MAGNITUDE_BIN
) -> pd.DataFrame:
    """Se queda con los eventos completos, los de magnitud >= Mc."""
    binned = np.round(quakes["mag"].to_numpy(dtype=float) / magnitude_bin)
    return quakes[binned * magnitude_bin >= mc - magnitude_bin / 4]


def fmd(magnitudes: np.ndarray, magnitude_bin: float = MAGNITUDE_BIN):
    """Distribución de frecuencia-magnitud, no acumulada y acumulada.

    Devuelve los centros de bin, cuántos eventos caen en cada uno y cuántos hay
    de esa magnitud para arriba (que es la forma en que se escribe
    Gutenberg-Richter).
    """
    m = np.round(magnitudes / magnitude_bin)
    indices = (m - m.min()).astype(int)
    non_cumulative = np.bincount(indices)
    centers = np.round((m.min() + np.arange(len(non_cumulative))) * magnitude_bin, 4)

    # Acumulada "de acá para arriba": se suma desde el final hacia atrás.
    cumulative = np.cumsum(non_cumulative[::-1])[::-1]

    return centers, non_cumulative, cumulative


def mc_maxc(
    centers: np.ndarray,
    non_cumulative: np.ndarray,
    correction: float = MAXC_CORRECTION,
) -> float:
    """Máxima curvatura: el bin más poblado de la FMD no acumulada.

    Es donde el catálogo deja de crecer y empieza a perder eventos: por encima
    de esa magnitud los registra a todos, por debajo se le escapan.
    """
    return float(centers[int(np.argmax(non_cumulative))] + correction)


def estimate(
    quakes: pd.DataFrame,
    magnitude_bin: float = MAGNITUDE_BIN,
    maxc_correction: float = MAXC_CORRECTION,
    min_events: int = MIN_EVENTS,
) -> dict:
    """Devuelve {mc, b, d} para el catálogo dado."""
    magnitudes = quakes["mag"].to_numpy(dtype=float)
    centers, non_cumulative, _ = fmd(magnitudes, magnitude_bin)

    mc = mc_maxc(centers, non_cumulative, maxc_correction)
    complete = trim_to_mc(quakes, mc, magnitude_bin)

    if len(complete) < min_events:
        raise ValueError(
            f"Only {len(complete)} events remain above Mc={mc:.2f} "
            f"(minimum {min_events}). Widen the window or lower minmagnitude."
        )

    log.info(
        "Mc=%.2f by maximum curvature | %d complete events out of %d | "
        "b=%.1f and d=%.1f (standard, not fitted).",
        mc,
        len(complete),
        len(quakes),
        B_STANDARD,
        D_STANDARD,
    )

    return {
        "mc": mc,
        "b": B_STANDARD,
        "d": D_STANDARD,
        "n_events_catalog": len(quakes),
        "n_events_complete": len(complete),
    }
