"""Features causales de cada secuencia y objetivos derivados.

Una fila del resumen es una secuencia sísmica y hasta ahora sólo se la describía
con lo que se sabe del sismo principal (`mainshock_mag`, `mainshock_depth`).
Acá se suman columnas que **existirían en el instante de predecir**, es decir,
calculables con lo que ya había pasado cuando ocurrió el sismo principal:

- La sismicidad previa en los alrededores (`n_prior_*`, `days_since_prior`,
  `prior_max_mag_365d`). No depende del declustering: se calcula sobre silver
  recortado a Mc, así que no hereda ninguna decisión del método.
- El contexto de la región (`region_mc`, `region_log10_eta0`), que cambia de
  una región a otra cuando se concatenan varias.
- `has_aftershock`, el objetivo de clasificación: `n_aftershocks > 0`.

Todo lo que sea *posterior* al sismo principal (conteos de réplicas en una
ventana, duración, extensión) es fuga y no va acá.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from sismos import OUTPUT_DIR, partition
from sismos.geo import EARTH_RADIUS_KM
from sismos.parameters import MAGNITUDE_BIN, trim_to_mc

log = logging.getLogger(__name__)

# Ventanas (en días) para contar la sismicidad previa. La de 2 días capta una
# secuencia en curso; la de 30, la actividad reciente; la de 365, el régimen.
PRIOR_WINDOWS_DAYS = (2, 30, 365)

# Radio por defecto de la "vecindad" del epicentro del sismo principal.
PRIOR_RADIUS_KM = 100.0

SECONDS_PER_DAY = 86400.0

FEATURES_DIR = OUTPUT_DIR / "features"


def features_path(seed, mainshock, method="", **query) -> Path:
    """El resumen ya enriquecido, en parquet, antes de publicarse como csv.

    `method` lleva el tope de distancia y el radio de la sismicidad previa: los
    dos cambian los valores, así que dos corridas que difieran en ellos no
    pueden compartir archivo.
    """
    return FEATURES_DIR / (
        f"features_{partition(**query)}{method}"
        f"_mainshock={mainshock}_seed={seed}.parquet"
    )


def prior_suffix(prior_radius_km=PRIOR_RADIUS_KM) -> str:
    """Sufijo de nombre: vacío con el radio por defecto, igual que `method_suffix`."""
    if float(prior_radius_km) == PRIOR_RADIUS_KM:
        return ""
    number = float(prior_radius_km)
    return f"_prior={int(number) if number.is_integer() else number}"


def _seconds(times: pd.Series) -> np.ndarray:
    """Segundos desde 1970 como float, sin depender de la resolución del dtype."""
    epoch = pd.Timestamp("1970-01-01", tz="UTC")
    return (times - epoch).dt.total_seconds().to_numpy(dtype=float)


def prior_seismicity(
    summary: pd.DataFrame,
    silver: pd.DataFrame,
    mc: float,
    starttime,
    prior_radius_km: float = PRIOR_RADIUS_KM,
    magnitude_bin: float = MAGNITUDE_BIN,
) -> pd.DataFrame:
    """Sismicidad previa al sismo principal de cada secuencia.

    Para cada fila de `summary` mira los eventos de `silver` (recortado a Mc) a
    menos de `prior_radius_km` del epicentro del sismo principal y **anteriores**
    a su `mainshock_time`. El propio sismo principal queda excluido. Devuelve un
    DataFrame con el mismo índice que `summary`:

    - `n_prior_{d}d`: cuántos hubo en los `d` días previos (d en 2, 30, 365).
    - `prior_max_mag_365d`: la mayor magnitud entre los de los últimos 365
      días. NaN **estructural** si no hubo ninguno.
    - `days_since_prior`: días desde el más reciente de la vecindad, sin límite
      de ventana. NaN **estructural** si no hubo ninguno desde `starttime`.
    - `prior_window_truncated`: True si la ventana de 365 días arranca antes
      que el catálogo, o sea que los conteos son un piso y no el valor real.
    """
    from sklearn.neighbors import BallTree

    catalog = trim_to_mc(silver, mc, magnitude_bin).reset_index(drop=True)
    start = pd.Timestamp(starttime)
    if start.tzinfo is None:
        start = start.tz_localize("UTC")
    start_s = (start - pd.Timestamp("1970-01-01", tz="UTC")).total_seconds()

    longest = max(PRIOR_WINDOWS_DAYS)
    out = {f"n_prior_{d}d": np.zeros(len(summary), dtype=int) for d in PRIOR_WINDOWS_DAYS}
    out["prior_max_mag_365d"] = np.full(len(summary), np.nan)
    out["days_since_prior"] = np.full(len(summary), np.nan)

    t0 = _seconds(summary["mainshock_time"])
    out["prior_window_truncated"] = t0 - longest * SECONDS_PER_DAY < start_s

    if len(catalog) and len(summary):
        times = _seconds(catalog["time"])
        magnitudes = catalog["mag"].to_numpy(dtype=float)
        ids = catalog["id"].to_numpy()
        tree = BallTree(
            np.radians(catalog[["latitude", "longitude"]].to_numpy(dtype=float)),
            metric="haversine",
        )
        centers = np.radians(
            summary[["mainshock_lat", "mainshock_lon"]].to_numpy(dtype=float)
        )
        # Margen mínimo: el filtro por tiempo y por id es exacto, y esto sólo
        # evita perder un evento que cae justo en el borde del radio.
        found = tree.query_radius(
            centers, r=prior_radius_km * (1 + 1e-9) / EARTH_RADIUS_KM
        )
        mainshock_ids = summary["mainshock_id"].to_numpy()

        for row, indices in enumerate(found):
            before = indices[(times[indices] < t0[row]) & (ids[indices] != mainshock_ids[row])]
            if before.size == 0:
                continue

            age = (t0[row] - times[before]) / SECONDS_PER_DAY
            out["days_since_prior"][row] = age.min()

            for d in PRIOR_WINDOWS_DAYS:
                out[f"n_prior_{d}d"][row] = int((age <= d).sum())

            recent = age <= longest
            if recent.any():
                out["prior_max_mag_365d"][row] = magnitudes[before][recent].max()

    return pd.DataFrame(out, index=summary.index)


def build_features(
    summary: pd.DataFrame,
    silver: pd.DataFrame,
    mc: float,
    log10_eta0: float,
    starttime,
    prior_radius_km: float = PRIOR_RADIUS_KM,
) -> pd.DataFrame:
    """Devuelve el resumen con las columnas nuevas **al final**.

    Las columnas que ya existían no se tocan ni cambian de lugar: lo que ya
    consume el notebook de análisis sigue funcionando igual.
    """
    prior = prior_seismicity(
        summary, silver, mc, starttime, prior_radius_km=prior_radius_km
    )

    out = summary.copy()
    for column in prior.columns:
        out[column] = prior[column]

    out["region_mc"] = float(mc)
    out["region_log10_eta0"] = float(log10_eta0)
    out["has_aftershock"] = out["n_aftershocks"] > 0

    # Una columna entera en nulo hace fallar la publicación sin decir por qué;
    # mejor avisar acá, con el motivo más probable.
    for column in prior.columns:
        if out[column].isna().all():
            log.warning(
                "Feature %r is null in every row: the window is too short or "
                "the neighborhood too small to ever contain a prior event.",
                column,
            )

    log.info(
        "Features: %d sequences | structural nulls: days_since_prior %d, "
        "prior_max_mag_365d %d | window truncated in %d.",
        len(out),
        int(out["days_since_prior"].isna().sum()),
        int(out["prior_max_mag_365d"].isna().sum()),
        int(out["prior_window_truncated"].sum()),
    )

    return out
