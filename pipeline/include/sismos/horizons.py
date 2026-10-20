"""Objetivos por horizonte: "parado a t horas del sismo principal, ¿cuántas réplicas más habrá?".

`n_aftershocks` cuenta las réplicas de *toda* la secuencia, así que cualquier
cosa que se sepa después del sismo principal es fuga para predecirlo. Acá la
pregunta se reformula con un instante de predicción explícito. Parados a `t`
horas de `mainshock_time`, se conoce lo que pasó hasta `t` (features) y se
quiere predecir lo que pasa entre `t` y `T` días (objetivo `y_aft`).

El formato es **largo**: una fila por (`cluster_id`, `horizon_hours`), con
`horizon_hours` como una feature más. `clusters_*.csv` no se ensancha. Al
entrenar hay que separar train y test **por `cluster_id`** (`GroupKFold`): las
filas de un mismo cluster comparten secuencia y, mezcladas entre los dos lados,
filtran información.

Qué es una réplica: la misma definición que `n_aftershocks` —un evento del
cluster, distinto del sismo principal, que no lo precede—. Por eso, si la
secuencia entera cabe en `T` y no está censurada, `n_aft_T == n_aftershocks`.

Censura: si `mainshock_time + T` pasa de `endtime`, el catálogo no alcanza para
ver toda la ventana del objetivo. Esas filas se **marcan** (`censored_T`), no
se borran.

Advertencias de causalidad: la pertenencia al cluster y `eta_0` se ajustan con
el catálogo completo, así que no son recomputables "hasta t"; y el sismo
principal es el mayor del cluster, que sólo se sabe una vez que ocurrió.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from sismos import OUTPUT_DIR, partition
from sismos.features import FEATURES_DIR

log = logging.getLogger(__name__)

HORIZONS_HOURS = (1, 6, 24, 72, 168)  # 1 h, 6 h, 24 h, 72 h, 7 d
HORIZON_DAYS = 30

HORIZONS_DIR = OUTPUT_DIR / "delivery"

COLUMNS = (
    "cluster_id",
    "horizon_hours",
    "n_aft_t",
    "max_aft_mag_t",
    "hours_since_last_t",
    "y_aft",
    "y_has_aft",
    "n_aft_T",
    "censored_T",
    "censored_t",
)


def horizon_suffix(horizon_days=HORIZON_DAYS) -> str:
    """Sufijo de nombre: vacío con T por defecto."""
    number = float(horizon_days)
    if number == HORIZON_DAYS:
        return ""
    return f"_T={int(number) if number.is_integer() else number}"


def horizons_path(seed, mainshock, method="", horizon_days=HORIZON_DAYS, **query) -> Path:
    return HORIZONS_DIR / (
        f"horizons_{partition(**query)}{method}{horizon_suffix(horizon_days)}"
        f"_mainshock={mainshock}_seed={seed}.csv"
    )


def horizons_cache_path(seed, mainshock, method="", horizon_days=HORIZON_DAYS, **query) -> Path:
    return FEATURES_DIR / (
        f"horizons_{partition(**query)}{method}{horizon_suffix(horizon_days)}"
        f"_mainshock={mainshock}_seed={seed}.parquet"
    )


def combined_horizons_path(tag, seed, mainshock, method="", horizon_days=HORIZON_DAYS) -> Path:
    return HORIZONS_DIR / (
        f"horizons_combined_{tag}{method}{horizon_suffix(horizon_days)}"
        f"_mainshock={mainshock}_seed={seed}.csv"
    )


def _aftershock_delays(events: pd.DataFrame, summary: pd.DataFrame) -> pd.DataFrame:
    """Una fila por réplica: su cluster, su magnitud y cuántas horas después del principal."""
    base = summary[["cluster_id", "mainshock_id", "mainshock_time"]]
    merged = events[["cluster_id", "id", "time", "mag"]].merge(base, on="cluster_id")
    merged = merged[merged["id"] != merged["mainshock_id"]]
    delay = (merged["time"] - merged["mainshock_time"]).dt.total_seconds() / 3600.0
    merged = merged.assign(delay_h=delay)
    # Los premonitores (delay < 0) no son réplicas: mismo criterio que n_foreshocks.
    return merged[merged["delay_h"] >= 0][["cluster_id", "mag", "delay_h"]]


def build_horizons(
    summary: pd.DataFrame,
    events: pd.DataFrame,
    endtime,
    horizons_hours=HORIZONS_HOURS,
    horizon_days: float = HORIZON_DAYS,
) -> pd.DataFrame:
    """Tabla larga de features y objetivo por (cluster, horizonte)."""
    horizon_hours_total = horizon_days * 24.0
    if any(h >= horizon_hours_total or h <= 0 for h in horizons_hours):
        raise ValueError(
            f"Every horizon must be in (0, T) with T={horizon_hours_total:g} h, "
            f"got {list(horizons_hours)}."
        )

    end = pd.Timestamp(endtime)
    if end.tzinfo is None:
        end = end.tz_localize("UTC")

    aft = _aftershock_delays(events, summary)
    clusters = summary[["cluster_id", "mainshock_time"]].reset_index(drop=True)
    ids = pd.Index(clusters["cluster_id"])

    until_T = aft[aft["delay_h"] <= horizon_hours_total]
    n_T = until_T.groupby("cluster_id").size().reindex(ids, fill_value=0).to_numpy()

    censored_T = (
        clusters["mainshock_time"] + pd.Timedelta(days=horizon_days) > end
    ).to_numpy()

    parts = []
    for t in horizons_hours:
        seen = aft[aft["delay_h"] <= t]
        grouped = seen.groupby("cluster_id").agg(
            n=("delay_h", "size"), max_mag=("mag", "max"), last=("delay_h", "max")
        )
        grouped = grouped.reindex(ids)

        n_t = grouped["n"].fillna(0).to_numpy().astype(int)
        # El sismo principal cuenta como el "último evento" si todavía no hubo réplicas.
        last = grouped["last"].fillna(0.0).to_numpy()

        y = n_T - n_t
        part = pd.DataFrame(
            {
                "cluster_id": clusters["cluster_id"].to_numpy(),
                "horizon_hours": t,
                "n_aft_t": n_t,
                "max_aft_mag_t": grouped["max_mag"].to_numpy(),
                "hours_since_last_t": t - last,
                "y_aft": y,
                "y_has_aft": y > 0,
                "n_aft_T": n_T,
                "censored_T": censored_T,
                "censored_t": (
                    clusters["mainshock_time"] + pd.Timedelta(hours=t) > end
                ).to_numpy(),
            }
        )
        parts.append(part)

    out = pd.concat(parts, ignore_index=True)[list(COLUMNS)]
    out = out.sort_values(["cluster_id", "horizon_hours"], kind="stable").reset_index(drop=True)

    log.info(
        "Horizons: %d rows (%d clusters x %d horizons) | censored by T: %d clusters | "
        "positives by horizon: %s",
        len(out),
        len(clusters),
        len(horizons_hours),
        int(censored_T.sum()),
        {int(t): int(g["y_has_aft"].sum()) for t, g in out.groupby("horizon_hours")},
    )
    return out


def check_invariants(horizons: pd.DataFrame, summary: pd.DataFrame, horizon_days=HORIZON_DAYS):
    """Levanta si la tabla viola alguna identidad que tiene que cumplir por construcción."""
    problems = []
    T_hours = horizon_days * 24.0

    ordered = horizons.sort_values(["cluster_id", "horizon_hours"])
    monotonic = ordered.groupby("cluster_id")["n_aft_t"].apply(
        lambda s: bool((s.diff().dropna() >= 0).all())
    )
    if not monotonic.all():
        problems.append(f"n_aft_t is not monotonic in {int((~monotonic).sum())} clusters")

    if not ((horizons["n_aft_t"] + horizons["y_aft"]) == horizons["n_aft_T"]).all():
        problems.append("n_aft_t + y_aft != n_aft_T")

    if (horizons["y_aft"] < 0).any():
        problems.append("negative y_aft")

    if not (horizons["hours_since_last_t"].between(0, horizons["horizon_hours"])).all():
        problems.append("hours_since_last_t outside [0, horizon_hours]")

    # La secuencia entera cabe en T y no está censurada: n_aft_T es n_aftershocks.
    joined = horizons.drop_duplicates("cluster_id").merge(
        summary[["cluster_id", "n_aftershocks", "duration_days", "mainshock_time"]],
        on="cluster_id",
    )
    # duration_days incluye premonitores, así que es una cota superior de la
    # distancia al último evento: si cabe en T, todas las réplicas caben.
    fits = (joined["duration_days"] * 24.0 <= T_hours) & ~joined["censored_T"]
    mismatch = joined[fits & (joined["n_aft_T"] != joined["n_aftershocks"])]
    if len(mismatch):
        problems.append(
            f"n_aft_T != n_aftershocks in {len(mismatch)} clusters that fit in T"
        )

    # Nulos: sólo max_aft_mag_t, y exactamente cuando no hubo réplicas hasta t.
    unexpected = [
        c for c in horizons.columns if c != "max_aft_mag_t" and horizons[c].isna().any()
    ]
    if unexpected:
        problems.append(f"unexpected nulls in {unexpected}")
    if not (horizons["max_aft_mag_t"].isna() == (horizons["n_aft_t"] == 0)).all():
        problems.append("max_aft_mag_t null does not match n_aft_t == 0")

    if problems:
        raise ValueError("Horizons invariants violated:\n  - " + "\n  - ".join(problems))
